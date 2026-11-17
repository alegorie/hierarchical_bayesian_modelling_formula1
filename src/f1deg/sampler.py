"""Chunked NUTS runner with exact sampler-state checkpoints.

Backend: NumPyro's low-level ``hmc()`` kernel (``init_kernel`` / ``sample_kernel``).
Every transition is a pure function of the ``HMCState`` named-tuple, which holds
*everything* the kernel needs: unconstrained position, its gradient and potential
energy, the current energy, iteration counter ``i``, the full adaptation state
(step size, dual-averaging state, Welford mass-matrix accumulator, adaptation
window index, adaptation RNG key) and the chain's PRNG key.  Advancing a chain by
k transitions with one ``lax.scan`` and then by m transitions with another is
bit-identical to a single scan of k+m transitions (tests/test_resume.py), so a
run can be stopped at any committed checkpoint and continued in a new process
without re-adapting, duplicating or skipping draws, or re-using random numbers.

What is NOT safe: interrupting inside a transition.  A chunk is the unit of
commit; after a crash the uncommitted tail (at most ``checkpoint_every``
transitions) is recomputed from the last committed state.
"""
from __future__ import annotations

import json
import os
import shutil
from functools import partial
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, random
import numpyro
from numpyro.infer.hmc import hmc
from numpyro.infer.initialization import init_to_median, init_to_uniform
from numpyro.infer.util import initialize_model

from .checkpoint import FitStore, LockError, atomic_savez, crash_point, pause_point
from .config import Config, ModelSpec, SamplerConfig, canonical_json, derive_seed, sha256_of
from .data import Dataset
from .model import ModelData, f1_model, hyper_sites, latent_site, model_data, model_spec_fingerprint_fields

LATENT_SITE = "z_inno"   # kept for backwards compatibility; the runner uses latent_site(spec)
STAT_NAMES = ["num_steps", "accept_prob", "energy", "potential_energy", "diverging", "step_size", "mean_accept_prob"]


# --------------------------------------------------------------------------- #
# state (de)serialisation
# --------------------------------------------------------------------------- #
def _is_key(x) -> bool:
    return jax.dtypes.issubdtype(getattr(x, "dtype", None), jax.dtypes.prng_key)


def state_to_arrays(state) -> Dict[str, np.ndarray]:
    leaves, treedef = jax.tree_util.tree_flatten(state)
    out: Dict[str, np.ndarray] = {"__treedef__": np.array(str(treedef)), "__n__": np.array(len(leaves))}
    for i, l in enumerate(leaves):
        if _is_key(l):
            out[f"leaf_{i:04d}__key"] = np.asarray(random.key_data(l))
        else:
            out[f"leaf_{i:04d}"] = np.asarray(l)
    return out


def arrays_to_state(arrays: Dict[str, np.ndarray], template):
    leaves_t, treedef = jax.tree_util.tree_flatten(template)
    if str(arrays["__treedef__"]) != str(treedef):
        raise ValueError("checkpoint state structure does not match the current model/sampler configuration")
    n = int(arrays["__n__"])
    assert n == len(leaves_t)
    leaves = []
    for i in range(n):
        if f"leaf_{i:04d}__key" in arrays:
            leaves.append(random.wrap_key_data(jnp.asarray(arrays[f"leaf_{i:04d}__key"])))
        else:
            a = arrays[f"leaf_{i:04d}"]
            leaves.append(jnp.asarray(a, dtype=leaves_t[i].dtype))
    return jax.tree_util.tree_unflatten(treedef, leaves)


def states_equal(s1, s2) -> Tuple[bool, List[str]]:
    l1 = jax.tree_util.tree_leaves(s1); l2 = jax.tree_util.tree_leaves(s2)
    bad = []
    for k, (a, b) in enumerate(zip(l1, l2)):
        a = np.asarray(random.key_data(a)) if _is_key(a) else np.asarray(a)
        b = np.asarray(random.key_data(b)) if _is_key(b) else np.asarray(b)
        if not np.array_equal(a, b, equal_nan=True):
            bad.append(f"leaf {k}")
    return (len(bad) == 0), bad


# --------------------------------------------------------------------------- #
# initialisation
# --------------------------------------------------------------------------- #
def _init_with_states(site=None, alpha0=None, base=None):
    if site is None:
        return partial(_init_with_states, alpha0=alpha0, base=base)
    if site["name"] == "alpha_lap":
        return alpha0
    return base(site)


def make_init_strategy(spec: ModelSpec, ds: Dataset, kind: str = "median"):
    """Prior-median initialisation for all sites; for the centred latent states a crude
    data-based starting point y - gamma_ext * fuel (changes only where the chains start).
    NumPyro expects a functools.partial."""
    base_init = init_to_median(num_samples=15) if kind == "median" else init_to_uniform
    if latent_site(spec) != "alpha_lap":
        return base_init
    alpha0 = jnp.asarray(np.asarray(ds.y) - spec.gamma_fixed_value * np.asarray(ds.fuel_c))
    return partial(_init_with_states, alpha0=alpha0, base=base_init)


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
class ChunkedNUTS:
    def __init__(self, cfg: Config, sampler: SamplerConfig, spec: ModelSpec, ds: Dataset, fit_name: str,
                 fits_root: Path, log: Callable[[str], None] = print, keep_sites: Optional[List[str]] = None):
        self.cfg, self.sampler, self.spec, self.ds = cfg, sampler, spec, ds
        self.fit_name = fit_name
        self.log = log
        self.md: ModelData = model_data(ds)
        self.keep_sites = keep_sites
        self.fingerprint, self.fingerprint_fields = self._fingerprint()
        self.fit_id = f"{fit_name}__{self.fingerprint[:12]}"
        self.store = FitStore(Path(fits_root), self.fit_id)
        self._kernel = None

    # ------------------------------------------------------------ identity
    def _fingerprint(self) -> Tuple[str, Dict[str, Any]]:
        fields = {
            "data": self.ds.fingerprint,
            "model": model_spec_fingerprint_fields(self.spec),
            "sampler": self.cfg.sampler_fingerprint_fields(self.sampler),
            "versions": {k: v for k, v in self.cfg.versions().items() if k in ("jax", "numpyro", "numpy", "python")},
            "hyper_block": hyper_sites(self.spec) if self.sampler.dense_mass_hyper else [],
            "runner": "ChunkedNUTS-v1",
        }
        return sha256_of(fields), fields

    def chain_seed(self, c: int) -> int:
        return derive_seed(self.cfg.seed, "chain", self.fit_name, self.spec.name, c)

    # ------------------------------------------------------------ kernel
    def _build_kernel(self):
        if self._kernel is not None:
            return self._kernel
        spec, md, sc = self.spec, self.md, self.sampler
        init_strategy = make_init_strategy(spec, self.ds, sc.init_strategy)
        lat = latent_site(spec)
        dense = [tuple(hyper_sites(spec))] if sc.dense_mass_hyper else False
        param_infos, pot_fn, post_fn = [], None, None
        for c in range(sc.chains):
            key = random.key(self.chain_seed(c))
            k_init, _ = random.split(key)
            mi, pot_fn, post_fn, _ = initialize_model(k_init, f1_model, model_args=(md, spec), init_strategy=init_strategy)
            param_infos.append(mi)
        init_kernel, sample_kernel = hmc(potential_fn=pot_fn, algo="NUTS")
        states = []
        for c in range(sc.chains):
            key = random.key(self.chain_seed(c))
            _, k_run = random.split(key)
            st = init_kernel(param_infos[c], num_warmup=sc.warmup, target_accept_prob=sc.target_accept, max_tree_depth=sc.max_treedepth,
                             dense_mass=dense, find_heuristic_step_size=sc.find_heuristic_step_size, rng_key=k_run)
            states.append(st)
        keep = self.keep_sites
        unravel_names = None

        def collect(st):
            cons = post_fn(st.z)
            latent = cons.pop(lat, None)
            if keep is not None:
                cons = {k: v for k, v in cons.items() if k in keep}
            stats = {"num_steps": st.num_steps, "accept_prob": st.accept_prob, "energy": st.energy, "potential_energy": st.potential_energy,
                     "diverging": st.diverging, "step_size": st.adapt_state.step_size, "mean_accept_prob": st.mean_accept_prob, "i": st.i}
            return cons, latent, stats

        def chunk(state, n):
            def body(s, _):
                s2 = sample_kernel(s)
                return s2, collect(s2)
            return lax.scan(body, state, None, length=n)

        self._kernel = {"states": states, "chunk": chunk, "post_fn": post_fn, "pot_fn": pot_fn, "sample_kernel": sample_kernel}
        self._jit_cache: Dict[Tuple[int, str], Callable] = {}
        return self._kernel

    def _chunk_fn(self, n: int):
        key = (n, self.sampler.chain_method)
        if key not in self._jit_cache:
            chunk = self._kernel["chunk"]
            f = lambda s: chunk(s, n)
            if self.sampler.chain_method == "sequential":
                self._jit_cache[key] = jax.jit(f)
            elif self.sampler.chain_method == "vectorized":
                self._jit_cache[key] = jax.jit(jax.vmap(f))
            elif self.sampler.chain_method == "parallel":
                if jax.local_device_count() < self.sampler.chains:
                    raise RuntimeError(f"chain_method=parallel needs >= {self.sampler.chains} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={self.sampler.chains} before importing jax")
                self._jit_cache[key] = jax.pmap(f)
            else:
                raise ValueError(self.sampler.chain_method)
        return self._jit_cache[key]

    def _advance(self, states, n):
        """Advance all chains by n transitions; returns (states, outputs per chain)."""
        f = self._chunk_fn(n)
        if self.sampler.chain_method == "sequential":
            new_states, outs = [], []
            for s in states:
                s2, o = f(s)
                new_states.append(s2); outs.append(o)
            return new_states, outs
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
        s2, o = f(stacked)
        C = len(states)
        new_states = [jax.tree_util.tree_map(lambda a: a[c], s2) for c in range(C)]
        outs = [jax.tree_util.tree_map(lambda a: a[c], o) for c in range(C)]
        return new_states, outs

    # ------------------------------------------------------------ preflight
    def preflight(self, out_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Compile, time two transitions, size the state, test write/read/restore,
        check disk space.  Numbers are *estimates* (early transitions are cheap)."""
        t0 = time.time()
        K = self._build_kernel()
        t_build = time.time() - t0
        sc = self.sampler
        st = K["states"][0]
        n_params = int(sum(np.prod(np.shape(l)) for l in jax.tree_util.tree_leaves(st.z)))
        arrays = state_to_arrays(st)
        state_bytes = int(sum(a.nbytes for a in arrays.values()))
        t0 = time.time(); f = self._chunk_fn(2); s2, o = f(st) if sc.chain_method == "sequential" else (None, None)
        if s2 is not None:
            jax.block_until_ready(s2)
        t_compile_and_2 = time.time() - t0
        t0 = time.time(); s3, o3 = f(st); jax.block_until_ready(s3); t_2 = time.time() - t0
        # write / read / restore round trip
        tmpdir = Path(tempfile.mkdtemp(prefix="f1deg_preflight_"))
        try:
            p = tmpdir / "state.npz"
            atomic_savez(p, arrays)
            loaded = arrays_to_state(dict(np.load(p, allow_pickle=False)), st)
            ok, bad = states_equal(st, loaded)
            # advancing the restored state must give the same result
            s4, _ = f(loaded); jax.block_until_ready(s4)
            ok2, bad2 = states_equal(s3, s4)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        du = shutil.disk_usage(str(self.store.dir.parent if self.store.dir.parent.exists() else Path(".")))
        per_draw_bytes = 8 * sum(int(np.prod(np.shape(v))) for k, v in jax.tree_util.tree_map(lambda a: a[0], o3[0]).items()) if sc.chain_method == "sequential" else None
        latent_bytes_per_saved_draw = 4 * int(len(self.ds.inno_idx)) if sc.latent_dtype == "float32" else 8 * int(len(self.ds.inno_idx))
        est_posterior_bytes = sc.chains * sc.draws * ((per_draw_bytes or 0) + latent_bytes_per_saved_draw / max(1, sc.save_latent_thin))
        info = {
            "n_free_parameters": n_params, "state_bytes_per_chain": state_bytes, "build_seconds": round(t_build, 2),
            "compile_plus_2_transitions_seconds": round(t_compile_and_2, 2), "two_transitions_seconds": round(t_2, 3),
            "rough_seconds_per_transition_early": round(t_2 / 2, 4),
            "note": "early transitions use small trees; per-transition cost typically grows 10-100x after adaptation",
            "write_read_restore_bitwise": bool(ok and ok2), "restore_mismatch": bad + bad2,
            "disk_free_gb": round(du.free / 1e9, 2), "estimated_posterior_gb": round(est_posterior_bytes / 1e9, 3),
            "estimated_checkpoint_gb_per_generation": round(sc.chains * state_bytes / 1e9, 4),
            "ram_available_gb": _ram_available_gb(), "devices": [str(d) for d in jax.devices()], "x64": bool(jax.config.jax_enable_x64),
            "chain_method": sc.chain_method, "fingerprint": self.fingerprint, "fit_id": self.fit_id,
        }
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"preflight_{self.fit_name}.json", "w") as fh:
                json.dump(info, fh, indent=1)
        return info

    # ------------------------------------------------------------ run / resume
    def run(self, resume: bool = True, force_refit: bool = False, max_seconds: Optional[float] = None, progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        sc = self.sampler
        total = sc.warmup + sc.draws
        manifest = {"fit_id": self.fit_id, "fit_name": self.fit_name, "fingerprint": self.fingerprint, "fingerprint_fields": self.fingerprint_fields,
                    "chains": sc.chains, "warmup": sc.warmup, "draws_target": sc.draws, "checkpoint_every": sc.checkpoint_every,
                    "save_latent_thin": sc.save_latent_thin, "latent_dtype": sc.latent_dtype, "model": self.spec.to_dict(),
                    "data_sizes": self.ds.sizes(), "versions": self.cfg.versions(), "precision": self.cfg.precision}
        if force_refit and self.store.exists():
            self.log(f"[{self.fit_name}] FORCE_REFIT: removing {self.store.dir}")
            shutil.rmtree(self.store.dir)
        self.store.create(manifest)
        self.store.acquire_lock()
        try:
            return self._run_locked(resume, total, max_seconds, progress)
        finally:
            self.store.release_lock()

    def _run_locked(self, resume, total, max_seconds, progress):
        sc = self.sampler
        K = self._build_kernel()
        rec = self.store.latest_commit() if resume else None
        removed = self.store.cleanup_orphans(rec)
        if removed:
            self.log(f"[{self.fit_name}] removed {len(removed)} uncommitted/orphan files: {removed[:4]}{'...' if len(removed) > 4 else ''}")
        if rec is None:
            states = K["states"]
            i, gen = 0, 0
            new_files = {}
            for c, st in enumerate(states):
                p = self.store.state_path(c, 0)
                new_files[self.store.rel(p)] = atomic_savez(p, state_to_arrays(st))
            rec = self.store.commit(0, 0, "init", new_files, None, {"draws_target": sc.draws})
            self.log(f"[{self.fit_name}] new run {self.fit_id}: {sc.chains} chains x ({sc.warmup}+{sc.draws}), checkpoint every {sc.checkpoint_every}")
            resumed = False
        else:
            i, gen = int(rec["iteration"]), int(rec["gen"])
            states = []
            for c in range(sc.chains):
                p = self.store.state_path(c, gen)
                states.append(arrays_to_state(dict(np.load(p, allow_pickle=False)), K["states"][c]))
            if i >= total:
                self.log(f"[{self.fit_name}] already complete at iteration {i} (gen {gen}); nothing to do")
                self.store.write_status({"status": "completed", "iteration": i, "gen": gen, "resumed": False})
                return {"status": "completed", "iteration": i, "gen": gen, "resumed": False, "fit_id": self.fit_id}
            self.log(f"[{self.fit_name}] RESUMING {self.fit_id} from committed generation {gen} at iteration {i}/{total} ({rec['phase']})")
            resumed = True
            man = self.store.read_manifest()
            if man.get("draws_target") != sc.draws:
                self.store.log_extension({"time": time.time(), "from_draws": man.get("draws_target"), "to_draws": sc.draws, "at_iteration": i})
                self.log(f"[{self.fit_name}] posterior target extended {man.get('draws_target')} -> {sc.draws} (kernel/adaptation unchanged)")
        t_start = time.time(); it_start = i
        last_hb = time.time()
        while i < total:
            boundary = sc.warmup if i < sc.warmup else total
            n = int(min(sc.checkpoint_every, boundary - i))
            phase = "warmup" if i < sc.warmup else "sampling"
            t0 = time.time()
            states, outs = self._advance(states, n)
            jax.block_until_ready(states)
            dt = time.time() - t0
            pause_point("after_advance", gen + 1, self.store.dir)   # test hook: new states in memory, nothing written
            new_files = {}
            for c in range(sc.chains):
                cons, latent, stats = outs[c]
                arrays = {f"stat__{k}": np.asarray(v) for k, v in stats.items()}
                if phase == "sampling":
                    for k, v in cons.items():
                        arrays[f"post__{k}"] = np.asarray(v)
                    if latent is not None:
                        it = np.arange(i, i + n)
                        keep = (it - sc.warmup) % sc.save_latent_thin == 0
                        arrays["latent__iter"] = it[keep]
                        arrays[f"latent__{latent_site(self.spec)}"] = np.asarray(latent)[keep].astype(sc.latent_dtype)
                    p = self.store.chunk_path(c, "draws", i, i + n)
                else:
                    p = self.store.chunk_path(c, "warmup", i, i + n)
                new_files[self.store.rel(p)] = atomic_savez(p, arrays)
                crash_point("after_chunk_write")
                ps = self.store.state_path(c, gen + 1)
                new_files[self.store.rel(ps)] = atomic_savez(ps, state_to_arrays(states[c]))
            crash_point("after_state_write")
            i += n; gen += 1
            rec = self.store.commit(gen, i, phase, new_files, rec, {"draws_target": sc.draws, "chunk_seconds": dt})
            div = int(sum(int(np.asarray(o[2]["diverging"]).sum()) for o in outs))
            with open(self.store.dir / "timing.jsonl", "a") as tf:   # append-only provenance (survives commit GC)
                tf.write(json.dumps({"gen": gen, "phase": phase, "iteration": i, "n": n, "chunk_seconds": round(dt, 3), "divergences": div, "time": time.time(), "pid": os.getpid()}) + "\n")
            msg = {"fit": self.fit_name, "phase": phase, "iteration": i, "total": total, "gen": gen, "chunk_seconds": round(dt, 2),
                   "sec_per_transition_per_chain": round(dt / (n * sc.chains), 4), "divergences_in_chunk": div,
                   "mean_depth": float(np.mean([np.log2(np.asarray(o[2]["num_steps"])).astype(int).mean() + 1 for o in outs])),
                   "eta_remaining_min": round((total - i) * dt / n / 60, 1), "resumed": resumed}
            self.log(f"[{self.fit_name}] {phase} {i}/{total} gen {gen} | {dt:.1f}s/chunk | ~{msg['eta_remaining_min']} min left | div {div}")
            if progress is not None:
                progress(msg)
            self.store.write_status({"status": "running", **msg})
            if time.time() - last_hb > sc.heartbeat_seconds:
                self.store.heartbeat(); last_hb = time.time()
            if max_seconds is not None and time.time() - t_start > max_seconds:
                self.store.write_status({"status": "paused_budget", **msg})
                self.log(f"[{self.fit_name}] time budget reached at iteration {i}; checkpoint committed, run again to continue")
                return {"status": "paused_budget", "iteration": i, "gen": gen, "resumed": resumed, "fit_id": self.fit_id}
        elapsed = time.time() - t_start
        status = {"status": "completed", "iteration": i, "gen": gen, "resumed": resumed, "fit_id": self.fit_id,
                  "wall_seconds_this_process": round(elapsed, 1), "transitions_this_process": i - it_start}
        self.store.write_status(status)
        self.log(f"[{self.fit_name}] completed: {i} iterations/chain in {elapsed/60:.1f} min (this process)")
        return status

    # ------------------------------------------------------------ reference (tests)
    def run_unchunked_reference(self) -> Tuple[List[Any], List[Any]]:
        """Run every chain with ONE lax.scan over warmup+draws transitions from the
        initial states (no checkpoints).  Used by the tests to show that chunking
        does not change the Markov chain (bitwise)."""
        K = self._build_kernel()
        total = self.sampler.warmup + self.sampler.draws
        f = jax.jit(lambda s: K["chunk"](s, total))
        states, outs = [], []
        for s in K["states"]:
            s2, o = f(s)
            jax.block_until_ready(s2)
            states.append(s2); outs.append(jax.tree_util.tree_map(np.asarray, o))
        return states, outs

    def final_states(self):
        """Committed final sampler states (for equality checks)."""
        K = self._build_kernel()
        rec = self.store.latest_commit()
        return [arrays_to_state(dict(np.load(self.store.state_path(c, rec["gen"]), allow_pickle=False)), K["states"][c]) for c in range(self.sampler.chains)]

    # ------------------------------------------------------------ loading
    def load_posterior(self, verify: bool = True) -> "PosteriorDraws":
        rec = self.store.latest_commit(verify_all=verify)
        if rec is None:
            raise FileNotFoundError(f"no valid commit in {self.store.dir}")
        return load_posterior_from_store(self.store, rec, self.sampler.warmup)


class PosteriorDraws:
    """Posterior draws assembled from committed chunks: dict site -> (chains, draws, ...)."""

    def __init__(self, posterior: Dict[str, np.ndarray], sample_stats: Dict[str, np.ndarray], latent: Optional[np.ndarray], latent_iters: Optional[np.ndarray],
                 warmup_stats: Dict[str, np.ndarray], meta: Dict[str, Any]):
        self.posterior, self.sample_stats, self.latent, self.latent_iters, self.warmup_stats, self.meta = posterior, sample_stats, latent, latent_iters, warmup_stats, meta

    @property
    def n_chains(self) -> int:
        return int(next(iter(self.posterior.values())).shape[0])

    @property
    def n_draws(self) -> int:
        return int(next(iter(self.posterior.values())).shape[1])

    def flat(self, name: str) -> np.ndarray:
        a = self.posterior[name]
        return a.reshape((-1,) + a.shape[2:])

    def to_inference_data(self, coords: Optional[Dict[str, Any]] = None, dims: Optional[Dict[str, List[str]]] = None):
        import arviz as az

        return az.from_dict({"posterior": self.posterior, "sample_stats": self.sample_stats}, coords=coords, dims=dims)


def load_posterior_from_store(store: FitStore, rec: Dict[str, Any], warmup: int) -> PosteriorDraws:
    chains = int(json.load(open(store.manifest_path))["chains"])
    post: Dict[str, List[np.ndarray]] = {}
    stats: Dict[str, List[np.ndarray]] = {}
    wstats: Dict[str, List[np.ndarray]] = {}
    latents, liters = [], []
    for c in range(chains):
        parts = store.list_chunks(rec, c, "draws")
        expect = warmup
        p_c, s_c, l_c, li_c = {}, {}, [], []
        for s, e, path in parts:
            if s != expect:
                raise ValueError(f"chain {c}: non-contiguous chunks ({expect} expected, found {s}); refusing to assemble")
            expect = e
            d = np.load(path, allow_pickle=False)
            for k in d.files:
                if k.startswith("post__"):
                    p_c.setdefault(k[6:], []).append(d[k])
                elif k.startswith("stat__"):
                    s_c.setdefault(k[6:], []).append(d[k])
                elif k == "latent__iter":
                    li_c.append(d[k])
                elif k.startswith("latent__"):
                    l_c.append(d[k])
        for k, v in p_c.items():
            post.setdefault(k, []).append(np.concatenate(v))
        for k, v in s_c.items():
            stats.setdefault(k, []).append(np.concatenate(v))
        if l_c:
            latents.append(np.concatenate(l_c)); liters.append(np.concatenate(li_c))
        wparts = store.list_chunks(rec, c, "warmup")
        w_c: Dict[str, List[np.ndarray]] = {}
        for s, e, path in wparts:
            d = np.load(path, allow_pickle=False)
            for k in d.files:
                if k.startswith("stat__"):
                    w_c.setdefault(k[6:], []).append(d[k])
        for k, v in w_c.items():
            wstats.setdefault(k, []).append(np.concatenate(v))
    n = min(len(v[0]) for v in post.values()) if post else 0
    posterior = {k: np.stack([a[:n] for a in v]) for k, v in post.items()}
    sample_stats = {k: np.stack([a[:n] for a in v]) for k, v in stats.items()}
    if "num_steps" in sample_stats:
        sample_stats["tree_depth"] = (np.log2(np.maximum(sample_stats["num_steps"], 1)).astype(int) + 1)
    warmup_stats = {k: np.stack([a[:min(len(x) for x in v)] for a in v]) for k, v in wstats.items()}
    latent = np.stack([l[:min(len(x) for x in latents)] for l in latents]) if latents else None
    latent_iters = liters[0][:latent.shape[1]] if latents else None
    meta = {"fit_id": store.fit_id, "gen": rec["gen"], "iteration": rec["iteration"], "phase": rec["phase"], "chains": chains, "draws": n}
    return PosteriorDraws(posterior, sample_stats, latent, latent_iters, warmup_stats, meta)


def _ram_available_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return round(int(line.split()[1]) / 1e6, 2)
    except Exception:
        return None
    return None

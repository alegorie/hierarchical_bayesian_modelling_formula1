"""Mode-driven orchestration used by the notebook and by scripts/run_pipeline.py.

Stages (each persisted with status + fingerprint, skipped when unchanged):
    data -> preflight -> fit[exp] -> diagnostics[exp] -> prior_predictive -> ppc[exp] -> loo[exp]
         -> baseline -> recovery -> forecast -> results[exp] -> export -> report

v1.1: the experiment MATRIX (cfg.experiments) is never filtered; `cfg.run_fits` selects the queue.
Whether a fit exists is always read from the stage registry / checkpoint store, never assumed
from the mode — the reviewer-response matrix and experiments_status.csv are built from that.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import baseline as B
from . import data as D
from . import diagnostics as DG
from . import export as EX
from . import loo as LOO
from . import paper as PAPER
from . import ppc as PPC
from . import recovery as RC
from . import report as RP
from . import results as RS
from . import synthetic as SY
from .config import Config, DataConfig, ExperimentSpec, ModelSpec, derive_seed, sha256_of, spec_fingerprint_dict
from .data import Dataset
from .link import mu_site, to_rate
from .model import hyper_sites, prior_table
from .stages import StageRegistry

POSTPROC_VERSION = 7   # bump when post-processing code (diagnostics/PPC/recovery tables/forecast/export/report) changes: fits are reused, derived stages recompute

# recovery design: (scenario) -> list of (suffix, spec overrides, well-specified?)   [the first entry is always the primary spec]
RECOVERY_FITS = {
    "baseline": [("est", {}, True)],
    "known_gamma": [("est", {}, True), ("fixed", "GAMMA_FIXED_AT_TRUTH", True)],
    "weak_signal": [("est", {}, True)],
    "no_interaction": [("est", {}, True), ("noint", {"interaction": False}, True)],
    "signed_rates": [("est", {}, False), ("signed", {"rate_link": "signed_stint"}, True)],
    "heavy_tails": [("est", {}, True)],
    "identity_rates": [("est", {}, False), ("ident", {"rate_link": "identity"}, True)],
    "identity_zero": [("est", {}, False), ("ident", {"rate_link": "identity"}, True)],
    "race_effect": [("est", {}, False), ("race", {"race_effect": True}, True)],
    "phase_confounded": [("est", {}, True)],
}


class Pipeline:
    def __init__(self, cfg: Config, log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.dirs = cfg.dirs()
        self.log_path = self.dirs["logs"] / "pipeline.log"
        self._user_log = log or print
        self.registry = StageRegistry(self.dirs["root"] / "stages.json", self.log)
        # the primary spec enters through the version-stable dictionary, so the run ID of an existing run does not
        # change merely because ModelSpec gained (unused) fields
        self.run_id = f"{cfg.mode}-{sha256_of({'seed': cfg.seed, 'data': cfg.data.__dict__, 'primary': spec_fingerprint_dict(cfg.primary, keep_name=True), 'sampler': cfg.sampler_fingerprint_fields(cfg.sampler)})[:8]}"
        self._datasets: Dict[str, Dataset] = {}
        self._raw = None
        self._runners: Dict[str, Any] = {}
        self.status_messages: List[Dict[str, Any]] = []
        self.problems: List[Dict[str, Any]] = []
        cfg.save(self.dirs["root"] / "config_used.yaml")

    # ------------------------------------------------------------ logging
    def log(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        self._user_log(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------ data
    def _dcfg(self, variant: Dict[str, Any]) -> DataConfig:
        v = dict(variant)
        years = v.pop("years", None)
        d = dataclasses.replace(self.cfg.data, **v)
        if years == "train":
            d = dataclasses.replace(d, years=list(self.cfg.forecast.train_years))
        elif years == "test":
            d = dataclasses.replace(d, years=list(self.cfg.forecast.test_years))
        elif years is not None:
            d = dataclasses.replace(d, years=list(years))
        return d

    def dataset(self, variant: Optional[Dict[str, Any]] = None) -> Dataset:
        variant = variant or {}
        key = sha256_of(variant)[:12]
        if key in self._datasets:
            return self._datasets[key]
        dcfg = self._dcfg(variant)
        if self.cfg.mode == "SMOKE":
            ds = SY.generate_from_config(self.cfg, "baseline", 0, dcfg=dcfg)
        else:
            if self._raw is None:
                p = Path(self.cfg.data_csv)
                if not p.exists():
                    raise FileNotFoundError(f"data file not found: {p}. Add f1_cache_combined_2022_2024.csv to data/ (see README) — nothing is fabricated.")
                self._raw = D.load_raw(p)
            clean_df, steps, excl = D.clean(self._raw, dcfg)
            ds = D.build_dataset(clean_df, self._raw, dcfg)
            ds.laps.attrs["steps"] = steps.to_dict(orient="list")
            ds.laps.attrs["excluded_races"] = excl["excluded_races"].to_dict(orient="list")
            ds.laps.attrs["excluded_short_stints"] = int(len(excl["excluded_short_stints"]))
        self._datasets[key] = ds
        return ds

    def stage_data(self) -> Dict[str, Any]:
        def fn():
            ds = self.dataset()
            tdir = self.dirs["tables"]
            out: Dict[str, Any] = {"sizes": ds.sizes(), "fingerprint": ds.fingerprint, "source": ds.source}
            ds.summary_counts().to_csv(tdir / "data_counts.csv", index=False)
            if self.cfg.mode == "SMOKE":
                steps = pd.DataFrame(ds.truth["cleaning_steps"])
                steps.to_csv(tdir / "data_cleaning_steps.csv", index=False)
                pd.DataFrame({"note": ["synthetic data: no real races excluded"]}).to_csv(tdir / "data_exclusions.csv", index=False)
            else:
                aud = D.audit_raw(self._raw)
                for k, v in aud.items():
                    v.to_csv(tdir / f"audit_{k}.csv", index=False)
                pd.DataFrame(ds.laps.attrs["steps"]).to_csv(tdir / "data_cleaning_steps.csv", index=False)
                pd.DataFrame(ds.laps.attrs["excluded_races"]).to_csv(tdir / "data_exclusions.csv", index=False)
                pd.DataFrame({"Team": list(self.cfg.data.team_map), "Constructor": list(self.cfg.data.team_map.values())}).to_csv(tdir / "data_team_map.csv", index=False)
                out["excluded_races"] = ds.laps.attrs["excluded_races"]
            defs = pd.DataFrame([
                ("TyreLife", "абсолютний вік комплекту (FastF1), 1 = перше коло на комплекті", "кола"),
                ("set_start_age", "TyreLife першого кола сирого стінта (>1 — вживаний/прогрітий комплект); не обнуляється центруванням", "кола"),
                ("first_retained_age", "TyreLife першого збереженого після очищення кола", "кола"),
                ("rel_age", "TyreLife − first_retained_age: внутрішньостінтовий «годинник» латентного стану", "кола"),
                ("dage", "приріст TyreLife між сусідніми збереженими колами (0 на першому); реальні Δa після вилучення кіл", "кола"),
                ("fuel_kg", "110·(L_r − lap)/(L_r − 1), L_r — максимальний номер кола гонки у сирих даних; лінійне вигорання — припущення (у т.ч. для скорочених гонок)", "кг"),
                ("fuel_c", "fuel_kg − 55 (детерміноване центрування на середину гонки)", "кг"),
                ("y", "LapTime − медіана LapTime очищених кіл гонки (ретроспективне центрування)", "с"),
                ("y_prefix", "LapTime − медіана очищених кіл першої чверті гонки (без майбутніх кіл; для прогнозу)", "с"),
                ("min_stint_laps", f"≥{self.cfg.data.min_stint_laps} ЗБЕРЕЖЕНИХ кіл (не обов'язково послідовних)", "кола"),
                ("компаунди", "SOFT/MEDIUM/HARD — відносні назви комплектів у межах вікенду; відповідність C1–C5 не відома й не вигадується", "—"),
            ], columns=["величина", "визначення", "одиниці"])
            defs.to_csv(tdir / "data_definitions.csv", index=False)
            ds.stints.to_csv(self.dirs["data"] / "stints.csv", index=False)
            ds.laps[["lap_index", "stint_id", "RaceKey", "Driver", "Constructor", "Compound", "LapNumber", "TyreLife", "rel_age", "dage", "LapTime_seconds", "y", "y_prefix", "fuel_c"]].to_csv(self.dirs["data"] / "laps_model_table.csv", index=False)
            out["tables"] = [str(p) for p in tdir.glob("data_*.csv")] + [str(p) for p in tdir.glob("audit_*.csv")]
            return out
        return self.registry.run("data", {"data": self.cfg.data.__dict__, "mode": self.cfg.mode, "seed": self.cfg.seed, "smoke": self.cfg.smoke}, fn)

    # ------------------------------------------------------------ fits
    def experiments(self) -> List[ExperimentSpec]:
        """The QUEUE of fits sampled in this run (a subset of the matrix `cfg.experiments`, which stays complete).

        `cfg.fits_override` (CLI --fits, notebook) wins in every mode; otherwise FULL uses `cfg.run_fits` (config.yaml)
        or, if that is empty, the whole matrix ordered by priority (mandatory first), then tier; SMOKE and PILOT use
        compact fixed designs.  Fits that are complete on disk are reported whether or not they are in the queue."""
        ex = {e.name: e for e in self.cfg.experiments}
        if self.cfg.fits_override is not None:
            names = list(self.cfg.fits_override)
        elif self.cfg.mode == "SMOKE":
            names = ["primary", "gauss", "gamma_fixed", "pooled", "signed_full", "unpooled", "primary_train", "gauss_train", "pooled_train", "unpooled_train"]
        elif self.cfg.mode == "PILOT":
            names = ["primary", "primary_train"]   # compact design: timing/diagnostics + one forecast training fit
        elif self.cfg.run_fits:
            names = list(self.cfg.run_fits)
        else:
            rank = {"mandatory": 0, "recommended": 1, "optional": 2, "": 1}
            names = [e.name for e in sorted(self.cfg.experiments, key=lambda e: (rank.get(e.priority, 1), {1: 0, 3: 1, 2: 2}.get(e.tier, 3)))]
        unknown = [n for n in names if n not in ex]
        if unknown:
            raise KeyError(f"unknown experiments in the queue: {unknown}; known: {sorted(ex)}")
        return [ex[n] for n in names]

    def primary_experiment(self) -> ExperimentSpec:
        """The experiment whose results fill the main tables/figures: the one named `primary` if it is in the
        matrix (queued or not), otherwise the first queued one."""
        for e in self.cfg.experiments:
            if e.name == "primary":
                return e
        return self.experiments()[0]

    def fit_state(self, exp: ExperimentSpec) -> Dict[str, Any]:
        """Evidence-based state of one experiment, read from the checkpoint store on disk (the stage registry is
        only a cache and is consulted for the derived stages).

        implemented  the spec exists in the matrix (always true here)
        started      the fit directory with a manifest and at least one commit record exists
        completed    the committed iteration has reached warmup + draws
        diagnosed    a diagnostics result for THIS fit exists; `gates_pass` says whether all operational gates were met
        usable       completed + diagnosed + 0 divergences + R-hat and ESS gates met (an E-BFMI failure alone leaves a
                     fit usable "with a caveat" and is reported separately)"""
        name = exp.name
        r = self.runner(exp)
        store = r.store
        total = int(r.sampler.warmup + r.sampler.draws)
        started = bool(store.exists() and store._commit_files())
        status = store.read_status() if started else {}
        iteration = int(status.get("iteration", 0) or 0)
        completed = bool(started and iteration >= total)
        dg = self.registry.result(f"diagnostics_{name}") if completed else None
        if dg and (self.registry.result(f"fit_{name}") or {}).get("fit_id", r.fit_id) != r.fit_id:
            dg = None                                         # the registry describes another fit of the same name
        gates = (dg or {}).get("gates") or {}
        energy = (dg or {}).get("energy") or {}
        g = self.cfg.gates
        conv_ok = bool(dg) and gates.get("worst_rhat") is not None and gates["worst_rhat"] < g["rhat_max"] and min(gates.get("min_ess_bulk") or 0, gates.get("min_ess_tail") or 0) >= g["ess_min"]
        return {"experiment": name, "in_queue": name in [e.name for e in self.experiments()], "started": started, "iteration": iteration, "total_iterations": total,
                "completed": completed, "diagnosed": bool(dg), "gates_pass": bool(gates.get("pass")) if dg else None, "failures": "; ".join(gates.get("failures", [])) if dg else "",
                "only_ebfmi_fails": bool(gates.get("only_ebfmi_fails")) if dg else None, "families_failed": list(gates.get("families_failed", [])) if dg else [],
                "worst_rhat": gates.get("worst_rhat"), "min_ess": min(gates.get("min_ess_bulk") or np.inf, gates.get("min_ess_tail") or np.inf) if dg else None,
                "divergences": energy.get("divergences"), "ebfmi_min": energy.get("ebfmi_min"), "sampling_wall_seconds": (dg or {}).get("sampling_wall_seconds"),
                "usable": bool(completed and dg and conv_ok and energy.get("divergences", 1) == 0), "fit_id": r.fit_id, "fit_dir": str(store.dir),
                "has_ppc": completed and self.registry.status(f"ppc_{name}") in ("completed", "needs_attention"),
                "has_loo": completed and self.registry.status(f"loo_{name}") in ("completed", "needs_attention")}

    def states(self) -> Dict[str, Dict[str, Any]]:
        return {e.name: self.fit_state(e) for e in self.cfg.experiments}

    def completed_experiments(self) -> List[ExperimentSpec]:
        """Every experiment of the MATRIX whose fit is complete on disk (queued in this run or not)."""
        return [e for e in self.cfg.experiments if self.fit_state(e)["completed"]]

    def _safe(self, fn, *args, **kw):
        """Run a post-processing stage; a failure is logged and recorded (reports/pipeline_problems.csv) instead of
        stopping the remaining stages.  Sampling stages are never wrapped."""
        try:
            return fn(*args, **kw)
        except Exception as err:  # noqa
            label = getattr(fn, "__name__", str(fn)) + "(" + ", ".join(getattr(a, "name", str(a)) for a in args) + ")"
            self.problems.append({"stage": label, "error": repr(err)})
            self.log(f"[pipeline] {label} FAILED and was skipped: {err!r}")
            return None

    def runner(self, exp: ExperimentSpec):
        from .sampler import ChunkedNUTS

        if exp.name in self._runners:
            return self._runners[exp.name]
        ds = self.dataset(exp.data_variant)
        sampler = dataclasses.replace(self.cfg.sampler, **exp.sampler)
        r = ChunkedNUTS(self.cfg, sampler, exp.model, ds, exp.name, self.dirs["fits"], log=self.log)
        self._runners[exp.name] = r
        return r

    def stage_preflight(self, exp: Optional[ExperimentSpec] = None) -> Dict[str, Any]:
        exp = exp or self.experiments()[0]
        r = self.runner(exp)

        def fn():
            info = r.preflight(self.dirs["diagnostics"])
            if not info["write_read_restore_bitwise"]:
                raise RuntimeError("preflight: write/read/restore of the sampler state is not bitwise exact — refusing to start a long run")
            return info
        return self.registry.run(f"preflight_{exp.name}", {"fp": r.fingerprint}, fn, deps=[])

    def stage_fit(self, exp: ExperimentSpec) -> Dict[str, Any]:
        r = self.runner(exp)
        max_s = float(self.cfg.budget.get("max_hours_per_fit", 12)) * 3600

        def fn():
            st = r.run(resume=self.cfg.resume, force_refit=self.cfg.force_refit, max_seconds=max_s, progress=lambda m: self.status_messages.append(m))
            st["fit_dir"] = str(r.store.dir)
            if st["status"] != "completed":
                st["_status"] = "running"
            return st
        res = self.registry.run(f"fit_{exp.name}", {"fp": r.fingerprint, "draws": r.sampler.draws}, fn, deps=[], force=self.cfg.force_refit,
                                outputs_exist=lambda res: Path(res.get("fit_dir", "")).exists())
        return res

    def posterior(self, exp: ExperimentSpec):
        return self.runner(exp).load_posterior(verify=False)

    @staticmethod
    def _release(runner) -> None:
        """Drop the compiled kernel of a finished fit and JAX's compilation caches.  Every model variant is
        jit-compiled separately; without this, a process that fits dozens of models (the recovery stage) grows by
        hundreds of megabytes per fit until the operating system kills it.  Stored draws are unaffected."""
        import gc
        import jax

        runner._kernel = None
        if hasattr(runner, "_jit_cache"):
            runner._jit_cache = {}
        try:
            jax.clear_caches()
        except Exception:  # pragma: no cover
            pass
        gc.collect()

    def stage_diagnostics(self, exp: ExperimentSpec) -> Dict[str, Any]:
        r = self.runner(exp)

        def fn():
            p = r.load_posterior(verify=True)
            ds = self.dataset(exp.data_variant)
            timing = DG.timing_from_store(r.store.dir)
            info = DG.full_diagnostics(p, ds.levels, self.cfg.gates, self.cfg.interval_prob, self.cfg.extra_interval_probs, self.dirs["diagnostics"], exp.name,
                                       wall_seconds=timing["sampling_seconds"] or None, max_treedepth=r.sampler.max_treedepth)
            info["timing"] = timing
            info["_status"] = "completed" if info["gates"]["pass"] else "needs_attention"
            return info
        return self.registry.run(f"diagnostics_{exp.name}", {"fp": r.fingerprint, "gates": self.cfg.gates, "draws": r.sampler.draws, "v": POSTPROC_VERSION}, fn, deps=[f"fit_{exp.name}"])

    def fit_with_bounded_extension(self, exp: ExperimentSpec) -> Dict[str, Any]:
        """Fit, diagnose; if only ESS/MCSE/R-hat gates fail (no divergences), extend the
        posterior draw target (compatible continuation) at most `max_extension_rounds`
        times.  Every round is logged; failures remain visible as needs_attention."""
        rounds = 0
        while True:
            fit = self.stage_fit(exp)
            if fit.get("status") != "completed":
                self.log(f"[{exp.name}] fit paused ({fit.get('status')}); rerun to continue")
                return {"fit": fit, "diagnostics": None, "extension_rounds": rounds}
            diag = self.stage_diagnostics(exp)
            g = diag["gates"]
            if g["pass"] or rounds >= int(self.cfg.budget.get("max_extension_rounds", 0)) or self.cfg.mode == "SMOKE":
                self._release(self.runner(exp))
                return {"fit": fit, "diagnostics": diag, "extension_rounds": rounds}
            if diag["energy"]["divergences"] > 0:
                self.log(f"[{exp.name}] gates failed with divergences — not extending (look for errors/parameterisation first)")
                self._release(self.runner(exp))
                return {"fit": fit, "diagnostics": diag, "extension_rounds": rounds}
            if g.get("only_ebfmi_fails"):
                # E-BFMI describes the energy distribution of the chains, not the Monte Carlo error: more draws cannot change it
                self.log(f"[{exp.name}] only E-BFMI fails — not extending (more draws do not change E-BFMI; see primary_centred)")
                self._release(self.runner(exp))
                return {"fit": fit, "diagnostics": diag, "extension_rounds": rounds}
            r = self.runner(exp)
            new_draws = int(round(r.sampler.draws * float(self.cfg.budget.get("extension_factor", 2.0))))
            self.log(f"[{exp.name}] extension round {rounds+1}: draws {r.sampler.draws} -> {new_draws} (gates: {g['failures']})")
            r.sampler = dataclasses.replace(r.sampler, draws=new_draws)
            rounds += 1

    # ------------------------------------------------------------ predictive checks
    def stage_prior_predictive(self, exp: Optional[ExperimentSpec] = None) -> Dict[str, Any]:
        exp = exp or self.experiments()[0]
        ds = self.dataset(exp.data_variant)
        n = 200 if self.cfg.mode != "SMOKE" else 60
        return self.registry.run(f"prior_predictive_{exp.name}", {"model": spec_fingerprint_dict(exp.model, keep_name=True), "data": ds.fingerprint, "n": n},
                                 lambda: PPC.prior_predictive(ds, exp.model, n, derive_seed(self.cfg.seed, "prior_pred", exp.name), self.dirs["ppc"], exp.name), deps=[])

    def stage_ppc(self, exp: ExperimentSpec) -> Dict[str, Any]:
        r = self.runner(exp)

        def fn():
            p = r.load_posterior(verify=False)
            ds = self.dataset(exp.data_variant)
            cond = PPC.ppc_conditional(ds, p, exp.model, r.sampler.warmup, self.cfg.ppc.draws, derive_seed(self.cfg.seed, "ppc_cond", exp.name), self.dirs["ppc"], exp.name, self.cfg.ppc.quantiles)
            rep = PPC.ppc_replicated_stints(ds, p, exp.model, self.cfg.ppc.draws, derive_seed(self.cfg.seed, "ppc_rep", exp.name), self.dirs["ppc"], exp.name)
            return {"conditional": cond, "replicated": rep}
        return self.registry.run(f"ppc_{exp.name}", {"fp": r.fingerprint, "n": self.cfg.ppc.draws, "draws": r.sampler.draws, "v": POSTPROC_VERSION}, fn, deps=[f"fit_{exp.name}"])

    def stage_loo(self, exp: ExperimentSpec) -> Dict[str, Any]:
        """PSIS-LOO / WAIC from the stored draws of a completed fit (no sampling)."""
        r = self.runner(exp)

        def fn():
            p = r.load_posterior(verify=False)
            ds = self.dataset(exp.data_variant)
            return LOO.loo_for_fit(ds, p, exp.model, r.sampler.warmup, self.dirs["loo"], exp.name)
        return self.registry.run(f"loo_{exp.name}", {"fp": r.fingerprint, "draws": r.sampler.draws, "v": POSTPROC_VERSION}, fn, deps=[f"fit_{exp.name}"])

    def stage_baseline(self) -> Dict[str, Any]:
        ds = self.dataset()
        gamma_ext = self.cfg.primary.gamma_fixed_value
        return self.registry.run("baseline_ols", {"data": ds.fingerprint, "gamma": gamma_ext}, lambda: B.ols_baseline(ds, gamma_ext, self.dirs["tables"], "ols"), deps=[])

    # ------------------------------------------------------------ synthetic recovery
    def recovery_scenarios(self) -> List[str]:
        rc = self.cfg.recovery
        if self.cfg.mode == "FULL":
            return [s for s in rc.scenarios if s in RECOVERY_FITS]
        return ["baseline", "known_gamma"] if self.cfg.mode == "PILOT" else ["baseline"]

    def stage_recovery(self, scenarios: Optional[List[str]] = None, replicates: Optional[int] = None, table_tag: str = "", replicate_start: int = 0) -> Dict[str, Any]:
        """Fit the model to data simulated by the independent generator and score the recovery of the truth.

        Every (scenario, replicate) is fitted with the primary specification and, where the scenario calls for
        it, with the matching variant (RECOVERY_FITS).  Fits whose specification does not contain the generating
        process (e.g. the log-link model on identity-link data) are flagged `well_specified = False`; they show
        what misspecification does and are excluded from the pooled coverage."""
        from .sampler import ChunkedNUTS

        rc = self.cfg.recovery
        scen = list(scenarios) if scenarios is not None else self.recovery_scenarios()
        reps = int(replicates) if replicates is not None else int(rc.replicates)
        results: Dict[str, Any] = {}
        for scenario in scen:
            for rep in range(int(replicate_start), int(replicate_start) + reps):
                ds = SY.generate_from_config(self.cfg, scenario, rep)
                for suffix, over, well in RECOVERY_FITS[scenario]:
                    if over == "GAMMA_FIXED_AT_TRUTH":
                        over = {"gamma": "fixed", "gamma_fixed_value": SY.SCENARIOS[scenario].gamma_fixed}
                    spec = self.cfg.primary.replace(name=f"rec_{scenario}_{rep}_{suffix}", **over)
                    sampler = dataclasses.replace(self.cfg.sampler, **rc.sampler)
                    r = ChunkedNUTS(self.cfg, sampler, spec, ds, spec.name, self.dirs["recovery"] / "fits", log=self.log)

                    def fn(r=r, ds=ds, spec=spec, scenario=scenario, rep=rep, suffix=suffix, well=well, sampler=sampler):
                        st = r.run(resume=self.cfg.resume, force_refit=self.cfg.force_refit)
                        if st["status"] != "completed":
                            return {"_status": "running", **st}
                        p = r.load_posterior(verify=False)
                        tab = RC.recovery_table(p, ds, self.cfg.interval_prob, tag=spec.name)
                        tab["variant"] = suffix; tab["replicate"] = rep; tab["well_specified"] = well
                        tab.to_csv(self.dirs["recovery"] / f"recovery_{spec.name}.csv", index=False)
                        idn = RC.identification_table(p, ds, spec, self.cfg.interval_prob)
                        if "table" in idn:
                            idn["table"].to_csv(self.dirs["recovery"] / f"identification_{spec.name}.csv", index=False)
                            idn["table"] = str(self.dirs["recovery"] / f"identification_{spec.name}.csv")
                        nu_pairs = (ds.truth["nu_stint"], np.median(p.flat("nu_stint"), axis=0))
                        np.savez(self.dirs["recovery"] / f"nu_pairs_{spec.name}.npz", truth=nu_pairs[0], est=nu_pairs[1])
                        sign = RS.sign_table(p, ds, self.cfg.interval_prob)
                        sign.to_csv(self.dirs["recovery"] / f"sign_{spec.name}.csv", index=False)
                        # sampler health of the short recovery chains (they are examples, not production fits)
                        div = int(p.sample_stats["diverging"].sum())
                        cov = tab[tab["element"] != ""]["covered"].mean() if (tab["element"] != "").any() else np.nan
                        return {"csv": str(self.dirs["recovery"] / f"recovery_{spec.name}.csv"), "identification": idn, "coverage_share_effects": float(cov),
                                "n_params": int(len(tab)), "scenario": scenario, "replicate": rep, "variant": suffix, "well_specified": well, "fit_id": r.fit_id,
                                "divergences": div, "chains": int(p.n_chains), "draws": int(p.n_draws),
                                "share_true_nu_negative": float((ds.truth["nu_stint"] < 0).mean()),
                                "scalar_z_abs_max": float(tab[tab["element"] == ""]["z"].abs().max())}
                    results[spec.name] = self.registry.run(f"recovery_{spec.name}", {"fp": r.fingerprint, "draws": sampler.draws, "v": POSTPROC_VERSION}, fn, deps=[])
                    self._release(r)
        # combined tables (only fits of THIS call; `table_tag` keeps a larger replicate study apart from the default design)
        frames = [pd.read_csv(v["csv"]) for v in results.values() if isinstance(v, dict) and "csv" in v]
        if frames:
            allrec = pd.concat(frames, ignore_index=True)
            allrec["element"] = allrec["element"].fillna("").astype(str)
            allrec.to_csv(self.dirs["tables"] / f"synthetic_recovery{table_tag}.csv", index=False)
            summ = RC.recovery_summary(allrec)
            summ.to_csv(self.dirs["tables"] / f"synthetic_recovery_summary{table_tag}.csv", index=False)
            health = pd.DataFrame([{"fit": k, **{kk: v.get(kk) for kk in ("scenario", "replicate", "variant", "well_specified", "divergences", "chains", "draws", "coverage_share_effects", "scalar_z_abs_max", "share_true_nu_negative")}}
                                   for k, v in results.items() if isinstance(v, dict) and "csv" in v])
            health.to_csv(self.dirs["tables"] / f"synthetic_recovery_fits{table_tag}.csv", index=False)
            # pooled coverage of the well-specified fits with its uncertainty (binomial; elements are correlated, so the SE is approximate)
            eff = allrec[(allrec["element"] != "") & allrec["well_specified"]]
            n = len(eff); k = int(eff["covered"].sum())
            cov = {"n_elements": n, "covered": k, "share": k / n if n else np.nan, "binomial_se": float(np.sqrt(k / n * (1 - k / n) / n)) if n else np.nan,
                   "n_fits": int(health["well_specified"].sum()), "nominal": self.cfg.interval_prob,
                   "note": "лише коректно специфіковані fits; приклад recovery (не SBC); елементи корельовані, тому SE приблизна"}
            json.dump(cov, open(self.dirs["tables"] / f"synthetic_recovery_coverage{table_tag}.json", "w"), indent=1)
            results["_coverage"] = cov
        return results

    # ------------------------------------------------------------ forecast
    def stage_forecast(self) -> Dict[str, Any]:
        fc = self.cfg.forecast
        matrix = {e.name: e for e in self.cfg.experiments}
        names = [n for n in fc.models if n in matrix]
        train_ds = self.dataset({"years": "train"})
        test_ds = self.dataset({"years": "test"})
        models = {}
        deps = []
        skipped = []
        for n in names:
            if not self.fit_state(matrix[n])["completed"]:
                skipped.append(n)
                continue
            r = self.runner(matrix[n])
            models[n] = (r.load_posterior(verify=False), matrix[n].model, train_ds.levels)
            deps.append(f"fit_{n}")
        if skipped:
            self.log(f"[forecast] training fits not completed (skipped, reported as not run): {skipped}")
        if not models:
            return {"_status": "failed", "note": "no completed training fits", "skipped_models": skipped}
        # y_prefix = y_full + (median_full - median_q1): shift the training-scale baseline prior by the mean training difference
        offset = float((train_ds.races["race_median_full"] - train_ds.races["race_median_q1"]).mean())
        max_stints = 60 if self.cfg.mode == "SMOKE" else (150 if self.cfg.mode == "PILOT" else None)

        def fn():
            seed = derive_seed(self.cfg.seed, "forecast")
            info = F.evaluate_models(test_ds, models, fc, self.cfg.primary.gamma_fixed_value, offset, self.dirs["forecast"], seed, log=self.log, max_stints=max_stints)
            info["train_offset_full_minus_q1_s"] = offset
            info["population"] = "clean green-flag laps of stints that survived the retrospective cleaning rules (conditional population, not all race situations)"
            info["train_years"] = fc.train_years; info["test_years"] = fc.test_years
            info["note"] = "retrospective transfer check (test year was part of the old study), not an untouched confirmatory hold-out"
            info["models_scored"] = sorted(models); info["models_not_run"] = skipped
            info["extra"] = {}
            for sch in fc.extra_schemes:
                tag = str(sch.get("tag", "_extra"))
                ex = F.evaluate_models(test_ds, models, fc, self.cfg.primary.gamma_fixed_value, offset, self.dirs["forecast"], seed, log=self.log, max_stints=max_stints,
                                       prefix_fraction=sch.get("prefix_fraction"), min_prefix_laps=sch.get("min_prefix_laps"), tag=tag)
                ex.pop("summary", None)
                info["extra"][tag] = ex
            return info
        from . import forecast as F  # noqa: E402 (late import keeps jax out of module import)
        return self.registry.run("forecast", {"models": sorted(models), "fc": fc.__dict__, "test": test_ds.fingerprint, "max_stints": max_stints, "v": POSTPROC_VERSION}, fn, deps=deps)

    # ------------------------------------------------------------ results tables
    def results_tables(self, exp: ExperimentSpec) -> Dict[str, Any]:
        r = self.runner(exp)
        p = r.load_posterior(verify=False)
        ds = self.dataset(exp.data_variant)
        prob = self.cfg.interval_prob
        out = {"global_rate": RS.global_rate(p, prob)}
        out.update(RS.compound_tables(p, ds, prob))
        out["tracks"] = RS.track_table(p, ds, prob)
        out["scales"] = RS.scales_table(p, ds, prob, spec=exp.model)
        out["interaction"] = RS.interaction_table(p, ds, prob)
        out["team"] = RS.group_contrast_table(p, ds, prob, "team")
        out["driver"] = RS.group_contrast_table(p, ds, prob, "driver")
        out["sigma_cmp"] = RS.sigma_comparison(p, prob)
        out["residual"] = RS.residual_table(p, prob)
        out["identification"] = RC.identification_table(p, ds, exp.model, prob)
        out["sign"] = RS.sign_table(p, ds, prob)
        rates = {}
        comps = ds.levels["compound"]
        C = p.flat("compound_effect"); mu = p.flat(mu_site(p.posterior))
        for i, c in enumerate(comps):
            rates[c] = to_rate(mu + C[:, i], p.posterior)
        out["rate_draws"] = rates
        return out

    def example_panels(self, exp: ExperimentSpec, picks: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        r = self.runner(exp); p = r.load_posterior(verify=False); ds = self.dataset(exp.data_variant)
        rng = np.random.default_rng(derive_seed(self.cfg.seed, "examples"))
        lo, hi = (1 - self.cfg.interval_prob) / 2, 1 - (1 - self.cfg.interval_prob) / 2
        panels = []
        W = r.sampler.warmup
        for j, why in picks:
            m = ds.stint_codes == j
            lap_idx = np.flatnonzero(m)
            inno_pos = {li: k for k, li in enumerate(ds.inno_idx)}
            fits, preds = [], []
            for c in range(p.latent.shape[0]):
                for t in range(p.latent.shape[1]):
                    d = int(p.latent_iters[t]) - W
                    pr = PPC._draw_params(p, c, d, exp.model)
                    if exp.model.process_noise and exp.model.latent_param == "centred":
                        alpha = np.asarray(p.latent[c, t, lap_idx], dtype=np.float64)
                    else:
                        alpha = pr["baseline_stint"][j] + pr["nu_stint"][j] * ds.rel_age[m]
                        if pr["sigma_eta"] is not None:
                            inc = np.array([np.sqrt(ds.dage[li]) * p.latent[c, t, inno_pos[li]] if li in inno_pos else 0.0 for li in lap_idx])
                            alpha = alpha + pr["sigma_eta"] * np.cumsum(inc)
                    gam = pr["gamma_track"][ds.track_of_stint[j]] if pr["gamma_track"] is not None else exp.model.gamma_fixed_value
                    fit = alpha + gam * ds.fuel_c[m]
                    fits.append(fit); preds.append(PPC._obs_sample(exp.model, fit, pr["sigma_eps_core"], pr["lam"], pr["eta"], rng))
            Fm = np.stack(fits); Pm = np.stack(preds)
            row = ds.stints.iloc[j]
            nu = p.flat("nu_stint")[:, j]
            panels.append({"age": ds.rel_age[m], "y": ds.y[m], "fit_med": np.median(Fm, 0), "fit_lo": np.quantile(Fm, lo, 0), "fit_hi": np.quantile(Fm, hi, 0),
                           "pred_lo": np.quantile(Pm, lo, 0), "pred_hi": np.quantile(Pm, hi, 0), "compound": row["Compound"],
                           "title": f"{row['Location']} {row['Year']} — {row['Driver']} ({row['Compound']}, стінт {int(row['Stint'])}) · {why}\nν_s = {np.median(nu):.3f} [{np.quantile(nu, lo):.3f}; {np.quantile(nu, hi):.3f}] с/коло"})
        return panels

    def forecast_example_panels(self, n: int = 4) -> List[Dict[str, Any]]:
        p = self.dirs["forecast"] / "forecast_laps.csv"
        if not p.exists():
            return []
        laps = pd.read_csv(p)
        test_ds = self.dataset({"years": "test"})
        model = [m for m in laps["model"].unique() if m != "ols_fuel_ext"]
        if not model:
            return []
        g = laps[(laps["model"] == model[0]) & (laps["scheme"] == "fixed_origin")]
        stints = g.groupby("stint_code")["lap_in_stint"].count().sort_values(ascending=False).index[:n]
        panels = []
        for j in stints:
            m = test_ds.stint_codes == j
            y = test_ds.laps.loc[m, "y_prefix"].to_numpy(float); age = test_ds.rel_age[m]
            gg = g[g["stint_code"] == j].sort_values("lap_in_stint")
            n_pre = int(gg["lap_in_stint"].min())
            fit_med = np.full(len(y), np.nan); lo = fit_med.copy(); hi = fit_med.copy()
            fit_med[n_pre:] = gg["pred_median"].to_numpy(); lo[n_pre:] = gg["pred_lo94"].to_numpy(); hi[n_pre:] = gg["pred_hi94"].to_numpy()
            row = test_ds.stints.iloc[j]
            panels.append({"age": age, "y": y, "fit_med": fit_med, "fit_lo": lo, "fit_hi": hi, "pred_lo": lo, "pred_hi": hi, "compound": row["Compound"], "split": n_pre,
                           "title": f"{row['Location']} {row['Year']} — {row['Driver']} ({row['Compound']}) · модель {model[0]}; префікс {n_pre} кіл, прогноз {len(y)-n_pre} кіл"})
        return panels

    # ------------------------------------------------------------ data-level sensitivity (no model fit)
    def stage_data_sensitivity(self) -> Dict[str, Any]:
        """How the cleaning threshold and the minimum stint length change the data and the fuel-adjusted OLS slopes
        (one factor at a time).  This needs no model fit; the model-level sensitivity is the job of the experiments
        clean_110 / clean_107 / minlaps_8 / minlaps_10.  The scan is what shows which thresholds are informative: 1.20 and 1.40
        change a few dozen laps only."""
        base = self.cfg.data
        variants = [("основний", {})] + [(f"поріг {t:.2f}", {"max_rel_laptime": t}) for t in (1.07, 1.10, 1.20, 1.40) if t != base.max_rel_laptime] + \
                   [(f"мін. {m} кіл", {"min_stint_laps": m}) for m in (8, 10) if m != base.min_stint_laps]
        gamma_ext = self.cfg.primary.gamma_fixed_value

        def fn():
            rows = []
            for label, var in variants:
                ds = self.dataset(var)
                S = ds.n_stints
                adj = PPC.grouped_ols_slopes(ds.stint_codes, ds.rel_age, ds.y - gamma_ext * ds.fuel_c, S)
                row = {"варіант": label, "поріг": var.get("max_rel_laptime", base.max_rel_laptime), "мін. кіл": var.get("min_stint_laps", base.min_stint_laps),
                       "кіл": ds.n_laps, "відрізків": S, "гонок": int(ds.stints["race"].nunique()), "max y, с": float(ds.y.max()),
                       "квантиль 0,99 y, с": float(np.quantile(ds.y, 0.99)), "частка y > 3 с": float(np.mean(ds.y > 3.0)),
                       "медіана МНК-нахилу з паливною поправкою, с/коло": float(np.nanmedian(adj)), "частка від'ємних нахилів": float(np.nanmean(adj < 0))}
                for i, c in enumerate(ds.levels["compound"]):
                    row[f"медіана {c}, с/коло"] = float(np.nanmedian(adj[ds.comp_of_stint == i]))
                rows.append(row)
            df = pd.DataFrame(rows)
            path = self.dirs["tables"] / "data_threshold_sensitivity.csv"
            df.to_csv(path, index=False)
            return {"csv": str(path), "rows": rows, "gamma_ext": gamma_ext}
        return self.registry.run("data_sensitivity", {"data": self.cfg.data.__dict__, "mode": self.cfg.mode, "gamma": gamma_ext, "smoke": self.cfg.smoke, "seed": self.cfg.seed, "v": 2}, fn, deps=[])

    # ------------------------------------------------------------ export + report
    def _data_label(self, e: ExperimentSpec) -> str:
        v = e.data_variant or {}
        if v.get("years") == "train":
            return "навчальні сезони"
        if not v:
            return "усі сезони"
        return "усі сезони; " + ", ".join(f"{k}={val}" for k, val in v.items())

    def _recovery_tables(self) -> Dict[str, Any]:
        """Recovery summaries on disk: the pipeline's own design (""), the larger replicate study ("_study"), the check
        on a larger synthetic data set ("_large") and on one of the size of the real data ("_realsize");
        scripts/run_recovery_study.py writes the last three under <outputs_root>/recovery_study/tables,
        <outputs_root>/recovery_large/tables and <outputs_root>/recovery_realsize/tables.  If the user has not run the
        first two, the tables shipped with the code (results_sandbox/, short chains, made during the revision) are
        used and `source` says so."""
        out = {"summary": {}, "coverage": {}, "all": {}, "source": {}}
        oroot = Path(self.cfg.outputs_root)
        shipped = Path(__file__).resolve().parents[2] / "results_sandbox"
        roots_of = {"": [self.dirs["tables"], oroot / "recovery_study" / "tables"],
                    "_study": [self.dirs["tables"], oroot / "recovery_study" / "tables", shipped / "recovery_study" / "tables"],
                    "_large": [self.dirs["tables"], oroot / "recovery_large" / "tables", oroot / "recovery_study_large" / "tables", shipped / "recovery_study_large" / "tables"],
                    "_realsize": [self.dirs["tables"], oroot / "recovery_realsize" / "tables"]}
        well = {(sc, suf): bool(w) for sc, fits in RECOVERY_FITS.items() for suf, _, w in fits}
        for label, roots in roots_of.items():
            for root in roots:
                p = root / f"synthetic_recovery_summary{label}.csv"
                a = root / f"synthetic_recovery{label}.csv"
                if p.exists():
                    out["summary"][label] = pd.read_csv(p)
                    out["source"][label] = "results_sandbox" if shipped in root.parents else "run"
                    if a.exists():
                        out["all"][label] = pd.read_csv(a)
                    c = root / f"synthetic_recovery_coverage{label}.json"
                    if c.exists():
                        out["coverage"][label] = json.load(open(c, encoding="utf-8"))
                    break
                if a.exists():
                    # table of a run made with v1.0 (no summary file, no `variant` / `well_specified` columns): the summary is
                    # computed here from the stored rows; the variant is read from the tag and the specification flag from the design
                    allrec = pd.read_csv(a)
                    allrec["element"] = allrec["element"].fillna("").astype(str)
                    if "variant" not in allrec:
                        allrec["variant"] = allrec["tag"].astype(str).str.rsplit("_", n=1).str[-1] if "tag" in allrec else "est"
                    if "well_specified" not in allrec:
                        allrec["well_specified"] = [well.get((sc, v), True) for sc, v in zip(allrec["scenario"], allrec["variant"])]
                    out["summary"][label] = RC.recovery_summary(allrec)
                    out["all"][label] = allrec
                    out["source"][label] = "run"
                    eff = allrec[(allrec["element"] != "") & allrec["well_specified"]]
                    n = len(eff); k = int(eff["covered"].sum())
                    out["coverage"][label] = {"n_elements": n, "covered": k, "share": k / n if n else np.nan, "binomial_se": float(np.sqrt(k / n * (1 - k / n) / n)) if n else np.nan,
                                              "n_fits": int(allrec.loc[allrec["well_specified"], "tag"].nunique()) if "tag" in allrec else None, "nominal": self.cfg.interval_prob,
                                              "note": "обчислено з таблиці попереднього запуску (v1.0); лише коректно специфіковані fits"}
                    break
        return out

    def stage_export(self) -> Dict[str, Any]:
        primary = self.primary_experiment()
        states = self.states()
        done = [e for e in self.cfg.experiments if states[e.name]["completed"]]
        fp = {e.name: states[e.name]["fit_id"] for e in done}
        derived = {n: self.registry.fingerprint_for(n) for n in ["forecast", "baseline_ols", "data_sensitivity"] + [f"{k}_{e.name}" for e in done for k in ("diagnostics", "ppc", "loo")]}

        def fn():
            ex = EX.Exporter(self.dirs["figures"], self.dirs["tables"], self.run_id, self.cfg.interval_prob)
            ds = self.dataset()
            tdir = self.dirs["tables"]
            outputs: Dict[str, Any] = {"figures": {}, "tables": {}}
            bundle: Dict[str, Any] = {"primary": primary.name, "prob": self.cfg.interval_prob}
            # ---- priors of every specification in the matrix
            for e in self.cfg.experiments:
                outputs["tables"][f"priors_{e.name}"] = ex.table(prior_table(e.model), f"priors_{e.name}", f"Апріорні розподіли моделі «{e.name}» (згенеровано з коду моделі)")
            bundle["priors"] = prior_table(primary.model)
            # ---- data
            steps = pd.read_csv(tdir / "data_cleaning_steps.csv")
            bundle["cleaning"] = steps
            bundle["counts"] = {"laps": ds.n_laps, "stints": ds.n_stints, "races": int(ds.stints["race"].nunique()), "tracks": len(ds.levels["track"]), "drivers": len(ds.levels["driver"]),
                                "constructors": len(ds.levels["team"]), "innovations": int(len(ds.inno_idx)), "share_used_sets": float(ds.stints["used_set"].mean()),
                                "share_gap_transitions": float(np.mean(ds.dage[ds.inno_idx] > 1)) if len(ds.inno_idx) else float("nan")}
            if (tdir / "ols_stint_slopes.csv").exists():
                slopes = pd.read_csv(tdir / "ols_stint_slopes.csv"); by_track = pd.read_csv(tdir / "ols_by_track.csv")
                outputs["figures"]["data"] = ex.fig_data_overview(steps, ds.y, slopes, by_track, self.cfg.primary.gamma_fixed_value)
                bundle["ols"] = {k: v for k, v in (self.registry.result("baseline_ols") or {}).items() if isinstance(v, (int, float))}
            dsens = tdir / "data_threshold_sensitivity.csv"
            if dsens.exists():
                bundle["data_sensitivity"] = pd.read_csv(dsens)
            # ---- primary results
            if primary in done:
                R = self.results_tables(primary)
                bundle["R"] = R
                T = lambda key, df, name, cap: outputs["tables"].__setitem__(key, ex.table(df, name, cap))
                T("compound", R["compound"], f"compound_{primary.name}", "Ефекти компаундів: множники та швидкості деградації (медіана, ETI, кількість відрізків)")
                T("compound_contrasts", R["compound_contrasts"], f"compound_contrasts_{primary.name}", "Попарні контрасти компаундів зі спільних апостеріорних реалізацій")
                T("order", R["order_probabilities"], f"order_probabilities_{primary.name}", "Апостеріорні ймовірності порядку (загальний рівень і за трасами) з MCSE або межею 3/ESS")
                T("tracks", R["tracks"], f"tracks_{primary.name}", "Трасові ефекти: швидкості, відносне нормування, γ_q, кількість відрізків/кіл/гонок")
                T("scales", R["scales"], f"scales_{primary.name}", "Масштаби неоднорідності: гіперпараметри, реалізовані SD і розклад дисперсії лінійного предиктора швидкості")
                T("interaction", R["interaction"], f"interaction_{primary.name}", "Взаємодія траса×компаунд: ефект, невизначеність, кількість відрізків, повні швидкості")
                T("team", R["team"], f"constructors_{primary.name}", "Ефекти конструкторів із ранговими ймовірностями")
                T("driver", R["driver"], f"drivers_{primary.name}", "Ефекти пілотів із ранговими ймовірностями")
                T("sigma_cmp", R["sigma_cmp"], f"sigma_comparison_{primary.name}", "Порівняння масштабів через апостеріорні різниці та відношення")
                T("residual", R["residual"], f"residual_{primary.name}", "Параметри похибок спостережень, процесного шуму, базових рівнів і паливного коефіцієнта")
                T("sign", R["sign"], f"sign_{primary.name}", "Знак швидкості деградації за рівнями: частка апостеріорної маси (з позначкою «за побудовою»)")
                if "table" in R["identification"]:
                    T("identification", R["identification"]["table"], f"identification_{primary.name}", "Розділення ν і γ за трасами: кореляції, SD швидкості проти SD сукупного нахилу")
                outputs["figures"]["compound"] = ex.fig_compounds(R["compound"], R["compound_contrasts"], R["order_probabilities"], R["rate_draws"], primary.name)
                if len(R["tracks"]):
                    outputs["figures"]["tracks"] = ex.fig_tracks(R["tracks"], primary.name)
                outputs["figures"]["scales"] = ex.fig_scales(R["scales"], primary.name)
                if len(R["interaction"]):
                    outputs["figures"]["interaction"] = ex.fig_interaction(R["interaction"], primary.name)
                if len(R["team"]) or len(R["driver"]):
                    outputs["figures"]["groups"] = ex.fig_groups(R["team"], R["driver"], R["sigma_cmp"], primary.name)
                p = self.posterior(primary)
                outputs["figures"]["diagnostics"] = ex.fig_diagnostics(p, hyper_sites(primary.model), primary.name)
                picks = RS.select_example_stints(ds, p)
                outputs["figures"]["examples"] = ex.fig_examples(self.example_panels(primary, picks), primary.name, "smoothing")
                dgp = self.registry.result(f"diagnostics_{primary.name}") or {}
                if dgp.get("hyper_csv") and Path(dgp["hyper_csv"]).exists():
                    bundle["hyper"] = pd.read_csv(dgp["hyper_csv"])
                    outputs["tables"]["mcmc_hyper"] = ex.table(bundle["hyper"], f"mcmc_hyper_{primary.name}", "Діагностика MCMC для кожного гіперпараметра: R̂, ESS (основна частина/хвости), MCSE", float_fmt="%.6g")
                # predictive checks
                pp = self.registry.result(f"prior_predictive_{primary.name}"); pc = self.registry.result(f"ppc_{primary.name}")
                if pp and pc and Path(pp["npz"]).exists() and Path(pc["conditional"]["npz"]).exists():
                    ppc_tab = pd.read_csv(pc["conditional"]["table_csv"])
                    outputs["figures"]["ppc"] = ex.fig_prior_and_ppc(pp["npz"], pp, pc["conditional"]["npz"], ppc_tab, pc["replicated"]["npz"], pd.read_csv(pc["replicated"]["table_csv"]), primary.name)
                    outputs["tables"]["ppc"] = ex.table(ppc_tab, f"ppc_conditional_{primary.name}", "Умовна апостеріорна предиктивна перевірка: статистики спостережень проти реплік (усічених і неусічених)")
                    outputs["tables"]["ppc_resid"] = ex.table(pd.read_csv(pc["conditional"]["residual_csv"]), f"ppc_residual_dependence_{primary.name}", "Залежність залишків від віку шини, компаунда й траси: спостережено проти реплік")
                    outputs["tables"]["ppc_rep"] = ex.table(pd.read_csv(pc["replicated"]["table_csv"]), f"ppc_replicated_{primary.name}", "Репліки нових відрізків: квантилі нахилів МНК за компаундами")
                    bundle["ppc"] = ppc_tab; bundle["pit_tails"] = pc["conditional"].get("pit_tails")
            # ---- gates of every completed fit
            rows = []
            for e in done:
                dg = self.registry.result(f"diagnostics_{e.name}")
                if dg:
                    rows.append({"model": e.name, "status": dg["status"], "worst_rhat": dg["gates"]["worst_rhat"], "min_ess_bulk": dg["gates"]["min_ess_bulk"], "min_ess_tail": dg["gates"]["min_ess_tail"],
                                 "max_mcse_rel": dg["gates"]["max_mcse_rel"], "divergences": dg["energy"]["divergences"], "ebfmi_min": dg["energy"]["ebfmi_min"], "tree_depth_mean": dg["energy"].get("tree_depth_mean"),
                                 "tree_depth_max": dg["energy"].get("tree_depth_max"), "mean_accept": dg["energy"].get("mean_accept"),
                                 "chains": dg["n_chains"], "draws": dg["n_draws"], "sampling_wall_s": dg.get("sampling_wall_seconds"), "min_ess_per_s": dg.get("min_ess_per_second_sampling"),
                                 "families_failed": ",".join(dg["gates"].get("families_failed", [])), "failures": "; ".join(dg["gates"]["failures"])})
            if rows:
                bundle["gates"] = pd.DataFrame(rows)
                outputs["tables"]["gates"] = ex.table(bundle["gates"], "diagnostics_gates", "Операційні критерії MCMC за всіма завершеними моделями (неокруглені значення)", float_fmt="%.6g")
            # ---- model comparison / sensitivity across completed fits
            comp_rows, sens_rows = [], []
            # v1.2: every completed fixed-scale fit is paired with the completed hierarchical fit that differs from it ONLY in
            # group_scale (same data, same everything else): unpooled <-> primary, unpooled_train <-> primary_train
            counterpart: Dict[str, str] = {}
            for e in done:
                if getattr(e.model, "group_scale", "hier") == "fixed":
                    for h in done:
                        same_but_scales = dataclasses.replace(e.model, name=h.model.name, group_scale="hier", track_fixed_scale=None, driver_fixed_scale=None,
                                                              team_fixed_scale=None, interaction_fixed_scale=None)
                        if h.data_variant == e.data_variant and h.sampler == e.sampler and getattr(h.model, "group_scale", "hier") == "hier" and same_but_scales == dataclasses.replace(
                                h.model, track_fixed_scale=None, driver_fixed_scale=None, team_fixed_scale=None, interaction_fixed_scale=None):
                            counterpart[e.name] = h.name
                            break
            keep_effects = set(counterpart) | set(counterpart.values())
            effects_by_model: Dict[str, Dict[str, pd.DataFrame]] = {}
            for e in done:
                try:
                    Rr = R if (e.name == primary.name and primary in done) else self.results_tables(e)
                except Exception as err:  # noqa
                    self.log(f"[export] results for {e.name} failed: {err!r}"); self.problems.append({"stage": f"results_tables({e.name})", "error": repr(err)}); continue
                if e.name in keep_effects:
                    effects_by_model[e.name] = {"tracks": Rr["tracks"], "team": Rr["team"], "driver": Rr["driver"], "interaction": Rr["interaction"]}
                gr = Rr["global_rate"]; st_ = states[e.name]
                comp = Rr["compound"].set_index("compound")
                row = {"model": e.name, "tier": e.tier, "data": self._data_label(e), "differs": e.differs_from_primary, "link": gr.get("link", "log"), "gates_pass": st_["gates_pass"], "usable": st_["usable"],
                       "worst_rhat": st_["worst_rhat"], "min_ess": st_["min_ess"], "divergences": st_["divergences"], "ebfmi_min": st_["ebfmi_min"],
                       "global_rate": gr["median"], "global_lo": gr["lo"], "global_hi": gr["hi"], "P(global_rate>0)": gr.get("P(>0)")}
                for c in comp.index:
                    row[f"mult_{c}"] = comp.loc[c, "multiplier_median"]; row[f"rate_{c}"] = comp.loc[c, "rate_median_s_per_lap"]
                sc = Rr["scales"]
                for _, s_ in sc[sc["kind"].str.startswith("hyper")].iterrows():
                    row[f"hyperSD_{s_['component']}"] = s_["median"]
                # realised SDs of the fitted effects exist with or without learned scales, so they are the comparable quantity
                # between `primary` and a fixed-scale specification; the fixed scales themselves are listed as such
                for _, s_ in sc[sc["kind"].str.startswith("realised")].iterrows():
                    row[f"realSD_{s_['component']}"] = s_["median"]; row[f"realSD_{s_['component']}_lo"] = s_["lo"]; row[f"realSD_{s_['component']}_hi"] = s_["hi"]
                for _, s_ in sc[sc["kind"] == "fixed prior scale"].iterrows():
                    row[f"fixedSD_{s_['component']}"] = s_["median"]
                row["group_scale"] = getattr(e.model, "group_scale", "hier")
                rs = Rr["residual"].set_index("site")
                for s_ in ("sigma_eps_core", "lambda_skew", "nu_df", "sigma_eta"):
                    if s_ in rs.index:
                        row[s_] = float(rs.loc[s_, "median"]) if not isinstance(rs.loc[s_], pd.DataFrame) else float(rs.loc[s_]["median"].iloc[0])
                sg = Rr["sign"]
                v = sg[sg["quantity"] == "частка відрізків із P(ν_s < 0) > 0.5"]
                if len(v):
                    row["share_stints_P(nu<0)>0.5"] = float(v["value"].iloc[0])
                sens_rows.append({"model": e.name, "quantity": "загальна швидкість, с/коло", "median": gr["median"], "lo": gr["lo"], "hi": gr["hi"]})
                if "SOFT" in comp.index and "HARD" in comp.index:
                    cc = Rr["compound_contrasts"].set_index("contrast")
                    k = [i for i in cc.index if i.startswith("SOFT") and i.endswith("HARD")]
                    if k:
                        row.update({"ratio_SH": cc.loc[k[0], "ratio_median"], "ratio_SH_lo": cc.loc[k[0], "ratio_lo"], "ratio_SH_hi": cc.loc[k[0], "ratio_hi"],
                                    "diff_SH": cc.loc[k[0], "rate_diff_median_s_per_lap"], "diff_SH_lo": cc.loc[k[0], "rate_diff_lo"], "diff_SH_hi": cc.loc[k[0], "rate_diff_hi"]})
                        sens_rows.append({"model": e.name, "quantity": "Soft / Hard (відношення швидкостей)", "median": cc.loc[k[0], "ratio_median"], "lo": cc.loc[k[0], "ratio_lo"], "hi": cc.loc[k[0], "ratio_hi"]})
                        sens_rows.append({"model": e.name, "quantity": "Soft − Hard, с/коло", "median": cc.loc[k[0], "rate_diff_median_s_per_lap"], "lo": cc.loc[k[0], "rate_diff_lo"], "hi": cc.loc[k[0], "rate_diff_hi"]})
                t = sc[(sc["component"] == RS.COMPONENT_UK["track"]) & sc["kind"].str.startswith("hyper")]
                if len(t):
                    sens_rows.append({"model": e.name, "quantity": f"σ_T, {t['units'].iloc[0]}", "median": t["median"].iloc[0], "lo": t["lo"].iloc[0], "hi": t["hi"].iloc[0]})
                comp_rows.append(row)
                if e.name != primary.name:
                    ex.table(Rr["sign"], f"sign_{e.name}", f"Знак швидкості деградації за рівнями, модель «{e.name}»")
                    ex.table(Rr["compound"], f"compound_{e.name}", f"Ефекти компаундів, модель «{e.name}»")
                    ex.table(Rr["scales"], f"scales_{e.name}", f"Масштаби неоднорідності, модель «{e.name}»")
                    if getattr(e.model, "group_scale", "hier") == "fixed":
                        # the group effects themselves are what this specification is compared on
                        for key, name, cap in (("tracks", "tracks", "Трасові ефекти"), ("team", "constructors", "Ефекти конструкторів"), ("driver", "drivers", "Ефекти пілотів"),
                                               ("interaction", "interaction", "Взаємодія траса×компаунд")):
                            if len(Rr[key]):
                                ex.table(Rr[key], f"{name}_{e.name}", f"{cap}, модель «{e.name}» (масштаби ефектів фіксовані, не оцінюються)")
            # v1.2: what learning the scales does to the effects themselves — the same effect under learned scales and under
            # fixed scales, for two fits on the SAME data that differ in nothing else
            for fx_name, h_name in counterpart.items():
                if fx_name in effects_by_model and h_name in effects_by_model:
                    pe = RS.pooling_effect_table(effects_by_model[h_name], effects_by_model[fx_name], h_name, fx_name)
                    if len(pe):
                        bundle.setdefault("pooling_effects", {})[fx_name] = pe
                        outputs["tables"][f"pooling_effects_{fx_name}"] = ex.table(
                            pe, f"pooling_effects_{h_name}_vs_{fx_name}",
                            f"Ті самі групові ефекти за навчених масштабів («{h_name}») і за фіксованих («{fx_name}»): медіани, {int(round(100 * self.cfg.interval_prob))} % інтервали, кількість відрізків", float_fmt="%.6g")
            if comp_rows:
                bundle["comparison"] = pd.DataFrame(comp_rows)
                outputs["tables"]["model_comparison"] = ex.table(bundle["comparison"], "model_comparison", "Порівняння завершених моделей матриці експериментів: основні оцінки та діагностичний стан", float_fmt="%.6g")
                outputs["tables"]["sensitivity"] = ex.table(pd.DataFrame(sens_rows), "sensitivity", "Чутливість основних оцінок до специфікації")
                outputs["figures"]["sensitivity"] = ex.fig_sensitivity(pd.DataFrame(sens_rows))
            # ---- PSIS-LOO / WAIC
            loo = {e.name: self.registry.result(f"loo_{e.name}") for e in done if states[e.name]["has_loo"]}
            loo = {k: v for k, v in loo.items() if v and "elpd_loo" in v}
            if loo:
                ref = primary.name if primary.name in loo else next(iter(loo))
                bundle["loo"] = LOO.compare_table(loo, reference=ref)
                outputs["tables"]["loo"] = ex.table(bundle["loo"], "loo_comparison", f"PSIS-LOO і WAIC (умовно на латентні стани); різниці — відносно «{ref}» лише для моделей на тих самих колах", float_fmt="%.6g")
            # ---- synthetic recovery
            rec = self._recovery_tables()
            bundle["recovery_summary"] = rec["summary"]; bundle["recovery_coverage"] = rec["coverage"]; bundle["recovery_source"] = rec["source"]
            for label, summ in rec["summary"].items():
                shipped_note = (" Таблиці взято з results_sandbox/ (скорочені ланцюги, виконано під час редакції v1.1), бо власного дослідження в outputs не знайдено."
                                if rec["source"].get(label) == "results_sandbox" else "")
                outputs["tables"][f"recovery_summary{label}"] = ex.table(summ, f"synthetic_recovery_summary{label}", "Відновлення параметрів на синтетичних даних: покриття, зміщення, RMSE за сценаріями." + shipped_note,
                                                                         float_fmt="%.6g")
                f_ = ex.fig_recovery_summary(summ, label, note=shipped_note)
                if f_:
                    outputs["figures"][f"recovery_summary{label}"] = f_
            recs = sorted(self.dirs["recovery"].glob("recovery_*.csv"))
            if recs:
                rec1 = pd.read_csv(recs[0]); tag = recs[0].stem.replace("recovery_", "")
                npz = self.dirs["recovery"] / f"nu_pairs_{tag}.npz"
                pairs = (np.load(npz)["truth"], np.load(npz)["est"]) if npz.exists() else None
                outputs["figures"]["recovery"] = ex.fig_recovery(rec1, pairs, tag)
            # ---- forecast
            fdir = self.dirs["forecast"]
            if (fdir / "forecast_summary.csv").exists():
                summ = pd.read_csv(fdir / "forecast_summary.csv"); pairs_t = pd.read_csv(fdir / "forecast_paired_differences.csv")
                bundle["forecast"] = summ; bundle["forecast_pairs"] = pairs_t; bundle["forecast_info"] = self.registry.result("forecast") or {}
                outputs["tables"]["forecast"] = ex.table(summ, "forecast_summary", "Прогнозні оцінки (фіксований початок і ковзний прогноз на одне коло окремо), агрегація за гонками")
                outputs["tables"]["forecast_pairs"] = ex.table(pairs_t, "forecast_paired_differences", "Парні різниці моделей за гонками (гонка — одиниця вибірки; t-інтервал)")
                outputs["figures"]["forecast"] = ex.fig_forecast(summ, pairs_t)
                extra = {}
                for q in sorted(fdir.glob("forecast_summary_*.csv")):
                    tag = q.stem.replace("forecast_summary", "")
                    pq = fdir / f"forecast_paired_differences{tag}.csv"
                    s2 = pd.read_csv(q) if q.stat().st_size > 2 else pd.DataFrame()
                    if len(s2):
                        extra[tag] = (s2, pd.read_csv(pq) if pq.exists() and pq.stat().st_size > 2 else None)
                        ex.table(s2, f"forecast_summary{tag}", f"Прогнозні оцінки, схема «{tag.strip('_')}»")
                        if extra[tag][1] is not None:
                            ex.table(extra[tag][1], f"forecast_paired_differences{tag}", f"Парні різниці моделей за гонками, схема «{tag.strip('_')}»")
                bundle["forecast_extra"] = extra
                # v1.2: does adaptive partial pooling help the forecast?  pooled_train -> unpooled_train -> primary_train
                from . import forecast as F  # noqa: E402 (late import keeps jax out of module import)
                fc_ = self.cfg.forecast
                schemes = {"": f"відомо {int(round(100 * fc_.prefix_fraction))} % кіл відрізка (основна схема)"}
                for sch in fc_.extra_schemes:
                    schemes[str(sch.get("tag", "_extra"))] = f"відомо {int(round(100 * float(sch.get('prefix_fraction', fc_.prefix_fraction))))} % кіл відрізка"
                abl = F.hierarchy_ablation(fdir, schemes, self.cfg.interval_prob)
                if len(abl):
                    abl.to_csv(fdir / "forecast_hierarchy_ablation.csv", index=False)
                    bundle["hierarchy_ablation"] = abl
                    outputs["tables"]["hierarchy_ablation"] = ex.table(abl, "forecast_hierarchy_ablation",
                                                                      "Внесок групових ефектів і навчання їхніх масштабів у прогноз: парні різниці за гонками між моделями без групових ефектів, "
                                                                      "з ефектами за фіксованих масштабів і з навченими масштабами (гонка — одиниця вибірки; t-інтервал)", float_fmt="%.6g")
                elif (fdir / "forecast_hierarchy_ablation.csv").exists():
                    (fdir / "forecast_hierarchy_ablation.csv").unlink()      # no pair of the ladder is scored any more: do not leave a stale table
                pan = self.forecast_example_panels()
                if pan:
                    outputs["figures"]["forecast_examples"] = ex.fig_examples(pan, "forecast", "forecasting")
            # ---- registry + paper tables
            bundle["experiments"] = RP.experiments_status(self.cfg.experiments, states, {n: (s_["sampling_wall_seconds"] or 0) / 3600.0 if s_.get("sampling_wall_seconds") else None for n, s_ in states.items()})
            outputs["paper"] = PAPER.build(bundle, self.dirs["paper"])
            outputs["captions"] = str(ex.cap_path)
            return outputs
        return self.registry.run("export", {"fits": fp, "derived": derived, "prob": self.cfg.interval_prob, "run_id": self.run_id, "v": POSTPROC_VERSION}, fn, deps=[], force=True)

    def stage_report(self) -> Dict[str, Any]:
        primary = self.primary_experiment()
        states = self.states()

        def fn():
            rdir = self.dirs["reports"]
            ds = self.dataset()
            counts = {"laps": ds.n_laps, "stints": ds.n_stints, "races": int(ds.stints["race"].nunique()), "tracks": len(ds.levels["track"]), "drivers": len(ds.levels["driver"]),
                      "constructors": len(ds.levels["team"]), "innovations": int(len(ds.inno_idx))}
            out: Dict[str, Any] = {}
            st_p = states[primary.name]
            dg = self.registry.result(f"diagnostics_{primary.name}") if st_p["completed"] else None
            if st_p["completed"] and dg:
                R = self.results_tables(primary)
                pc = self.registry.result(f"ppc_{primary.name}") or {}
                fs = self.dirs["forecast"] / "forecast_summary.csv"; fpair = self.dirs["forecast"] / "forecast_paired_differences.csv"
                ctx = {"mode": self.cfg.mode, "run_id": self.run_id, "counts": counts, "global_rate": R["global_rate"], "compound": R["compound"], "contrasts": R["compound_contrasts"],
                       "order": R["order_probabilities"], "scales": R["scales"], "residual": R["residual"], "diagnostics": dg, "sigma_cmp": R["sigma_cmp"], "sign": R["sign"],
                       "ppc_headline": (pc.get("conditional") or {}).get("headline"), "pit_tails": (pc.get("conditional") or {}).get("pit_tails"),
                       "forecast": pd.read_csv(fs) if fs.exists() else None, "forecast_pairs": pd.read_csv(fpair) if fpair.exists() else None,
                       "hierarchy_ablation": pd.read_csv(self.dirs["forecast"] / "forecast_hierarchy_ablation.csv") if (self.dirs["forecast"] / "forecast_hierarchy_ablation.csv").exists() else None,
                       "loo": self.registry.result(f"loo_{primary.name}") if st_p["has_loo"] else None,
                       "identification": {k: v for k, v in R["identification"].items() if k != "table"}, "interval_prob": self.cfg.interval_prob, "state": st_p, "states": states}
                (rdir / "results_summary_uk.md").write_text(RP.results_summary_uk(ctx), encoding="utf-8")
                RP.old_conclusions_status(ctx).to_csv(rdir / "old_conclusions_status.csv", index=False)
                out["summary_md"] = str(rdir / "results_summary_uk.md"); out["old_conclusions_csv"] = str(rdir / "old_conclusions_status.csv")
            else:
                (rdir / "results_summary_uk.md").write_text(f"## Результати\n\nFit `{primary.name}` не завершено ({RP.state_label(st_p)}) — зведення результатів не побудовано.\n", encoding="utf-8")
            RP.reviewer_response_matrix(states, rdir / "reviewer_response_matrix.csv")
            (rdir / "editorial_checklist.md").write_text(RP.editorial_checklist_md(), encoding="utf-8")
            hours = {n: (s_["sampling_wall_seconds"] / 3600.0 if s_.get("sampling_wall_seconds") else None) for n, s_ in states.items()}
            RP.experiments_status(self.cfg.experiments, states, hours).to_csv(rdir / "experiments_status.csv", index=False)
            self.registry.table().to_csv(rdir / "stages_status.csv", index=False)
            pd.DataFrame(self.problems, columns=["stage", "error"]).to_csv(rdir / "pipeline_problems.csv", index=False)
            out.update({"response_matrix_csv": str(rdir / "reviewer_response_matrix.csv"), "experiments_csv": str(rdir / "experiments_status.csv"), "checklist_md": str(rdir / "editorial_checklist.md"),
                        "problems": len(self.problems), "completed_fits": [n for n, s_ in states.items() if s_["completed"]]})
            return out
        return self.registry.run("report", {"mode": self.cfg.mode, "run_id": self.run_id, "v": POSTPROC_VERSION}, fn, deps=[], force=True)

    # ------------------------------------------------------------ run all
    def postprocess(self) -> None:
        """Everything that needs no MCMC: diagnostics, predictive checks and LOO of every completed fit of the matrix,
        the forecast evaluation, tables, figures and reports.  This is what REPORT_ONLY runs."""
        prim = self.primary_experiment()
        self._safe(self.stage_baseline)
        self._safe(self.stage_data_sensitivity)
        self._safe(self.stage_prior_predictive, prim)
        for e in self.completed_experiments():
            self._safe(self.stage_diagnostics, e)
            if e.tier != 3:
                self._safe(self.stage_ppc, e)
            self._safe(self.stage_loo, e)
        self._safe(self.stage_forecast)
        self.stage_export()
        self.stage_report()

    def run_all(self) -> Dict[str, Any]:
        t0 = time.time()
        self.log(f"=== pipeline {self.run_id} mode={self.cfg.mode}{' (REPORT_ONLY: no sampling)' if self.cfg.report_only else ''} ===")
        self.stage_data()
        summary: Dict[str, Any] = {}
        if not self.cfg.report_only:
            exps = self.experiments()
            if exps:
                self.stage_preflight(exps[0])
            for e in exps:
                summary[e.name] = self.fit_with_bounded_extension(e)
            self.stage_recovery()
        self.postprocess()
        self.log(f"=== done in {(time.time()-t0)/60:.1f} min; problems: {len(self.problems)} ===")
        return summary

"""MCMC diagnostics and operational gates.

Gates (operational criteria, NOT proof of model correctness):
    rank-normalised split R-hat < 1.01, bulk & tail ESS >= 400, MCSE(mean) <= 5% of the
    posterior SD for the main estimands and key hyper-parameters, 0 divergences,
    E-BFMI >= 0.3 in every chain.  All checks use unrounded values.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .sampler import PosteriorDraws

try:
    import arviz as az
except Exception:  # pragma: no cover
    az = None

MAIN_ESTIMANDS = ["mu_log_nu", "compound_effect", "track_effect", "team_effect", "driver_effect", "interaction_effect", "gamma_track",
                  "sigma_track", "sigma_driver", "sigma_team", "sigma_interaction", "sigma_stint", "sigma_compound", "sigma_stint_add",
                  "baseline_mean", "compound_base_effect", "sigma_stint_base", "mu_log_gamma", "sigma_log_gamma", "sigma_eta",
                  "sigma_eps_core", "lambda_skew", "nu_df",
                  # v1.1 variants (absent from v1.0 fits, so the gates of existing fits are evaluated on the same set as before)
                  "mu_nu_add", "sigma_track_add", "sigma_driver_add", "sigma_team_add", "sigma_interaction_add", "race_effect", "sigma_race", "sigma_race_add"]
# scalar hyper-parameters in the order used for the paper's MCMC table
HYPER_SCALARS = ["mu_log_nu", "mu_nu_add", "sigma_track", "sigma_track_add", "sigma_interaction", "sigma_interaction_add", "sigma_stint", "sigma_stint_add",
                 "sigma_team", "sigma_team_add", "sigma_driver", "sigma_driver_add", "sigma_race", "sigma_race_add", "sigma_compound",
                 "baseline_mean", "sigma_stint_base", "mu_log_gamma", "sigma_log_gamma", "sigma_eta", "sigma_eps_core", "lambda_skew", "nu_df"]
LARGE_BLOCKS = ["nu_stint", "baseline_stint", "z_stint_deg", "z_stint_base"]


def _flat_names(name: str, shape: Tuple[int, ...], labels: Optional[Dict[str, List[str]]] = None, dims: Optional[List[str]] = None) -> List[str]:
    if len(shape) == 0:
        return [name]
    idx = np.indices(shape).reshape(len(shape), -1).T
    out = []
    for ix in idx:
        parts = []
        for d, i in enumerate(ix):
            if labels and dims and d < len(dims) and dims[d] in labels:
                parts.append(str(labels[dims[d]][i]))
            else:
                parts.append(str(i))
        out.append(f"{name}[{','.join(parts)}]")
    return out


SITE_DIMS = {"compound_effect": ["compound"], "track_effect": ["track"], "driver_effect": ["driver"], "team_effect": ["team"],
             "interaction_effect": ["track", "compound"], "gamma_track": ["track"], "compound_base_effect": ["compound"],
             "nu_stint": ["stint"], "baseline_stint": ["stint"], "z_stint_deg": ["stint"], "z_stint_base": ["stint"], "z_log_gamma": ["track"]}


def summarise(pdraws: PosteriorDraws, sites: List[str], levels: Dict[str, List[str]], prob: float = 0.94, extra_probs: List[float] = ()) -> pd.DataFrame:
    """Per-element summary with unrounded diagnostics."""
    rows = []
    lo, hi = (1 - prob) / 2, 1 - (1 - prob) / 2
    for s in sites:
        if s not in pdraws.posterior:
            continue
        a = pdraws.posterior[s]
        shape = a.shape[2:]
        names = _flat_names(s, shape, levels, SITE_DIMS.get(s))
        a2 = a.reshape(a.shape[0], a.shape[1], -1)
        for k, nm in enumerate(names):
            x = a2[:, :, k]
            flat = x.reshape(-1)
            row = {"parameter": nm, "site": s, "mean": float(flat.mean()), "sd": float(flat.std(ddof=1)), "median": float(np.median(flat)),
                   f"eti{int(prob*100)}_lo": float(np.quantile(flat, lo)), f"eti{int(prob*100)}_hi": float(np.quantile(flat, hi))}
            for p in extra_probs:
                row[f"eti{int(p*100)}_lo"] = float(np.quantile(flat, (1 - p) / 2)); row[f"eti{int(p*100)}_hi"] = float(np.quantile(flat, 1 - (1 - p) / 2))
            if az is not None and x.shape[1] >= 4:
                row["rhat"] = float(az.rhat(x)); row["ess_bulk"] = float(az.ess(x, method="bulk")); row["ess_tail"] = float(az.ess(x, method="tail", prob=(0.05, 0.95)))
                row["mcse_mean"] = float(az.mcse(x, method="mean")); row["mcse_sd"] = float(az.mcse(x, method="sd"))
                row["mcse_rel"] = row["mcse_mean"] / row["sd"] if row["sd"] > 0 else np.nan
                row["mcse_q05"] = float(az.mcse(x, method="quantile", prob=0.05)); row["mcse_q95"] = float(az.mcse(x, method="quantile", prob=0.95))
            rows.append(row)
    return pd.DataFrame(rows)


def energy_diagnostics(pdraws: PosteriorDraws) -> Dict[str, Any]:
    E = pdraws.sample_stats["energy"]
    dE = np.diff(E, axis=1)
    ebfmi = (dE ** 2).mean(axis=1) / E.var(axis=1, ddof=1)
    depth = pdraws.sample_stats.get("tree_depth")
    div = pdraws.sample_stats["diverging"]
    out = {"ebfmi_per_chain": [float(v) for v in ebfmi], "ebfmi_min": float(ebfmi.min()), "divergences": int(div.sum()),
           "divergences_per_chain": [int(v) for v in div.sum(axis=1)], "mean_accept": float(pdraws.sample_stats["accept_prob"].mean()),
           "step_size_per_chain": [float(v) for v in pdraws.sample_stats["step_size"][:, -1]]}
    if depth is not None:
        out["tree_depth_mean"] = float(depth.mean()); out["tree_depth_max"] = int(depth.max())
        out["tree_depth_saturation_share"] = float((depth >= pdraws.meta.get("max_treedepth", depth.max())).mean()) if depth.max() > 0 else 0.0
        out["tree_depth_hist"] = {int(k): int(v) for k, v in zip(*np.unique(depth, return_counts=True))}
    return out


def evaluate_gates(summary: pd.DataFrame, energy: Dict[str, Any], gates: Dict[str, float], main_sites: List[str]) -> Dict[str, Any]:
    main = summary[summary["site"].isin(main_sites)] if "site" in summary else summary
    fails: List[str] = []
    if "rhat" in main:
        bad = main[main["rhat"] >= gates["rhat_max"]]
        if len(bad):
            fails.append(f"R-hat >= {gates['rhat_max']} for {len(bad)} parameters (worst {bad['parameter'].iloc[bad['rhat'].argmax()]} = {bad['rhat'].max():.4f})")
        bad = main[(main["ess_bulk"] < gates["ess_min"]) | (main["ess_tail"] < gates["ess_min"])]
        if len(bad):
            fails.append(f"ESS < {gates['ess_min']} for {len(bad)} parameters (min bulk {main['ess_bulk'].min():.0f}, min tail {main['ess_tail'].min():.0f})")
        bad = main[main["mcse_rel"] > gates["mcse_rel_max"]]
        if len(bad):
            fails.append(f"MCSE(mean) > {gates['mcse_rel_max']*100:.0f}% of SD for {len(bad)} parameters (worst {bad['mcse_rel'].max():.3f})")
    if energy["divergences"] > gates["divergences_max"]:
        fails.append(f"{energy['divergences']} divergent transitions")
    if energy["ebfmi_min"] < gates["ebfmi_min"]:
        fails.append(f"E-BFMI min {energy['ebfmi_min']:.3f} < {gates['ebfmi_min']}")
    return {"pass": len(fails) == 0, "failures": fails, "n_parameters_checked": int(len(main)),
            "worst_rhat": float(main["rhat"].max()) if "rhat" in main and len(main) else None,
            "min_ess_bulk": float(main["ess_bulk"].min()) if "ess_bulk" in main and len(main) else None,
            "min_ess_tail": float(main["ess_tail"].min()) if "ess_tail" in main and len(main) else None,
            "max_mcse_rel": float(main["mcse_rel"].max()) if "mcse_rel" in main and len(main) else None}


def latent_block_diagnostics(pdraws: PosteriorDraws, levels: Dict[str, List[str]]) -> pd.DataFrame:
    """Compact diagnostics for the large blocks (per-stint parameters and the thinned
    lap-level innovations): max R-hat, min ESS.  Thinned draws understate ESS."""
    rows = []
    for s in LARGE_BLOCKS:
        if s not in pdraws.posterior or az is None:
            continue
        a = pdraws.posterior[s]
        if a.shape[1] < 4:
            continue
        rh = np.asarray(az.rhat(a)); eb = np.asarray(az.ess(a, method="bulk")); et = np.asarray(az.ess(a, method="tail", prob=(0.05, 0.95)))
        rows.append({"block": s, "n": int(np.prod(a.shape[2:])), "rhat_max": float(np.nanmax(rh)), "rhat_q99": float(np.nanquantile(rh, 0.99)),
                     "ess_bulk_min": float(np.nanmin(eb)), "ess_tail_min": float(np.nanmin(et)), "share_rhat_gt_1.01": float(np.nanmean(rh > 1.01)), "thinned": False})
    if pdraws.latent is not None and az is not None and pdraws.latent.shape[1] >= 4:
        L = pdraws.latent.astype(np.float64)
        # subsample columns for speed if huge
        cols = np.arange(L.shape[2]) if L.shape[2] <= 5000 else np.linspace(0, L.shape[2] - 1, 5000).astype(int)
        rh = np.asarray(az.rhat(L[:, :, cols])); eb = np.asarray(az.ess(L[:, :, cols], method="bulk"))
        rows.append({"block": "lap-level latent block (alpha_lap or z_inno; thinned draws)", "n": int(L.shape[2]), "rhat_max": float(np.nanmax(rh)), "rhat_q99": float(np.nanquantile(rh, 0.99)),
                     "ess_bulk_min": float(np.nanmin(eb)), "ess_tail_min": np.nan, "share_rhat_gt_1.01": float(np.nanmean(rh > 1.01)), "thinned": True})
    return pd.DataFrame(rows)


def hyper_table(summary: pd.DataFrame, prob: float = 0.94) -> pd.DataFrame:
    """Diagnostics of every scalar hyper-parameter and of the vector estimands' worst element — the table the
    reviewer asked for (R-hat, bulk/tail ESS, MCSE) — from the per-element summary of `summarise`."""
    lo, hi = f"eti{int(prob*100)}_lo", f"eti{int(prob*100)}_hi"
    rows = []
    for s in HYPER_SCALARS:
        g = summary[summary["site"] == s]
        if len(g) == 1:
            r = g.iloc[0]
            rows.append({"parameter": s, "kind": "scalar", "n_elements": 1, "mean": r["mean"], "sd": r["sd"], "lo": r.get(lo), "hi": r.get(hi), "rhat": r.get("rhat"),
                         "ess_bulk": r.get("ess_bulk"), "ess_tail": r.get("ess_tail"), "mcse_rel": r.get("mcse_rel")})
    for s in ("compound_effect", "compound_base_effect", "track_effect", "gamma_track", "team_effect", "driver_effect", "interaction_effect", "race_effect"):
        g = summary[summary["site"] == s]
        if len(g) and "rhat" in g:
            rows.append({"parameter": s, "kind": "vector (worst element)", "n_elements": int(len(g)), "mean": np.nan, "sd": np.nan, "lo": np.nan, "hi": np.nan,
                         "rhat": float(g["rhat"].max()), "ess_bulk": float(g["ess_bulk"].min()), "ess_tail": float(g["ess_tail"].min()), "mcse_rel": float(g["mcse_rel"].max())})
    return pd.DataFrame(rows)


def timing_from_store(store_dir: Path) -> Dict[str, Any]:
    """Wall-clock per phase from the append-only timing log (provenance; resumed runs
    add their own chunks, re-computed chunks after a crash are counted again)."""
    secs = {"warmup": 0.0, "sampling": 0.0}; n = {"warmup": 0, "sampling": 0}; pids = set()
    p = Path(store_dir) / "timing.jsonl"
    if p.exists():
        for line in open(p):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ph = rec.get("phase")
            if ph in secs:
                secs[ph] += float(rec.get("chunk_seconds", 0)); n[ph] += int(rec.get("n", 0)); pids.add(rec.get("pid"))
    return {"warmup_seconds": round(secs["warmup"], 1), "sampling_seconds": round(secs["sampling"], 1), "transitions_warmup": n["warmup"],
            "transitions_sampling": n["sampling"], "processes": len(pids),
            "seconds_per_transition_all_chains_sampling": round(secs["sampling"] / n["sampling"], 4) if n["sampling"] else None}


def full_diagnostics(pdraws: PosteriorDraws, levels: Dict[str, List[str]], gates: Dict[str, float], prob: float, extra_probs: List[float],
                     out_dir: Path, fit_name: str, wall_seconds: Optional[float] = None, max_treedepth: int = 10) -> Dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdraws.meta["max_treedepth"] = max_treedepth
    sites = [s for s in pdraws.posterior if s not in LARGE_BLOCKS and not s.endswith("_raw") and s not in ("mu_y_last",)]
    summ = summarise(pdraws, sites, levels, prob, extra_probs)
    energy = energy_diagnostics(pdraws)
    main_sites = [s for s in MAIN_ESTIMANDS if s in pdraws.posterior]
    gate = evaluate_gates(summ, energy, gates, main_sites)
    latent = latent_block_diagnostics(pdraws, levels)
    summ.to_csv(out_dir / f"summary_{fit_name}.csv", index=False)
    latent.to_csv(out_dir / f"latent_blocks_{fit_name}.csv", index=False)
    hyper_table(summ, prob).to_csv(out_dir / f"hyper_{fit_name}.csv", index=False)
    # which gate families fail: convergence of the estimands (R-hat/ESS/MCSE), divergences, energy (E-BFMI)
    f = " ".join(gate["failures"])
    gate["families_failed"] = [k for k, pat in (("rhat", "R-hat"), ("ess", "ESS <"), ("mcse", "MCSE"), ("divergences", "divergent"), ("ebfmi", "E-BFMI")) if pat in f]
    gate["only_ebfmi_fails"] = gate["families_failed"] == ["ebfmi"]
    ess_per_sec = None
    if wall_seconds and "ess_bulk" in summ and len(summ):
        ess_per_sec = float(summ.loc[summ["site"].isin(main_sites), "ess_bulk"].min() / wall_seconds)
    info = {"fit": fit_name, "gates": gate, "energy": energy, "n_chains": pdraws.n_chains, "n_draws": pdraws.n_draws, "hyper_csv": str(out_dir / f"hyper_{fit_name}.csv"),
            "min_ess_per_second_sampling": ess_per_sec, "sampling_wall_seconds": wall_seconds, "summary_csv": str(out_dir / f"summary_{fit_name}.csv"), "latent_csv": str(out_dir / f"latent_blocks_{fit_name}.csv"),
            "status": "ok" if gate["pass"] else "needs_attention"}
    with open(out_dir / f"diagnostics_{fit_name}.json", "w") as f:
        json.dump(info, f, indent=1, default=float)
    return info

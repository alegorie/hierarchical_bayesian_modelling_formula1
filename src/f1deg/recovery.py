"""Parameter-recovery evaluation on synthetic data and ν–γ identification diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data import Dataset
from .link import group_is_additive, link_name, mu_site, to_rate
from .sampler import PosteriorDraws

SCALARS = ["mu_log_nu", "sigma_track", "sigma_driver", "sigma_team", "sigma_interaction", "sigma_stint", "sigma_stint_base", "sigma_eta",
           "sigma_eps_core", "lambda_skew", "nu_df", "baseline_mean", "sigma_race",
           # identity-link fits: the truth dictionary of an identity-link scenario carries the same names
           "mu_nu_add", "sigma_track_add", "sigma_driver_add", "sigma_team_add", "sigma_interaction_add", "sigma_stint_add", "sigma_race_add"]
VECTORS = ["compound_effect", "track_effect", "driver_effect", "team_effect", "gamma_track", "compound_base_effect", "interaction_effect", "race_effect"]


def recovery_table(p: PosteriorDraws, ds: Dataset, prob: float = 0.94, tag: str = "") -> pd.DataFrame:
    assert ds.truth is not None
    tr = ds.truth
    lo, hi = (1 - prob) / 2, 1 - (1 - prob) / 2
    rows = []

    def add(name, truth, draws, element=""):
        q = np.quantile(draws, [lo, 0.5, hi])
        sd = draws.std(ddof=1); mean = draws.mean()
        rows.append({"scenario": tr["scenario"], "replicate_seed": tr["seed"], "parameter": name, "element": element, "truth": float(truth), "post_mean": float(mean),
                     "post_median": float(q[1]), f"eti{int(prob*100)}_lo": float(q[0]), f"eti{int(prob*100)}_hi": float(q[2]), "post_sd": float(sd),
                     "covered": bool(q[0] <= truth <= q[2]), "z": float((truth - mean) / sd) if sd > 0 else np.nan, "bias": float(mean - truth),
                     "rel_bias": float((mean - truth) / abs(truth)) if truth != 0 else np.nan, "tag": tag})

    # A fit and a scenario are on the same scale of the rate predictor only if they use the same kind of link.
    # Otherwise (e.g. the log-link model fitted to identity-link data) effects are not comparable element by element
    # and only link-free quantities (rates in s/lap, baselines, noise, gamma) are scored.
    truth_additive = bool(tr.get("additive", False))
    same_scale = truth_additive == group_is_additive(p.posterior)
    rate_scale_sites = {"compound_effect", "track_effect", "driver_effect", "team_effect", "interaction_effect", "race_effect"}
    for s in SCALARS:
        if s in p.posterior and s in tr:
            add(s + (" (centred scale, approx.)" if s == "baseline_mean" else ""), tr[s], p.flat(s))
    for v in VECTORS:
        if v in rate_scale_sites and not same_scale:
            continue
        if v in p.posterior and v in tr:
            a = p.flat(v); t = np.asarray(tr[v])
            if a.shape[1:] != t.shape:
                continue
            flat_t = t.reshape(-1); flat_a = a.reshape(a.shape[0], -1)
            labels = _labels(v, ds)
            for k in range(flat_t.shape[0]):
                add(v, flat_t[k], flat_a[:, k], labels[k] if labels is not None and k < len(labels) else str(k))
    # stint-level rates: summary over stints
    if "nu_stint" in p.posterior:
        a = p.flat("nu_stint"); t = np.asarray(tr["nu_stint"])
        q = np.quantile(a, [lo, hi], axis=0)
        cov = np.mean((q[0] <= t) & (t <= q[1]))
        rows.append({"scenario": tr["scenario"], "replicate_seed": tr["seed"], "parameter": "nu_stint (summary over stints)", "element": f"n={len(t)}", "truth": float(t.mean()),
                     "post_mean": float(a.mean()), "post_median": float(np.median(a)), f"eti{int(prob*100)}_lo": np.nan, f"eti{int(prob*100)}_hi": np.nan,
                     "post_sd": float(a.std(axis=0).mean()), "covered": bool(cov >= prob - 0.1), "z": np.nan, "bias": float((a.mean(0) - t).mean()),
                     "rel_bias": float(((a.mean(0) - t) / t).mean()), "tag": f"{tag} coverage_share={cov:.3f}; corr(truth, post mean)={np.corrcoef(t, a.mean(0))[0,1]:.3f}",
                     # numeric copies of what the tag says (v1.1), so that summaries need no string parsing
                     "stint_coverage": float(cov), "stint_corr": float(np.corrcoef(t, a.mean(0))[0, 1]), "stint_rmse": float(np.sqrt(np.mean((a.mean(0) - t) ** 2))),
                     "stint_share_truth_negative": float((t < 0).mean()), "stint_share_postmedian_negative": float((np.median(a, axis=0) < 0).mean())})
    # derived contrasts (identified quantities): SOFT-HARD additive rate at the geometric-mean level, interaction SD realised
    if "compound_effect" in p.posterior and "compound_effect" in tr:
        S = p.posterior
        ce = p.flat("compound_effect"); mu = p.flat(mu_site(S))
        comps = ds.levels["compound"]
        t_rate = (lambda e: e) if truth_additive else np.exp
        t_mu = tr["mu_nu_add"] if truth_additive else tr["mu_log_nu"]
        for c_i, c in enumerate(comps):            # link-free: compound rates in s/lap at the central level of the other factors
            add(f"rate_{c} (s/lap)", t_rate(t_mu + tr["compound_effect"][c_i]), to_rate(mu + ce[:, c_i], S))
        add("rate_global (s/lap)", t_rate(t_mu), to_rate(mu, S))
        if "SOFT" in comps and "HARD" in comps:
            i, j = comps.index("SOFT"), comps.index("HARD")
            tr_c = t_rate(t_mu + tr["compound_effect"][i]) - t_rate(t_mu + tr["compound_effect"][j])
            est_i, est_j = to_rate(mu + ce[:, i], S), to_rate(mu + ce[:, j], S)
            add("rate_SOFT_minus_HARD (s/lap)", tr_c, est_i - est_j)
            add("ratio_SOFT_over_HARD", t_rate(t_mu + tr["compound_effect"][i]) / t_rate(t_mu + tr["compound_effect"][j]), est_i / est_j)
    out = pd.DataFrame(rows)
    out["fit_link"] = link_name(p.posterior); out["truth_link"] = "identity" if truth_additive else ("signed_stint" if tr.get("spec", {}).get("signed_stint_sd") is not None else "log")
    return out


def _labels(v: str, ds: Dataset) -> Optional[List[str]]:
    if v in ("compound_effect", "compound_base_effect"):
        return ds.levels["compound"]
    if v in ("track_effect", "gamma_track"):
        return ds.levels["track"]
    if v == "driver_effect":
        return ds.levels["driver"]
    if v == "team_effect":
        return ds.levels["team"]
    if v == "interaction_effect":
        return [f"{t}×{c}" for t in ds.levels["track"] for c in ds.levels["compound"]]
    return None


def identification_table(p: PosteriorDraws, ds: Dataset, spec, prob: float = 0.94) -> Dict[str, Any]:
    """ν–γ identification diagnostics: joint posterior correlations, prior-to-posterior
    SD ratios and the implied net (identified) slopes."""
    out: Dict[str, Any] = {}
    if "gamma_track" not in p.posterior:
        out["note"] = "gamma fixed in this model"
        return out
    S = p.posterior
    mu = p.flat(mu_site(S)); mlg = p.flat("mu_log_gamma")
    out["corr_mu_log_nu__mu_log_gamma"] = float(np.corrcoef(mu, mlg)[0, 1])
    gam = p.flat("gamma_track"); T = p.flat("track_effect") if "track_effect" in p.posterior else np.zeros((len(mu), gam.shape[1]))
    rate_track = to_rate(mu[:, None] + T, S)                    # s/lap at the central level of the other factors
    rows = []
    b = ds.fuel_b_of_race
    race_track = ds.races.set_index("RaceKey").loc[ds.levels["race"], "Location"].tolist() if "Location" in ds.races else None
    for q, tname in enumerate(ds.levels["track"]):
        # mean fuel burn per lap for races at this track
        bq = float(np.mean([b[k] for k, r in enumerate(ds.levels["race"]) if race_track is not None and race_track[k] == tname])) if race_track else float(np.mean(b))
        net = rate_track[:, q] - bq * gam[:, q]                  # identified net trend ν - b_r γ (s/lap)
        rows.append({"track": tname, "corr_rate_gamma": float(np.corrcoef(rate_track[:, q], gam[:, q])[0, 1]), "gamma_median": float(np.median(gam[:, q])),
                     "gamma_sd": float(gam[:, q].std()), "rate_median": float(np.median(rate_track[:, q])), "rate_sd": float(rate_track[:, q].std()),
                     "fuel_b_kg_per_lap": bq, "net_slope_median": float(np.median(net)), "net_slope_sd": float(net.std()),
                     "rate_sd_over_net_sd": float(rate_track[:, q].std() / net.std()) if net.std() > 0 else np.nan})
    tab = pd.DataFrame(rows)
    # prior-to-posterior shrinkage for mu_log_gamma and sigma_log_gamma
    out["mu_log_gamma_prior_sd"] = float(spec.mu_log_gamma_scale); out["mu_log_gamma_post_sd"] = float(mlg.std())
    out["mu_log_gamma_sd_ratio_post_over_prior"] = float(mlg.std() / spec.mu_log_gamma_scale)
    out["mu_log_gamma_post_mean"] = float(mlg.mean()); out["mu_log_gamma_prior_mean"] = float(spec.mu_log_gamma_loc)
    out["gamma_prior_median_s_per_kg"] = float(np.exp(spec.mu_log_gamma_loc))
    out["corr_track_rate_gamma_mean"] = float(tab["corr_rate_gamma"].mean())
    out["table"] = tab
    return out


SCENARIO_ORDER = ["baseline", "known_gamma", "weak_signal", "no_interaction", "heavy_tails", "phase_confounded", "race_effect", "signed_rates", "identity_rates", "identity_zero"]
VARIANT_ORDER = ["est", "fixed", "noint", "race", "signed", "ident"]


def recovery_summary(allrec: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recovery rows over replicates and elements, per (scenario, fit variant, parameter).

    coverage     share of truths inside the interval (nominal = the interval probability), with a binomial SE
                 (elements of one fit are correlated, so the SE is only indicative)
    mean_z       mean of (truth − posterior mean)/posterior SD: ≈ 0 if unbiased, |mean_z|·sqrt(n) >> 2 flags a bias
    rms_z        ≈ 1 if the posterior SD matches the actual error
    For the per-stint rates one row per fit exists (parameter contains "summary"); its coverage/correlation are averaged."""
    d = allrec.copy()
    d["element"] = d["element"].fillna("").astype(str)
    if "variant" not in d:      # tables written by v1.0: the fit variant is the last token of the tag (rec_<scenario>_<rep>_<variant>)
        d["variant"] = d["tag"].astype(str).str.rsplit("_", n=1).str[-1] if "tag" in d else "est"
    if "well_specified" not in d:
        d["well_specified"] = True
    rows = []
    reg = d[~d["parameter"].str.contains("summary")]
    for (scen, var, well, par), g in reg.groupby(["scenario", "variant", "well_specified", "parameter"], sort=False):
        n = len(g); cov = float(g["covered"].mean()); z = g["z"].to_numpy(float); b = g["bias"].to_numpy(float)
        rows.append({"scenario": scen, "variant": var, "well_specified": bool(well), "parameter": par, "n": n, "n_replicates": int(g["replicate_seed"].nunique()),
                     "coverage": cov, "coverage_se": float(np.sqrt(cov * (1 - cov) / n)) if n else np.nan, "mean_z": float(np.nanmean(z)), "rms_z": float(np.sqrt(np.nanmean(z ** 2))),
                     "mean_bias": float(np.nanmean(b)), "rmse": float(np.sqrt(np.nanmean(b ** 2))), "mean_post_sd": float(g["post_sd"].mean()), "mean_truth": float(g["truth"].mean())})
    st = d[d["parameter"].str.contains("summary")]
    if len(st) and "stint_coverage" in st:
        for (scen, var, well), g in st.groupby(["scenario", "variant", "well_specified"], sort=False):
            rows.append({"scenario": scen, "variant": var, "well_specified": bool(well), "parameter": "nu_stint (per-stint rates)", "n": int(len(g)), "n_replicates": int(g["replicate_seed"].nunique()),
                         "coverage": float(g["stint_coverage"].mean()), "coverage_se": float(g["stint_coverage"].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan,
                         "mean_z": np.nan, "rms_z": np.nan, "mean_bias": float(g["bias"].mean()), "rmse": float(g["stint_rmse"].mean()), "mean_post_sd": float(g["post_sd"].mean()),
                         "mean_truth": float(g["truth"].mean()), "stint_corr": float(g["stint_corr"].mean()),
                         "share_truth_negative": float(g["stint_share_truth_negative"].mean()), "share_postmedian_negative": float(g["stint_share_postmedian_negative"].mean())})
    out = pd.DataFrame(rows)
    if len(out):
        # canonical order of scenarios and fit variants: tables and figures must not depend on the order in which
        # the tables of several workers were merged (the order of parameters inside a group is kept)
        so = {k: i for i, k in enumerate(SCENARIO_ORDER)}; vo = {k: i for i, k in enumerate(VARIANT_ORDER)}
        out = (out.assign(_s=[so.get(k, len(so)) for k in out["scenario"]], _v=[vo.get(k, len(vo)) for k in out["variant"]])
                  .sort_values(["_s", "_v"], kind="stable").drop(columns=["_s", "_v"]).reset_index(drop=True))
    return out

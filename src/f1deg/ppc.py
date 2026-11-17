"""Prior predictive and posterior predictive checks.

Three kinds of predictive distributions are kept apart (and labelled so in the
figures):
  * prior predictive              — joint consequences of the priors (Gelman et al. §2.4)
  * conditional PPC               — y_rep | fitted latent paths α̂ (observation model only)
  * replicated-stint PPC          — new stint effects + new latent paths from the posterior
                                    hyper-parameters (tests hierarchy + process noise)
Cleaned data are truncated (lap time <= 1.30 x race-fastest); a "truncated" variant of
every statistic applies the same race-specific cut to the replicates so that like is
compared with like.  Raw sample skewness is reported but flagged: it is unreliable
when the tail index is near 2-3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as st

from .config import ModelSpec, derive_seed
from .data import Dataset
from .hansen import np_cdf, np_rvs
from .link import mu_site, to_rate
from .sampler import PosteriorDraws


def grouped_ols_slopes(codes: np.ndarray, x: np.ndarray, y: np.ndarray, n_groups: int) -> np.ndarray:
    """Vectorised per-group OLS slope of y on x (NaN for groups with no x-variation)."""
    n = np.bincount(codes, minlength=n_groups).astype(float)
    sx = np.bincount(codes, weights=x, minlength=n_groups); sy = np.bincount(codes, weights=y, minlength=n_groups)
    sxx = np.bincount(codes, weights=x * x, minlength=n_groups); sxy = np.bincount(codes, weights=x * y, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = sxx - sx * sx / n
        slope = (sxy - sx * sy / n) / den
    slope[(den <= 1e-12) | (n < 3)] = np.nan
    return slope


def _obs_sample(spec: ModelSpec, mu: np.ndarray, sc: float, lam: Optional[float], eta: Optional[float], rng: np.random.Generator) -> np.ndarray:
    if spec.likelihood == "skewt":
        return np_rvs(mu, sc, lam, eta, size=mu.shape, rng=rng)
    if spec.likelihood == "studentt":
        return mu + sc * rng.standard_t(eta, size=mu.shape)
    return mu + sc * rng.normal(size=mu.shape)


def _obs_cdf(spec: ModelSpec, y: np.ndarray, mu: np.ndarray, sc: float, lam: Optional[float], eta: Optional[float]) -> np.ndarray:
    if spec.likelihood == "skewt":
        return np_cdf(y, mu, sc, lam, eta)
    if spec.likelihood == "studentt":
        return st.t.cdf((y - mu) / sc, eta)
    return st.norm.cdf((y - mu) / sc)


def _stats(y: np.ndarray, qs: List[float]) -> Dict[str, float]:
    y = y[np.isfinite(y)]
    out = {f"q{int(q*100):02d}": float(np.quantile(y, q)) for q in qs}
    out.update({"mean": float(y.mean()), "iqr": float(np.quantile(y, .75) - np.quantile(y, .25)), "mad": float(np.median(np.abs(y - np.median(y)))),
                "p_gt_2": float((y > 2).mean()), "p_gt_3": float((y > 3).mean()), "p_gt_5": float((y > 5).mean()), "p_lt_m2": float((y < -2).mean()),
                "skew_raw_unreliable": float(st.skew(y)), "max": float(y.max()), "n": int(len(y))})
    return out


# --------------------------------------------------------------------------- #
# prior predictive
# --------------------------------------------------------------------------- #
def prior_predictive(ds: Dataset, spec: ModelSpec, n: int, seed: int, out_dir: Path, tag: str = "primary") -> Dict[str, Any]:
    import jax
    from numpyro.infer import Predictive
    from .model import f1_model, model_data

    md = model_data(ds)
    spec = spec.replace(latent_param="noncentred")   # the generative (non-centred) form of the same model is needed for prior sampling
    pred = Predictive(f1_model, num_samples=n)
    s = pred(jax.random.key(seed), md, spec, observe=False)
    s = {k: np.asarray(v) for k, v in s.items()}
    y = s["y"]                      # (n, laps)
    nu = s["nu_stint"]              # (n, S)
    ages = [10, 20, 30]
    out = {"n": n, "observed": _stats(ds.y, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
           "y_rep_quantiles": {f"q{int(q*100):02d}": [float(v) for v in np.quantile(np.quantile(y, q, axis=1), [0.05, 0.5, 0.95])] for q in (0.01, 0.05, 0.5, 0.95, 0.99)},
           "nu_quantiles_s_per_lap": {k: float(v) for k, v in zip(["q05", "q25", "q50", "q75", "q95"], np.quantile(nu, [.05, .25, .5, .75, .95]))},
           "cumulative_degradation_s": {f"age_{a}": {k: float(v) for k, v in zip(["q05", "q50", "q95"], np.quantile(nu * a, [.05, .5, .95]))} for a in ages},
           "baseline_sd_s": {k: float(v) for k, v in zip(["q05", "q50", "q95"], np.quantile(s["baseline_stint"].std(axis=1), [.05, .5, .95]))},
           "sigma_eps_core_s": {k: float(v) for k, v in zip(["q05", "q50", "q95"], np.quantile(s["sigma_eps_core"], [.05, .5, .95]))}}
    if "gamma_track" in s:
        out["fuel_trend_half_race_s"] = {k: float(v) for k, v in zip(["q05", "q50", "q95"], np.quantile(s["gamma_track"] * 55.0, [.05, .5, .95]))}
    if "nu_df" in s:
        out["nu_df"] = {k: float(v) for k, v in zip(["q05", "q50", "q95"], np.quantile(s["nu_df"], [.05, .5, .95]))}
    # histograms for the figure
    edges = np.linspace(-15, 25, 161)
    hist_rep = np.stack([np.histogram(np.clip(y[i], -15, 25), bins=edges, density=True)[0] for i in range(min(n, 40))])
    hist_obs = np.histogram(np.clip(ds.y, -15, 25), bins=edges, density=True)[0]
    lnu_edges = np.linspace(-6, 1, 71)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"prior_predictive_{tag}.npz", edges=edges, hist_rep=hist_rep, hist_obs=hist_obs, lnu_edges=lnu_edges,
             lnu_hist=np.histogram(np.log(nu.ravel()), bins=lnu_edges, density=True)[0], nu_flat=nu.ravel()[:200000])
    with open(out_dir / f"prior_predictive_{tag}.json", "w") as f:
        json.dump(out, f, indent=1)
    out["npz"] = str(out_dir / f"prior_predictive_{tag}.npz")
    return out


# --------------------------------------------------------------------------- #
# posterior predictive: conditional on fitted latent paths
# --------------------------------------------------------------------------- #
def _draw_params(p: PosteriorDraws, c: int, d: int, spec: ModelSpec) -> Dict[str, Any]:
    g = lambda k: p.posterior[k][c, d]
    out = {"baseline_stint": g("baseline_stint"), "nu_stint": g("nu_stint"), "sigma_eps_core": float(g("sigma_eps_core")),
           "lam": float(g("lambda_skew")) if "lambda_skew" in p.posterior else None, "eta": float(g("nu_df")) if "nu_df" in p.posterior else None,
           "sigma_eta": float(g("sigma_eta")) if "sigma_eta" in p.posterior else None,
           "gamma_track": g("gamma_track") if "gamma_track" in p.posterior else None}
    return out


def _mu_conditional(ds: Dataset, pr: Dict[str, Any], latent: Optional[np.ndarray], spec: ModelSpec) -> np.ndarray:
    """Conditional mean given one posterior draw; `latent` is the stored lap-level block:
    alpha_lap (centred parameterisation) or z_inno innovations (non-centred)."""
    if latent is not None and spec.process_noise and spec.latent_param == "centred":
        alpha = np.asarray(latent, dtype=np.float64)
    else:
        alpha = pr["baseline_stint"][ds.stint_codes] + pr["nu_stint"][ds.stint_codes] * ds.rel_age
        if latent is not None and pr["sigma_eta"] is not None:
            inno_full = np.zeros(ds.n_laps); inno_full[ds.inno_idx] = np.sqrt(ds.dage[ds.inno_idx]) * np.asarray(latent, dtype=np.float64)
            cs = np.cumsum(inno_full)
            alpha = alpha + pr["sigma_eta"] * (cs - cs[ds.first_pos][ds.stint_codes])
    gam = pr["gamma_track"][ds.track_of_lap()] if pr["gamma_track"] is not None else spec.gamma_fixed_value
    return alpha + gam * ds.fuel_c


def ppc_conditional(ds: Dataset, p: PosteriorDraws, spec: ModelSpec, warmup: int, n_draws: int, seed: int, out_dir: Path, tag: str,
                    quantiles: List[float]) -> Dict[str, Any]:
    """y_rep | fitted latent paths, with data-like truncation and residual-dependence checks."""
    rng = np.random.default_rng(seed)
    assert p.latent is not None, "conditional PPC needs saved latent innovations"
    C, T = p.latent.shape[0], p.latent.shape[1]
    pairs = [(c, t) for c in range(C) for t in range(T)]
    if len(pairs) > n_draws:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, n_draws).astype(int)]
    cut = ds.laps["race_cut_y"].to_numpy(float)
    age_bins = np.array([0, 3, 6, 10, 15, 20, 30, 80])
    age_bin = np.digitize(ds.rel_age, age_bins) - 1
    comp_lap = ds.comp_of_stint[ds.stint_codes]; track_lap = ds.track_of_lap()
    rows_rep, rows_rep_trunc, resid_age_obs, resid_age_rep, resid_comp_obs, resid_comp_rep, resid_track_obs, resid_track_rep = [], [], [], [], [], [], [], []
    pit_all = []
    edges = np.linspace(-6, 10, 161); hist_rep = []
    for (c, t) in pairs:
        d = int(p.latent_iters[t]) - warmup
        pr = _draw_params(p, c, d, spec)
        mu = _mu_conditional(ds, pr, p.latent[c, t], spec)
        yrep = _obs_sample(spec, mu, pr["sigma_eps_core"], pr["lam"], pr["eta"], rng)
        rows_rep.append(_stats(yrep, quantiles))
        keep = yrep <= cut
        rows_rep_trunc.append(_stats(yrep[keep], quantiles))
        r_obs = ds.y - mu; r_rep = yrep - mu
        resid_age_obs.append(np.bincount(age_bin, weights=r_obs, minlength=len(age_bins) - 1) / np.maximum(np.bincount(age_bin, minlength=len(age_bins) - 1), 1))
        resid_age_rep.append(np.bincount(age_bin, weights=r_rep, minlength=len(age_bins) - 1) / np.maximum(np.bincount(age_bin, minlength=len(age_bins) - 1), 1))
        resid_comp_obs.append(np.bincount(comp_lap, weights=r_obs, minlength=len(ds.levels["compound"])) / np.maximum(np.bincount(comp_lap, minlength=len(ds.levels["compound"])), 1))
        resid_comp_rep.append(np.bincount(comp_lap, weights=r_rep, minlength=len(ds.levels["compound"])) / np.maximum(np.bincount(comp_lap, minlength=len(ds.levels["compound"])), 1))
        resid_track_obs.append(np.bincount(track_lap, weights=r_obs, minlength=len(ds.levels["track"])) / np.maximum(np.bincount(track_lap, minlength=len(ds.levels["track"])), 1))
        resid_track_rep.append(np.bincount(track_lap, weights=r_rep, minlength=len(ds.levels["track"])) / np.maximum(np.bincount(track_lap, minlength=len(ds.levels["track"])), 1))
        if len(pit_all) < 20:
            pit_all.append(_obs_cdf(spec, ds.y, mu, pr["sigma_eps_core"], pr["lam"], pr["eta"]))
        if len(hist_rep) < 40:
            hist_rep.append(np.histogram(np.clip(yrep[keep], -6, 10), bins=edges, density=True)[0])
    rep = pd.DataFrame(rows_rep); rep_t = pd.DataFrame(rows_rep_trunc)
    obs = _stats(ds.y, quantiles)
    table = []
    for k in obs:
        if k == "n":
            continue
        for label, R in (("unfiltered", rep), ("truncated_like_data", rep_t)):
            v = R[k].to_numpy()
            table.append({"statistic": k, "replicates": label, "observed": obs[k], "rep_mean": float(v.mean()), "rep_q05": float(np.quantile(v, .05)),
                          "rep_q95": float(np.quantile(v, .95)), "bayes_p_two_sided": float(2 * min((v >= obs[k]).mean(), (v <= obs[k]).mean())),
                          "note": "large n: p-values near 0 can reflect tiny practical misfit" if k.startswith("q") else ("unreliable at low tail index" if "skew" in k else "")})
    table = pd.DataFrame(table)
    # residual dependence tables
    def dep_table(obs_list, rep_list, labels, name):
        O = np.stack(obs_list); R = np.stack(rep_list)
        return pd.DataFrame({"group": labels, "obs_resid_mean": O.mean(0), "rep_resid_mean": R.mean(0), "rep_resid_q05": np.quantile(R, .05, 0),
                             "rep_resid_q95": np.quantile(R, .95, 0), "obs_minus_rep_mean": O.mean(0) - R.mean(0), "kind": name})
    dep = pd.concat([dep_table(resid_age_obs, resid_age_rep, [f"age {age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins) - 1)], "rel_age bin"),
                     dep_table(resid_comp_obs, resid_comp_rep, ds.levels["compound"], "compound"),
                     dep_table(resid_track_obs, resid_track_rep, ds.levels["track"], "track")], ignore_index=True)
    pit = np.concatenate(pit_all)
    pit_hist = np.histogram(pit, bins=20, range=(0, 1))[0] / len(pit)
    # share of observations in the outer 5 % / 1 % of their conditional predictive distribution (nominal: 0.05 / 0.01)
    pit_tails = {"pit_gt_0.95": float((pit > 0.95).mean()), "pit_gt_0.99": float((pit > 0.99).mean()), "pit_lt_0.05": float((pit < 0.05).mean()),
                 "pit_lt_0.01": float((pit < 0.01).mean()), "n_pit_draws": len(pit_all)}
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"ppc_conditional_{tag}.csv", index=False)
    dep.to_csv(out_dir / f"ppc_conditional_residuals_{tag}.csv", index=False)
    np.savez(out_dir / f"ppc_conditional_{tag}.npz", edges=edges, hist_rep=np.stack(hist_rep), hist_obs=np.histogram(np.clip(ds.y, -6, 10), bins=edges, density=True)[0],
             pit_hist=pit_hist, rep_skew=rep["skew_raw_unreliable"].to_numpy(), rep_q99=rep_t["q99"].to_numpy() if "q99" in rep_t else rep["q95"].to_numpy(),
             rep_q95=rep_t["q95"].to_numpy(), rep_p_gt_3=rep_t["p_gt_3"].to_numpy())
    return {"n_draws": len(pairs), "table_csv": str(out_dir / f"ppc_conditional_{tag}.csv"), "residual_csv": str(out_dir / f"ppc_conditional_residuals_{tag}.csv"),
            "npz": str(out_dir / f"ppc_conditional_{tag}.npz"), "pit_hist": [float(v) for v in pit_hist], "pit_tails": pit_tails,
            "headline": {r["statistic"]: {"observed": r["observed"], "rep_mean": r["rep_mean"], "p": r["bayes_p_two_sided"]} for _, r in table[table["replicates"] == "truncated_like_data"].iterrows()
                         if r["statistic"] in ("q05", "q50", "q95", "q99", "iqr", "mad", "p_gt_3", "skew_raw_unreliable")}}


# --------------------------------------------------------------------------- #
# posterior predictive: replicated stints (new stint effects + new latent paths)
# --------------------------------------------------------------------------- #
def ppc_replicated_stints(ds: Dataset, p: PosteriorDraws, spec: ModelSpec, n_draws: int, seed: int, out_dir: Path, tag: str) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    C, D = p.n_chains, p.n_draws
    idx = [(int(i // D), int(i % D)) for i in np.linspace(0, C * D - 1, min(n_draws, C * D)).astype(int)]
    S = ds.n_stints
    obs_slopes = grouped_ols_slopes(ds.stint_codes, ds.rel_age, ds.y, S)
    comp = ds.comp_of_stint
    rep_slopes = []
    sites = p.posterior
    mu_name = mu_site(sites)
    for (c, d) in idx:
        g = lambda k: p.posterior[k][c, d]
        log_nu = g(mu_name) + g("compound_effect")[comp]          # group predictor on the model's own scale (log, or s/lap under the identity link)
        if "track_effect" in p.posterior:
            log_nu = log_nu + g("track_effect")[ds.track_of_stint] + g("driver_effect")[ds.driver_of_stint] + g("team_effect")[ds.team_of_stint]
        if "interaction_effect" in p.posterior:
            log_nu = log_nu + g("interaction_effect")[ds.track_of_stint, comp]
        if "race_effect" in p.posterior:                           # replicated stints of the SAME races: fitted race effects
            log_nu = log_nu + g("race_effect")[ds.race_of_stint]
        if "sigma_stint" in p.posterior:
            nu = np.exp(log_nu + float(g("sigma_stint")) * rng.normal(size=S))
        elif "sigma_stint_add" in p.posterior:
            nu = to_rate(log_nu, sites) + float(g("sigma_stint_add")) * rng.normal(size=S)
        else:
            nu = to_rate(log_nu, sites)
        beta = float(g("baseline_mean")) + g("compound_base_effect")[comp] + float(g("sigma_stint_base")) * rng.normal(size=S)
        alpha = beta[ds.stint_codes] + nu[ds.stint_codes] * ds.rel_age
        if "sigma_eta" in p.posterior:
            inno = np.zeros(ds.n_laps); inno[ds.inno_idx] = np.sqrt(ds.dage[ds.inno_idx]) * rng.normal(size=len(ds.inno_idx))
            cs = np.cumsum(inno); alpha = alpha + float(g("sigma_eta")) * (cs - cs[ds.first_pos][ds.stint_codes])
        gam = g("gamma_track")[ds.track_of_lap()] if "gamma_track" in p.posterior else spec.gamma_fixed_value
        mu = alpha + gam * ds.fuel_c
        yrep = _obs_sample(spec, mu, float(g("sigma_eps_core")), float(g("lambda_skew")) if "lambda_skew" in p.posterior else None,
                           float(g("nu_df")) if "nu_df" in p.posterior else None, rng)
        rep_slopes.append(grouped_ols_slopes(ds.stint_codes, ds.rel_age, yrep, S))
    R = np.stack(rep_slopes)   # (draws, S)
    rows = []
    for ci, cname in enumerate(ds.levels["compound"]):
        m = comp == ci
        o = obs_slopes[m]; o = o[np.isfinite(o)]
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            rq = np.nanquantile(R[:, m], q, axis=1)
            rows.append({"compound": cname, "statistic": f"slope q{int(q*100)}", "observed": float(np.quantile(o, q)), "rep_mean": float(rq.mean()),
                         "rep_q05": float(np.quantile(rq, .05)), "rep_q95": float(np.quantile(rq, .95)), "bayes_p_two_sided": float(2 * min((rq >= np.quantile(o, q)).mean(), (rq <= np.quantile(o, q)).mean()))})
        neg = np.nanmean(R[:, m] < 0, axis=1)
        rows.append({"compound": cname, "statistic": "share of negative raw slopes", "observed": float((o < 0).mean()), "rep_mean": float(neg.mean()),
                     "rep_q05": float(np.quantile(neg, .05)), "rep_q95": float(np.quantile(neg, .95)), "bayes_p_two_sided": float(2 * min((neg >= (o < 0).mean()).mean(), (neg <= (o < 0).mean()).mean()))})
    tab = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_dir / f"ppc_replicated_{tag}.csv", index=False)
    np.savez(out_dir / f"ppc_replicated_{tag}.npz", obs_slopes=obs_slopes, rep_slopes_q=np.nanquantile(R, [.05, .5, .95], axis=0), comp=comp)
    return {"n_draws": len(idx), "table_csv": str(out_dir / f"ppc_replicated_{tag}.csv"), "npz": str(out_dir / f"ppc_replicated_{tag}.npz")}

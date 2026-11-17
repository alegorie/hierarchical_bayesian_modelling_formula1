"""Time-correct predictive evaluation.

Scheme (fixed before looking at any test score):
  * training fit on the training years (e.g. 2022-2023); evaluation on the test
    year (2024).  2024 was part of the old study, so this is a retrospective
    transfer check, not an untouched confirmatory hold-out.
  * target: for every test stint with >= min_prefix_laps + min_target_laps retained
    laps, the first ceil(prefix_fraction * n) retained laps are the calibration
    prefix and the remaining laps are the target.
  * fixed-origin forecast: all target laps predicted from the prefix only.
  * rolling one-step-ahead forecast: each target lap predicted after assimilating
    the previous target laps (a different task; scored separately).
  * information used for a test stint: the training posterior, the stint's
    prefix laps, the race length L_r (fuel proxy) and the race centring constant
    computed from the first quarter of the race only (see data.py).  New
    tracks/drivers/constructors get effects drawn from the population
    distribution (recorded).  Nothing from the stint's future enters.

Inference for the stint-level unknowns (beta_s, u_s, latent path) given the
prefix is done with a bootstrap particle filter whose particles also carry a
posterior draw of the global parameters, so the forecast integrates over
posterior uncertainty.  The filter's effective sample size per lap is recorded.

Scores: MAE and pooled RMSE of the predictive median, CRPS (sample estimate),
ETI coverage and width at two levels, per race and pooled; paired model
differences with race as the sampling unit.

v1.1 (review round 1)
  * log predictive density (LPD) of every target lap: log (1/K) sum_k p(y | particle k) for the
    particle forecasts and the Student-t predictive density for the OLS baseline.  The log score is
    strictly proper and local, i.e. it reacts to the tails, which MAE/CRPS of a median do not.  For the
    rolling scheme the sum of LPDs over a stint's target laps is the exact one-step-ahead (prequential,
    "leave-future-out") log predictive density of those laps given the prefix.
  * every test stint is flagged `origin_after_q1`: the forecast origin lies after the first quarter of
    the race, so the centring constant (median of the first-quarter laps) contains no lap that is later
    than the origin.  `forecast_summary_strict.csv` repeats the scores on that leak-free subset.
  * paired differences use the t quantile with n_races - 1 degrees of freedom (the normal quantile 1.88
    used in v1.0 is anti-conservative for 23 races); the method is recorded in the table.
  * the rate link of the training fit is honoured (f1deg.link); a race effect, if present, is drawn from
    its population distribution because a test race is by definition a new race.
None of these additions consumes random numbers, so MAE/CRPS/coverage of v1.0 are reproduced bit for bit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as st

from .config import ForecastConfig, ModelSpec, derive_seed, fixed_group_scales
from .data import Dataset
from scipy.special import logsumexp

from .hansen import np_logpdf, np_rvs
from .link import group_is_additive, hyper, mu_site
from .sampler import PosteriorDraws


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #
def crps_samples(samples: np.ndarray, y: float) -> float:
    """CRPS of an ensemble forecast (Gneiting & Raftery 2007):
    CRPS = E|X - y| - 0.5 E|X - X'|, with E|X - X'| = (2/K^2) sum_i (2i - K + 1) x_(i) (0-indexed sorted)."""
    x = np.sort(np.asarray(samples, float))
    K = len(x)
    term1 = float(np.mean(np.abs(x - y)))
    i = np.arange(K)
    e_xx = 2.0 * float(np.sum((2 * i - K + 1) * x)) / (K * K)
    return term1 - 0.5 * e_xx


def obs_logpdf(spec: ModelSpec, y, mu, sc, lam, eta):
    """log p(y | mu, observation parameters) for the three likelihoods (NumPy, vectorised over particles)."""
    if spec.likelihood == "skewt":
        return np_logpdf(y, mu, sc, lam, eta)
    if spec.likelihood == "studentt":
        return st.t.logpdf((y - mu) / sc, eta) - np.log(sc)
    return st.norm.logpdf((y - mu) / sc) - np.log(sc)


def mixture_lpd(logp: np.ndarray) -> float:
    """log (1/K) sum_k exp(logp_k): log predictive density of an equally weighted particle cloud."""
    return float(logsumexp(logp) - np.log(len(logp)))


def score_predictive(samples: np.ndarray, y: float, probs=(0.94, 0.5), lpd: Optional[float] = None) -> Dict[str, float]:
    med = float(np.median(samples)); mean = float(samples.mean())
    out = {"y": float(y), "pred_median": med, "pred_mean": mean, "abs_err": abs(med - y), "sq_err": (med - y) ** 2, "crps": crps_samples(samples, y),
           "lpd": float(lpd) if lpd is not None else np.nan}
    for p in probs:
        lo, hi = np.quantile(samples, [(1 - p) / 2, 1 - (1 - p) / 2])
        out[f"cover{int(p*100)}"] = float(lo <= y <= hi); out[f"width{int(p*100)}"] = float(hi - lo)
        out[f"pred_lo{int(p*100)}"] = float(lo); out[f"pred_hi{int(p*100)}"] = float(hi)
    return out


# --------------------------------------------------------------------------- #
# particle filter forecast for one stint
# --------------------------------------------------------------------------- #
class GlobalDraws:
    """Posterior draws of the global parameters of a training fit, indexed by
    training-level labels so that test stints can be mapped (or flagged new)."""

    def __init__(self, p: PosteriorDraws, spec: ModelSpec, train_levels: Dict[str, List[str]], n_use: int, seed: int):
        rng = np.random.default_rng(seed)
        C, D = p.n_chains, p.n_draws
        sel = rng.choice(C * D, size=min(n_use, C * D), replace=False)
        self.ci, self.di = sel // D, sel % D
        self.n = len(sel)
        self.spec = spec
        self.levels = train_levels
        g = lambda k: p.posterior[k][self.ci, self.di] if k in p.posterior else None
        S = p.posterior
        self.additive = group_is_additive(S)          # identity link: the group predictor is the rate itself
        self.mu_log_nu = g(mu_site(S)); self.C = g("compound_effect"); self.T = g("track_effect"); self.D = g("driver_effect"); self.G = g("team_effect")
        self.I = g("interaction_effect"); self.sigma_T = g(hyper("sigma_track", S)); self.sigma_D = g(hyper("sigma_driver", S)); self.sigma_G = g(hyper("sigma_team", S))
        self.sigma_I = g(hyper("sigma_interaction", S)); self.sigma_R = g(hyper("sigma_race", S))
        # v1.2, group_scale == "fixed": there is no posterior of sigma_T, sigma_D, sigma_G, sigma_I.  The population from which
        # the effect of a level unseen in training is drawn is then the model's own prior N(0, s_X^2) with the fixed scale s_X,
        # exactly as the hierarchical model draws it from N(0, sigma_X^2) with posterior draws of sigma_X.
        self.fixed_group_scales = fixed_group_scales(spec) if getattr(spec, "group_scale", "hier") == "fixed" else None
        if self.fixed_group_scales is not None:
            const = lambda v: np.full(self.n, float(v))
            if self.T is not None:
                self.sigma_T, self.sigma_D, self.sigma_G = const(self.fixed_group_scales["track"]), const(self.fixed_group_scales["driver"]), const(self.fixed_group_scales["team"])
            if self.I is not None:
                self.sigma_I = const(self.fixed_group_scales["interaction"])
        self.sigma_U = g("sigma_stint"); self.sigma_U_add = g("sigma_stint_add"); self.beta0 = g("baseline_mean"); self.b = g("compound_base_effect"); self.sigma_beta = g("sigma_stint_base")
        self.gamma = g("gamma_track"); self.mu_lg = g("mu_log_gamma"); self.sig_lg = g("sigma_log_gamma"); self.sigma_eta = g("sigma_eta")
        self.sc = g("sigma_eps_core"); self.lam = g("lambda_skew"); self.eta = g("nu_df")

    def index(self, kind: str, label: str) -> Optional[int]:
        lv = self.levels.get(kind, [])
        return lv.index(label) if label in lv else None


def pf_forecast_stint(gd: GlobalDraws, y: np.ndarray, fuel_c: np.ndarray, dage: np.ndarray, rel_age: np.ndarray, n_prefix: int,
                      comp_i: int, track_label: str, driver_label: str, team_label: str, K: int, rng: np.random.Generator, offset: float = 0.0) -> Dict[str, Any]:
    spec = gd.spec
    n = len(y)
    d = rng.integers(gd.n, size=K)
    q = gd.index("track", track_label); dr = gd.index("driver", driver_label); tm = gd.index("team", team_label)
    new = {"track": q is None, "driver": dr is None, "team": tm is None}
    log_nu = gd.mu_log_nu[d] + gd.C[d, comp_i]
    if gd.T is not None:
        log_nu = log_nu + (gd.T[d, q] if q is not None else gd.sigma_T[d] * rng.normal(size=K))
        log_nu = log_nu + (gd.D[d, dr] if dr is not None else gd.sigma_D[d] * rng.normal(size=K))
        log_nu = log_nu + (gd.G[d, tm] if tm is not None else gd.sigma_G[d] * rng.normal(size=K))
    if gd.I is not None:
        log_nu = log_nu + (gd.I[d, q, comp_i] if q is not None else gd.sigma_I[d] * rng.normal(size=K))
    if gd.sigma_R is not None:                      # a test race is a new race: its effect comes from the population
        log_nu = log_nu + gd.sigma_R[d] * rng.normal(size=K)
    group_rate = (lambda eta: eta) if gd.additive else np.exp      # `log_nu` is the group predictor on the model's own scale
    if gd.sigma_U is not None:
        nu = np.exp(log_nu + gd.sigma_U[d] * rng.normal(size=K))
    elif gd.sigma_U_add is not None:
        nu = group_rate(log_nu) + gd.sigma_U_add[d] * rng.normal(size=K)
    else:
        nu = group_rate(log_nu)
    beta = gd.beta0[d] + gd.b[d, comp_i] + gd.sigma_beta[d] * rng.normal(size=K) + offset
    if gd.gamma is not None:
        gamma = gd.gamma[d, q] if q is not None else np.exp(gd.mu_lg[d] + gd.sig_lg[d] * rng.normal(size=K))
    else:
        gamma = np.full(K, spec.gamma_fixed_value)
    sigma_eta = gd.sigma_eta[d] if gd.sigma_eta is not None else np.zeros(K)
    sc = gd.sc[d]; lam = gd.lam[d] if gd.lam is not None else None; eta = gd.eta[d] if gd.eta is not None else None

    def logpdf(yv, mu):
        return obs_logpdf(spec, yv, mu, sc, lam, eta)

    def sample_obs(mu):
        if spec.likelihood == "skewt":
            return np_rvs(mu, sc, lam, eta, size=mu.shape, rng=rng)
        if spec.likelihood == "studentt":
            return mu + sc * rng.standard_t(eta, size=mu.shape)
        return mu + sc * rng.normal(size=mu.shape)

    alpha = beta.copy()
    ess = []
    P = {"nu": nu, "alpha": alpha, "beta": beta, "gamma": gamma, "sigma_eta": sigma_eta, "sc": sc, "lam": lam, "eta": eta, "d": d}

    def propagate(P, j):
        if dage[j] > 0:
            P["alpha"] = P["alpha"] + P["nu"] * dage[j] + P["sigma_eta"] * np.sqrt(dage[j]) * rng.normal(size=K)

    def assimilate(P, j):
        mu = P["alpha"] + P["gamma"] * fuel_c[j]
        lw = logpdf(y[j], mu)
        lw = lw - lw.max(); w = np.exp(lw); w /= w.sum()
        ess.append(float(1.0 / np.sum(w * w)))
        # systematic resampling
        u = (rng.uniform() + np.arange(K)) / K
        idx = np.searchsorted(np.cumsum(w), u)
        idx = np.minimum(idx, K - 1)
        for k in ("nu", "alpha", "beta", "gamma", "sigma_eta", "d"):
            P[k] = P[k][idx]
        nonlocal sc, lam, eta
        sc = sc[idx]
        if lam is not None:
            lam = lam[idx]
        if eta is not None:
            eta = eta[idx]
        P["sc"], P["lam"], P["eta"] = sc, lam, eta

    for j in range(n_prefix):
        if j > 0:
            propagate(P, j)
        assimilate(P, j)
    # fixed-origin cloud (A) and rolling cloud (B)
    A = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    rows_fixed, rows_roll = [], []
    for j in range(n_prefix, n):
        propagate(A, j)
        muA = A["alpha"] + A["gamma"] * fuel_c[j]
        # use the (possibly resampled) observation parameters of cloud A
        if spec.likelihood == "skewt":
            yA = np_rvs(muA, A["sc"], A["lam"], A["eta"], size=muA.shape, rng=rng)
        elif spec.likelihood == "studentt":
            yA = muA + A["sc"] * rng.standard_t(A["eta"], size=muA.shape)
        else:
            yA = muA + A["sc"] * rng.normal(size=muA.shape)
        lpdA = mixture_lpd(obs_logpdf(spec, y[j], muA, A["sc"], A["lam"], A["eta"]))        # p(y_j | prefix): fixed origin
        rows_fixed.append({"lap_in_stint": j, "rel_age": float(rel_age[j]), **score_predictive(yA, y[j], lpd=lpdA)})
        propagate(P, j)
        muB = P["alpha"] + P["gamma"] * fuel_c[j]
        yB = sample_obs(muB)
        lpdB = mixture_lpd(logpdf(y[j], muB))                                                 # p(y_j | y_1..y_{j-1}): one step ahead
        rows_roll.append({"lap_in_stint": j, "rel_age": float(rel_age[j]), **score_predictive(yB, y[j], lpd=lpdB)})
        assimilate(P, j)
    return {"fixed": rows_fixed, "rolling": rows_roll, "ess_prefix": ess[:n_prefix], "ess_min": float(min(ess[:n_prefix])) if n_prefix > 0 else float(K), "new_groups": new,
            "unique_draws_after_prefix": int(len(np.unique(A["d"]))),      # distinct posterior draws surviving the prefix (static-parameter depletion)
            "nu_posterior_prefix": [float(np.median(A["nu"])), float(np.quantile(A["nu"], .03)), float(np.quantile(A["nu"], .97))]}


# --------------------------------------------------------------------------- #
# OLS baseline forecast for one stint (prefix only, external gamma)
# --------------------------------------------------------------------------- #
def ols_forecast_stint(y, fuel_c, rel_age, n_prefix: int, gamma_ext: float, K: int, rng: np.random.Generator) -> Dict[str, Any]:
    x = rel_age[:n_prefix]; yy = y[:n_prefix] - gamma_ext * fuel_c[:n_prefix]
    X = np.column_stack([np.ones(n_prefix), x])
    beta, res, rank, _ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    dof = max(n_prefix - 2, 1)
    s2 = float(resid @ resid / dof)
    XtX_inv = np.linalg.pinv(X.T @ X)
    rows = []
    for j in range(n_prefix, len(y)):
        xj = np.array([1.0, rel_age[j]])
        mean = float(xj @ beta + gamma_ext * fuel_c[j])
        sd = float(np.sqrt(s2 * (1.0 + xj @ XtX_inv @ xj)))
        samples = mean + sd * rng.standard_t(dof, size=K)
        lpd = float(st.t.logpdf((y[j] - mean) / sd, dof) - np.log(sd)) if sd > 0 else -np.inf    # exact Student-t predictive density
        rows.append({"lap_in_stint": j, "rel_age": float(rel_age[j]), **score_predictive(samples, y[j], lpd=lpd)})
    return {"fixed": rows, "slope": float(beta[1]), "resid_sd": float(np.sqrt(s2))}


# --------------------------------------------------------------------------- #
# evaluation driver
# --------------------------------------------------------------------------- #
def split_test_stints(ds_test: Dataset, fc: ForecastConfig, prefix_fraction: Optional[float] = None, min_prefix_laps: Optional[int] = None) -> List[Tuple[int, int]]:
    """(stint index, number of prefix laps) for every test stint that has enough laps on both sides of the split."""
    frac = fc.prefix_fraction if prefix_fraction is None else float(prefix_fraction)
    min_pre = fc.min_prefix_laps if min_prefix_laps is None else int(min_prefix_laps)
    out = []
    for j in range(ds_test.n_stints):
        n = int((ds_test.stint_codes == j).sum())
        n_pre = int(np.ceil(frac * n))
        if n_pre >= min_pre and n - n_pre >= fc.min_target_laps:
            out.append((j, n_pre))
    return out


def evaluate_models(ds_test: Dataset, models: Dict[str, Tuple[PosteriorDraws, ModelSpec, Dict[str, List[str]]]], fc: ForecastConfig, gamma_ext: float,
                    train_offset: float, out_dir: Path, seed: int, log=print, max_stints: Optional[int] = None,
                    prefix_fraction: Optional[float] = None, min_prefix_laps: Optional[int] = None, tag: str = "") -> Dict[str, Any]:
    """Run the forecast for every test stint under every model (+ OLS baseline).

    `tag` ("" for the main scheme) is appended to the output file names and, when not empty, to the seed
    labels, so that an additional scheme (e.g. a short prefix) never overwrites or re-uses the main one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = split_test_stints(ds_test, fc, prefix_fraction, min_prefix_laps)
    if max_stints is not None:
        splits = splits[:max_stints]
    lab = (lambda *a: a) if not tag else (lambda *a: a + (tag,))
    gds = {name: GlobalDraws(p, spec, levels, fc.posterior_draws_used, derive_seed(seed, *lab("gd", name))) for name, (p, spec, levels) in models.items()}
    st_tab = ds_test.stints
    lap_no = ds_test.laps["LapNumber"].to_numpy(float); L_r = ds_test.laps["L_r"].to_numpy(float)
    lap_rows, stint_rows = [], []
    for k, (j, n_pre) in enumerate(splits):
        m = ds_test.stint_codes == j
        y = ds_test.laps.loc[m, "y_prefix"].to_numpy(float)
        fuel = ds_test.fuel_c[m]; dage = ds_test.dage[m]; rel = ds_test.rel_age[m]
        row = st_tab.iloc[j]
        origin_lap = float(lap_no[m][n_pre - 1])                    # last lap that the forecast is allowed to see
        q1_end = float(np.ceil(L_r[m][0] / 4.0))                    # the centring constant uses cleaned laps with LapNumber <= q1_end
        base = {"stint_code": int(j), "stint_id": row["stint_id"], "race": row["race"], "Location": row["Location"], "Compound": row["Compound"],
                "Driver": row["Driver"], "Constructor": row["Constructor"], "n_laps": int(m.sum()), "n_prefix": n_pre,
                "origin_lap": origin_lap, "origin_after_q1": bool(origin_lap >= q1_end)}
        rng = np.random.default_rng(derive_seed(seed, *lab("ols", row["stint_id"])))
        r = ols_forecast_stint(y, fuel, rel, n_pre, gamma_ext, fc.particles, rng)
        for rr in r["fixed"]:
            lap_rows.append({**base, "model": "ols_fuel_ext", "scheme": "fixed_origin", **rr})
        stint_rows.append({**base, "model": "ols_fuel_ext", "ols_slope": r["slope"], "ols_resid_sd": r["resid_sd"]})
        for name, gd in gds.items():
            rng = np.random.default_rng(derive_seed(seed, *lab("pf", name, row["stint_id"])))
            r = pf_forecast_stint(gd, y, fuel, dage, rel, n_pre, int(row["comp_code"]), row["Location"], row["Driver"], row["Constructor"], fc.particles, rng, offset=train_offset)
            for rr in r["fixed"]:
                lap_rows.append({**base, "model": name, "scheme": "fixed_origin", **rr})
            for rr in r["rolling"]:
                lap_rows.append({**base, "model": name, "scheme": "rolling_one_step", **rr})
            stint_rows.append({**base, "model": name, "pf_ess_min": r["ess_min"], "pf_unique_draws": r["unique_draws_after_prefix"], "new_track": r["new_groups"]["track"],
                               "new_driver": r["new_groups"]["driver"], "new_team": r["new_groups"]["team"], "nu_median_prefix": r["nu_posterior_prefix"][0],
                               "nu_lo": r["nu_posterior_prefix"][1], "nu_hi": r["nu_posterior_prefix"][2]})
        if (k + 1) % 50 == 0:
            log(f"  forecast{tag}: {k+1}/{len(splits)} stints")
    laps = pd.DataFrame(lap_rows); stints = pd.DataFrame(stint_rows)
    laps.to_csv(out_dir / f"forecast_laps{tag}.csv", index=False); stints.to_csv(out_dir / f"forecast_stints{tag}.csv", index=False)
    summary = summarise_scores(laps)
    summary.to_csv(out_dir / f"forecast_summary{tag}.csv", index=False)
    pairs = paired_differences(laps)
    pairs.to_csv(out_dir / f"forecast_paired_differences{tag}.csv", index=False)
    strict = laps[laps["origin_after_q1"]] if len(laps) else laps
    summary_strict = summarise_scores(strict); summary_strict.to_csv(out_dir / f"forecast_summary_strict{tag}.csv", index=False)
    pairs_strict = paired_differences(strict); pairs_strict.to_csv(out_dir / f"forecast_paired_differences_strict{tag}.csv", index=False)
    pf_rows = stints[stints["model"] != "ols_fuel_ext"] if len(stints) else stints
    ess_by_model = (pf_rows.groupby("model").agg(ess_min_q05=("pf_ess_min", lambda v: float(np.quantile(v, .05))), ess_min_q50=("pf_ess_min", "median"),
                                                 share_ess_lt_50=("pf_ess_min", lambda v: float((v < 50).mean())), unique_draws_q05=("pf_unique_draws", lambda v: float(np.quantile(v, .05))),
                                                 unique_draws_q50=("pf_unique_draws", "median")).reset_index() if len(pf_rows) else pd.DataFrame())
    ess_by_model.to_csv(out_dir / f"forecast_filter_health{tag}.csv", index=False)
    n_ols = laps[laps["model"] == "ols_fuel_ext"] if len(laps) else laps
    info = {"n_test_stints": len(splits), "n_target_laps": int(len(n_ols)), "n_target_laps_strict": int(n_ols["origin_after_q1"].sum()) if len(n_ols) else 0,
            "n_test_stints_strict": int(sum(1 for _ in stints[(stints["model"] == "ols_fuel_ext") & stints["origin_after_q1"]].index)) if len(stints) else 0,
            "prefix_fraction": fc.prefix_fraction if prefix_fraction is None else float(prefix_fraction),
            "min_prefix_laps": fc.min_prefix_laps if min_prefix_laps is None else int(min_prefix_laps),
            "laps_csv": str(out_dir / f"forecast_laps{tag}.csv"),
            "stints_csv": str(out_dir / f"forecast_stints{tag}.csv"), "summary_csv": str(out_dir / f"forecast_summary{tag}.csv"), "paired_csv": str(out_dir / f"forecast_paired_differences{tag}.csv"),
            "summary_strict_csv": str(out_dir / f"forecast_summary_strict{tag}.csv"), "paired_strict_csv": str(out_dir / f"forecast_paired_differences_strict{tag}.csv"),
            "filter_health_csv": str(out_dir / f"forecast_filter_health{tag}.csv"),
            "summary": summary.to_dict(orient="records"),
            # v1.0 key kept: quantiles of the minimum filter ESS pooled over models (per-model values are in forecast_filter_health.csv)
            "pf_ess_min_quantiles": {k: float(v) for k, v in zip(["q05", "q50"], np.quantile(pf_rows["pf_ess_min"], [.05, .5]))} if len(pf_rows) else {}}
    with open(out_dir / f"forecast_info{tag}.json", "w") as f:
        json.dump(info, f, indent=1, default=float)
    return info


def summarise_scores(laps: pd.DataFrame) -> pd.DataFrame:
    """Scores per (model, scheme): pooled over laps and as a mean over races with its standard error
    (race = sampling unit).  LPD is the mean log predictive density per target lap (higher is better)."""
    rows = []
    if not len(laps):
        return pd.DataFrame(rows)
    has_lpd = "lpd" in laps
    for (model, scheme), g in laps.groupby(["model", "scheme"]):
        agg = {"mae": ("abs_err", "mean"), "rmse": ("sq_err", lambda v: np.sqrt(v.mean())), "crps": ("crps", "mean")}
        if has_lpd:
            agg["lpd"] = ("lpd", "mean")
        by_race = g.groupby("race").agg(**agg)
        nr = len(by_race)
        se = lambda col: by_race[col].std(ddof=1) / np.sqrt(nr) if nr > 1 else np.nan
        row = {"model": model, "scheme": scheme, "n_laps": len(g), "n_races": g["race"].nunique(), "MAE": g["abs_err"].mean(), "RMSE_pooled": np.sqrt(g["sq_err"].mean()),
               "CRPS": g["crps"].mean(), "coverage94": g["cover94"].mean(), "width94": g["width94"].mean(), "coverage50": g["cover50"].mean(), "width50": g["width50"].mean(),
               "MAE_race_mean": by_race["mae"].mean(), "MAE_race_se": se("mae"), "CRPS_race_mean": by_race["crps"].mean(), "CRPS_race_se": se("crps")}
        if has_lpd:
            row.update({"LPD": g["lpd"].mean(), "LPD_race_mean": by_race["lpd"].mean(), "LPD_race_se": se("lpd"), "n_lpd_nonfinite": int((~np.isfinite(g["lpd"])).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(laps: pd.DataFrame, scheme: str = "fixed_origin", prob: float = 0.94, pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    """Paired model differences with the race as the sampling unit.

    For every pair (a, b) and metric the per-race means are differenced on the races both models have;
    the interval is mean ± t_{(1+prob)/2, n-1} · SE.  MAE and CRPS are losses (a better <=> difference < 0);
    LPD is a gain (a better <=> difference > 0); `share_races_a_better` accounts for the direction.

    `pairs` = None: every unordered pair, a < b alphabetically (the tables of the forecast stage).  A list of
    (a, b) gives exactly those ordered differences a − b (pairs with a model that was not scored are skipped)."""
    rows = []
    if not len(laps):
        return pd.DataFrame(rows)
    g = laps[laps["scheme"] == scheme]
    models = sorted(g["model"].unique())
    metrics = {"mae": ("abs_err", -1), "crps": ("crps", -1)}
    if "lpd" in g:
        metrics["lpd"] = ("lpd", +1)
    per_race = g.groupby(["model", "race"]).agg(**{k: (v[0], "mean") for k, v in metrics.items()}).reset_index()
    todo = [(a, b) for a in models for b in models if a < b] if pairs is None else [(a, b) for a, b in pairs if a in models and b in models]
    for a, b in todo:
        A = per_race[per_race["model"] == a].set_index("race"); B = per_race[per_race["model"] == b].set_index("race")
        common = A.index.intersection(B.index)
        for metric, (_, direction) in metrics.items():
            diff = (A.loc[common, metric] - B.loc[common, metric]).to_numpy()
            diff = diff[np.isfinite(diff)]
            n = len(diff)
            se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            tq = float(st.t.ppf(0.5 + prob / 2.0, n - 1)) if n > 1 else np.nan
            rows.append({"model_a": a, "model_b": b, "metric": metric, "n_races": n, "mean_diff_a_minus_b": diff.mean() if n else np.nan, "se_race": se,
                         "ci94_lo": diff.mean() - tq * se if n > 1 else np.nan, "ci94_hi": diff.mean() + tq * se if n > 1 else np.nan,
                         "t_quantile": tq, "ci_method": f"t({n-1}), {prob:.0%}" if n > 1 else "",
                         "share_races_a_better": float((direction * diff > 0).mean()) if n else np.nan,
                         "higher_is_better": direction > 0})
    return pd.DataFrame(rows)


# Three nested specifications trained on the same seasons (v1.2).  Each step isolates one question:
#   pooled_train    -> unpooled_train : do the group effects T, D, G, I help at all (with fixed prior scales)?
#   unpooled_train  -> primary_train  : does LEARNING their scales (adaptive hierarchical partial pooling) help?
HIERARCHY_LADDER = [("primary_train", "unpooled_train", "навчені масштаби − фіксовані масштаби (primary_train − unpooled_train)"),
                    ("unpooled_train", "pooled_train", "групові ефекти з фіксованими масштабами − без групових ефектів (unpooled_train − pooled_train)"),
                    ("primary_train", "pooled_train", "навчені масштаби − без групових ефектів (primary_train − pooled_train)")]


def hierarchy_ablation(out_dir: Path, schemes: Dict[str, str], prob: float = 0.94) -> pd.DataFrame:
    """Ordered paired differences a − b by race between the specifications of HIERARCHY_LADDER, for every forecast scheme
    on disk (`schemes`: file tag -> label; "" is the main scheme) and for up to four sets of stints:

      * all stints;
      * stints whose forecast origin lies after the window used for centring (no leakage through the centring constant);
      * stints whose track, driver and constructor were all present in the training data;
      * stints with at least one level unseen in training.  For these the hierarchical model draws the new effect from
        N(0, sigma_X^2) with the learned sigma_X, the fixed-scale model from its prior N(0, s_X^2): a difference found only
        here is about predicting NEW levels, not about pooling among the known ones.

    A row exists only when both models of the pair were scored (and the set is not empty).  `reading` states what the
    interval allows one to say and nothing more: an interval that contains zero is reported as "difference not confirmed",
    never as equivalence."""
    rows = []
    for tag, label in schemes.items():
        f = Path(out_dir) / f"forecast_laps{tag}.csv"
        if not f.exists() or f.stat().st_size < 3:
            continue
        laps = pd.read_csv(f)
        if not len(laps):
            continue
        subsets = [("усі відрізки", laps)]
        if "origin_after_q1" in laps:
            subsets.append(("початок прогнозу після вікна центрування", laps[laps["origin_after_q1"].astype(bool)]))
        fs = Path(out_dir) / f"forecast_stints{tag}.csv"
        if fs.exists() and fs.stat().st_size > 2 and "stint_id" in laps:
            st_tab = pd.read_csv(fs)
            if {"stint_id", "new_track", "new_driver", "new_team"} <= set(st_tab.columns):
                # the flags depend only on the levels of the training data, which all fits of the ladder share
                fl = st_tab.dropna(subset=["new_track"]).copy()
                fl["has_new"] = fl[["new_track", "new_driver", "new_team"]].astype(bool).any(axis=1)
                has_new = fl.groupby("stint_id")["has_new"].max()
                flag = laps["stint_id"].map(has_new)
                known, new = laps[flag == False], laps[flag == True]      # noqa: E712 (NaN = stint without a particle-filter row)
                if len(known) and len(new):
                    subsets.append(("траса, пілот і конструктор були в навчальних даних", known))
                    subsets.append(("є рівень, якого не було в навчальних даних", new))
        for sub_label, d in subsets:
            if not len(d):
                continue
            pr = paired_differences(d, prob=prob, pairs=[(a, b) for a, b, _ in HIERARCHY_LADDER])
            names = {(a, b): lab for a, b, lab in HIERARCHY_LADDER}
            for _, r in pr.iterrows():
                lo, hi = r["ci94_lo"], r["ci94_hi"]
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    reading = "замало гонок для інтервалу"
                elif lo <= 0.0 <= hi:
                    reading = "інтервал містить нуль: різницю не підтверджено"
                else:
                    a_better = (r["mean_diff_a_minus_b"] > 0) == bool(r["higher_is_better"])
                    reading = f"інтервал не містить нуля: краща {'перша' if a_better else 'друга'} модель пари"
                rows.append({"scheme": label, "stints": sub_label, "comparison": names[(r["model_a"], r["model_b"])], "model_a": r["model_a"], "model_b": r["model_b"], "metric": r["metric"],
                             "higher_is_better": bool(r["higher_is_better"]), "n_races": int(r["n_races"]), "mean_diff_a_minus_b": r["mean_diff_a_minus_b"], "se_race": r["se_race"],
                             "ci94_lo": lo, "ci94_hi": hi, "share_races_a_better": r["share_races_a_better"], "reading": reading})
    return pd.DataFrame(rows)

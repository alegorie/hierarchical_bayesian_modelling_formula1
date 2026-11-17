"""Estimands computed from posterior draws (tables; figures live in export.py).

Every table states the estimand explicitly:
  * multiplier  exp(X)                     — dimensionless, relative to the geometric mean of the group
  * rate        exp(mu_nu + X)             — s per lap of tyre age at the geometric-mean level of all OTHER factors
  * full rate   exp(mu_nu + C_c + T_q + I_qc) — s/lap for compound c at track q (driver/constructor at their geometric mean)
  * hyper-SD    sigma_X                    — prior/population scale of the effects
  * realised SD sd(X_1..X_K) per draw      — finite-group spread of the fitted effects (NOT the same object)
Intervals are equal-tailed (ETI) at `prob` (default 94%).

v1.1: every table is link-aware (f1deg.link).  Under the identity link the effects are additive in
s/lap, so "multiplier" columns are empty and `effect_*` columns carry the additive effects; rates
are computed with the model's own link in all cases, and `sign_table` reports how much posterior
mass lies at or below zero (trivially none under the log link — stated as "by construction").
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import fixed_group_scales
from .data import Dataset
from .link import group_is_additive, hyper, link_name, mu_site, predictor_units, to_rate
from .sampler import PosteriorDraws

try:
    import arviz as az
except Exception:  # pragma: no cover
    az = None


def _eti(x: np.ndarray, prob: float, axis=0):
    lo, hi = (1 - prob) / 2, 1 - (1 - prob) / 2
    return np.quantile(x, lo, axis=axis), np.quantile(x, hi, axis=axis)


def prob_with_mcse(ind: np.ndarray) -> Tuple[float, float, float]:
    """Posterior probability of an event from (chain, draw) indicator draws with an
    MCSE based on the indicator's ESS.  For events seen in all/none of the draws the
    MCSE is undefined; we return the rule-of-three bound 3/ESS as the uncertainty."""
    p = float(ind.mean())
    n = ind.size
    try:
        ess = float(az.ess(ind.astype(float), method="bulk")) if az is not None and ind.ndim == 2 and ind.shape[1] >= 4 else float(n)
    except Exception:
        ess = float(n)
    if not np.isfinite(ess) or ess <= 0:
        ess = float(n)
    if p in (0.0, 1.0):
        return p, 3.0 / ess, ess
    return p, float(np.sqrt(p * (1 - p) / ess)), ess


def compound_tables(p: PosteriorDraws, ds: Dataset, prob: float) -> Dict[str, pd.DataFrame]:
    comps = ds.levels["compound"]
    S = p.posterior
    add = group_is_additive(S)
    C = S["compound_effect"]; mu = S[mu_site(S)]
    rate = to_rate(mu[..., None] + C, S)
    mult = np.full_like(rate, np.nan) if add else np.exp(C)        # a multiplier exists only under a multiplicative link
    rows = []
    for i, c in enumerate(comps):
        m = mult[..., i].reshape(-1); r = rate[..., i].reshape(-1); e = C[..., i].reshape(-1)
        rows.append({"compound": c, "n_stints": int((ds.comp_of_stint == i).sum()), "n_laps": int(np.isin(ds.stint_codes, np.flatnonzero(ds.comp_of_stint == i)).sum()),
                     "multiplier_median": np.median(m), "multiplier_lo": _eti(m, prob)[0], "multiplier_hi": _eti(m, prob)[1],
                     "rate_median_s_per_lap": np.median(r), "rate_lo": _eti(r, prob)[0], "rate_hi": _eti(r, prob)[1], "rate_mean": r.mean(), "rate_sd": r.std(),
                     "effect_median": np.median(e), "effect_lo": _eti(e, prob)[0], "effect_hi": _eti(e, prob)[1], "effect_scale": predictor_units(S),
                     "P(rate>0)": float((r > 0).mean())})
    tab = pd.DataFrame(rows)
    # contrasts (joint draws)
    crow = []
    for i, a in enumerate(comps):
        for j, b in enumerate(comps):
            if i >= j:
                continue
            diff = (rate[..., i] - rate[..., j])
            # exp(C_i - C_j) under the log link; the plain ratio of rates otherwise (unstable if a rate is near zero)
            ratio = rate[..., i] / rate[..., j] if add else np.exp(C[..., i] - C[..., j])
            pr, se, ess = prob_with_mcse(diff > 0)
            crow.append({"contrast": f"{a} − {b}", "rate_diff_median_s_per_lap": float(np.median(diff)), "rate_diff_lo": float(_eti(diff.reshape(-1), prob)[0]),
                         "rate_diff_hi": float(_eti(diff.reshape(-1), prob)[1]), "ratio_median": float(np.median(ratio)), "ratio_lo": float(_eti(ratio.reshape(-1), prob)[0]),
                         "ratio_hi": float(_eti(ratio.reshape(-1), prob)[1]), "P(diff>0)": pr, "P_mcse_or_bound": se, "ess_indicator": ess})
    contrasts = pd.DataFrame(crow)
    # order probabilities
    orows = []
    if all(c in comps for c in ("SOFT", "MEDIUM", "HARD")):
        iS, iM, iH = (comps.index(c) for c in ("SOFT", "MEDIUM", "HARD"))
        events = {"SOFT > MEDIUM > HARD": (rate[..., iS] > rate[..., iM]) & (rate[..., iM] > rate[..., iH]), "SOFT > HARD": rate[..., iS] > rate[..., iH],
                  "SOFT > MEDIUM": rate[..., iS] > rate[..., iM], "MEDIUM > HARD": rate[..., iM] > rate[..., iH]}
        for k, ind in events.items():
            pr, se, ess = prob_with_mcse(ind)
            orows.append({"scope": "global (geometric-mean track)" if not add else "global (mean track)", "event": k, "probability": pr, "mcse_or_rule_of_three_bound": se, "ess_indicator": ess, "n_draws": ind.size})
        if "interaction_effect" in S and "track_effect" in S:
            T = S["track_effect"]; I = S["interaction_effect"]
            for q, tname in enumerate(ds.levels["track"]):
                full = to_rate(mu[..., None] + C + T[..., q][..., None] + I[..., q, :], S)
                ind = (full[..., iS] > full[..., iM]) & (full[..., iM] > full[..., iH])
                pr, se, ess = prob_with_mcse(ind)
                ind2 = full[..., iS] > full[..., iH]
                pr2, se2, _ = prob_with_mcse(ind2)
                n_cells = {c: int(((ds.track_of_stint == q) & (ds.comp_of_stint == comps.index(c))).sum()) for c in ("SOFT", "MEDIUM", "HARD")}
                orows.append({"scope": f"track {tname}", "event": "SOFT > MEDIUM > HARD (full rate)", "probability": pr, "mcse_or_rule_of_three_bound": se, "ess_indicator": ess,
                              "n_draws": ind.size, "P(SOFT>HARD)": pr2, "n_stints_S/M/H": f"{n_cells['SOFT']}/{n_cells['MEDIUM']}/{n_cells['HARD']}",
                              "min_cell_n": min(n_cells.values())})
    order = pd.DataFrame(orows)
    return {"compound": tab, "compound_contrasts": contrasts, "order_probabilities": order}


def track_table(p: PosteriorDraws, ds: Dataset, prob: float) -> pd.DataFrame:
    S = p.posterior
    if "track_effect" not in S:
        return pd.DataFrame()
    add = group_is_additive(S)
    T = S["track_effect"]; mu = S[mu_site(S)]
    rate = to_rate(mu[..., None] + T, S)
    mult = np.full_like(rate, np.nan) if add else np.exp(T)
    med_lap = ds.laps.groupby("Location")["LapTime_seconds"].median()
    rows = []
    for q, t in enumerate(ds.levels["track"]):
        m = mult[..., q].reshape(-1); r = rate[..., q].reshape(-1); e = T[..., q].reshape(-1)
        ns = int((ds.track_of_stint == q).sum()); nl = int((ds.track_of_lap() == q).sum())
        ref = float(med_lap.get(t, np.nan))
        rel = 100 * r / ref if ref == ref else np.full_like(r, np.nan)
        rows.append({"track": t, "n_stints": ns, "n_laps": nl, "n_races": int(ds.stints.loc[ds.stints["track_code"] == q, "race"].nunique()),
                     "multiplier_median": np.median(m), "multiplier_lo": _eti(m, prob)[0], "multiplier_hi": _eti(m, prob)[1],
                     "rate_median_s_per_lap": np.median(r), "rate_lo": _eti(r, prob)[0], "rate_hi": _eti(r, prob)[1],
                     "effect_median": np.median(e), "effect_lo": _eti(e, prob)[0], "effect_hi": _eti(e, prob)[1],
                     "reference_lap_time_s": ref, "relative_rate_pct_per_lap": float(np.median(rel)),
                     "relative_rate_lo": float(_eti(rel, prob)[0]), "relative_rate_hi": float(_eti(rel, prob)[1]),
                     "P(effect>0)": prob_with_mcse(T[..., q] > 0)[0], "P(rate>0)": float((r > 0).mean())})
    if "gamma_track" in S:
        g = S["gamma_track"]
        for q in range(len(rows)):
            gq = g[..., q].reshape(-1)
            rows[q].update({"gamma_median_s_per_kg": np.median(gq), "gamma_lo": _eti(gq, prob)[0], "gamma_hi": _eti(gq, prob)[1]})
    return pd.DataFrame(rows)


COMPONENT_UK = {"compound": "компаунд", "track": "траса", "driver": "пілот", "team": "конструктор", "interaction": "траса×компаунд",
                "race": "гонка (траса×сезон)", "stint": "відрізок (залишок)"}


def scales_table(p: PosteriorDraws, ds: Dataset, prob: float, spec=None) -> pd.DataFrame:
    """Scales of heterogeneity on the scale of the rate's linear predictor (log s/lap, or s/lap under
    the identity link): hyper-SDs, realised SDs of the fitted effects and a design-weighted variance
    decomposition of the linear predictor across stints (marginal variances + covariance remainder).

    For a specification with fixed group scales (``spec.group_scale == "fixed"``, v1.2) there are no hyper-SDs of
    T, D, G, I; the fixed prior scale is listed instead (kind "fixed prior scale", no interval), so that it is never
    mistaken for an estimate.  The realised SDs and the variance decomposition are computed in the same way."""
    S = p.posterior
    units = predictor_units(S)
    rows = []
    pairs = [("sigma_track", "track_effect", "track"), ("sigma_compound", "compound_effect", "compound"), ("sigma_driver", "driver_effect", "driver"),
             ("sigma_team", "team_effect", "team"), ("sigma_interaction", "interaction_effect", "interaction"), ("sigma_race", "race_effect", "race")]
    fixed = fixed_group_scales(spec) if (spec is not None and getattr(spec, "group_scale", "hier") == "fixed") else {}
    for hyper_base, eff, key in pairs:
        label = COMPONENT_UK[key]
        h = hyper(hyper_base, S)
        if h in S:
            x = p.flat(h)
            rows.append({"component": label, "quantity": f"hyper-SD {h}", "median": np.median(x), "lo": _eti(x, prob)[0], "hi": _eti(x, prob)[1], "kind": "hyper-SD (population scale)", "units": units})
        elif key in fixed and eff in S:
            rows.append({"component": label, "quantity": f"fixed prior scale of {eff} (not estimated)", "median": fixed[key], "lo": np.nan, "hi": np.nan, "kind": "fixed prior scale", "units": units})
        if eff is not None and eff in S:
            e = p.flat(eff); e = e.reshape(e.shape[0], -1)
            sdr = e.std(axis=1, ddof=1)
            rows.append({"component": label, "quantity": f"realised SD of {eff} ({e.shape[1]} values)", "median": np.median(sdr), "lo": _eti(sdr, prob)[0], "hi": _eti(sdr, prob)[1], "kind": "realised finite-group SD", "units": units})
    stint_same_scale = None          # stint deviation on the same scale as the group predictor?
    if "sigma_stint" in S:
        x = p.flat("sigma_stint")
        rows.append({"component": COMPONENT_UK["stint"], "quantity": "hyper-SD sigma_stint", "median": np.median(x), "lo": _eti(x, prob)[0], "hi": _eti(x, prob)[1], "kind": "hyper-SD (population scale)", "units": "лог с/коло"})
        stint_same_scale = "sigma_stint"
    if "sigma_stint_add" in S:
        x = p.flat("sigma_stint_add")
        rows.append({"component": COMPONENT_UK["stint"], "quantity": "hyper-SD sigma_stint_add", "median": np.median(x), "lo": _eti(x, prob)[0], "hi": _eti(x, prob)[1], "kind": "hyper-SD (population scale)", "units": "с/коло"})
        if group_is_additive(S):
            stint_same_scale = "sigma_stint_add"
    # design-weighted variance decomposition of the linear predictor across stints (with covariance terms)
    comp = {}
    comp["compound"] = p.flat("compound_effect")[:, ds.comp_of_stint]
    if "track_effect" in S:
        comp["track"] = p.flat("track_effect")[:, ds.track_of_stint]; comp["driver"] = p.flat("driver_effect")[:, ds.driver_of_stint]; comp["team"] = p.flat("team_effect")[:, ds.team_of_stint]
    if "interaction_effect" in S:
        comp["interaction"] = p.flat("interaction_effect")[:, ds.track_of_stint, ds.comp_of_stint]
    if "race_effect" in S:
        comp["race"] = p.flat("race_effect")[:, ds.race_of_stint]
    if stint_same_scale is not None and "z_stint_deg" in S:
        comp["stint"] = p.flat(stint_same_scale)[:, None] * p.flat("z_stint_deg")
    total = sum(comp.values())
    var_total = total.var(axis=1, ddof=1)
    what = "ν_s" if group_is_additive(S) else "log ν_s"
    for k, v in comp.items():
        vv = v.var(axis=1, ddof=1)
        rows.append({"component": COMPONENT_UK[k], "quantity": f"var share of {what} across stints (design-weighted)", "median": np.median(vv / var_total), "lo": _eti(vv / var_total, prob)[0],
                     "hi": _eti(vv / var_total, prob)[1], "kind": "variance decomposition (marginal variances; covariances below)", "units": "частка"})
    cov_sum = var_total - sum(v.var(axis=1, ddof=1) for v in comp.values())
    rows.append({"component": "2×коваріації (перехресний дизайн)", "quantity": "share", "median": np.median(cov_sum / var_total), "lo": _eti(cov_sum / var_total, prob)[0],
                 "hi": _eti(cov_sum / var_total, prob)[1], "kind": "variance decomposition", "units": "частка"})
    rows.append({"component": "разом", "quantity": f"SD of {what} across stints" + ("" if stint_same_scale or "z_stint_deg" not in S else " (group part only: stint deviation is on another scale)"),
                 "median": np.median(np.sqrt(var_total)), "lo": _eti(np.sqrt(var_total), prob)[0], "hi": _eti(np.sqrt(var_total), prob)[1], "kind": "total", "units": units})
    return pd.DataFrame(rows)


def interaction_table(p: PosteriorDraws, ds: Dataset, prob: float) -> pd.DataFrame:
    S = p.posterior
    if "interaction_effect" not in S or "track_effect" not in S:
        return pd.DataFrame()
    add = group_is_additive(S)
    I = S["interaction_effect"]; C = S["compound_effect"]; T = S["track_effect"]; mu = S[mu_site(S)]
    rows = []
    for q, t in enumerate(ds.levels["track"]):
        for c, cn in enumerate(ds.levels["compound"]):
            x = I[..., q, c]; xf = x.reshape(-1)
            m = np.full_like(xf, np.nan) if add else np.exp(xf)
            full = to_rate(mu + C[..., c] + T[..., q] + x, S).reshape(-1)
            n = int(((ds.track_of_stint == q) & (ds.comp_of_stint == c)).sum())
            pr, se, _ = prob_with_mcse(x > 0)
            rows.append({"track": t, "compound": cn, "n_stints": n, "multiplier_median": np.median(m), "multiplier_lo": _eti(m, prob)[0], "multiplier_hi": _eti(m, prob)[1],
                         "effect_median": np.median(xf), "effect_lo": _eti(xf, prob)[0], "effect_hi": _eti(xf, prob)[1],
                         "interval_width_log": float(np.diff(_eti(xf, prob))[0]), "P(I>0)": pr, "full_rate_median_s_per_lap": np.median(full), "full_rate_lo": _eti(full, prob)[0],
                         "full_rate_hi": _eti(full, prob)[1], "P(full_rate>0)": float((full > 0).mean()), "weakly_informed": n < 5})
    return pd.DataFrame(rows)


def group_contrast_table(p: PosteriorDraws, ds: Dataset, prob: float, kind: str) -> pd.DataFrame:
    site, lv, code = {"team": ("team_effect", "team", ds.team_of_stint), "driver": ("driver_effect", "driver", ds.driver_of_stint)}[kind]
    if site not in p.posterior:
        return pd.DataFrame()
    E = p.posterior[site]
    add = group_is_additive(p.posterior)
    mult = np.full_like(E, np.nan) if add else np.exp(E)
    K = E.shape[-1]
    rank = (-E.reshape(-1, K)).argsort(axis=1).argsort(axis=1) + 1       # ranks of the effects (identical to ranks of exp(effect))
    rows = []
    for k, name in enumerate(ds.levels[lv]):
        m = mult[..., k].reshape(-1); e = E[..., k].reshape(-1)
        rows.append({kind: name, "n_stints": int((code == k).sum()), "n_laps": int(np.isin(ds.stint_codes, np.flatnonzero(code == k)).sum()),
                     "multiplier_median": np.median(m), "multiplier_lo": _eti(m, prob)[0], "multiplier_hi": _eti(m, prob)[1],
                     "effect_median": np.median(e), "effect_lo": _eti(e, prob)[0], "effect_hi": _eti(e, prob)[1], "P(effect>0)": prob_with_mcse(E[..., k] > 0)[0],
                     "expected_rank": rank[:, k].mean(), "P(rank=1)": (rank[:, k] == 1).mean(), "P(rank<=3)": (rank[:, k] <= 3).mean()})
    tab = pd.DataFrame(rows).sort_values("effect_median", ascending=False)
    # pairwise ratio of the extreme pair + sigma comparison
    return tab


def pooling_effect_table(hier: Dict[str, pd.DataFrame], fixed: Dict[str, pd.DataFrame], hier_name: str = "primary", fixed_name: str = "unpooled") -> pd.DataFrame:
    """Group effects under learned scales (`hier`) and under fixed scales (`fixed`), level by level (v1.2).

    Inputs are the tables of `track_table`, `group_contrast_table` (team, driver) and `interaction_table` of two fits on the
    SAME data.  Output: one row per level with the posterior median and interval of the effect on the scale of the rate's
    linear predictor in both fits, their difference and the number of stints.  With adaptive partial pooling the effects of
    groups whose scale is estimated small are pulled towards zero; the table shows by how much, it does not judge which
    estimate is better (that is the question of the forecast comparison)."""
    rows = []

    def add(kind: str, a: pd.DataFrame, b: pd.DataFrame, keys: List[str], cols: Tuple[str, str, str], n_col: str = "n_stints"):
        if a is None or b is None or not len(a) or not len(b) or not all(c in a and c in b for c in cols):
            return
        m = a.merge(b, on=keys, suffixes=("_h", "_f"))
        for _, r in m.iterrows():
            rows.append({"group": kind, "level": " / ".join(str(r[k]) for k in keys), "n_stints": int(r[f"{n_col}_h"]) if f"{n_col}_h" in r else np.nan,
                         f"effect_{hier_name}": r[f"{cols[0]}_h"], f"lo_{hier_name}": r[f"{cols[1]}_h"], f"hi_{hier_name}": r[f"{cols[2]}_h"],
                         f"effect_{fixed_name}": r[f"{cols[0]}_f"], f"lo_{fixed_name}": r[f"{cols[1]}_f"], f"hi_{fixed_name}": r[f"{cols[2]}_f"],
                         "difference_hier_minus_fixed": r[f"{cols[0]}_h"] - r[f"{cols[0]}_f"],
                         "abs_effect_ratio_hier_over_fixed": abs(r[f"{cols[0]}_h"]) / abs(r[f"{cols[0]}_f"]) if abs(r[f"{cols[0]}_f"]) > 1e-12 else np.nan})

    eff = ("effect_median", "effect_lo", "effect_hi")
    add(COMPONENT_UK["track"], hier.get("tracks"), fixed.get("tracks"), ["track"], eff)
    add(COMPONENT_UK["team"], hier.get("team"), fixed.get("team"), ["team"], eff)
    add(COMPONENT_UK["driver"], hier.get("driver"), fixed.get("driver"), ["driver"], eff)
    add(COMPONENT_UK["interaction"], hier.get("interaction"), fixed.get("interaction"), ["track", "compound"], eff)
    return pd.DataFrame(rows)


def sigma_comparison(p: PosteriorDraws, prob: float) -> pd.DataFrame:
    rows = []
    S = p.posterior
    sg, sd_, st_ = hyper("sigma_team", S), hyper("sigma_driver", S), hyper("sigma_track", S)
    unit = "с/коло" if group_is_additive(S) else "log-rate"
    if sg in S and sd_ in S:
        g = p.flat(sg); d = p.flat(sd_)
        diff = g - d; ratio = g / d
        pr, se, _ = prob_with_mcse(S[sg] > S[sd_])
        rows.append({"comparison": f"σ_G − σ_D ({unit})", "median": np.median(diff), "lo": _eti(diff, prob)[0], "hi": _eti(diff, prob)[1], "P(>0)": pr, "mcse": se})
        rows.append({"comparison": "σ_G / σ_D", "median": np.median(ratio), "lo": _eti(ratio, prob)[0], "hi": _eti(ratio, prob)[1], "P(>1)": pr, "mcse": se})
    if st_ in S and "compound_effect" in S:
        t = p.flat(st_); c = p.flat("compound_effect").std(axis=1, ddof=1)
        pr, se, _ = prob_with_mcse(S[st_] > S["compound_effect"].std(axis=-1, ddof=1))
        rows.append({"comparison": f"σ_T − realised SD(C) ({unit})", "median": np.median(t - c), "lo": _eti(t - c, prob)[0], "hi": _eti(t - c, prob)[1], "P(>0)": pr, "mcse": se})
    for a, lab in ((hyper("sigma_race", S), "σ_R"),):
        if a in S and st_ in S:
            r = p.flat(a); t = p.flat(st_)
            share = r ** 2 / (r ** 2 + t ** 2)
            pr, se, _ = prob_with_mcse(S[a] > S[st_])
            rows.append({"comparison": "σ_R² / (σ_T² + σ_R²): сезонна частка «трасової» дисперсії", "median": np.median(share), "lo": _eti(share, prob)[0], "hi": _eti(share, prob)[1], "P(>0)": pr, "mcse": se})
    return pd.DataFrame(rows)


def residual_table(p: PosteriorDraws, prob: float) -> pd.DataFrame:
    rows = []
    for s, label in (("sigma_eps_core", "σ_core (s)"), ("sigma_eps_sd", "implied SD σ_core·sqrt(η/(η−2)) (s) — diverges as η→2"), ("lambda_skew", "λ"), ("nu_df", "η"), ("sigma_eta", "σ_η (s/√lap)"),
                     ("sigma_stint_base", "σ_β (s)"), ("baseline_mean", "β_0 (s)"), ("mu_log_gamma", "μ_γ"), ("sigma_log_gamma", "σ_γ")):
        if s in p.posterior:
            x = p.flat(s)
            rows.append({"parameter": label, "site": s, "median": np.median(x), "mean": x.mean(), "lo": _eti(x, prob)[0], "hi": _eti(x, prob)[1]})
    if "nu_df" in p.posterior:
        x = p.flat("nu_df")
        rows.append({"parameter": "P(η < 2.5)", "site": "nu_df", "median": float((x < 2.5).mean()), "mean": np.nan, "lo": np.nan, "hi": np.nan})
    return pd.DataFrame(rows)


def global_rate(p: PosteriorDraws, prob: float) -> Dict[str, float]:
    """Global level of the rate in s/lap: exp(μ_ν) (geometric-mean level, log link) or μ_ν^add (identity link)."""
    S = p.posterior
    r = to_rate(p.flat(mu_site(S)), S)
    lo, hi = _eti(r, prob)
    return {"median": float(np.median(r)), "lo": float(lo), "hi": float(hi), "mean": float(r.mean()), "P(>0)": float((r > 0).mean()), "link": link_name(S)}


def sign_table(p: PosteriorDraws, ds: Dataset, prob: float) -> pd.DataFrame:
    """How much posterior mass lies at or below zero, level by level (reviewer remark A2.2).

    Under the log link the group- and stint-level rates are positive BY CONSTRUCTION, so the table then
    only documents that fact (column `by_construction`); under `signed_stint` only the stint level is
    free; under the identity link every row is an empirical statement."""
    S = p.posterior
    link = link_name(S)
    mu = S[mu_site(S)]
    rows = []

    def add_row(level, quantity, value, n=None, constructed=False, note=""):
        rows.append({"level": level, "quantity": quantity, "value": float(value), "n": n, "link": link, "by_construction": bool(constructed), "note": note})

    group_free = link == "identity"
    stint_free = link in ("identity", "signed_stint")
    g = to_rate(mu, S)
    add_row("глобальний рівень", "P(rate > 0)", (g > 0).mean(), constructed=not group_free)
    C = S["compound_effect"]
    for i, c in enumerate(ds.levels["compound"]):
        add_row(f"компаунд {c}", "P(rate > 0)", (to_rate(mu + C[..., i], S) > 0).mean(), n=int((ds.comp_of_stint == i).sum()), constructed=not group_free)
    if "track_effect" in S:
        T = S["track_effect"]
        pt = np.array([(to_rate(mu + T[..., q], S) > 0).mean() for q in range(T.shape[-1])])
        add_row("траси", "частка трас із P(rate > 0) ≥ 0.975", (pt >= 0.975).mean(), n=len(pt), constructed=not group_free)
        add_row("траси", "мінімум P(rate > 0) за трасами", pt.min(), n=len(pt), constructed=not group_free, note=str(ds.levels["track"][int(pt.argmin())]))
        if "interaction_effect" in S:
            I = S["interaction_effect"]
            pc = np.array([[(to_rate(mu + C[..., c] + T[..., q] + I[..., q, c], S) > 0).mean() for c in range(C.shape[-1])] for q in range(T.shape[-1])])
            add_row("комірки траса×компаунд", "частка комірок із P(full rate > 0) ≥ 0.975", (pc >= 0.975).mean(), n=int(pc.size), constructed=not group_free)
            add_row("комірки траса×компаунд", "мінімум P(full rate > 0)", pc.min(), n=int(pc.size), constructed=not group_free)
    nu = p.flat("nu_stint")
    pneg = (nu < 0).mean(axis=0)
    add_row("відрізки", "середня P(ν_s < 0)", pneg.mean(), n=int(nu.shape[1]), constructed=not stint_free)
    add_row("відрізки", "частка відрізків із P(ν_s < 0) > 0.5", (pneg > 0.5).mean(), n=int(nu.shape[1]), constructed=not stint_free)
    add_row("відрізки", "частка відрізків із P(ν_s < 0) > 0.9", (pneg > 0.9).mean(), n=int(nu.shape[1]), constructed=not stint_free)
    add_row("відрізки", "частка відрізків із P(ν_s > 0) ≥ 0.975", ((1 - pneg) >= 0.975).mean(), n=int(nu.shape[1]), constructed=not stint_free)
    med = np.median(nu, axis=0)
    add_row("відрізки", "мінімальна апостеріорна медіана ν_s, с/коло", med.min(), n=int(nu.shape[1]), constructed=False)
    add_row("відрізки", "5-й процентиль апостеріорних медіан ν_s, с/коло", np.quantile(med, 0.05), n=int(nu.shape[1]), constructed=False)
    return pd.DataFrame(rows)


def select_example_stints(ds: Dataset, p: PosteriorDraws, min_laps: int = 15) -> List[Tuple[int, str]]:
    """Transparent selection rule: among stints with >= min_laps retained laps —
    the highest and lowest posterior-median rate, the longest stint, and the
    longest SOFT stint (if any)."""
    nu = np.median(p.flat("nu_stint"), axis=0)
    n = np.bincount(ds.stint_codes, minlength=ds.n_stints)
    cand = np.flatnonzero(n >= min_laps)
    if len(cand) < 4:
        cand = np.argsort(-n)[:8]
    picks = [(int(cand[np.argmax(nu[cand])]), "найвища медіанна ν_s"), (int(cand[np.argmin(nu[cand])]), "найнижча медіанна ν_s"), (int(cand[np.argmax(n[cand])]), "найдовший стінт")]
    comps = ds.levels["compound"]
    if "SOFT" in comps:
        soft = [j for j in cand[np.argsort(-n[cand])] if ds.comp_of_stint[j] == comps.index("SOFT")]
        if soft:
            picks.append((int(soft[0]), "найдовший стінт на SOFT"))
    seen, out = set(), []
    for j, why in picks:
        if j not in seen:
            out.append((j, why)); seen.add(j)
    return out[:4]

"""PSIS-LOO and WAIC from stored posterior draws (reviewer remarks A2.2, A2.3).

What is computed
----------------
For every retained lap n and every stored posterior draw d (the draws for which the lap-level
latent block was saved, i.e. every `save_latent_thin`-th draw of every chain) the pointwise
log-likelihood conditional on the latent state,

    ll[d, n] = log p(y_n | α_n^(d) + γ_q^(d) f_n, σ_core^(d), λ^(d), η^(d)),

then
  * PSIS-LOO   elpd_loo = Σ_n log Σ_d w_n^(d) p(y_n | θ^(d)),  w ∝ PSIS(1 / p(y_n | θ^(d)))
               (Vehtari, Gelman & Gabry 2017; smoothing as in Vehtari et al. 2024), with the Pareto
               shape k̂_n of every lap as the reliability diagnostic;
  * WAIC       lppd − Σ_n Var_d(ll[d, n]).

What it means — and what it does not
------------------------------------
The model has one latent innovation per lap, so "leave one lap out, conditional on the latent
state" asks how well lap n is predicted when its own state α_n is inferred from the neighbouring
laps and the hierarchy.  The posterior of α_n is strongly informed by y_n itself, hence the
importance ratios are heavy-tailed for many laps and k̂_n > 0.7 is expected for a non-negligible
share of them; the estimate is then unreliable and is REPORTED AS SUCH (columns share_k_gt_0.7,
share_k_gt_1).  Conditional LOO is informative about the observation model (Gaussian vs Student-t
vs skew-t) and nearly blind to the grouping structure of the rates, because the per-lap states
absorb what the hierarchy leaves unexplained.  The question "does the hierarchy predict better?"
is answered by the leave-future-out evaluation in `forecast.py` (prefix-only forecasts of a new
season, with the log predictive density as a strictly proper score), which needs no importance
sampling at all.  Both are exported; neither is declared a winner criterion in advance.

Comparisons are meaningful only between fits on the SAME laps (same dataset fingerprint); the
comparison table enforces this.

The PSIS implementation is self-contained (NumPy) and vectorised over laps; tests compare it with
exact leave-one-out for a conjugate model and with ArviZ where its API is available.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .config import ModelSpec
from .data import Dataset
from .sampler import PosteriorDraws

K_GOOD, K_BAD = 0.7, 1.0


# --------------------------------------------------------------------------- #
# pointwise log-likelihood
# --------------------------------------------------------------------------- #
def pointwise_loglik(ds: Dataset, p: PosteriorDraws, spec: ModelSpec, warmup: int, max_draws: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ll[d, n] for the draws that have a stored latent block (or, for models without process
    noise, for every `thin`-th draw).  Returns (ll, info)."""
    from .forecast import obs_logpdf
    from .ppc import _draw_params, _mu_conditional

    if p.latent is not None:
        C, T = p.latent.shape[0], p.latent.shape[1]
        pairs = [(c, t, int(p.latent_iters[t]) - warmup) for c in range(C) for t in range(T)]
        latent = lambda c, t: p.latent[c, t]
    else:
        C, D = p.n_chains, p.n_draws
        thin = max(1, D // 200)
        pairs = [(c, None, d) for c in range(C) for d in range(0, D, thin)]
        latent = lambda c, t: None
    if max_draws is not None and len(pairs) > max_draws:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, max_draws).astype(int)]
    ll = np.empty((len(pairs), ds.n_laps), dtype=np.float64)
    for i, (c, t, d) in enumerate(pairs):
        pr = _draw_params(p, c, d, spec)
        mu = _mu_conditional(ds, pr, latent(c, t), spec)
        ll[i] = obs_logpdf(spec, ds.y, mu, pr["sigma_eps_core"], pr["lam"], pr["eta"])
    info = {"n_draws": len(pairs), "n_chains": int(C), "n_obs": int(ds.n_laps), "has_latent": p.latent is not None,
            "nonfinite": int((~np.isfinite(ll)).sum())}
    return ll, info


# --------------------------------------------------------------------------- #
# Pareto smoothed importance sampling
# --------------------------------------------------------------------------- #
def _gpdfit_batch(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Zhang & Stephens (2009) estimate of the generalised Pareto parameters for every row of
    `x` (B, M), rows sorted ascending and strictly positive in the last column.  Returns (k, sigma).
    The weakly informative prior of Vehtari et al. (k shrunk towards 0.5 with weight 10) is applied."""
    prior_bs, prior_k = 3.0, 10.0
    B, n = x.shape
    m_est = 30 + int(np.sqrt(n))
    j = np.arange(1, m_est + 1, dtype=float)
    b = 1.0 - np.sqrt(m_est / (j - 0.5))                                   # (m,)
    q25 = x[:, int(n / 4 + 0.5) - 1]                                       # (B,)
    b = b[None, :] / (prior_bs * q25[:, None]) + 1.0 / x[:, -1][:, None]   # (B, m)
    k = np.log1p(-b[:, :, None] * x[:, None, :]).mean(axis=2)             # (B, m)
    len_scale = n * (np.log(-(b / k)) - k - 1.0)                           # (B, m) profile log-likelihood
    w = 1.0 / np.exp(len_scale[:, None, :] - len_scale[:, :, None]).sum(axis=2)   # (B, m)
    w = np.where(w >= 10.0 * np.finfo(float).eps, w, 0.0)
    w = w / w.sum(axis=1, keepdims=True)
    b_post = (b * w).sum(axis=1)                                           # (B,)
    k_post = np.log1p(-b_post[:, None] * x).mean(axis=1)
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return k_post, sigma


def psis_log_weights(log_ratios: np.ndarray, r_eff: float = 1.0, batch: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Pareto smoothed, self-normalised log importance weights.

    log_ratios: (S, N) raw log importance ratios (for LOO: -ll).  Returns (log_w (S, N), k (N,)).
    Columns whose tail is degenerate (all tail ratios equal) are left unsmoothed with k = -inf."""
    S, N = log_ratios.shape
    lw = log_ratios - log_ratios.max(axis=0, keepdims=True)
    M = int(min(np.ceil(0.2 * S), np.ceil(3.0 * np.sqrt(S / r_eff))))
    khat = np.full(N, -np.inf)
    if M < 5:
        return lw - logsumexp(lw, axis=0, keepdims=True), khat
    sti = (np.arange(M) + 0.5) / M
    for start in range(0, N, batch):
        sl = slice(start, min(N, start + batch))
        block = lw[:, sl]                                                 # (S, b)
        order = np.argsort(block, axis=0)
        tail_idx = order[S - M:, :]                                        # indices of the M largest, ascending
        cutoff = np.take_along_axis(block, order[S - M - 1:S - M, :], axis=0)[0]   # (b,) log of the largest non-tail ratio
        tail = np.take_along_axis(block, tail_idx, axis=0)                 # (M, b) ascending
        x = (np.exp(tail) - np.exp(cutoff)[None, :]).T                     # (b, M) exceedances
        ok = x[:, -1] > 10.0 * np.finfo(float).tiny
        ok &= x[:, int(M / 4 + 0.5) - 1] > 0                               # the Zhang–Stephens scale needs a positive lower quartile
        if ok.any():
            xo = np.maximum(x[ok], np.finfo(float).tiny)
            k, sigma = _gpdfit_batch(xo)
            fin = np.isfinite(k) & np.isfinite(sigma) & (sigma > 0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                kk = np.where(np.abs(k) < 1e-10, 1e-10, k)
                q = sigma[:, None] * np.expm1(-kk[:, None] * np.log1p(-sti)[None, :]) / kk[:, None]     # (b_ok, M) GPD quantiles
                sm = np.log(q + np.exp(cutoff[ok])[:, None])
            sm = np.minimum(sm, 0.0)                                       # truncate at the largest raw ratio (log 1 after shifting)
            cols = np.flatnonzero(ok)
            for jj, col in enumerate(cols):
                if fin[jj] and np.all(np.isfinite(sm[jj])):
                    block[tail_idx[:, col], col] = sm[jj]
                    khat[start + col] = k[jj]
        lw[:, sl] = block
    return lw - logsumexp(lw, axis=0, keepdims=True), khat


def psis_loo(ll: np.ndarray, r_eff: float = 1.0) -> Dict[str, Any]:
    """PSIS-LOO from a pointwise log-likelihood matrix (S, N)."""
    S, N = ll.shape
    lw, k = psis_log_weights(-ll, r_eff=r_eff)
    elpd_i = logsumexp(lw + ll, axis=0)
    lppd_i = logsumexp(ll, axis=0) - np.log(S)
    p_i = lppd_i - elpd_i
    return {"elpd_i": elpd_i, "lppd_i": lppd_i, "p_loo_i": p_i, "k": k, "elpd": float(elpd_i.sum()), "se": float(np.sqrt(N * elpd_i.var(ddof=1))),
            "p_loo": float(p_i.sum()), "lppd": float(lppd_i.sum()), "n_obs": int(N), "n_draws": int(S),
            "share_k_gt_0.7": float((k > K_GOOD).mean()), "share_k_gt_1": float((k > K_BAD).mean()), "max_k": float(np.max(k)) if N else np.nan}


def waic(ll: np.ndarray) -> Dict[str, Any]:
    S, N = ll.shape
    lppd_i = logsumexp(ll, axis=0) - np.log(S)
    p_i = ll.var(axis=0, ddof=1)
    e_i = lppd_i - p_i
    return {"elpd_i": e_i, "p_waic_i": p_i, "elpd": float(e_i.sum()), "se": float(np.sqrt(N * e_i.var(ddof=1))), "p_waic": float(p_i.sum()),
            "share_p_gt_0.4": float((p_i > 0.4).mean())}


# --------------------------------------------------------------------------- #
# stage helpers
# --------------------------------------------------------------------------- #
def loo_for_fit(ds: Dataset, p: PosteriorDraws, spec: ModelSpec, warmup: int, out_dir: Path, tag: str, max_draws: Optional[int] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ll, info = pointwise_loglik(ds, p, spec, warmup, max_draws)
    L = psis_loo(ll); W = waic(ll)
    stint = ds.stint_codes
    np.savez_compressed(out_dir / f"loo_pointwise_{tag}.npz", elpd_loo_i=L["elpd_i"].astype(np.float32), pareto_k=L["k"].astype(np.float32),
                        elpd_waic_i=W["elpd_i"].astype(np.float32), lppd_i=L["lppd_i"].astype(np.float32), stint_code=stint.astype(np.int32))
    # where do the unreliable laps sit?  first laps of stints have no earlier neighbour to inform the state
    first = np.zeros(ds.n_laps, bool); first[ds.first_pos] = True
    last = np.zeros(ds.n_laps, bool); last[np.r_[ds.first_pos[1:] - 1, ds.n_laps - 1]] = True
    summ = {"fit": tag, "data_fingerprint": ds.fingerprint, **info,
            "elpd_loo": L["elpd"], "se_elpd_loo": L["se"], "p_loo": L["p_loo"], "lppd": L["lppd"], "elpd_loo_per_lap": L["elpd"] / ds.n_laps,
            "share_k_gt_0.7": L["share_k_gt_0.7"], "share_k_gt_1": L["share_k_gt_1"], "max_k": L["max_k"],
            "share_k_gt_0.7_first_laps": float((L["k"][first] > K_GOOD).mean()), "share_k_gt_0.7_last_laps": float((L["k"][last] > K_GOOD).mean()),
            "share_k_gt_0.7_inner_laps": float((L["k"][~(first | last)] > K_GOOD).mean()) if (~(first | last)).any() else np.nan,
            "elpd_waic": W["elpd"], "se_elpd_waic": W["se"], "p_waic": W["p_waic"], "share_p_waic_gt_0.4": W["share_p_gt_0.4"],
            "reliable": bool(L["share_k_gt_0.7"] <= 0.01),
            "note": "conditional on the lap-level latent states; see the module docstring of f1deg.loo for the interpretation",
            "pointwise_npz": str(out_dir / f"loo_pointwise_{tag}.npz")}
    with open(out_dir / f"loo_{tag}.json", "w") as f:
        json.dump(summ, f, indent=1, default=float)
    return summ


def compare_table(summaries: Dict[str, Dict[str, Any]], reference: str = "primary") -> pd.DataFrame:
    """One row per fit; differences to the reference are filled only for fits on the same laps."""
    rows = []
    ref = summaries.get(reference)
    ref_pw = np.load(ref["pointwise_npz"]) if ref and Path(ref["pointwise_npz"]).exists() else None
    for name, s in summaries.items():
        row = {"model": name, "n_laps": s["n_obs"], "n_draws": s["n_draws"], "elpd_loo": s["elpd_loo"], "se": s["se_elpd_loo"], "p_loo": s["p_loo"],
               "elpd_loo_per_lap": s["elpd_loo_per_lap"], "share_k_gt_0.7": s["share_k_gt_0.7"], "share_k_gt_1": s["share_k_gt_1"], "reliable": s["reliable"],
               "elpd_waic": s["elpd_waic"], "p_waic": s["p_waic"], "same_data_as_reference": bool(ref is not None and s["data_fingerprint"] == ref["data_fingerprint"])}
        if ref_pw is not None and row["same_data_as_reference"] and name != reference and Path(s["pointwise_npz"]).exists():
            pw = np.load(s["pointwise_npz"])
            d = pw["elpd_loo_i"].astype(float) - ref_pw["elpd_loo_i"].astype(float)
            row["d_elpd_loo_vs_ref"] = float(d.sum()); row["se_d_elpd_loo"] = float(np.sqrt(len(d) * d.var(ddof=1)))
            # laps of one stint are dependent: a stint-clustered standard error is the honest one
            st = pw["stint_code"]; by = np.bincount(st, weights=d); n_st = len(by)
            row["se_d_elpd_loo_stint_clustered"] = float(np.sqrt(n_st * by.var(ddof=1)))
            dw = pw["elpd_waic_i"].astype(float) - ref_pw["elpd_waic_i"].astype(float)
            row["d_elpd_waic_vs_ref"] = float(dw.sum())
        rows.append(row)
    return pd.DataFrame(rows)

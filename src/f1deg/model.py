"""NumPyro model variants, prior table and symbol map.

Full specification (primary model; units in brackets)
-----------------------------------------------------
Lap n of stint s(n) at track q(s), compound c(s), driver d(s), constructor g(s);
a_n relative tyre age [laps], f_n centred fuel proxy [kg], y_n centred lap time [s].

Latent state (non-centred, innovations only for laps with Δa_n > 0):
    α_n = β_s + ν_s a_n + σ_η Σ_{j≤n, j∈s, Δa_j>0} sqrt(Δa_j) z_j,     z_j ~ N(0,1)            [s]
Baseline:
    β_s = β_0 + b_{c(s)} + σ_β z^β_s,  b ~ ZeroSumNormal(fixed scale)                           [s]
Rate (log link):
    log ν_s = μ_ν + C_{c(s)} + T_{q(s)} + D_{d(s)} + G_{g(s)} + I_{q(s),c(s)} + σ_U u_s         [log s/lap]
    C: zero-sum, fixed scale (primary) or σ_C·ZSN (variant);  T, D, G: σ_X·ZSN;  I: σ_I·ZSN(2 axes)
Fuel/evolution:
    γ_q = exp(μ_γ + σ_γ z^γ_q)   [s/kg]      (or fixed external value)
Observation:
    y_n ~ HansenSkewT(α_n + γ_{q(n)} f_n, σ_core, λ, η),  η = 2 + η̃, η̃ ~ Gamma(2, rate 0.5)
    (variants: StudentT(η, ·, σ_core), Normal(·, σ))

Identifiability constraints: zero-sum on C, T, D, G (each axis) and on both axes of I.
Within one stint only (β_s + γ_q f_{s,0}, ν_s − b_r γ_q) enter the likelihood; γ_q is separated from ν_s
by the exchangeable prior on β_s (stint baselines may not trend with the stint's start lap) together
with the hierarchy on the rate — see docs/MODEL_REGISTRY_uk.md §Ідентифіковність.

Variants added in v1.1 (review round 1; all are switched off by default, see config.V11_FIELD_DEFAULTS)
---------------------------------------------------------------------------------------------------
rate_link == "identity":  ν_s = μ^add + C_c + T_q + D_d + G_g + I_qc (+ R_r) + σ_U^add u_s       [s/lap]
    every effect is additive on the s/lap scale; nothing constrains the sign of the rate at any level.
race_effect:              log ν_s (or ν_s) += R_{r(s)},  R_r = σ_R z^R_r,  z^R_r ~ N(0,1)           (race = track × season)

Variant added in v1.2
---------------------
group_scale == "fixed":   the SAME linear predictor and zero-sum constraints, but the group effects have no learned scales:
    T ~ ZSN(s_T),  D ~ ZSN(s_D),  G ~ ZSN(s_G),  I ~ ZSN(s_I; both axes),   s_X fixed (config.fixed_group_scales)
    instead of X = σ_X·ZSN(1), σ_X ~ HalfNormal(a_X).  With the default s_X = a_X both models give X the same prior
    covariance (E[σ_X²] = a_X²); the hierarchical one additionally learns σ_X, i.e. how strongly to pool.
    Everything else (C, U_s, β_s, γ_q, the latent state, the likelihood) is the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from numpyro.distributions import constraints

from .config import GROUP_SCALES, ModelSpec, fixed_group_scales, spec_fingerprint_dict
from .data import Dataset
from .hansen import HansenSkewT


class ModelData(NamedTuple):
    y: jnp.ndarray
    fuel_c: jnp.ndarray
    rel_age: jnp.ndarray
    sqrt_dage_inno: jnp.ndarray      # sqrt(Δa) at innovation laps
    inno_idx: jnp.ndarray
    stint_codes: jnp.ndarray
    first_pos: jnp.ndarray
    track_of_stint: jnp.ndarray
    comp_of_stint: jnp.ndarray
    driver_of_stint: jnp.ndarray
    team_of_stint: jnp.ndarray
    n_laps: int
    n_stints: int
    n_track: int
    n_comp: int
    n_driver: int
    n_team: int
    race_of_stint: Optional[jnp.ndarray] = None     # v1.1: used only when spec.race_effect
    n_race: int = 0


def model_data(ds: Dataset) -> ModelData:
    return ModelData(
        y=jnp.asarray(ds.y), fuel_c=jnp.asarray(ds.fuel_c), rel_age=jnp.asarray(ds.rel_age),
        sqrt_dage_inno=jnp.asarray(np.sqrt(ds.dage[ds.inno_idx])), inno_idx=jnp.asarray(ds.inno_idx),
        stint_codes=jnp.asarray(ds.stint_codes), first_pos=jnp.asarray(ds.first_pos),
        track_of_stint=jnp.asarray(ds.track_of_stint), comp_of_stint=jnp.asarray(ds.comp_of_stint),
        driver_of_stint=jnp.asarray(ds.driver_of_stint), team_of_stint=jnp.asarray(ds.team_of_stint),
        n_laps=ds.n_laps, n_stints=ds.n_stints, n_track=len(ds.levels["track"]), n_comp=len(ds.levels["compound"]),
        n_driver=len(ds.levels["driver"]), n_team=len(ds.levels["team"]),
        race_of_stint=jnp.asarray(ds.race_of_stint), n_race=len(ds.levels["race"]),
    )


# --------------------------------------------------------------------------- #
# prior table (single source of truth, also consumed by the model)
# --------------------------------------------------------------------------- #
@dataclass
class PriorEntry:
    symbol: str
    python_name: str
    role: str
    distribution: str
    units: str
    support: str
    block: str


def prior_table(spec: ModelSpec) -> pd.DataFrame:
    P: List[PriorEntry] = []
    add = lambda *a: P.append(PriorEntry(*a))
    ident = spec.rate_link == "identity"
    U = "с/коло" if ident else "log(с/коло)"          # units of the linear predictor of the rate
    sfx = "_add" if ident else ""                     # hyper-parameters of the identity link live on another scale -> other names
    if ident:
        add("μ_ν^add", "mu_nu_add", "глобальний рівень швидкості деградації (може бути < 0)", f"N({spec.mu_nu_add_loc}, {spec.mu_nu_add_scale}²)", U, "ℝ", "rate")
        add("C_c", "compound_effect", "адитивний ефект компаунда (zero-sum, фіксована шкала)", f"ZeroSumNormal(σ={spec.compound_scale_add}), ΣC=0", U, "ℝ³, ΣC=0", "rate")
    else:
        add("μ_ν", "mu_log_nu", "глобальний лог-рівень швидкості деградації", f"N({spec.mu_log_nu_loc}, {spec.mu_log_nu_scale}²)", U, "ℝ", "rate")
        if spec.compound_prior == "fixed":
            add("C_c", "compound_effect", "ефект компаунда (zero-sum, фіксована шкала)", f"ZeroSumNormal(σ={spec.compound_scale}), ΣC=0", U, "ℝ³, ΣC=0", "rate")
        else:
            add("σ_C", "sigma_compound", "ієрархічна шкала ефектів компаунда", f"HalfNormal({spec.compound_hier_scale})", U, "ℝ₊", "rate")
            add("C_c", "compound_effect", "ефект компаунда C = σ_C·C̃, C̃ ~ ZSN(1)", "σ_C · ZeroSumNormal(1)", U, "ℝ³, ΣC=0", "rate")
    sc = (lambda log_scale, add_scale: add_scale if ident else log_scale)
    if spec.group_scale not in GROUP_SCALES:
        raise ValueError(f"unknown group_scale {spec.group_scale!r}; known: {GROUP_SCALES}")
    fixed = spec.group_scale == "fixed"         # v1.2: group effects with fixed prior scales, no sigma_T, sigma_D, sigma_G, sigma_I
    gs = fixed_group_scales(spec)
    if spec.pooling == "hier":
        if fixed:
            add("T_q", "track_effect", "ефект траси (zero-sum, фіксована шкала; σ_T не оцінюється)", f"ZeroSumNormal(σ={gs['track']}), ΣT=0", U, "ℝ^Q, ΣT=0", "rate")
            add("D_d", "driver_effect", "ефект пілота (zero-sum, фіксована шкала; σ_D не оцінюється)", f"ZeroSumNormal(σ={gs['driver']}), ΣD=0", U, "ℝ^D, ΣD=0", "rate")
            add("G_g", "team_effect", "ефект конструктора (zero-sum, фіксована шкала; σ_G не оцінюється)", f"ZeroSumNormal(σ={gs['team']}), ΣG=0", U, "ℝ^G, ΣG=0", "rate")
        else:
            add("σ_T", "sigma_track" + sfx, "шкала трасових ефектів", f"HalfNormal({sc(spec.sigma_track_scale, spec.sigma_track_add_scale)})", U, "ℝ₊", "rate")
            add("T_q", "track_effect", "ефект траси T = σ_T·T̃, T̃ ~ ZSN(1)", "σ_T · ZeroSumNormal(1)", U, "ℝ^Q, ΣT=0", "rate")
            add("σ_D", "sigma_driver" + sfx, "шкала ефектів пілотів", f"HalfNormal({sc(spec.sigma_driver_scale, spec.sigma_driver_add_scale)})", U, "ℝ₊", "rate")
            add("D_d", "driver_effect", "ефект пілота (crossed з конструктором)", "σ_D · ZeroSumNormal(1)", U, "ℝ^D, ΣD=0", "rate")
            add("σ_G", "sigma_team" + sfx, "шкала ефектів конструкторів", f"HalfNormal({sc(spec.sigma_team_scale, spec.sigma_team_add_scale)})", U, "ℝ₊", "rate")
            add("G_g", "team_effect", "ефект конструктора", "σ_G · ZeroSumNormal(1)", U, "ℝ^G, ΣG=0", "rate")
    if spec.interaction:
        if fixed:
            add("I_qc", "interaction_effect", "взаємодія (zero-sum за обома осями, фіксована шкала; σ_I не оцінюється)", f"ZeroSumNormal(σ={gs['interaction']}; 2 осі)", U, "ℝ^{Q×3}, Σ_q I=Σ_c I=0", "rate")
        else:
            add("σ_I", "sigma_interaction" + sfx, "шкала взаємодії траса×компаунд", f"HalfNormal({sc(spec.sigma_interaction_scale, spec.sigma_interaction_add_scale)})", U, "ℝ₊", "rate")
            add("I_qc", "interaction_effect", "взаємодія (zero-sum за обома осями)", "σ_I · ZeroSumNormal(1; 2 осі)", U, "ℝ^{Q×3}, Σ_q I=Σ_c I=0", "rate")
    if spec.race_effect:
        add("σ_R", "sigma_race" + sfx, "шкала ефектів гонки (траса×сезон)", f"HalfNormal({sc(spec.sigma_race_scale, spec.sigma_race_add_scale)})", U, "ℝ₊", "rate")
        add("R_r", "race_effect", "ефект гонки R = σ_R·z^R, z^R ~ N(0,1)", "σ_R · N(0,1)", U, "ℝ^{N_r}", "rate")
    if spec.stint_rate_effect:
        if spec.rate_link == "log":
            add("σ_U", "sigma_stint", "залишкова стінтова неоднорідність log ν", f"HalfNormal({spec.sigma_stint_scale})", "log(с/коло)", "ℝ₊", "rate")
            add("u_s", "z_stint_deg", "стандартизований стінтовий ефект", "N(0,1)", "—", "ℝ", "rate")
        else:
            add("σ_U^add", "sigma_stint_add", "стінтове відхилення на шкалі с/коло (signed)", f"HalfNormal({spec.signed_stint_sd_scale})", "с/коло", "ℝ₊", "rate")
            add("u_s", "z_stint_deg", "стандартизоване стінтове відхилення", "N(0,1)", "—", "ℝ", "rate")
    add("β_0", "baseline_mean", "глобальний базовий темп (центрований)", f"N(0, {spec.baseline_mean_scale}²)", "с", "ℝ", "baseline")
    add("b_c", "compound_base_effect", "зсув базового темпу компаунда (zero-sum, фіксована шкала)", f"ZeroSumNormal(σ={spec.compound_base_scale})", "с", "ℝ³, Σb=0", "baseline")
    add("σ_β", "sigma_stint_base", "шкала стінтових базових рівнів", f"HalfNormal({spec.sigma_stint_base_scale})", "с", "ℝ₊", "baseline")
    add("z^β_s", "z_stint_base", "стандартизований стінтовий базовий рівень", "N(0,1)", "—", "ℝ", "baseline")
    if spec.gamma == "hier":
        add("μ_γ", "mu_log_gamma", "лог-рівень паливно-трасового коефіцієнта", f"N({spec.mu_log_gamma_loc:.4f}, {spec.mu_log_gamma_scale}²)  [= log 0.04 ± …]", "log(с/кг)", "ℝ", "fuel")
        add("σ_γ", "sigma_log_gamma", "шкала трасових відхилень log γ", f"HalfNormal({spec.sigma_log_gamma_scale})", "log(с/кг)", "ℝ₊", "fuel")
        add("z^γ_q", "z_log_gamma", "стандартизоване трасове відхилення", "N(0,1)", "—", "ℝ", "fuel")
        add("γ_q", "gamma_track", "паливно-трасовий коефіцієнт γ_q = exp(μ_γ + σ_γ z^γ_q)", "детерміновано", "с/кг", "ℝ₊", "fuel")
    else:
        add("γ", "gamma_fixed", "паливний коефіцієнт (зовнішнє припущення)", f"≡ {spec.gamma_fixed_value}", "с/кг", "const", "fuel")
    if spec.process_noise:
        add("σ_η", "sigma_eta", "шкала інновацій латентного стану", f"HalfNormal({spec.sigma_eta_scale})", "с/√коло", "ℝ₊", "process")
        if spec.latent_param == "centred":
            add("α_n", "alpha_lap", "латентний стан (centred): α_first ~ N(β_s-prior), α_n − α_{n−1} ~ N(ν_s Δa_n, σ_η² Δa_n)", "random walk (факторами)", "с", "ℝ", "process")
        else:
            add("z_j", "z_inno", "стандартизовані інновації (лише Δa>0; non-centred)", "N(0,1)", "—", "ℝ", "process")
    if spec.likelihood == "skewt":
        add("σ_core", "sigma_eps_core", "core-масштаб похибки (аналог масштабу Student-t)", f"HalfNormal({spec.sigma_eps_scale})", "с", "ℝ₊", "obs")
        add("λ", "lambda_skew", "асиметрія Гансена", f"TruncatedNormal({spec.lambda_loc}, {spec.lambda_scale}²; −{spec.lambda_bound}, {spec.lambda_bound})", "—", f"(−{spec.lambda_bound}, {spec.lambda_bound})", "obs")
        if spec.fix_df is None:
            add("η", "nu_df", "індекс хвоста, η = 2 + η̃", f"2 + Gamma(shape={spec.df_gamma_shape}, rate={spec.df_gamma_rate}) (mean 6)", "—", "(2, ∞)", "obs")
        else:
            add("η", "nu_df", "індекс хвоста (зафіксовано)", f"≡ {spec.fix_df}", "—", "const", "obs")
        add("σ_ε", "sigma_eps_sd", "імплікований SD похибки = σ_core·sqrt(η/(η−2)) (похідна величина, розбігається при η→2)", "детерміновано", "с", "ℝ₊", "obs")
    elif spec.likelihood == "studentt":
        add("σ_core", "sigma_eps_core", "масштаб Student-t", f"HalfNormal({spec.sigma_eps_scale})", "с", "ℝ₊", "obs")
        if spec.fix_df is None:
            add("η", "nu_df", "ступені вільності, η = 2 + η̃", f"2 + Gamma(shape={spec.df_gamma_shape}, rate={spec.df_gamma_rate})", "—", "(2, ∞)", "obs")
        else:
            add("η", "nu_df", "ступені вільності (зафіксовано)", f"≡ {spec.fix_df}", "—", "const", "obs")
    else:
        add("σ", "sigma_eps_core", "SD гаусової похибки", f"HalfNormal({spec.sigma_eps_scale})", "с", "ℝ₊", "obs")
    df = pd.DataFrame([e.__dict__ for e in P])
    df.insert(0, "model", spec.name)
    return df


def latent_site(spec: ModelSpec) -> str:
    """Name of the lap-level latent block stored thinned: innovations (non-centred) or states (centred)."""
    return "alpha_lap" if (spec.process_noise and spec.latent_param == "centred") else "z_inno"


def site_names(spec: ModelSpec) -> List[str]:
    """Sampled + deterministic site names (excluding the big latent block)."""
    return [n for n in prior_table(spec)["python_name"].tolist() if n not in ("z_inno", "alpha_lap")] + ["nu_stint", "baseline_stint"]


HYPER_SITES_ALL = ["mu_log_nu", "sigma_compound", "sigma_track", "sigma_driver", "sigma_team", "sigma_interaction", "sigma_stint",
                   "sigma_stint_add", "baseline_mean", "sigma_stint_base", "mu_log_gamma", "sigma_log_gamma", "sigma_eta",
                   "sigma_eps_core", "lambda_skew", "nu_df_m2",
                   # v1.1 (appended: the order of the v1.0 names above is part of the fingerprint of existing fits)
                   "mu_nu_add", "sigma_track_add", "sigma_driver_add", "sigma_team_add", "sigma_interaction_add", "sigma_race", "sigma_race_add"]


def hyper_sites(spec: ModelSpec) -> List[str]:
    """Global hyper-parameters that are *sampled* sites (used for the dense mass block)."""
    names = set(prior_table(spec)["python_name"].tolist())
    if spec.likelihood in ("skewt", "studentt") and spec.fix_df is None:
        names.add("nu_df_m2")
    return [s for s in HYPER_SITES_ALL if s in names]


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
def f1_model(md: ModelData, spec: ModelSpec, observe: bool = True):
    S, Q, Cn, Dn, Gn = md.n_stints, md.n_track, md.n_comp, md.n_driver, md.n_team

    # ---- rate hierarchy ------------------------------------------------------
    # `eta_group` is the linear predictor of the rate without the stint deviation: log ν (log / signed_stint links)
    # or ν itself in s/lap (identity link).  Site names of the effects are shared between links; the hyper-parameters
    # of the identity link carry the suffix `_add` because they live on a different scale.
    ident = spec.rate_link == "identity"
    sfx = "_add" if ident else ""
    if ident:
        mu_nu = numpyro.sample("mu_nu_add", dist.Normal(spec.mu_nu_add_loc, spec.mu_nu_add_scale))
        C = numpyro.sample("compound_effect", dist.ZeroSumNormal(spec.compound_scale_add, event_shape=(Cn,)))
    else:
        mu_nu = numpyro.sample("mu_log_nu", dist.Normal(spec.mu_log_nu_loc, spec.mu_log_nu_scale))
        if spec.compound_prior == "fixed":
            C = numpyro.sample("compound_effect", dist.ZeroSumNormal(spec.compound_scale, event_shape=(Cn,)))
        else:
            sigma_C = numpyro.sample("sigma_compound", dist.HalfNormal(spec.compound_hier_scale))
            C_raw = numpyro.sample("compound_raw", dist.ZeroSumNormal(1.0, event_shape=(Cn,)))
            C = numpyro.deterministic("compound_effect", sigma_C * C_raw)
    eta_group = mu_nu + C[md.comp_of_stint]
    if spec.group_scale not in GROUP_SCALES:
        raise ValueError(f"unknown group_scale {spec.group_scale!r}; known: {GROUP_SCALES}")
    fixed = spec.group_scale == "fixed"
    gs = fixed_group_scales(spec) if fixed else None
    if spec.pooling == "hier":
        if fixed:
            # v1.2: the same effects with the same zero-sum constraints, but with FIXED prior scales — there is no sigma_T,
            # sigma_D, sigma_G to learn, hence no adaptive partial pooling (the effects are sampled directly, like C)
            T = numpyro.sample("track_effect", dist.ZeroSumNormal(gs["track"], event_shape=(Q,)))
            Dd = numpyro.sample("driver_effect", dist.ZeroSumNormal(gs["driver"], event_shape=(Dn,)))
            G = numpyro.sample("team_effect", dist.ZeroSumNormal(gs["team"], event_shape=(Gn,)))
        else:
            sigma_T = numpyro.sample("sigma_track" + sfx, dist.HalfNormal(spec.sigma_track_add_scale if ident else spec.sigma_track_scale))
            T = numpyro.deterministic("track_effect", sigma_T * numpyro.sample("track_raw", dist.ZeroSumNormal(1.0, event_shape=(Q,))))
            sigma_D = numpyro.sample("sigma_driver" + sfx, dist.HalfNormal(spec.sigma_driver_add_scale if ident else spec.sigma_driver_scale))
            Dd = numpyro.deterministic("driver_effect", sigma_D * numpyro.sample("driver_raw", dist.ZeroSumNormal(1.0, event_shape=(Dn,))))
            sigma_G = numpyro.sample("sigma_team" + sfx, dist.HalfNormal(spec.sigma_team_add_scale if ident else spec.sigma_team_scale))
            G = numpyro.deterministic("team_effect", sigma_G * numpyro.sample("team_raw", dist.ZeroSumNormal(1.0, event_shape=(Gn,))))
        eta_group = eta_group + T[md.track_of_stint] + Dd[md.driver_of_stint] + G[md.team_of_stint]
    if spec.interaction:
        if fixed:
            I = numpyro.sample("interaction_effect", dist.ZeroSumNormal(gs["interaction"], event_shape=(Q, Cn)))
        else:
            sigma_I = numpyro.sample("sigma_interaction" + sfx, dist.HalfNormal(spec.sigma_interaction_add_scale if ident else spec.sigma_interaction_scale))
            I = numpyro.deterministic("interaction_effect", sigma_I * numpyro.sample("interaction_raw", dist.ZeroSumNormal(1.0, event_shape=(Q, Cn))))
        eta_group = eta_group + I[md.track_of_stint, md.comp_of_stint]
    if spec.race_effect:
        sigma_R = numpyro.sample("sigma_race" + sfx, dist.HalfNormal(spec.sigma_race_add_scale if ident else spec.sigma_race_scale))
        R = numpyro.deterministic("race_effect", sigma_R * numpyro.sample("race_raw", dist.Normal(0.0, 1.0).expand([md.n_race])))
        eta_group = eta_group + R[md.race_of_stint]
    if spec.stint_rate_effect:
        u = numpyro.sample("z_stint_deg", dist.Normal(0.0, 1.0).expand([S]))
        if spec.rate_link == "log":
            sigma_U = numpyro.sample("sigma_stint", dist.HalfNormal(spec.sigma_stint_scale))
            nu = numpyro.deterministic("nu_stint", jnp.exp(eta_group + sigma_U * u))
        elif spec.rate_link == "signed_stint":  # signed stint deviation on the natural scale, positive group level
            sigma_U = numpyro.sample("sigma_stint_add", dist.HalfNormal(spec.signed_stint_sd_scale))
            nu = numpyro.deterministic("nu_stint", jnp.exp(eta_group) + sigma_U * u)
        elif ident:                             # everything additive: the rate may be negative at every level
            sigma_U = numpyro.sample("sigma_stint_add", dist.HalfNormal(spec.signed_stint_sd_scale))
            nu = numpyro.deterministic("nu_stint", eta_group + sigma_U * u)
        else:
            raise ValueError(f"unknown rate_link {spec.rate_link!r}")
    else:
        nu = numpyro.deterministic("nu_stint", eta_group if ident else jnp.exp(eta_group))

    # ---- baselines -----------------------------------------------------------
    beta0 = numpyro.sample("baseline_mean", dist.Normal(0.0, spec.baseline_mean_scale))
    b = numpyro.sample("compound_base_effect", dist.ZeroSumNormal(spec.compound_base_scale, event_shape=(Cn,)))
    sigma_beta = numpyro.sample("sigma_stint_base", dist.HalfNormal(spec.sigma_stint_base_scale))
    centred = bool(spec.process_noise and spec.latent_param == "centred")
    if not centred:
        zb = numpyro.sample("z_stint_base", dist.Normal(0.0, 1.0).expand([S]))
        beta = numpyro.deterministic("baseline_stint", beta0 + b[md.comp_of_stint] + sigma_beta * zb)

    # ---- fuel / evolution ----------------------------------------------------
    if spec.gamma == "hier":
        mu_lg = numpyro.sample("mu_log_gamma", dist.Normal(spec.mu_log_gamma_loc, spec.mu_log_gamma_scale))
        sig_lg = numpyro.sample("sigma_log_gamma", dist.HalfNormal(spec.sigma_log_gamma_scale))
        zg = numpyro.sample("z_log_gamma", dist.Normal(0.0, 1.0).expand([Q]))
        gamma = numpyro.deterministic("gamma_track", jnp.exp(mu_lg + sig_lg * zg))
        gamma_lap = gamma[md.track_of_stint][md.stint_codes]
    else:
        gamma_lap = spec.gamma_fixed_value

    # ---- latent state --------------------------------------------------------
    if centred:
        # Centred parameterisation of the SAME model: the states alpha_n are the parameters; the
        # baseline prior acts on the first lap of each stint and the random-walk prior on increments.
        sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(spec.sigma_eta_scale))
        alpha = numpyro.sample("alpha_lap", dist.ImproperUniform(constraints.real, (), event_shape=(md.n_laps,)))
        beta = numpyro.deterministic("baseline_stint", alpha[md.first_pos])
        numpyro.sample("baseline_prior", dist.Normal(beta0 + b[md.comp_of_stint], sigma_beta), obs=beta)
        d_alpha = alpha[md.inno_idx] - alpha[md.inno_idx - 1]
        numpyro.sample("rw_prior", dist.Normal(nu[md.stint_codes[md.inno_idx]] * md.sqrt_dage_inno ** 2, sigma_eta * md.sqrt_dage_inno), obs=d_alpha)
    else:
        alpha = beta[md.stint_codes] + nu[md.stint_codes] * md.rel_age
        if spec.process_noise:
            sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(spec.sigma_eta_scale))
            z = numpyro.sample("z_inno", dist.Normal(0.0, 1.0).expand([md.inno_idx.shape[0]]))
            inno_full = jnp.zeros(md.n_laps).at[md.inno_idx].set(md.sqrt_dage_inno * z)
            cs = jnp.cumsum(inno_full)
            W = cs - cs[md.first_pos][md.stint_codes]
            alpha = alpha + sigma_eta * W
    mu_y = alpha + gamma_lap * md.fuel_c

    # ---- observation model ---------------------------------------------------
    obs = md.y if observe else None
    if spec.likelihood == "skewt":
        sigma_c = numpyro.sample("sigma_eps_core", dist.HalfNormal(spec.sigma_eps_scale))
        lam = numpyro.sample("lambda_skew", dist.TruncatedNormal(spec.lambda_loc, spec.lambda_scale, low=-spec.lambda_bound, high=spec.lambda_bound))
        if spec.fix_df is None:
            eta = numpyro.deterministic("nu_df", 2.0 + numpyro.sample("nu_df_m2", dist.Gamma(spec.df_gamma_shape, spec.df_gamma_rate)))
        else:
            eta = numpyro.deterministic("nu_df", jnp.asarray(float(spec.fix_df)))
        numpyro.deterministic("sigma_eps_sd", sigma_c * jnp.sqrt(eta / (eta - 2.0)))
        numpyro.sample("y", HansenSkewT(mu_y, sigma_c, lam, eta), obs=obs)
    elif spec.likelihood == "studentt":
        sigma_c = numpyro.sample("sigma_eps_core", dist.HalfNormal(spec.sigma_eps_scale))
        if spec.fix_df is None:
            eta = numpyro.deterministic("nu_df", 2.0 + numpyro.sample("nu_df_m2", dist.Gamma(spec.df_gamma_shape, spec.df_gamma_rate)))
        else:
            eta = numpyro.deterministic("nu_df", jnp.asarray(float(spec.fix_df)))
        numpyro.sample("y", dist.StudentT(eta, mu_y, sigma_c), obs=obs)
    elif spec.likelihood == "gaussian":
        sigma_c = numpyro.sample("sigma_eps_core", dist.HalfNormal(spec.sigma_eps_scale))
        numpyro.sample("y", dist.Normal(mu_y, sigma_c), obs=obs)
    else:
        raise ValueError(spec.likelihood)
    numpyro.deterministic("mu_y_last", mu_y[-1])   # tiny hook for tests


def model_spec_fingerprint_fields(spec: ModelSpec) -> Dict[str, Any]:
    """Model fields entering the fit fingerprint (version-stable, see config.spec_fingerprint_dict)."""
    return spec_fingerprint_dict(spec)


# --------------------------------------------------------------------------- #
# latent-path reconstruction (NumPy, from stored draws)
# --------------------------------------------------------------------------- #
def reconstruct_alpha(ds: Dataset, baseline_stint: np.ndarray, nu_stint: np.ndarray, sigma_eta: Optional[np.ndarray], z_inno: Optional[np.ndarray], stint_index: int) -> np.ndarray:
    """Latent path α for one stint for a set of draws.
    baseline_stint, nu_stint: (draws, S); sigma_eta: (draws,); z_inno: (draws, n_inno) or None."""
    m = ds.stint_codes == stint_index
    a = baseline_stint[:, stint_index][:, None] + nu_stint[:, stint_index][:, None] * ds.rel_age[m][None, :]
    if z_inno is not None and sigma_eta is not None:
        # innovations belonging to this stint, in lap order
        lap_idx = np.flatnonzero(m)
        inno_pos = {li: k for k, li in enumerate(ds.inno_idx)}
        incr = np.zeros((z_inno.shape[0], m.sum()))
        for j, li in enumerate(lap_idx):
            k = inno_pos.get(li)
            if k is not None:
                incr[:, j] = np.sqrt(ds.dage[li]) * z_inno[:, k]
        a = a + sigma_eta[:, None] * np.cumsum(incr, axis=1)
    return a

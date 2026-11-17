"""Independent synthetic-data generator for smoke runs and parameter recovery.

Written from the mathematical specification in README §Model with plain NumPy
(no code shared with the NumPyro model or the JAX Hansen implementation): the
observation noise uses the inverse-CDF reference sampler, the latent path is
built lap by lap with an explicit loop, and zero-sum effects are drawn and
centred by hand.  A shared bug between generator and fit code would therefore
have to be a shared *misunderstanding*, not shared code.

The generator writes a lap table in the raw FastF1 column layout and pushes it
through the SAME cleaning/indexing code as the real data (data.clean,
data.build_dataset), so stint resets, Δage gaps, used tyres, race-centring and
the fuel proxy are all exercised exactly as in production.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import DataConfig, derive_seed
from .data import COMPOUND_ORDER, Dataset, build_dataset, clean
from .hansen import np_rvs


@dataclass
class TruthSpec:
    """Generating hyper-parameters (scenario).  Rates in s/lap, log-scale effects dimensionless."""
    mu_log_nu: float = -2.9
    compound_effects: tuple = (0.30, -0.02, -0.28)   # SOFT, MEDIUM, HARD (sum zero)
    sigma_track: float = 0.40
    sigma_driver: float = 0.05
    sigma_team: float = 0.10
    sigma_interaction: float = 0.20
    sigma_stint: float = 0.20
    baseline_mean: float = -0.5
    compound_base: tuple = (-0.6, 0.0, 0.6)           # SOFT faster baseline
    sigma_stint_base: float = 0.8
    gamma_mode: str = "hier"                            # hier | fixed
    gamma_fixed: float = 0.03
    mu_log_gamma: float = np.log(0.035)
    sigma_log_gamma: float = 0.2
    sigma_eta: float = 0.15
    likelihood: str = "skewt"
    sigma_core: float = 0.25
    lambda_skew: float = 0.6
    eta_df: float = 4.0
    signed_stint_sd: Optional[float] = None             # if set: nu_s = exp(group) + sd * u_s (can be < 0)
    # ---- v1.1 ---------------------------------------------------------------------------------
    sigma_race: float = 0.0                             # race (track x season) effect on the rate predictor; 0 = none
    additive: bool = False                              # identity link: nu_s = mu_add + C + T + D + G + I (+R) + sd_add * u_s  [s/lap]
    mu_nu_add: float = 0.05
    compound_effects_add: tuple = (0.025, -0.005, -0.020)
    sigma_track_add: float = 0.025
    sigma_driver_add: float = 0.004
    sigma_team_add: float = 0.006
    sigma_interaction_add: float = 0.012
    sigma_stint_add: float = 0.02
    sigma_race_add: float = 0.0
    compound_by_phase: float = 0.0                      # probability that all cars of a race use the same compound order (real races: ~0.8)


SCENARIOS: Dict[str, TruthSpec] = {
    "baseline": TruthSpec(),
    "known_gamma": TruthSpec(gamma_mode="fixed", gamma_fixed=0.03),
    "weak_signal": TruthSpec(mu_log_nu=-4.0, sigma_stint=0.10, sigma_interaction=0.10),
    "no_interaction": TruthSpec(sigma_interaction=0.0),
    "signed_rates": TruthSpec(mu_log_nu=-3.2, sigma_stint=0.0, signed_stint_sd=0.04),
    "heavy_tails": TruthSpec(eta_df=2.3, lambda_skew=0.7),
    # v1.1
    "identity_rates": TruthSpec(additive=True),                                  # truth generated with the identity link
    "identity_zero": TruthSpec(additive=True, mu_nu_add=0.0, compound_effects_add=(0.01, 0.0, -0.01)),   # no degradation on average: can the model say so?
    "race_effect": TruthSpec(sigma_track=0.30, sigma_race=0.30),                 # half of the "track" variance is season-specific
    "phase_confounded": TruthSpec(compound_by_phase=0.8),                        # compound nearly determined by stint order, as in real races
}


def _zero_sum(x: np.ndarray, axes=(-1,)) -> np.ndarray:
    for ax in axes:
        x = x - x.mean(axis=ax, keepdims=True)
    return x


def generate(scenario: str = "baseline", seed: int = 0, n_tracks: int = 5, n_drivers: int = 8, n_teams: int = 4, years=(1, 2),
             stints_per_car_race: int = 2, laps_per_stint=(8, 16), race_laps=(45, 65), gap_prob: float = 0.04,
             used_tyre_prob: float = 0.2, dcfg: Optional[DataConfig] = None, truth: Optional[TruthSpec] = None) -> Dataset:
    ts = truth if truth is not None else SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    dcfg = dcfg or DataConfig()
    tracks = [f"T{k+1:02d}" for k in range(n_tracks)]
    drivers = [f"D{k+1:02d}" for k in range(n_drivers)]
    teams = [f"Team{k+1}" for k in range(n_teams)]
    team_of = {d: teams[k % n_teams] for k, d in enumerate(drivers)}
    # two drivers switch teams in the second season (crossed driver/constructor design)
    team_of_year = {y: dict(team_of) for y in years}
    if len(years) > 1 and n_drivers >= 2:
        y2 = years[1]
        team_of_year[y2][drivers[0]] = teams[1 % n_teams]
        team_of_year[y2][drivers[1]] = teams[0]
    comps = COMPOUND_ORDER
    # A second, independent stream for everything added in v1.1, so that the main stream — and with it every
    # v1.0 scenario — generates exactly the same data as before.
    rng2 = np.random.default_rng([int(seed), 11])
    add = bool(ts.additive)
    # ---- true effects (the standard-normal draws are identical for both links; only the scales differ) ----
    C = np.asarray(ts.compound_effects_add if add else ts.compound_effects, float); C = C - C.mean()
    T = _zero_sum(rng.normal(0, 1, n_tracks)) * (ts.sigma_track_add if add else ts.sigma_track)
    Dd = _zero_sum(rng.normal(0, 1, n_drivers)) * (ts.sigma_driver_add if add else ts.sigma_driver)
    G = _zero_sum(rng.normal(0, 1, n_teams)) * (ts.sigma_team_add if add else ts.sigma_team)
    I = _zero_sum(rng.normal(0, 1, (n_tracks, 3)), axes=(0, 1)) * (ts.sigma_interaction_add if add else ts.sigma_interaction)
    sig_race = ts.sigma_race_add if add else ts.sigma_race
    R_true: Dict[str, float] = {}
    b = np.asarray(ts.compound_base, float); b = b - b.mean()
    if ts.gamma_mode == "hier":
        gamma = np.exp(ts.mu_log_gamma + ts.sigma_log_gamma * rng.normal(0, 1, n_tracks))
    else:
        gamma = np.full(n_tracks, ts.gamma_fixed)
    rows = []
    stint_truth = []
    round_no = 0
    for y in years:
        for q, tr in enumerate(tracks):
            round_no += 1
            L = int(rng.integers(race_laps[0], race_laps[1] + 1))
            race = f"{y}::{tr} Grand Prix::R"
            base_lap = 90.0 + 10 * rng.normal()   # absolute lap-time level of this race (removed by centring)
            R_true[race] = float(sig_race * rng2.normal()) if sig_race > 0 else 0.0
            race_plan = list(rng2.permutation(3)) if ts.compound_by_phase > 0 else None   # the order most cars follow in this race
            for d_i, drv in enumerate(drivers):
                team = team_of_year[y][drv]
                g_i = teams.index(team)
                lap = 2
                stint_no = 1
                # compound plan: random permutation, no repeats within the race if possible
                plan = list(rng.permutation(3))[:stints_per_car_race] if stints_per_car_race <= 3 else list(rng.integers(0, 3, stints_per_car_race))
                if race_plan is not None and rng2.uniform() < ts.compound_by_phase:
                    plan = (race_plan * (1 + stints_per_car_race // 3))[:stints_per_car_race]
                for sidx in range(stints_per_car_race):
                    c_i = int(plan[sidx])
                    n_laps = int(rng.integers(laps_per_stint[0], laps_per_stint[1] + 1))
                    if lap + n_laps > L:
                        n_laps = L - lap
                    if n_laps < 3:
                        break
                    start_age = int(rng.integers(3, 6)) if rng.uniform() < used_tyre_prob else 1
                    log_nu_group = (ts.mu_nu_add if add else ts.mu_log_nu) + C[c_i] + T[q] + Dd[d_i] + G[g_i] + I[q, c_i] + R_true[race]
                    u_s = rng.normal()
                    if add:                                  # identity link: the predictor IS the rate (s/lap) and may be negative
                        nu_s = float(log_nu_group + ts.sigma_stint_add * u_s)
                    elif ts.signed_stint_sd is None:
                        nu_s = float(np.exp(log_nu_group + ts.sigma_stint * u_s))
                    else:
                        nu_s = float(np.exp(log_nu_group) + ts.signed_stint_sd * u_s)
                    beta_s = ts.baseline_mean + b[c_i] + ts.sigma_stint_base * rng.normal()
                    sid = f"{race}::{drv}::{stint_no}"
                    stint_truth.append({"stint_id": sid, "nu_true": nu_s, "beta_true_uncentred": beta_s, "log_nu_group": log_nu_group, "u_true": u_s,
                                        "compound": comps[c_i], "track": tr, "driver": drv, "team": team, "race": race})
                    # latent path lap by lap (explicit loop; tyre age advances every lap, retained or not)
                    alpha = beta_s
                    age_prev = start_age
                    for k in range(n_laps):
                        ln = lap + k
                        age = start_age + k
                        if k > 0:
                            da = age - age_prev
                            alpha = alpha + nu_s * da + ts.sigma_eta * np.sqrt(da) * rng.normal()
                        age_prev = age
                        fuel = 110.0 * (L - ln) / (L - 1.0)
                        mu = alpha + gamma[q] * (fuel - 55.0)
                        if ts.likelihood == "skewt":
                            eps = float(np_rvs(0.0, ts.sigma_core, ts.lambda_skew, ts.eta_df, size=(), rng=rng))
                        elif ts.likelihood == "studentt":
                            eps = float(rng.standard_t(ts.eta_df) * ts.sigma_core)
                        else:
                            eps = float(rng.normal(0, ts.sigma_core))
                        status = 2 if rng.uniform() < gap_prob else 1        # yellow laps are removed by cleaning -> Δage gaps
                        rows.append({"RaceKey": race, "Year": y, "RoundNumber": round_no, "EventName": f"{tr} Grand Prix", "Location": tr,
                                     "Driver": drv, "Team": team, "LapNumber": float(ln), "Stint": float(stint_no), "Compound": comps[c_i],
                                     "TyreLife": float(age), "FreshTyre": start_age == 1, "TrackStatus": status, "IsAccurate": True,
                                     "LapTime_seconds": base_lap + mu + eps, "PitOutTime_seconds": np.nan, "PitInTime_seconds": np.nan,
                                     "Deleted": False, "FastF1Generated": False, "SessionType": "R", "mu_true": mu, "alpha_true": alpha,
                                     "fuel_true": fuel - 55.0, "base_lap_true": base_lap})
                    lap += n_laps + 1   # one in-lap/out-lap gap between stints (removed as in reality)
                    stint_no += 1
            # make sure LapNumber max equals L for the race (fuel formula uses the raw maximum)
            rows.append({"RaceKey": race, "Year": y, "RoundNumber": round_no, "EventName": f"{tr} Grand Prix", "Location": tr, "Driver": drivers[0],
                         "Team": team_of_year[y][drivers[0]], "LapNumber": float(L), "Stint": float(stints_per_car_race + 1), "Compound": "HARD",
                         "TyreLife": 1.0, "FreshTyre": True, "TrackStatus": 1, "IsAccurate": False, "LapTime_seconds": np.nan,
                         "PitOutTime_seconds": np.nan, "PitInTime_seconds": np.nan, "Deleted": False, "FastF1Generated": False, "SessionType": "R",
                         "mu_true": np.nan, "alpha_true": np.nan, "fuel_true": np.nan, "base_lap_true": base_lap})
    raw = pd.DataFrame(rows)
    clean_df, steps, excl = clean(raw, dcfg)
    ds = build_dataset(clean_df, raw, dcfg, source=f"synthetic:{scenario}:{seed}")
    # ---- align truth with the dataset ---------------------------------------
    st = pd.DataFrame(stint_truth).set_index("stint_id")
    med = ds.laps.groupby("RaceKey")["LapTime_seconds"].median()
    stints = ds.stints
    nu_true = st.loc[stints["stint_id"], "nu_true"].to_numpy(float)
    beta_unc = st.loc[stints["stint_id"], "beta_true_uncentred"].to_numpy(float)
    # centred truth: the model's beta_s is relative to the race median of the *retained* laps.
    # LapTime = base_lap + mu + eps  =>  y = LapTime - median = (mu + eps) - (median - base_lap)
    race_base = raw.groupby("RaceKey")["base_lap_true"].first()
    shift = np.array([med[r] - race_base[r] for r in stints["race"]])
    truth = {
        "scenario": scenario, "seed": seed, "spec": ts.__dict__,
        "mu_log_nu": ts.mu_log_nu, "compound_effect": C, "track_effect": T, "driver_effect": Dd, "team_effect": G, "interaction_effect": I,
        "compound_base_effect": b, "gamma_track": gamma, "sigma_track": ts.sigma_track, "sigma_driver": ts.sigma_driver, "sigma_team": ts.sigma_team,
        "sigma_interaction": ts.sigma_interaction, "sigma_stint": ts.sigma_stint, "sigma_stint_base": ts.sigma_stint_base, "sigma_eta": ts.sigma_eta,
        "sigma_eps_core": ts.sigma_core, "lambda_skew": ts.lambda_skew, "nu_df": ts.eta_df,
        # beta_0 of the model lives on the race-centred scale: subtract the mean per-race centring shift (approximate, race medians are random)
        "baseline_mean": ts.baseline_mean - float(np.mean(shift)), "baseline_mean_uncentred": ts.baseline_mean,
        "nu_stint": nu_true, "baseline_stint": beta_unc - shift, "u_stint": st.loc[stints["stint_id"], "u_true"].to_numpy(float),
        "levels": {"track": tracks, "driver": drivers, "team": teams, "compound": list(comps)},
        "mu_y_true": (ds.laps["mu_true"].to_numpy(float) - np.array([med[r] - race_base[r] for r in ds.laps["RaceKey"]])),
        "alpha_true": (ds.laps["alpha_true"].to_numpy(float) - np.array([med[r] - race_base[r] for r in ds.laps["RaceKey"]])),
        "cleaning_steps": steps.to_dict(orient="list"),
    }
    # map truth effects to the dataset's level order
    truth["track_effect"] = np.array([T[tracks.index(t)] for t in ds.levels["track"]])
    truth["driver_effect"] = np.array([Dd[drivers.index(d)] for d in ds.levels["driver"]])
    truth["team_effect"] = np.array([G[teams.index(t)] for t in ds.levels["team"]])
    truth["gamma_track"] = np.array([gamma[tracks.index(t)] for t in ds.levels["track"]])
    truth["interaction_effect"] = np.array([[I[tracks.index(t), comps.index(c)] for c in ds.levels["compound"]] for t in ds.levels["track"]])
    truth["compound_effect"] = np.array([C[comps.index(c)] for c in ds.levels["compound"]])
    truth["compound_base_effect"] = np.array([b[comps.index(c)] for c in ds.levels["compound"]])
    # ---- v1.1 additions to the truth dictionary --------------------------------------------------
    truth["additive"] = add
    if sig_race > 0:
        truth["race_effect"] = np.array([R_true[r] for r in ds.levels["race"]])
        truth["sigma_race_add" if add else "sigma_race"] = sig_race
    if add:      # the identity-link truth lives on the s/lap scale: expose it under the names of the identity-link sites
        for k in ("mu_log_nu", "sigma_track", "sigma_driver", "sigma_team", "sigma_interaction", "sigma_stint"):
            truth.pop(k, None)
        truth.update({"mu_nu_add": ts.mu_nu_add, "sigma_track_add": ts.sigma_track_add, "sigma_driver_add": ts.sigma_driver_add, "sigma_team_add": ts.sigma_team_add,
                      "sigma_interaction_add": ts.sigma_interaction_add, "sigma_stint_add": ts.sigma_stint_add})
    elif ts.signed_stint_sd is not None:
        truth["sigma_stint_add"] = ts.signed_stint_sd; truth.pop("sigma_stint", None)
    ds.truth = truth
    return ds


def generate_from_config(cfg, scenario: str, replicate: int, dcfg: Optional[DataConfig] = None) -> Dataset:
    s = cfg.smoke
    seed = derive_seed(cfg.seed, "synthetic", scenario, replicate)
    return generate(scenario=scenario, seed=seed, n_tracks=s["n_tracks"], n_drivers=s["n_drivers"], n_teams=s["n_teams"], years=tuple(s["years"]),
                    stints_per_car_race=s["stints_per_car_race"], laps_per_stint=tuple(s["laps_per_stint"]), dcfg=dcfg)

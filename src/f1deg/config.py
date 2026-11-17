"""Configuration: one explicit object, loaded from YAML, with per-mode overrides,
deterministic seed derivation and canonical fingerprints.

Nothing in here touches JAX so that the configuration can be read before the
XLA backend is initialised (needed for `chain_method="parallel"`).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

MODES = ("SMOKE", "PILOT", "FULL", "REPORT_ONLY")


# --------------------------------------------------------------------------- #
# deterministic hashing helpers
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    """Canonical JSON (sorted keys, no whitespace, floats repr'd) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default, ensure_ascii=False)


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path: os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def derive_seed(master_seed: int, *labels: Any) -> int:
    """Stable 32-bit seed from a master seed and string labels (SHA-256, never
    Python's randomised ``hash``).  Every stage / chain / replicate / fold gets
    its own label and therefore its own independent, reproducible stream."""
    s = "|".join([str(int(master_seed))] + [str(l) for l in labels])
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:4], "little")


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    fuel_start_kg: float = 110.0
    max_rel_laptime: float = 1.30
    min_stint_laps: int = 5
    dry_compounds: List[str] = field(default_factory=lambda: ["SOFT", "MEDIUM", "HARD"])
    team_map: Dict[str, str] = field(
        default_factory=lambda: {
            "AlphaTauri": "RB / AlphaTauri",
            "RB": "RB / AlphaTauri",
            "Alfa Romeo": "Sauber / Alfa Romeo",
            "Kick Sauber": "Sauber / Alfa Romeo",
        }
    )
    drop_lap1: bool = True
    green_only: bool = True           # TrackStatus == 1 only
    drop_deleted: bool = True
    drop_pit_laps: bool = True
    years: Optional[List[int]] = None  # None = all years in the file
    tracks: Optional[List[str]] = None  # None = all tracks (PILOT restricts)


@dataclass
class SamplerConfig:
    backend: str = "numpyro_lowlevel"   # the only production backend (explicit, no silent fallback)
    chains: int = 4
    warmup: int = 2000
    draws: int = 2000
    target_accept: float = 0.95
    max_treedepth: int = 10
    dense_mass_hyper: bool = True        # dense mass-matrix block for the global hyper-parameters
    chain_method: str = "sequential"     # sequential | vectorized | parallel
    checkpoint_every: int = 25           # transitions between committed checkpoints
    init_strategy: str = "median"        # median | uniform
    find_heuristic_step_size: bool = True
    save_latent_thin: int = 10           # keep every k-th draw of the lap-level latent innovations
    latent_dtype: str = "float32"
    max_resume_attempts: int = 20        # bounded automatic resumption inside one process
    # extension policy: increasing `draws` after warm-up keeps the chain; anything else = new run
    heartbeat_seconds: float = 60.0


@dataclass
class ModelSpec:
    name: str = "primary"
    likelihood: str = "skewt"            # skewt | studentt | gaussian
    fix_df: Optional[float] = None       # fixes eta (skewt) / nu (studentt); None = estimated
    interaction: bool = True             # track x compound interaction on log-rate
    pooling: str = "hier"                # hier (track+driver+team) | compound_only
    stint_rate_effect: bool = True       # U_s residual stint heterogeneity of log-rate
    compound_prior: str = "fixed"        # fixed (zero-sum N(0, compound_scale^2)) | hier (sigma_C ~ HN)
    compound_scale: float = 0.5
    compound_hier_scale: float = 0.75
    gamma: str = "hier"                  # hier (per-track lognormal) | fixed
    gamma_fixed_value: float = 0.03      # s/kg external assumption when gamma == fixed
    mu_log_gamma_loc: float = -3.2188758248682006  # log(0.04)
    mu_log_gamma_scale: float = 0.4
    sigma_log_gamma_scale: float = 0.3
    process_noise: bool = True
    latent_param: str = "noncentred"     # noncentred (z_inno innovations; default) | centred (alpha_lap sampled, RW prior as factors) — same model, sampler-side choice
    rate_link: str = "log"               # log | signed_stint (stint deviation additive on s/lap scale) | identity (all effects additive, s/lap)
    # prior scales (log-rate hierarchy) -------------------------------------
    mu_log_nu_loc: float = -3.0
    mu_log_nu_scale: float = 0.7
    sigma_track_scale: float = 0.5
    sigma_driver_scale: float = 0.3
    sigma_team_scale: float = 0.3
    sigma_interaction_scale: float = 0.3
    sigma_stint_scale: float = 0.5
    signed_stint_sd_scale: float = 0.05  # s/lap, used when rate_link == signed_stint
    # baselines ---------------------------------------------------------------
    baseline_mean_scale: float = 1.0
    compound_base_scale: float = 0.7     # fixed-scale zero-sum compound pace offsets (s)
    sigma_stint_base_scale: float = 1.0
    # noise -------------------------------------------------------------------
    sigma_eta_scale: float = 0.1
    sigma_eps_scale: float = 0.5
    lambda_loc: float = 0.3
    lambda_scale: float = 0.25
    lambda_bound: float = 0.95
    df_gamma_shape: float = 2.0
    df_gamma_rate: float = 0.5           # eta = 2 + Gamma(shape, rate): mean 2 + 4 = 6
    # ---- fields added in v1.1 (review round 1) -------------------------------------------
    # Every field below is listed in V11_FIELD_DEFAULTS with the value that reproduces the
    # v1.0 model; fields left at that value are dropped from the fingerprint, so fits made
    # with v1.0 keep their fit ID (tests/rev2/test_rev2_fingerprint_compat.py).
    race_effect: bool = False            # R_r ~ N(0, sigma_R^2): race (= track x season) effect on the rate
    sigma_race_scale: float = 0.3
    # identity link (rate_link == "identity"): every effect is additive on the s/lap scale and
    # the rate may be negative at every level (stint, track, compound, global)
    mu_nu_add_loc: float = 0.0
    mu_nu_add_scale: float = 0.10
    compound_scale_add: float = 0.03
    sigma_track_add_scale: float = 0.05
    sigma_driver_add_scale: float = 0.02
    sigma_team_add_scale: float = 0.02
    sigma_interaction_add_scale: float = 0.03
    sigma_race_add_scale: float = 0.03
    # ---- fields added in v1.2 (also listed in V11_FIELD_DEFAULTS: existing fits keep their fit ID) ----
    # group_scale decides whether the scales of the group effects T, D, G, I are LEARNED:
    #   hier   T = sigma_T * ZSN(1), sigma_T ~ HalfNormal(sigma_track_scale), ... (adaptive partial pooling; default)
    #   fixed  T ~ ZSN(s_T), D ~ ZSN(s_D), G ~ ZSN(s_G), I ~ ZSN(s_I; both axes) with FIXED scales: the same linear
    #          predictor and the same zero-sum constraints, but no sigma_T, sigma_D, sigma_G, sigma_I at all
    # A fixed scale left at None equals the scale parameter a of the half-normal hyper-prior of the hierarchical
    # model (sigma ~ HN(a) has E[sigma^2] = a^2), so both models give every effect the same prior variance and
    # differ only in whether the scale is learned from the data.  See fixed_group_scales().
    group_scale: str = "hier"
    track_fixed_scale: Optional[float] = None
    driver_fixed_scale: Optional[float] = None
    team_fixed_scale: Optional[float] = None
    interaction_fixed_scale: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **kw) -> "ModelSpec":
        return dataclasses.replace(self, **kw)


# Fields added to ModelSpec after v1.0 and the value at which the model is exactly the v1.0 model.
V11_FIELD_DEFAULTS: Dict[str, Any] = {
    "race_effect": False, "sigma_race_scale": 0.3,
    "mu_nu_add_loc": 0.0, "mu_nu_add_scale": 0.10, "compound_scale_add": 0.03, "sigma_track_add_scale": 0.05,
    "sigma_driver_add_scale": 0.02, "sigma_team_add_scale": 0.02, "sigma_interaction_add_scale": 0.03,
    "sigma_race_add_scale": 0.03,
    # v1.2
    "group_scale": "hier", "track_fixed_scale": None, "driver_fixed_scale": None, "team_fixed_scale": None, "interaction_fixed_scale": None,
}

GROUP_SCALES = ("hier", "fixed")


def fixed_group_scales(spec: "ModelSpec") -> Dict[str, float]:
    """Scales of T, D, G, I used when ``spec.group_scale == "fixed"`` (on the scale of the rate's linear predictor:
    log s/lap, or s/lap under the identity link).

    Default (field left at None): the scale parameter a of the half-normal hyper-prior that the hierarchical model
    puts on the corresponding sigma.  For sigma ~ HN(a), E[sigma^2] = a^2, hence Var(T_q) is the same under both
    models (a^2 (1 - 1/Q) with the zero-sum constraint): the priors of the effects agree in mean and covariance,
    and the only difference is that the hierarchical model learns the scale."""
    ident = spec.rate_link == "identity"
    base = {"track": spec.sigma_track_add_scale if ident else spec.sigma_track_scale,
            "driver": spec.sigma_driver_add_scale if ident else spec.sigma_driver_scale,
            "team": spec.sigma_team_add_scale if ident else spec.sigma_team_scale,
            "interaction": spec.sigma_interaction_add_scale if ident else spec.sigma_interaction_scale}
    over = {"track": spec.track_fixed_scale, "driver": spec.driver_fixed_scale, "team": spec.team_fixed_scale, "interaction": spec.interaction_fixed_scale}
    out = {k: float(base[k] if over[k] is None else over[k]) for k in base}
    for k, v in out.items():
        if not (v > 0):
            raise ValueError(f"fixed scale of the {k} effects must be positive, got {v!r}")
    return out


def spec_fingerprint_dict(spec: "ModelSpec", keep_name: bool = False) -> Dict[str, Any]:
    """Model fields that define the posterior, in a form that is stable across package versions:
    a field added after v1.0 is included only when it differs from the value that reproduces
    the v1.0 model.  Fits made before the field existed therefore keep their fingerprint,
    while any variant that actually uses the new field gets a new one."""
    d = spec.to_dict()
    if not keep_name:
        d.pop("name", None)
    for k, v in V11_FIELD_DEFAULTS.items():
        if k in d and d[k] == v:
            d.pop(k)
    return d


# priority of each experiment with respect to the review (docs/RUN_PLAN_uk.md); "" = not classified
PRIORITIES = ("mandatory", "recommended", "optional")


@dataclass
class ExperimentSpec:
    """One fit in the experiment matrix."""
    name: str
    model: ModelSpec
    data_variant: Dict[str, Any] = field(default_factory=dict)   # overrides of DataConfig
    sampler: Dict[str, Any] = field(default_factory=dict)        # overrides of SamplerConfig
    tier: int = 1
    question: str = ""
    differs_from_primary: str = ""
    allows: str = ""
    priority: str = ""                   # mandatory | recommended | optional (review round 1); "" = original matrix
    reviewer_points: str = ""            # codes of the reviewer remarks the fit answers (report.REVIEWER_POINTS)


@dataclass
class ForecastConfig:
    train_years: List[int] = field(default_factory=lambda: [2022, 2023])
    test_years: List[int] = field(default_factory=lambda: [2024])
    prefix_fraction: float = 0.75
    min_prefix_laps: int = 4
    min_target_laps: int = 2
    particles: int = 4000
    posterior_draws_used: int = 400
    centering: str = "first_quarter"     # test-race centering without future laps
    # training fits scored by the forecast stage; fits that are not completed are skipped (and reported as such)
    models: List[str] = field(default_factory=lambda: ["primary_train", "gauss_train", "pooled_train", "ch_adapted_train", "noint_train",
                                                       "signed_full_train", "race_effect_train", "unpooled_train"])
    # additional forecast schemes evaluated with the same training fits (no sampling).  The short prefix asks more of
    # the hierarchical prior than the main scheme, in which three quarters of the stint are already observed.
    extra_schemes: List[Dict[str, Any]] = field(default_factory=lambda: [{"tag": "_p25", "prefix_fraction": 0.25, "min_prefix_laps": 2}])


@dataclass
class PPCConfig:
    draws: int = 200
    batch_laps: int = 20000
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


@dataclass
class RecoveryConfig:
    replicates: int = 1                  # SMOKE/PILOT: one replicate (example); FULL default 3
    scenarios: List[str] = field(default_factory=lambda: ["baseline", "known_gamma", "weak_signal", "no_interaction", "signed_rates",
                                                          "identity_rates", "race_effect", "phase_confounded"])   # the last three: v1.1
    sampler: Dict[str, Any] = field(default_factory=lambda: {"warmup": 300, "draws": 300, "chains": 2})


@dataclass
class Config:
    mode: str = "SMOKE"
    data_csv: str = "data/f1_cache_combined_2022_2024.csv"
    outputs_root: str = "outputs"
    seed: int = 20260610
    resume: bool = True
    force_refit: bool = False
    precision: str = "float64"
    interval_prob: float = 0.94          # ETI used for ALL main results (continuity with the paper)
    extra_interval_probs: List[float] = field(default_factory=lambda: [0.5, 0.8])
    data: DataConfig = field(default_factory=DataConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    primary: ModelSpec = field(default_factory=ModelSpec)
    experiments: List[ExperimentSpec] = field(default_factory=list)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    ppc: PPCConfig = field(default_factory=PPCConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    pilot_tracks: List[str] = field(default_factory=lambda: ["Sakhir", "Monza", "Jeddah"])
    # Queue of fits for FULL (order = priority).  Empty = the whole matrix, mandatory fits first.  The experiment
    # MATRIX (cfg.experiments) is never filtered: fits outside the queue stay visible in the reports as "not run".
    run_fits: List[str] = field(default_factory=list)
    # Runtime override of the queue in ANY mode (CLI --fits, notebook); None = use the mode's own queue.
    fits_override: Optional[List[str]] = None
    smoke: Dict[str, Any] = field(default_factory=lambda: {"n_tracks": 5, "n_drivers": 8, "n_teams": 4, "years": [1, 2], "stints_per_car_race": 2, "laps_per_stint": (8, 16)})
    gates: Dict[str, float] = field(default_factory=lambda: {"rhat_max": 1.01, "ess_min": 400, "mcse_rel_max": 0.05, "divergences_max": 0, "ebfmi_min": 0.3})
    budget: Dict[str, Any] = field(default_factory=lambda: {"max_extension_rounds": 2, "extension_factor": 2.0, "max_hours_per_fit": 12.0})
    run_label: str = ""
    report_only: bool = False            # regenerate figures/tables/reports from disk, no sampling
    report_only_source: str = "FULL"     # which mode's outputs REPORT_ONLY reads

    # ------------------------------------------------------------------ paths
    @property
    def out_dir(self) -> Path:
        return Path(self.outputs_root) / self.mode.lower()

    def dirs(self) -> Dict[str, Path]:
        base = self.out_dir
        d = {k: base / k for k in ("fits", "figures", "tables", "diagnostics", "reports", "data", "logs", "tests", "forecast", "recovery", "ppc", "loo", "paper")}
        for p in d.values():
            p.mkdir(parents=True, exist_ok=True)
        d["root"] = base
        return d

    # ------------------------------------------------------------ fingerprints
    def versions(self) -> Dict[str, str]:
        import numpy, pandas, scipy

        v = {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__, "platform": platform.platform()}
        try:
            import jax, numpyro

            v.update({"jax": jax.__version__, "numpyro": numpyro.__version__, "jaxlib": getattr(jax, "__version_info__", None) and jax.__version__})
        except Exception:  # pragma: no cover
            pass
        try:
            import arviz

            v["arviz"] = arviz.__version__
        except Exception:  # pragma: no cover
            pass
        return v

    def sampler_fingerprint_fields(self, sampler: SamplerConfig) -> Dict[str, Any]:
        """Fields of the sampler configuration that define the kernel/adaptation
        and therefore the Markov chain.  `draws` is deliberately NOT included:
        extending the posterior draw target continues a compatible chain."""
        return {
            "backend": sampler.backend,
            "chains": sampler.chains,
            "warmup": sampler.warmup,
            "target_accept": sampler.target_accept,
            "max_treedepth": sampler.max_treedepth,
            "dense_mass_hyper": sampler.dense_mass_hyper,
            "chain_method": sampler.chain_method,
            "init_strategy": sampler.init_strategy,
            "find_heuristic_step_size": sampler.find_heuristic_step_size,
            "precision": self.precision,
            "seed": self.seed,
        }

    # ------------------------------------------------------------- overrides
    def apply_mode_defaults(self) -> "Config":
        """Mode-specific sampler/experiment defaults (explicit, documented)."""
        c = copy.deepcopy(self)
        if c.mode == "SMOKE":
            c.sampler = dataclasses.replace(c.sampler, chains=2, warmup=150, draws=150, checkpoint_every=25, save_latent_thin=5)
            c.ppc.draws = 60
            c.forecast.particles = 1500
            c.forecast.posterior_draws_used = 100
            c.forecast.train_years = [1]
            c.forecast.test_years = [2]
            c.recovery.replicates = 1
            c.recovery.sampler = {"warmup": 120, "draws": 120, "chains": 2}
        elif c.mode == "PILOT":
            c.sampler = dataclasses.replace(c.sampler, chains=4, warmup=300, draws=300, checkpoint_every=25)
            c.ppc.draws = 100
            c.forecast.particles = 2000
            c.forecast.posterior_draws_used = 200
            c.recovery.replicates = 1
            c.recovery.sampler = {"warmup": 200, "draws": 200, "chains": 2}
            c.data.tracks = list(c.pilot_tracks)
            c.budget = {**c.budget, "max_extension_rounds": 0}   # PILOT measures; it does not chase the gates
        elif c.mode == "FULL":
            c.recovery.replicates = max(c.recovery.replicates, 3)
        return c

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: os.PathLike):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


def default_experiments(primary: ModelSpec) -> List[ExperimentSpec]:
    """The fixed, review-driven experiment matrix (no Cartesian product).

    Tier 1 = model variants on the full data; Tier 2 = targeted sensitivity; Tier 3 = forecast
    training fits (train years only).  `priority` classifies every fit with respect to the first
    review round (mandatory / recommended / optional) and `reviewer_points` names the remarks it
    answers; whether a fit has actually been RUN is never stored here — it is read from the stage
    registry and the checkpoint store (reports/experiments_status.csv).

    The specs of the experiments that existed in v1.0 are unchanged, so their fit IDs are too."""
    E = ExperimentSpec
    ex = [
        # ------------------------------------------------------------------ tier 1: full data
        E("primary", primary, tier=1, priority="mandatory", reviewer_points="A1 A4 A5 A6 A11 A12 A13",
          question="Основні оцінки швидкостей деградації, їх неоднорідність за компаундами/трасами/конструкторами/пілотами та взаємодія траса×компаунд.",
          differs_from_primary="—",
          allows="Усі основні estimands статті; conditional PPC; приклади відрізків."),
        E("noint", primary.replace(name="noint", interaction=False), tier=1, priority="mandatory", reviewer_points="A3 A5",
          question="Який внесок взаємодії траса×компаунд? Чи змінюються головні ефекти без неї?",
          differs_from_primary="без I_{q,c}",
          allows="Порівняння σ_U, компаундних контрастів і прогнозу з/без взаємодії."),
        E("pooled", primary.replace(name="pooled", pooling="compound_only", interaction=False), tier=1, priority="optional", reviewer_points="A3",
          question="Роль ієрархії: complete pooling за трасами/пілотами/конструкторами (лише компаунд + стінт).",
          differs_from_primary="без T, D, G, I; залишено C та U_s",
          allows="Оцінку внеску групової структури у прогноз і в розкид швидкостей (прогнозне порівняння вже є через pooled_train)."),
        E("ch_adapted", primary.replace(name="ch_adapted", pooling="compound_only", interaction=False, stint_rate_effect=False), tier=1, priority="optional", reviewer_points="A3 B1",
          question="Адаптована baseline Cappello–Hoegh: швидкість лише за компаундом, без стінтової неоднорідності.",
          differs_from_primary="log ν = μ + C_c (без T, D, G, I, U); всі інші блоки як у primary",
          allows="Чесно названу адаптацію (не точне відтворення: інші дані, трасовий γ, оцінюване η>2, стінтові baselines)."),
        # v1.2: the same group effects as in primary, but WITHOUT learned group-level scales (see ModelSpec.group_scale).  `pooled`
        # removes T, D, G, I altogether, so it cannot tell "the effects are useless" from "learning their scales is useless".
        E("unpooled", primary.replace(name="unpooled", group_scale="fixed"), tier=1, priority="recommended", reviewer_points="A3",
          question="Чи дає користь саме адаптивне ієрархічне об'єднання? Ті самі T, D, G, I, але з фіксованими апріорними масштабами (σ_T, σ_D, σ_G, σ_I не оцінюються).",
          differs_from_primary="T ~ ZSN(0.5), D, G, I ~ ZSN(0.3) з фіксованими масштабами; σ_T, σ_D, σ_G, σ_I не оцінюються",
          allows="Порівняння ефектів і швидкостей з primary: що змінює навчання масштабів (стискання ефектів пілотів, конструкторів, взаємодії); PPC і LOO."),
        E("gauss", primary.replace(name="gauss", likelihood="gaussian"), tier=1, priority="mandatory", reviewer_points="A3 A7",
          question="Чи потрібні важкі/асиметричні хвости? Що змінюється в rates при гаусових похибках?",
          differs_from_primary="ε ~ N(0, σ²)",
          allows="Порівняння PPC хвостів, rates/contrasts і прогнозних scores."),
        E("studentt", primary.replace(name="studentt", likelihood="studentt"), tier=1, priority="recommended", reviewer_points="A7",
          question="Чи потрібна асиметрія (λ) понад симетричні важкі хвости?",
          differs_from_primary="ε ~ Student-t(ν>2, 0, σ_core)",
          allows="Ізолювати внесок λ у PPC/прогноз; перевірити стабільність rates."),
        E("gamma_fixed", primary.replace(name="gamma_fixed", gamma="fixed"), tier=1, priority="mandatory", reviewer_points="A3",
          question="Ідентифікація ν–γ: що змінюється, якщо γ зафіксовано зовнішнім припущенням 0.03 с/кг?",
          differs_from_primary="γ_q ≡ 0.03 с/кг (не оцінюється)",
          allows="Чутливість рівнів ν_s та трасових ефектів до паливного припущення."),
        # ------------------------------------------------------------------ tier 2: sensitivity (full data)
        # Thresholds: an exact scan of the data (tables/data_threshold_sensitivity.csv) shows that 1.20 and 1.40 change only 36 and 19
        # of 57 143 laps, so they cannot show any sensitivity.  1.10 removes 467 laps (0.8 %, the whole tail y > 6.5 s) and 1.07
        # removes 3 560 laps (6.2 %) and 105 stints: these are the informative variants.  clean_120 / clean_140 stay in the matrix
        # (their specs and fit IDs are unchanged) but are optional.
        E("clean_110", primary.replace(name="clean_110"), data_variant={"max_rel_laptime": 1.10}, tier=2, priority="mandatory", reviewer_points="A7",
          question="Чутливість до порога відбору кіл: 1,10 вилучає 467 кіл (0,8 %), тобто весь хвіст y > 6,5 с.",
          differs_from_primary="дані: поріг 1.10", allows="Стабільність rates/contrasts і параметрів хвоста (η, λ, σ_core) без найповільніших кіл."),
        E("clean_120", primary.replace(name="clean_120"), data_variant={"max_rel_laptime": 1.20}, tier=2, priority="optional", reviewer_points="A7",
          question="Чутливість до порога очищення 1.20 (змінює лише 36 кіл із 57 143).", differs_from_primary="дані: поріг 1.20", allows="Стабільність rates/contrasts."),
        E("minlaps_8", primary.replace(name="minlaps_8"), data_variant={"min_stint_laps": 8}, tier=2, priority="mandatory", reviewer_points="A7",
          question="Чутливість до мінімальної довжини стінта (8 збережених кіл).", differs_from_primary="дані: ≥8 кіл", allows="Стабільність rates/contrasts."),
        E("signed_full", primary.replace(name="signed_full", rate_link="identity"), tier=2, priority="recommended", reviewer_points="A2",
          question="Чи є додатність деградації висновком із даних? Тотожна функція зв'язку: усі рівні швидкості можуть бути від'ємними.",
          differs_from_primary="ν_s = μ_ν + C_c + T_q + D_d + G_g + I_qc + σ_U u_s на шкалі с/коло (без exp)",
          allows="P(μ_ν>0), P(швидкість компаунда>0), частку трас і відрізків з P(ν<0)>0.5; LOO/прогноз проти log-зв'язку."),
        # optional: in the pilot check on real data (3 tracks, 2 x (250 + 250)) the centred parameterisation mixed worse than the
        # non-centred one, so it is a diagnostic experiment, not a remedy for the low E-BFMI (docs/MODEL_REGISTRY_uk.md, section 6)
        E("primary_centred", primary.replace(name="primary_centred", latent_param="centred"), tier=2, priority="optional", reviewer_points="A1",
          question="Та сама модель у центрованій параметризації латентного стану: чи відтворюються оцінки і чи змінюється E-BFMI? (діагностичний експеримент)",
          differs_from_primary="параметри — стани α_n (а не інновації z_j); апостеріорний розподіл той самий",
          allows="Незалежне підтвердження оцінок primary іншим sampler-геометричним шляхом; діагностику E-BFMI."),
        E("clean_107", primary.replace(name="clean_107"), data_variant={"max_rel_laptime": 1.07}, tier=2, priority="recommended", reviewer_points="A7",
          question="Чутливість до суворого порога відбору кіл: 1,07 вилучає 3 560 кіл (6,2 %) і 105 відрізків.",
          differs_from_primary="дані: поріг 1.07", allows="Стабільність rates/contrasts за помітно іншої вибірки; потребу у важких хвостах без повільних кіл."),
        E("clean_140", primary.replace(name="clean_140"), data_variant={"max_rel_laptime": 1.40}, tier=2, priority="optional", reviewer_points="A7",
          question="Чутливість до порога очищення 1.40 (додає лише 19 кіл).", differs_from_primary="дані: поріг 1.40", allows="Стабільність rates/contrasts."),
        E("minlaps_10", primary.replace(name="minlaps_10"), data_variant={"min_stint_laps": 10}, tier=2, priority="recommended", reviewer_points="A7",
          question="Чутливість до мінімальної довжини стінта (10 збережених кіл).", differs_from_primary="дані: ≥10 кіл", allows="Стабільність rates/contrasts."),
        E("skewt_df5", primary.replace(name="skewt_df5", fix_df=5.0), tier=2, priority="optional", reviewer_points="A7",
          question="Чи зміщують оцінки ν надто важкі хвости? Індекс хвоста зафіксовано на помірному значенні η = 5.",
          differs_from_primary="η ≡ 5 (не оцінюється)",
          allows="Проміжну точку між η→2 (primary) і гаусовими похибками для rates/contrasts і PPC."),
        E("signed_stint", primary.replace(name="signed_stint", rate_link="signed_stint"), tier=2, priority="optional", reviewer_points="A2",
          question="Обмежена signed-drift чутливість: чи підтримують дані від'ємні стінтові дрейфи?",
          differs_from_primary="ν_s = exp(η_group) + σ_U^{add}·u_s на шкалі с/коло (може бути < 0)",
          allows="P(ν_s<0) за стінтами; груповий рівень лишається додатним за побудовою (тому основний варіант — signed_full)."),
        E("race_effect", primary.replace(name="race_effect", race_effect=True), tier=2, priority="optional", reviewer_points="A3 A14",
          question="Наскільки трасові ефекти стабільні між сезонами? Додатковий ефект гонки (траса×сезон) R_r.",
          differs_from_primary="log ν_s += R_{r(s)}, R_r ~ N(0, σ_R²)",
          allows="Розклад «трасової» дисперсії на сталу (σ_T) і сезонну (σ_R) частини — пояснення, чому ієрархія не поліпшує прогноз на новий сезон."),
        E("comp_hier", primary.replace(name="comp_hier", compound_prior="hier"), tier=2, priority="optional", reviewer_points="A5",
          question="Чутливість до прайора компаундних ефектів: ієрархічний σ_C ~ HN(0.75) замість фіксованої шкали 0.5.",
          differs_from_primary="C_c = σ_C·ZSN, σ_C ~ HN(0.75)",
          allows="Перевірити, чи фіксована шкала впливає на компаундні контрасти; показати збіжність σ_C."),
        E("fuel_prior_wide", primary.replace(name="fuel_prior_wide", mu_log_gamma_scale=1.0), tier=2, priority="optional", reviewer_points="A3",
          question="Чутливість до прайора μ_γ (ширший: SD 1.0 на log-шкалі).",
          differs_from_primary="μ_γ ~ N(log 0.04, 1.0²)",
          allows="Показати, наскільки γ_q і ν інформуються даними, а наскільки прайором."),
        # ------------------------------------------------------------------ tier 3: forecast training fits
        E("primary_train", primary.replace(name="primary_train"), data_variant={"years": "train"}, tier=3, priority="mandatory", reviewer_points="A3 A14",
          question="Навчальний fit для часово коректної прогнозної перевірки (train 2022–2023 → test 2024).",
          differs_from_primary="дані: лише train-роки", allows="Prefix-only прогноз останньої чверті стінтів 2024 року."),
        E("gauss_train", primary.replace(name="gauss_train", likelihood="gaussian"), data_variant={"years": "train"}, tier=3, priority="mandatory", reviewer_points="A3",
          question="Навчальний fit гаусової альтернативи для прогнозного порівняння.", differs_from_primary="ε ~ N; дані train", allows="Предиктивне порівняння хвостів."),
        E("pooled_train", primary.replace(name="pooled_train", pooling="compound_only", interaction=False), data_variant={"years": "train"}, tier=3, priority="mandatory", reviewer_points="A3 A14",
          question="Навчальний fit complete-pooling альтернативи для прогнозного порівняння.", differs_from_primary="без T, D, G, I; дані train", allows="Внесок ієрархії у прогноз."),
        E("ch_adapted_train", primary.replace(name="ch_adapted_train", pooling="compound_only", interaction=False, stint_rate_effect=False), data_variant={"years": "train"}, tier=3,
          priority="mandatory", reviewer_points="A3 B1",
          question="Навчальний fit адаптації моделі Cappello–Hoegh для прогнозного порівняння з нею.",
          differs_from_primary="log ν = μ + C_c; дані train", allows="Прогнозне порівняння з адаптацією моделі Cappello–Hoegh на тих самих відрізках 2024 року; компаундні rates проти primary_train."),
        E("unpooled_train", primary.replace(name="unpooled_train", group_scale="fixed"), data_variant={"years": "train"}, tier=3, priority="recommended", reviewer_points="A3 A14",
          question="Навчальний fit моделі з фіксованими масштабами групових ефектів: чи поліпшує прогноз 2024 року саме навчання масштабів (primary_train проти unpooled_train)?",
          differs_from_primary="T, D, G, I з фіксованими масштабами; дані train",
          allows="Ізолювати внесок адаптивного часткового об'єднання у прогноз: pooled_train (без групових ефектів) → unpooled_train (ефекти без навчених масштабів) → primary_train."),
        E("noint_train", primary.replace(name="noint_train", interaction=False), data_variant={"years": "train"}, tier=3, priority="recommended", reviewer_points="A3 A5",
          question="Навчальний fit без взаємодії: чи дає I_qc прогнозний виграш на новому сезоні?",
          differs_from_primary="без I_{q,c}; дані train", allows="Ізолювати внесок взаємодії у прогноз (між pooled_train і primary_train)."),
        E("signed_full_train", primary.replace(name="signed_full_train", rate_link="identity"), data_variant={"years": "train"}, tier=3, priority="optional", reviewer_points="A2",
          question="Навчальний fit моделі з тотожною функцією зв'язку для прогнозного порівняння з log-зв'язком.",
          differs_from_primary="тотожний зв'язок; дані train", allows="Прогнозні scores специфікації, що допускає ν ≤ 0."),
        E("race_effect_train", primary.replace(name="race_effect_train", race_effect=True), data_variant={"years": "train"}, tier=3, priority="optional", reviewer_points="A3 A14",
          question="Навчальний fit з ефектом гонки: чи поліпшує прогноз відокремлення сезонної складової трасового ефекту?",
          differs_from_primary="log ν_s += R_r; дані train", allows="Прогноз нової гонки з R_r з популяції N(0, σ_R²)."),
    ]
    return ex


def load_config(path: Optional[os.PathLike] = None, **overrides) -> Config:
    """Load `config.yaml` (if given) on top of defaults, then apply overrides
    (``mode=...`` etc.) and mode defaults.  Experiments are rebuilt from the
    primary spec unless the YAML lists them explicitly."""
    cfg = Config()
    raw: Dict[str, Any] = {}
    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    cfg = _merge_into(cfg, raw)
    cfg = _merge_into(cfg, overrides)
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {cfg.mode!r}")
    if cfg.mode == "REPORT_ONLY":          # REPORT_ONLY = the source mode's configuration without sampling
        cfg.report_only = True
        cfg.mode = cfg.report_only_source
    if not cfg.experiments:
        cfg.experiments = default_experiments(cfg.primary)
    return cfg.apply_mode_defaults()


def _merge_into(cfg: Config, raw: Dict[str, Any]) -> Config:
    cfg = copy.deepcopy(cfg)
    for k, v in raw.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key {k!r}")
        cur = getattr(cfg, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            setattr(cfg, k, dataclasses.replace(cur, **v))
        elif k == "experiments" and isinstance(v, list):
            cfg.experiments = [ExperimentSpec(name=e["name"], model=ModelSpec(**e["model"]), data_variant=e.get("data_variant", {}), sampler=e.get("sampler", {}), tier=e.get("tier", 1), question=e.get("question", ""), differs_from_primary=e.get("differs_from_primary", ""), allows=e.get("allows", ""),
                                              priority=e.get("priority", ""), reviewer_points=e.get("reviewer_points", "")) for e in v]
        else:
            setattr(cfg, k, v)
    return cfg

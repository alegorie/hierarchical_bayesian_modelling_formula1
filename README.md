
```markdown
# Hierarchical Bayesian State-Space Model for Formula 1 Tyre Degradation

A reproducible Bayesian analysis of **Formula 1 tyre degradation from 2022–2024**, implemented in **PyMC**.

The project extends a single-driver state-space approach to a multi-season, multi-track and multi-driver setting. Its main goal is to estimate tyre degradation while separating it from important confounding effects such as **fuel burn-off, track evolution, driver differences, constructor differences and track-specific behaviour**.

The primary analysis is contained in:

```text
f1_tyre_degradation_hierarchical_ssm.ipynb
```

and uses the accompanying cached Formula 1 lap dataset:

```text
f1_cache_combined_2022_2024.csv
```

## Motivation

Raw lap-time trends within a stint are not direct measurements of tyre degradation. During a race, lap times are simultaneously affected by several processes:

- tyres generally lose performance with increasing tyre age;
- fuel mass decreases throughout the race, tending to make the car faster;
- track conditions may improve or deteriorate;
- drivers and constructors differ systematically;
- degradation varies across compounds and circuits;
- occasional slow laps create a heavy and asymmetric right tail in the lap-time distribution.

Consequently, a simple regression of lap time on tyre age can produce misleading degradation estimates, including negative raw slopes at some circuits. This repository models these effects jointly in a hierarchical Bayesian state-space framework.

## Research questions

The notebook is designed to investigate questions such as:

1. How quickly do **SOFT, MEDIUM and HARD** tyres degrade after accounting for fuel and other contextual effects?
2. How much does degradation vary between **tracks, drivers and constructors**?
3. Does a **track × compound interaction** materially improve the representation of degradation?
4. How large is the apparent fuel/track-evolution effect at each circuit?
5. How uncertain are driver, constructor and track degradation rankings?
6. Does the asymmetric **Hansen skewed-t likelihood** provide a better description than a symmetric Student-t alternative?

## Model overview

For each cleaned race lap, the outcome is the lap time centred by the median cleaned lap time of the corresponding race.

Conceptually, the observation model is

\[
y_n = \alpha_n + \gamma_{q(n)}\,\mathrm{fuel}_n + \varepsilon_n,
\]

where:

- \(y_n\) is the race-centred lap time;
- \(\alpha_n\) is latent racing pace;
- \(\gamma_{q(n)}\) is a track-specific coefficient capturing the fuel-related trend together with residual systematic track evolution;
- \(\varepsilon_n\) follows the selected robust likelihood.

Within each stint, latent pace evolves approximately as

\[
\alpha_{t+1} = \alpha_t + \nu_s\,\Delta\mathrm{age}_t + \eta_t,
\]

where \(\nu_s\) is the stint-specific degradation rate and \(\eta_t\) optionally introduces process noise.

The degradation rate is modelled hierarchically on the log scale using partially pooled effects for contextual factors including:

- tyre compound;
- track;
- driver;
- constructor;
- optional track × compound interaction.

The primary likelihood is the **Hansen skewed-t distribution**, which accommodates the strong asymmetric slow-lap tail observed in race data. A symmetric Student-t likelihood is available as a robustness comparison.

## Data

The included CSV contains cached Formula 1 race-lap data for **2022, 2023 and 2024**.

Raw dataset dimensions:

```text
74,605 rows × 51 columns
```

Examples of available fields include:

- season, event and circuit;
- driver and team;
- lap number and stint number;
- tyre compound and tyre life;
- lap and sector times;
- pit-in and pit-out indicators;
- track status;
- speed-trap measurements;
- lap validity/deletion flags.

The notebook performs its own cleaning before model fitting. Important exclusions include invalid/deleted laps, wet-weather compounds, pit in/out laps, unsuitable track-status observations and very short stints.

Only the dry compounds

```text
SOFT / MEDIUM / HARD
```

are used in the degradation model.

## Repository structure

```text
.
├── README.md
├── f1_cache_combined_2022_2024.csv
└── f1_tyre_degradation_hierarchical_ssm.ipynb
```

If the notebook has a different local filename, either rename it or open it directly in Jupyter; the dataset filename is controlled by `DATA_PATH` in the configuration cell.

## Notebook workflow

The notebook is organized as a complete analysis pipeline:

```text
0. Environment and configuration
1. Dataset inspection
2. Cleaning
3. Optional QUICK_RUN subsampling
4. Outcome, covariates and index construction
5. Exploratory data analysis
6. Model structure, priors and identifiability
7. PyMC implementation
8. Prior predictive checks
9. Posterior sampling
10. Convergence diagnostics
11. Posterior predictive checks
12. Posterior analysis of effects
13. Rankings with uncertainty
14. Example stint reconstruction
15. Discussion and limitations
```

## Installation

Python **3.12** is recommended.

A minimal environment can be created with Conda:

```bash
conda create -n f1ssm python=3.12 -y
conda activate f1ssm
pip install pymc arviz nutpie pandas plotly scipy h5netcdf h5py ipywidgets
```

Optional packages:

```bash
# Static Plotly image export
pip install kaleido

# Alternative NumPyro/JAX NUTS backend
pip install numpyro jax jaxlib

# Optional model graph rendering
pip install graphviz
```

The notebook was developed for modern PyMC/ArviZ versions and documents its tested package versions in the environment section.

## Running the analysis

Clone the repository and place the notebook and CSV in the same directory.

Then start Jupyter:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

Open the notebook and run the cells sequentially.

The default data path is:

```python
DATA_PATH = "f1_cache_combined_2022_2024.csv"
```

## Configuration

The main experiment settings are centralized near the beginning of the notebook.

### Full vs quick run

```python
QUICK_RUN = False
```

Set `QUICK_RUN=True` for a smaller representative five-track analysis using:

```python
["Sakhir", "Suzuka", "Barcelona", "Monza", "Jeddah"]
```

This is useful for development and model checks. The full analysis uses all available cleaned tracks.

### Sampling

Default settings are:

```python
CHAINS        = 4
N_TUNE        = 2000
N_DRAWS       = 2000
TARGET_ACCEPT = 0.95
SEED          = 20260610
```

The sampler can be selected with:

```python
SAMPLER = "auto"
```

Automatic selection attempts the supported high-performance backends before falling back to standard PyMC sampling.

### Model switches

The main specification can be modified through:

```python
LIKELIHOOD    = "skewt"
PROCESS_NOISE = True
INTERACTION   = True
```

Available likelihoods include:

```text
skewt      Hansen skewed-t; primary specification
studentt   symmetric Student-t robustness specification
```

To fit the Student-t model as an additional comparison and calculate PSIS-LOO, enable:

```python
RUN_STUDENT_T_COMPARISON = True
```

Note that this requires fitting a second model and therefore substantially increases runtime.

## Cleaning settings

Important defaults include:

```python
MIN_STINT_LAPS  = 5
MAX_REL_LAPTIME = 1.30
FUEL_START_KG   = 110.0
```

These values are documented and justified in the notebook rather than treated as hidden preprocessing choices.

## Diagnostics

The workflow includes explicit checks for posterior reliability, including:

- split-\(\hat R\);
- effective sample size;
- divergences;
- energy/BFMI diagnostics;
- posterior geometry and parameter dependence;
- prior predictive checks;
- posterior predictive checks.

Do not interpret degradation rankings or group differences from a run that has unresolved convergence problems.

## Outputs

The notebook produces interactive tables and Plotly figures for, among other things:

- raw within-stint degradation slopes;
- circuit-level degradation patterns;
- prior predictive behaviour;
- sampling diagnostics;
- posterior predictive checks;
- tyre compound effects;
- track, driver and constructor effects;
- track-specific fuel/evolution coefficients;
- variance decomposition;
- uncertainty-aware degradation rankings;
- reconstructed latent pace for representative stints.

When enabled, posterior samples are saved as an ArviZ `InferenceData` NetCDF file:

```text
idata_f1_tyre_degradation.nc
```

controlled by:

```python
SAVE_IDATA = True
IDATA_PATH = "idata_f1_tyre_degradation.nc"
```

## Interpretation of rankings

The repository deliberately avoids presenting posterior means as deterministic league tables.

For drivers, constructors and other grouped effects, the notebook reports quantities such as:

- expected posterior rank;
- probability of being ranked first;
- probability of being in the top three;
- posterior intervals for degradation multipliers.

This is important because hierarchical partial pooling naturally shrinks weakly observed groups toward the population mean. A mid-table ranking with high uncertainty can therefore be the correct statistical conclusion rather than missing information from the model.

## Reproducibility

The analysis fixes a random seed and centralizes all major modelling decisions in the configuration cell.

For a reproducible run:

1. use the same CSV input;
2. use the same Python/package versions;
3. retain the configuration values;
4. run the notebook from top to bottom;
5. archive the resulting `InferenceData` file together with the exact notebook version.

Because Bayesian MCMC results are stochastic, minor numerical differences between machines or sampler backends are expected even when the substantive posterior conclusions are unchanged.

## Computational considerations

The full hierarchical model is computationally intensive. Runtime depends strongly on:

- number of tracks included;
- interaction term being enabled;
- number of tuning and posterior draws;
- sampler backend;
- CPU/GPU hardware;
- whether an additional Student-t comparison is requested.

For experimentation, start with `QUICK_RUN=True` and shorter chains. Use the registered full configuration only for final inference after the model and preprocessing pipeline have been validated.

The notebook discusses these limitations in detail and should be consulted before citing individual posterior estimates.

## Methodological background

This work is a multi-driver, multi-track extension inspired by the state-space approach to Formula 1 tyre degradation described by Cappello & Hoegh. The notebook also uses standard Bayesian workflow ideas including hierarchical partial pooling, posterior predictive checking and PSIS-LOO model comparison.

See the notebook's **References** section for the complete bibliography used by the analysis.

## Citation

If you use this repository in academic work, please cite the associated paper or thesis once its final bibliographic information is available.

A repository citation can temporarily be given as:

```text
Markov, Y. & Tavrov, D. Hierarchical Bayesian State-Space Modelling of
Tyre Degradation in Formula 1 (2022–2024). Source code and data analysis repository.
```

## License

No software or data license is specified by the current repository materials. Add an explicit `LICENSE` file before redistribution or public release, and verify that redistribution of the cached Formula 1/FastF1-derived data is compatible with the relevant upstream terms.
```

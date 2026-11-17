"""f1deg — reproducible, crash-safe computational core for the revision of
"Hierarchical Bayesian state-space model for estimating tyre degradation of
Formula One cars" (Markov & Tavrov).

Modules
-------
config       configuration dataclasses, modes, deterministic seeds, fingerprints
data         loading, audit tables, cleaning, index construction, fuel proxy
hansen       Hansen (1994) skew-t: NumPy reference + JAX/NumPyro implementation
model        NumPyro model variants, prior table, symbol map
checkpoint   transactional checkpoint store (generations, checksums, locks)
sampler      chunked NUTS runner with exact sampler-state resume
stages       stage registry (pending/running/completed/failed, dependencies)
diagnostics  MCMC diagnostics and gates
synthetic    independent synthetic-data generator + recovery design
ppc          prior / posterior predictive checks
baseline     fuel-corrected OLS baselines
forecast     prefix-only particle-filter forecasts and proper scores (MAE, CRPS, log score)
loo          PSIS-LOO and WAIC from stored draws
link         rate-link helpers (log / signed_stint / identity)
export       figures and tables
report       Ukrainian summaries, experiment registry, reviewer response matrix
paper        paper-ready tables and the list of what is still missing
pipeline     mode-driven orchestration used by the notebook and the CLI
"""
__version__ = "1.2.0"

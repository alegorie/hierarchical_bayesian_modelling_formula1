"""Rate-link helpers shared by results / ppc / forecast / recovery (v1.1).

Three links exist (ModelSpec.rate_link):

    log           log ν_s = μ_ν + C + T + D + G + I (+R) + σ_U u_s          ν_s > 0 by construction
    signed_stint  ν_s = exp(μ_ν + C + T + D + G + I (+R)) + σ_U^add u_s     group level > 0, stint level signed
    identity      ν_s = μ_ν^add + C + T + D + G + I (+R) + σ_U^add u_s      signed at every level

Posterior draws do not carry the ModelSpec, so the link is recognised from the sampled sites:
`mu_nu_add` exists only under the identity link, `sigma_stint_add` without it only under signed_stint.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

IDENTITY_SUFFIX = "_add"


def link_name(sites: Iterable[str]) -> str:
    sites = set(sites)
    if "mu_nu_add" in sites:
        return "identity"
    if "sigma_stint_add" in sites:
        return "signed_stint"
    return "log"


def mu_site(sites: Iterable[str]) -> str:
    """Name of the global level of the rate's linear predictor."""
    return "mu_nu_add" if "mu_nu_add" in set(sites) else "mu_log_nu"


def group_is_additive(sites: Iterable[str]) -> bool:
    """True when the GROUP-level linear predictor is on the s/lap scale (identity link)."""
    return link_name(sites) == "identity"


def to_rate(eta, sites: Iterable[str]):
    """Group-level linear predictor -> rate in s per lap of tyre age."""
    return eta if group_is_additive(sites) else np.exp(eta)


def hyper(base: str, sites: Iterable[str]) -> str:
    """`sigma_track` -> `sigma_track_add` under the identity link (the scale differs, hence the name)."""
    return base + IDENTITY_SUFFIX if group_is_additive(sites) else base


def predictor_units(sites: Iterable[str]) -> str:
    return "с/коло" if group_is_additive(sites) else "лог с/коло"

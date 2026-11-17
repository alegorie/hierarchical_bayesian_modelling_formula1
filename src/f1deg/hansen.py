"""Hansen (1994) skewed Student-t distribution.

Two implementations that must agree (tests/test_hansen.py):

* ``np_*``   — NumPy/SciPy *reference* written from the textbook formulas
               (Hansen 1994; Jondeau & Rockinger 2003 inverse CDF).
* ``jax_*``  — the production implementation used in the likelihood, written
               in an algebraically different, boundary-stable form.

Parameterisation ("core scale", identical to the old notebook):
    y = mu + sigma * z,  z ~ Hansen(lambda, eta) standardised (mean 0, var 1 for eta > 2)
    sigma_core = sigma * sqrt((eta - 2) / eta)         <- the parameter with the prior
so that sigma_core is the exact analogue of the Student-t scale and stays finite
as eta -> 2.  The *standard deviation* of y is sigma_core * sqrt(eta/(eta-2)),
which diverges as eta -> 2 (and is reported only as a derived quantity).

Support: y in R; lambda in (-1, 1); eta > 2.
"""
from __future__ import annotations

import numpy as np
from scipy import special as sps
from scipy import stats as st

# --------------------------------------------------------------------------- #
# NumPy reference implementation (textbook form)
# --------------------------------------------------------------------------- #

def np_constants(lam, eta):
    lam = np.asarray(lam, float); eta = np.asarray(eta, float)
    logc = sps.gammaln((eta + 1) / 2) - sps.gammaln(eta / 2) - 0.5 * np.log(np.pi * (eta - 2))
    c = np.exp(logc)
    a = 4.0 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1.0 + 3.0 * lam ** 2 - a ** 2)
    return logc, c, a, b


def np_logpdf_std(z, lam, eta):
    """log density of the *standardised* Hansen skew-t at z."""
    z = np.asarray(z, float)
    logc, c, a, b = np_constants(lam, eta)
    den = np.where(z < -a / b, 1.0 - lam, 1.0 + lam)
    kern = 1.0 + ((b * z + a) / den) ** 2 / (eta - 2)
    return np.log(b) + logc - 0.5 * (eta + 1) * np.log(kern)


def np_logpdf(y, mu, sigma_core, lam, eta):
    sigma = sigma_core * np.sqrt(eta / (eta - 2.0))
    z = (np.asarray(y, float) - mu) / sigma
    return np_logpdf_std(z, lam, eta) - np.log(sigma)


def np_cdf_std(z, lam, eta):
    z = np.asarray(z, float)
    _, _, a, b = np_constants(lam, eta)
    s = np.sqrt(eta / (eta - 2.0))
    left = (1 - lam) * st.t.cdf((b * z + a) / (1 - lam) * s, eta)
    right = (1 + lam) * st.t.cdf((b * z + a) / (1 + lam) * s, eta) - lam
    return np.where(z < -a / b, left, right)


def np_cdf(y, mu, sigma_core, lam, eta):
    sigma = sigma_core * np.sqrt(eta / (eta - 2.0))
    return np_cdf_std((np.asarray(y, float) - mu) / sigma, lam, eta)


def np_ppf_std(u, lam, eta):
    """Inverse CDF (Jondeau & Rockinger 2003)."""
    u = np.asarray(u, float)
    _, _, a, b = np_constants(lam, eta)
    s = np.sqrt((eta - 2.0) / eta)
    left = ((1 - lam) * s * st.t.ppf(np.clip(u / (1 - lam), 1e-300, 1 - 1e-16), eta) - a) / b
    right = ((1 + lam) * s * st.t.ppf(np.clip((u + lam) / (1 + lam), 1e-300, 1 - 1e-16), eta) - a) / b
    return np.where(u < (1 - lam) / 2, left, right)


def np_ppf(u, mu, sigma_core, lam, eta):
    sigma = sigma_core * np.sqrt(eta / (eta - 2.0))
    return mu + sigma * np_ppf_std(u, lam, eta)


def np_rvs(mu, sigma_core, lam, eta, size=None, rng=None):
    """Inverse-CDF sampler (reference)."""
    rng = np.random.default_rng() if rng is None else rng
    mu, sigma_core, lam, eta = (np.asarray(v, float) for v in (mu, sigma_core, lam, eta))
    if size is None:
        size = np.broadcast(mu, sigma_core, lam, eta).shape
    u = rng.uniform(size=size)
    return np_ppf(u, mu, sigma_core, lam, eta)


def np_rvs_twopiece(mu, sigma_core, lam, eta, size=None, rng=None):
    """Independent two-piece sampler (the construction used in JAX): draw
    t ~ t_eta, draw the side with P(right) = (1+lambda)/2, scale by (1 +/- lambda)."""
    rng = np.random.default_rng() if rng is None else rng
    mu, sigma_core, lam, eta = (np.asarray(v, float) for v in (mu, sigma_core, lam, eta))
    if size is None:
        size = np.broadcast(mu, sigma_core, lam, eta).shape
    t = np.abs(st.t.rvs(eta, size=size, random_state=rng))
    sgn = np.where(rng.uniform(size=size) < (1 + lam) / 2, 1.0, -1.0)
    _, _, a, b = np_constants(lam, eta)
    s = np.sqrt((eta - 2.0) / eta)
    z = ((1 + sgn * lam) * sgn * t * s - a) / b
    sigma = sigma_core * np.sqrt(eta / (eta - 2.0))
    return mu + sigma * z


def np_moments_std(lam, eta):
    """Mean and variance of the standardised distribution (should be 0, 1)."""
    from scipy.integrate import quad

    f = lambda z: np.exp(np_logpdf_std(z, lam, eta))
    m0 = quad(f, -np.inf, np.inf, limit=400)[0]
    m1 = quad(lambda z: z * f(z), -np.inf, np.inf, limit=400)[0]
    m2 = quad(lambda z: z * z * f(z), -np.inf, np.inf, limit=400)[0]
    return m0, m1, m2


def implied_sd(sigma_core, eta):
    """Standard deviation of y implied by the core scale (diverges as eta -> 2)."""
    return sigma_core * np.sqrt(eta / (eta - 2.0))


# --------------------------------------------------------------------------- #
# JAX implementation (boundary-stable algebra, written in terms of t = (y-mu)/sigma_core)
# --------------------------------------------------------------------------- #
try:
    import jax
    import jax.numpy as jnp
    from jax.scipy import special as jsp
    from jax import random as jrandom
    import numpyro.distributions as ndist
    from numpyro.distributions import constraints
    from numpyro.distributions.util import promote_shapes, validate_sample

    HAVE_JAX = True
except Exception:  # pragma: no cover
    HAVE_JAX = False


if HAVE_JAX:

    def jax_logpdf(y, mu, sigma_core, lam, eta):
        """log p(y | mu, sigma_core, lambda, eta), stable as eta -> 2.

        With t = (y - mu)/sigma_core,  c~ = Gamma((eta+1)/2) / (sqrt(pi) Gamma(eta/2)),
        a~ = 4 lambda c~ / (eta - 1)  (= a / sqrt(eta-2)),  b^2 = 1 + 3 lambda^2 - a~^2 (eta-2):
            log p = log b + log c~ - 0.5 log eta - log sigma_core
                    - (eta+1)/2 * log(1 + ((b t / sqrt(eta) + a~) / (1 -/+ lambda))^2)
        branch: left piece iff t < -a~ sqrt(eta) / b.
        """
        t = (y - mu) / sigma_core
        logct = jsp.gammaln((eta + 1.0) / 2.0) - jsp.gammaln(eta / 2.0) - 0.5 * jnp.log(jnp.pi)
        ct = jnp.exp(logct)
        at = 4.0 * lam * ct / (eta - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam ** 2 - at ** 2 * (eta - 2.0))
        left = t < -at * jnp.sqrt(eta) / b
        den = jnp.where(left, 1.0 - lam, 1.0 + lam)
        arg = (b * t / jnp.sqrt(eta) + at) / den
        return jnp.log(b) + logct - 0.5 * jnp.log(eta) - jnp.log(sigma_core) - 0.5 * (eta + 1.0) * jnp.log1p(arg * arg)

    def _t_cdf(x, df):
        """Student-t CDF via the regularised incomplete beta."""
        x2 = x * x
        ib = jsp.betainc(df / 2.0, 0.5, df / (df + x2))
        return jnp.where(x >= 0, 1.0 - 0.5 * ib, 0.5 * ib)

    def jax_cdf(y, mu, sigma_core, lam, eta):
        t = (y - mu) / sigma_core
        logct = jsp.gammaln((eta + 1.0) / 2.0) - jsp.gammaln(eta / 2.0) - 0.5 * jnp.log(jnp.pi)
        at = 4.0 * lam * jnp.exp(logct) / (eta - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam ** 2 - at ** 2 * (eta - 2.0))
        left = t < -at * jnp.sqrt(eta) / b
        u = b * t + at * jnp.sqrt(eta)
        F_left = (1.0 - lam) * _t_cdf(u / (1.0 - lam), eta)
        F_right = (1.0 + lam) * _t_cdf(u / (1.0 + lam), eta) - lam
        return jnp.where(left, F_left, F_right)

    def jax_sample(key, mu, sigma_core, lam, eta, shape):
        """Two-piece construction: t ~ t_eta, side ~ Bernoulli((1+lambda)/2)."""
        k1, k2 = jrandom.split(key)
        tabs = jnp.abs(jrandom.t(k1, eta, shape))
        sgn = jnp.where(jrandom.uniform(k2, shape) < (1.0 + lam) / 2.0, 1.0, -1.0)
        logct = jsp.gammaln((eta + 1.0) / 2.0) - jsp.gammaln(eta / 2.0) - 0.5 * jnp.log(jnp.pi)
        at = 4.0 * lam * jnp.exp(logct) / (eta - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam ** 2 - at ** 2 * (eta - 2.0))
        return mu + sigma_core * ((1.0 + sgn * lam) * sgn * tabs - at * jnp.sqrt(eta)) / b

    class HansenSkewT(ndist.Distribution):
        """NumPyro distribution: Hansen skew-t in the core-scale parameterisation."""

        arg_constraints = {"loc": constraints.real, "scale": constraints.positive,
                           "skew": constraints.interval(-1.0, 1.0), "df": constraints.greater_than(2.0)}
        support = constraints.real
        reparametrized_params = ["loc", "scale"]

        def __init__(self, loc=0.0, scale=1.0, skew=0.0, df=5.0, *, validate_args=None):
            self.loc, self.scale, self.skew, self.df = promote_shapes(loc, scale, skew, df)
            batch_shape = jax.lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale), jnp.shape(skew), jnp.shape(df))
            super().__init__(batch_shape=batch_shape, validate_args=validate_args)

        @validate_sample
        def log_prob(self, value):
            return jax_logpdf(value, self.loc, self.scale, self.skew, self.df)

        def sample(self, key, sample_shape=()):
            shape = sample_shape + self.batch_shape
            return jax_sample(key, self.loc, self.scale, self.skew, self.df, shape)

        def cdf(self, value):
            return jax_cdf(value, self.loc, self.scale, self.skew, self.df)

        @property
        def mean(self):
            return self.loc  # standardised Hansen has zero mean (eta > 2)

        @property
        def variance(self):
            return (self.scale ** 2) * self.df / (self.df - 2.0)

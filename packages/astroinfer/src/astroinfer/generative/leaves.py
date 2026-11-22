"""Fundamental 1-D distributions. Conditioning enters only through the param
specs (mu, sigma, a, b may each be a constant, a hyperparameter name, or a
Conditional reading earlier variables). One TruncatedNormal class therefore
covers both the plain leaf and the 'conditional whose mean/sigma ride on
earlier draws' case."""
import jax
import jax.numpy as jnp
import numpy as np

from .base import Distribution
from .inversion import truncnorm_cdf, truncnorm_pdf, truncnorm_ppf
from .spec import ParamSpec, resolve, spec_deps


class TruncatedNormal(Distribution):
    def __init__(self, var: str, mu: ParamSpec, sigma: ParamSpec, a, b):
        self.var_names = [var]
        self.var = var
        self.mu, self.sigma = mu, sigma
        self.a, self.b = a, b

    def _ms(self, x, L):
        return resolve(self.mu, x, L), resolve(self.sigma, x, L)

    def cond_cdf(self, var, val, x, L):
        mu, sigma = self._ms(x, L)
        return truncnorm_cdf(val, mu, sigma, self.a, self.b)

    def cond_pdf(self, var, val, x, L):
        mu, sigma = self._ms(x, L)
        return truncnorm_pdf(val, mu, sigma, self.a, self.b)

    def cond_ppf(self, var, u, x, L):
        mu, sigma = self._ms(x, L)
        return truncnorm_ppf(u, mu, sigma, self.a, self.b)

    def bounds(self, var):
        return self.a, self.b

    def deps(self, var):
        return spec_deps(self.mu) + spec_deps(self.sigma)


class Uniform(Distribution):
    def __init__(self, var: str, a, b):
        self.var_names = [var]
        self.var = var
        self.a, self.b = a, b

    def cond_cdf(self, var, val, x, L):
        return jnp.clip((val - self.a) / (self.b - self.a), 0.0, 1.0)

    def cond_pdf(self, var, val, x, L):
        d = 1.0 / (self.b - self.a)
        return jnp.where((val >= self.a) & (val <= self.b), d, 0.0)

    def cond_ppf(self, var, u, x, L):
        return self.a + u * (self.b - self.a)

    def bounds(self, var):
        return self.a, self.b

    def deps(self, var):
        return []


class Beta(Distribution):
    """Beta on [0,1]. cond_cdf uses the regularized incomplete beta. Gradients
    w.r.t. the shape params flow through betainc; if you hit a gradient gap on
    older JAX, prefer TruncatedNormal or supply a closed-form quantile."""
    def __init__(self, var: str, alpha: ParamSpec, beta: ParamSpec):
        self.var_names = [var]
        self.var = var
        self.alpha, self.beta = alpha, beta
        self.a, self.b = 0.0, 1.0

    def _ab(self, x, L):
        return resolve(self.alpha, x, L), resolve(self.beta, x, L)

    def cond_cdf(self, var, val, x, L):
        al, be = self._ab(x, L)
        return jnp.clip(jax.scipy.special.betainc(al, be, jnp.clip(val, 0.0, 1.0)),
                        0.0, 1.0)

    def cond_pdf(self, var, val, x, L):
        al, be = self._ab(x, L)
        logp = ((al - 1) * jnp.log(val) + (be - 1) * jnp.log1p(-val)
                - jax.scipy.special.betaln(al, be))
        return jnp.where((val > 0.0) & (val < 1.0), jnp.exp(logp), 0.0)

    def bounds(self, var):
        return 0.0, 1.0

    def deps(self, var):
        return spec_deps(self.alpha) + spec_deps(self.beta)

# ------------------------------------------------------------
# A leaf built from a NumPyro distribution
# ------------------------------------------------------------

def _icdf_types():
    import numpyro.distributions as dist
    from numpyro.distributions.truncated import (
        LeftTruncatedDistribution,
        RightTruncatedDistribution,
        TwoSidedTruncatedDistribution,
    )
    plain = (dist.Uniform, dist.Normal, dist.LogNormal, dist.Exponential,
             dist.Cauchy, dist.Logistic)
    truncated = (TwoSidedTruncatedDistribution, LeftTruncatedDistribution,
                 RightTruncatedDistribution)
    return plain, truncated, (dist.Normal, dist.Cauchy, dist.Logistic)


def has_differentiable_icdf(d):
    """Whether NumPyro's ``d.icdf`` is a differentiable closed form here.

    True for Uniform, Normal, LogNormal, Exponential, Cauchy and Logistic,
    and for truncations (TruncatedNormal and the other truncated classes) of
    Normal, Cauchy and Logistic. Beta and Gamma are not on the list: their
    ``icdf`` needs tensorflow_probability.
    """
    plain, truncated, bases = _icdf_types()
    if type(d) in truncated:
        return type(d.base_dist) in bases
    return type(d) in plain


def _concrete_float(v):
    if isinstance(v, jax.core.Tracer):
        return None
    a = np.asarray(v)
    return float(a.reshape(())) if a.size == 1 else None


def support_bounds(support):
    """``(a, b)`` of a scalar NumPyro constraint, infinite on open sides.

    Returns None when a bound is not a concrete number (it depends on the
    parameters).
    """
    lo = getattr(support, "lower_bound", None)
    hi = getattr(support, "upper_bound", None)
    lo = -np.inf if lo is None else _concrete_float(lo)
    hi = np.inf if hi is None else _concrete_float(hi)
    return None if lo is None or hi is None else (lo, hi)


class NumPyroLeaf(Distribution):
    """One scalar variable with a NumPyro distribution.

    ``make_dist(x, L)`` returns the distribution given the drawn prefix
    ``x`` and the hyperparameters ``L``; ``deps`` names the variables of
    ``x`` it reads. Both arrive batched, with the sample shape (E, N) on the
    drawn values, so the returned distribution has that batch shape
    (NumPyro broadcasts its parameters).

    ``cond_cdf`` is ``d.cdf`` with the value clipped to the support,
    ``cond_pdf`` is ``exp(d.log_prob)``, evaluated at a point inside the
    support and masked to zero outside it, so that out-of-support values
    give no NaN gradients. ``cond_ppf`` is ``d.icdf`` for the distributions
    listed in :func:`has_differentiable_icdf`; for the others it returns
    None and ``GenerativeModel`` inverts the CDF by bisection, which needs
    finite ``bounds``.

    ``bounds`` is ``(a, b)``; by default it comes from the support of
    ``make_dist({}, {})`` when that call succeeds and its bounds are
    numbers, and is infinite otherwise. Pass it for a distribution whose
    parameters come from ``L`` or ``x``.
    """

    def __init__(self, var: str, make_dist, deps=(), bounds=None):
        self.var_names = [var]
        self.var = var
        self.make_dist = make_dist
        self._deps = [deps] if isinstance(deps, str) else list(deps)
        self._bounds = None if bounds is None else tuple(map(float, bounds))

    def cond_cdf(self, var, val, x, L):
        d = self.make_dist(x, L)
        lo = getattr(d.support, "lower_bound", None)
        hi = getattr(d.support, "upper_bound", None)
        if lo is not None:
            val = jnp.maximum(val, lo)
        if hi is not None:
            val = jnp.minimum(val, hi)
        return jnp.clip(d.cdf(val), 0.0, 1.0)

    def cond_pdf(self, var, val, x, L):
        d = self.make_dist(x, L)
        inside = d.support(val)
        safe = jnp.where(inside, val, d.support.feasible_like(val))
        return jnp.where(inside, jnp.exp(d.log_prob(safe)), 0.0)

    def cond_ppf(self, var, u, x, L):
        d = self.make_dist(x, L)
        if has_differentiable_icdf(d):
            return d.icdf(u)
        a, b = self.bounds(var)
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(
                f"{var}: {type(d).__name__} has no differentiable inverse CDF "
                "and unbounded support, so it cannot be drawn by bisection"
            )
        return None

    def bounds(self, var):
        if self._bounds is None:
            try:
                b = support_bounds(self.make_dist({}, {}).support)
            except Exception:  # noqa: BLE001 - parameters read from x or L
                b = None
            self._bounds = b or (-np.inf, np.inf)
        return self._bounds

    def deps(self, var):
        return list(self._deps)

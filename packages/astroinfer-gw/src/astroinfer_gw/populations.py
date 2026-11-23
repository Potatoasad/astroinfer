"""NumPyro distributions for gravitational-wave populations that NumPyro lacks.

    PowerLaw            p(x) proportional to x^alpha on [low, high], with closed-form cdf
                        and icdf, smooth in alpha through alpha = -1, and an optional fixed
                        support wider than [low, high]
    PowerLawRedshift    p(z) proportional to dV_c/dz (1 + z)^(lamb - 1) on [0, z_max],
                        tabulated on a fixed redshift grid
    ProjectionFactor    the isotropic projection factor, Beta(2, 4), with a polynomial cdf

All three are ordinary ``numpyro.distributions.Distribution`` subclasses, so they can be
fields of a ``StructuredDistribution`` (``GWParams.distribution(...)``) and sample sites
of a NumPyro model. They have ``support``, ``log_prob`` and ``cdf``, which is what
``StructuredDistribution.as_generative`` needs; the first two also have ``icdf``. The
conversion draws a distribution that is not on its list of known inverse CDFs (these
three are not) by bisection on its ``cdf``, inside the bounds of its ``support``, so the
support must be a fixed interval and the ``cdf`` should be cheap. A ``PowerLaw`` whose
bounds depend on other fields (the mass ratio above ``m_min / m_1``) therefore takes
``support=`` with fixed outer bounds.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes

from .physics import RedshiftToLuminosityDistance

__all__ = ["PowerLaw", "PowerLawRedshift", "ProjectionFactor"]

# Below this |alpha + 1|, PowerLaw uses series expansions in alpha + 1.
_SMALL = 1e-4


def _expm1_ratio(a, t):
    """expm1(a t) / a, which tends to t as a -> 0, with gradients in a everywhere."""
    small = jnp.abs(a) < _SMALL
    safe = jnp.where(small, 1.0, a)
    at = a * t
    return jnp.where(small, t * (1.0 + at / 2.0 + at**2 / 6.0), jnp.expm1(safe * t) / safe)


def _log1p_ratio(a, c):
    """log1p(a c) / a, which tends to c as a -> 0, with gradients in a everywhere."""
    small = jnp.abs(a) < _SMALL
    safe = jnp.where(small, 1.0, a)
    ac = a * c
    return jnp.where(small, c * (1.0 - ac / 2.0 + ac**2 / 3.0), jnp.log1p(safe * c) / safe)


class PowerLaw(Distribution):
    r"""Power law p(x) proportional to x^alpha on [low, high].

    .. math::

        p(x) = \frac{(\alpha+1)\,x^\alpha}{h^{\alpha+1} - l^{\alpha+1}}, \qquad l \le x \le h,

    and p(x) = 1 / (x \log(h/l)) at alpha = -1. ``log_prob``, ``cdf`` and ``icdf`` are
    closed forms written with ``expm1`` and ``log1p``, so they are smooth in alpha
    through alpha = -1 and differentiable in alpha, low and high.

    :param alpha: power-law index.
    :param low: lower bound, > 0.
    :param high: upper bound, > low.
    :param support: optional fixed ``(a, b)`` containing every [low, high] the
        distribution will be given. ``support`` is then ``interval(a, b)``, and the
        density and cdf are 0 on the parts of it outside [low, high]. Use it when low or
        high depend on other variables, so that ``as_generative`` has finite bounds
        for bisection. By default the support is ``interval(low, high)``.
    """

    arg_constraints = {
        "alpha": constraints.real,
        "low": constraints.positive,
        "high": constraints.positive,
    }
    reparametrized_params = ["alpha", "low", "high"]
    pytree_data_fields = ("alpha", "low", "high", "_support")

    def __init__(self, alpha, low, high, *, support=None, validate_args=None):
        self.alpha, self.low, self.high = promote_shapes(alpha, low, high)
        batch_shape = lax.broadcast_shapes(jnp.shape(alpha), jnp.shape(low), jnp.shape(high))
        if support is None:
            self._support = constraints.interval(low, high)
        else:
            a, b = support
            self._support = constraints.interval(float(a), float(b))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def _a(self):
        return self.alpha + 1.0

    def _log_norm(self, a):
        """log of the normalization, log((high^a - low^a) / a)."""
        log_low = jnp.log(self.low)
        return a * log_low + jnp.log(_expm1_ratio(a, jnp.log(self.high) - log_low))

    def log_prob(self, value):
        a = self._a()
        inside = (value >= self.low) & (value <= self.high)
        safe = jnp.where(inside, value, self.low)
        lp = self.alpha * jnp.log(safe) - self._log_norm(a)
        return jnp.where(inside, lp, -jnp.inf)

    def cdf(self, value):
        a = self._a()
        x = jnp.clip(value, self.low, self.high)
        log_low = jnp.log(self.low)
        return jnp.clip(
            _expm1_ratio(a, jnp.log(x) - log_low) / _expm1_ratio(a, jnp.log(self.high) - log_low),
            0.0, 1.0,
        )

    def icdf(self, q):
        a = self._a()
        log_low = jnp.log(self.low)
        ratio = _expm1_ratio(a, jnp.log(self.high) - log_low)
        return jnp.exp(log_low + _log1p_ratio(a, q * ratio))

    def sample(self, key, sample_shape=()):
        u = jax.random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)


@lru_cache(maxsize=None)
def _default_cosmology():
    return RedshiftToLuminosityDistance()


def _interp(x, xp, fp):
    """``jnp.interp`` broadcast over the leading axes of xp and fp."""
    return jnp.vectorize(jnp.interp, signature="(),(n),(n)->()")(x, xp, fp)


def _grid_interp(x, z, fp):
    """Linear interpolation at x of fp (..., n + 1) on the uniform grid z, clamped at the ends.

    Finds the interval by division rather than by search, which is much faster than
    ``jnp.interp`` for the many evaluations of a bisection.
    """
    if jnp.ndim(fp) > 1:
        return _interp(x, z, fp)
    n = z.shape[0] - 1
    t = jnp.clip((x - z[0]) / (z[1] - z[0]), 0.0, n)
    i = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, n - 1)
    w = t - i
    return fp[i] * (1.0 - w) + fp[i + 1] * w


class PowerLawRedshift(Distribution):
    r"""Redshift distribution p(z) proportional to dV_c/dz (1+z)^(lamb-1) on [0, z_max].

    The factor 1/(1+z) converts a rate per unit source-frame time into a rate per unit
    detector-frame time, so lamb = 0 is uniform in comoving volume and source-frame
    time. dV_c/dz comes from ``cosmology``, a
    :class:`~astroinfer_gw.physics.RedshiftToLuminosityDistance` (Planck15 by default;
    pass ``measurement.cosmology`` to share the measurement's).

    The density is normalized on a fixed grid of ``n_grid`` intervals in [0, z_max] with
    the trapezoid rule. ``cdf`` interpolates the cumulative trapezoid integral, and
    ``icdf`` interpolates its inverse, so both are differentiable in lamb.
    ``log_prob`` uses the same grid for dV_c/dz, and agrees with the derivative of
    ``cdf`` to the accuracy of the grid.

    :param lamb: power-law index of (1 + z).
    :param z_max: largest redshift; at most ``cosmology.z_max``.
    :param cosmology: the cosmology table; default Planck15.
    :param n_grid: number of grid intervals.
    """

    arg_constraints = {"lamb": constraints.real}
    reparametrized_params = ["lamb"]
    pytree_data_fields = ("lamb", "_z", "_dvdz")
    pytree_aux_fields = ("z_max",)

    def __init__(self, lamb, z_max=1.5, *, cosmology=None, n_grid=2000, validate_args=None):
        cosmology = _default_cosmology() if cosmology is None else cosmology
        self.z_max = float(z_max)
        if not 0.0 < self.z_max <= cosmology.z_max:
            raise ValueError(f"z_max must be in (0, {cosmology.z_max}] for {cosmology}, got {z_max}")
        z = np.linspace(0.0, self.z_max, int(n_grid) + 1)
        self._z = jnp.asarray(z)
        self._dvdz = jnp.asarray(np.interp(z, cosmology.z, cosmology.dV_c_dz) / 1e9)   # Gpc^3
        self.lamb = lamb
        super().__init__(batch_shape=jnp.shape(lamb), validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(0.0, self.z_max)

    def _cumulative(self):
        """The unnormalized cumulative integral on the grid, shape batch + (n_grid + 1,)."""
        lamb = jnp.asarray(self.lamb)[..., None]
        pdf = self._dvdz * (1.0 + self._z) ** (lamb - 1.0)
        steps = 0.5 * (pdf[..., 1:] + pdf[..., :-1]) * jnp.diff(self._z)
        zero = jnp.zeros(steps.shape[:-1] + (1,))
        return jnp.concatenate([zero, jnp.cumsum(steps, axis=-1)], axis=-1)

    def log_prob(self, value):
        total = self._cumulative()[..., -1]
        inside = (value >= 0.0) & (value <= self.z_max)
        safe = jnp.where(inside, value, 0.5 * self.z_max)
        lp = (jnp.log(_grid_interp(safe, self._z, self._dvdz))
              + (self.lamb - 1.0) * jnp.log1p(safe) - jnp.log(total))
        return jnp.where(inside, lp, -jnp.inf)

    def cdf(self, value):
        cum = self._cumulative()
        return _grid_interp(value, self._z, cum / cum[..., -1:])

    def icdf(self, q):
        cum = self._cumulative()
        return _interp(q, cum / cum[..., -1:], self._z)

    def sample(self, key, sample_shape=()):
        u = jax.random.uniform(key, sample_shape + self.batch_shape)
        return self.icdf(u)


class ProjectionFactor(Distribution):
    r"""The isotropic projection factor, Theta ~ Beta(2, 4), with a polynomial cdf.

    The same distribution as :func:`astroinfer_gw.physics.isotropic_projection_factor`,

    .. math::

        p(\Theta) = 20\,\Theta(1-\Theta)^3, \qquad
        F(\Theta) = 1 - (1-\Theta)^4 (1 + 4\Theta),

    but its ``cdf`` is the polynomial above rather than the regularized incomplete beta
    function NumPyro's ``Beta`` evaluates, so bisection on it (in ``as_generative``) is
    much cheaper. It has no ``icdf``.
    """

    arg_constraints = {}
    support = constraints.unit_interval
    reparametrized_params = []

    def __init__(self, *, validate_args=None):
        super().__init__(batch_shape=(), validate_args=validate_args)

    def log_prob(self, value):
        inside = (value >= 0.0) & (value <= 1.0)
        safe = jnp.where(inside, value, 0.5)
        lp = jnp.log(20.0) + jnp.log(safe) + 3.0 * jnp.log1p(-safe)
        return jnp.where(inside, lp, -jnp.inf)

    def cdf(self, value):
        x = jnp.clip(value, 0.0, 1.0)
        return 1.0 - (1.0 - x) ** 4 * (1.0 + 4.0 * x)

    def sample(self, key, sample_shape=()):
        return jax.random.beta(key, 2.0, 4.0, sample_shape + self.batch_shape)

    @property
    def mean(self):
        return jnp.asarray(1.0 / 3.0)

    @property
    def variance(self):
        return jnp.asarray(8.0 / 252.0)

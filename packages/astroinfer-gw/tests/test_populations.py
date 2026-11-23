import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
import astroinfer as ai
from astroinfer import parameter, parameter_dataclass

from astroinfer_gw.physics import RedshiftToLuminosityDistance
from astroinfer_gw.populations import PowerLaw, PowerLawRedshift, ProjectionFactor

jax.config.update("jax_enable_x64", True)

ALPHAS = [-2.35, -1.0, -1.0 + 1e-9, 0.0, 1.1]


def numerical_cdf(d, x, at):
    """Trapezoid integral of exp(d.log_prob) over the grid x, up to each value in ``at``."""
    p = np.exp(np.asarray(d.log_prob(jnp.asarray(x))))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    return np.interp(at, x, cum)


# ------------------------------------------------------------
# PowerLaw
# ------------------------------------------------------------

@pytest.mark.parametrize("alpha", ALPHAS)
def test_power_law_normalized_and_cdf_matches_integral(alpha):
    d = PowerLaw(alpha, 5.0, 80.0)
    x = np.geomspace(5.0, 80.0, 200_001)
    at = np.array([6.0, 10.0, 30.0, 80.0])
    np.testing.assert_allclose(numerical_cdf(d, x, at), np.asarray(d.cdf(at)), atol=1e-7)
    np.testing.assert_allclose(np.asarray(d.icdf(d.cdf(at))), at, rtol=1e-10)


@pytest.mark.parametrize("alpha", [-2.35, -1.0, 1.1])
def test_power_law_samples_follow_density(alpha):
    d = PowerLaw(alpha, 5.0, 80.0)
    x = np.asarray(d.sample(jax.random.key(0), (200_000,)))
    assert x.min() >= 5.0 and x.max() <= 80.0
    for v in (8.0, 20.0, 50.0):
        np.testing.assert_allclose(np.mean(x <= v), float(d.cdf(v)), atol=4e-3)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_power_law_gradients_match_finite_differences(alpha):
    def quantities(a, low, high):
        d = PowerLaw(a, low, high)
        return jnp.stack([d.log_prob(20.0), d.cdf(20.0), d.icdf(0.3)])

    args = (alpha, 5.0, 80.0)
    jac = jax.jacobian(quantities, argnums=(0, 1, 2))(*args)
    h = 1e-6
    for i in range(3):
        up, dn = list(args), list(args)
        up[i] += h
        dn[i] -= h
        fd = (quantities(*up) - quantities(*dn)) / (2 * h)
        assert np.all(np.isfinite(np.asarray(jac[i])))
        np.testing.assert_allclose(np.asarray(jac[i]), np.asarray(fd), rtol=1e-4, atol=1e-6)


def test_power_law_wider_support():
    d = PowerLaw(1.1, 0.3, 1.0, support=(0.0, 1.0))
    assert d.support.lower_bound == 0.0 and d.support.upper_bound == 1.0
    assert float(d.log_prob(0.2)) == -np.inf
    assert float(d.cdf(0.2)) == 0.0
    assert np.isfinite(float(d.log_prob(0.5)))
    # without support=, the support is [low, high]
    plain = PowerLaw(1.1, 0.3, 1.0)
    assert float(plain.support.lower_bound) == 0.3


# ------------------------------------------------------------
# PowerLawRedshift
# ------------------------------------------------------------

@pytest.mark.parametrize("lamb", [-1.0, 0.0, 2.7])
def test_redshift_normalized_and_cdf_matches_integral(lamb):
    d = PowerLawRedshift(lamb, z_max=1.5)
    z = np.linspace(0.0, 1.5, 300_001)
    at = np.array([0.1, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(numerical_cdf(d, z, at), np.asarray(d.cdf(at)), atol=1e-6)
    np.testing.assert_allclose(np.asarray(d.icdf(d.cdf(at))), at, atol=1e-8)
    assert float(d.log_prob(1.6)) == -np.inf and float(d.log_prob(-0.1)) == -np.inf


def test_redshift_is_comoving_volume_times_rate():
    cosmology = RedshiftToLuminosityDistance()
    d = PowerLawRedshift(2.0, z_max=1.5, cosmology=cosmology)
    z = np.array([0.2, 0.7, 1.3])
    ratio = np.exp(np.asarray(d.log_prob(z))) / (
        np.asarray(cosmology.differential_comoving_volume(z)) * (1 + z))
    np.testing.assert_allclose(ratio / ratio[0], 1.0, rtol=1e-4)


def test_redshift_samples_follow_density():
    d = PowerLawRedshift(2.7, z_max=1.5)
    z = np.asarray(d.sample(jax.random.key(1), (200_000,)))
    assert z.min() >= 0.0 and z.max() <= 1.5
    for v in (0.3, 0.8, 1.2):
        np.testing.assert_allclose(np.mean(z <= v), float(d.cdf(v)), atol=4e-3)


def test_redshift_gradient_in_lamb():
    def quantities(lamb):
        d = PowerLawRedshift(lamb, z_max=1.5)
        return jnp.stack([d.log_prob(0.5), d.cdf(0.5), d.icdf(0.5)])

    grad = jax.jacobian(quantities)(2.7)
    h = 1e-5
    fd = (quantities(2.7 + h) - quantities(2.7 - h)) / (2 * h)
    assert np.all(np.isfinite(np.asarray(grad))) and np.all(np.asarray(grad) != 0.0)
    np.testing.assert_allclose(np.asarray(grad), np.asarray(fd), rtol=1e-4)


def test_redshift_batched_lamb():
    d = PowerLawRedshift(jnp.array([0.0, 2.7]), z_max=1.5)
    assert d.batch_shape == (2,)
    one = PowerLawRedshift(2.7, z_max=1.5)
    np.testing.assert_allclose(np.asarray(d.cdf(jnp.array([0.5, 0.5])))[1], float(one.cdf(0.5)))
    np.testing.assert_allclose(np.asarray(d.log_prob(0.5))[1], float(one.log_prob(0.5)))
    assert d.sample(jax.random.key(0), (3,)).shape == (3, 2)


def test_redshift_rejects_z_max_beyond_cosmology():
    with pytest.raises(ValueError, match="z_max"):
        PowerLawRedshift(1.0, z_max=10.0)


# ------------------------------------------------------------
# ProjectionFactor
# ------------------------------------------------------------

def test_projection_factor_is_beta_2_4():
    d, beta = ProjectionFactor(), dist.Beta(2.0, 4.0)
    x = jnp.array([0.05, 0.3, 0.6, 0.95])
    np.testing.assert_allclose(np.asarray(d.log_prob(x)), np.asarray(beta.log_prob(x)), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(d.cdf(x)), np.asarray(beta.cdf(x)), rtol=1e-6)
    s = np.asarray(d.sample(jax.random.key(2), (100_000,)))
    np.testing.assert_allclose(s.mean(), float(d.mean), atol=3e-3)


# ------------------------------------------------------------
# As leaves of a structured distribution, and through as_generative
# ------------------------------------------------------------

@parameter_dataclass
class Source:
    mass_1: float
    mass_ratio: float
    redshift: float
    projection_factor: float


@parameter_dataclass
class Hyper:
    alpha: float
    beta: float
    lamb: float


M_MIN, M_MAX = 5.0, 80.0
LAMBDA = Hyper(alpha=-2.35, beta=1.1, lamb=2.7)


@pytest.fixture(scope="module")
def population():
    return Source.distribution(
        mass_1=lambda L: PowerLaw(L.alpha, M_MIN, M_MAX),
        mass_ratio=ai.conditional(
            lambda x, L: PowerLaw(L.beta, M_MIN / x["mass_1"], 1.0, support=(0.0, 1.0)),
            on="mass_1"),
        redshift=lambda L: PowerLawRedshift(L.lamb, z_max=1.5),
        projection_factor=ProjectionFactor(),
    ).given(Hyper)


def test_structured_population(population):
    theta = population.sample(LAMBDA, key=0, N=20_000)
    assert theta.batch_shape == (20_000,)
    assert float(jnp.min(theta.mass_ratio * theta.mass_1)) >= M_MIN * (1 - 1e-9)
    lp = population.logpdf(theta[:100], LAMBDA)
    expected = (PowerLaw(-2.35, M_MIN, M_MAX).log_prob(theta.mass_1[:100])
                + PowerLaw(1.1, M_MIN / theta.mass_1[:100], 1.0).log_prob(theta.mass_ratio[:100])
                + PowerLawRedshift(2.7).log_prob(theta.redshift[:100])
                + dist.Beta(2.0, 4.0).log_prob(theta.projection_factor[:100]))
    np.testing.assert_allclose(np.asarray(lp), np.asarray(expected), rtol=1e-8)


def test_as_generative(population):
    bank = population.as_generative()
    assert bank.order == ["mass_1", "mass_ratio", "redshift", "projection_factor"]
    seeds = bank.make_seeds(jax.random.key(3), N=40_000)
    draws = bank.sample(LAMBDA, seeds=seeds)
    native = population.sample(LAMBDA, key=4, N=40_000)
    for name in ["mass_1", "mass_ratio", "redshift", "projection_factor"]:
        a, b = np.asarray(draws[name]), np.asarray(native[name])
        np.testing.assert_allclose(a.mean(), b.mean(), rtol=0.02, err_msg=name)
    np.testing.assert_allclose(np.asarray(bank.logpdf(draws[:200], LAMBDA)),
                               np.asarray(population.logpdf(draws[:200], LAMBDA)), atol=1e-8)

    def mean_draws(L):
        d = bank.sample(L, seeds=seeds)
        return jnp.stack([d.mass_1.mean(), d.mass_ratio.mean(), d.redshift.mean()])

    jac = jax.jacobian(mean_draws)(LAMBDA)
    for name, row in [("alpha", 0), ("beta", 1), ("lamb", 2)]:
        g = float(getattr(jac, name)[row])
        assert np.isfinite(g) and g > 0, name
    # the gradient in lamb agrees with a finite difference on the same seeds
    h = 1e-4
    up = Hyper(alpha=-2.35, beta=1.1, lamb=2.7 + h)
    dn = Hyper(alpha=-2.35, beta=1.1, lamb=2.7 - h)
    fd = (mean_draws(up)[2] - mean_draws(dn)[2]) / (2 * h)
    np.testing.assert_allclose(float(jac.lamb[2]), float(fd), rtol=1e-3)

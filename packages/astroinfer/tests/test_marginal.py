"""Integrating values out of a model in closed form: astroinfer.marginal."""
from typing import ClassVar

import astroinfer as ai
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
import scipy.stats as st
from astroinfer import parameter, parameter_dataclass
from astroinfer.marginal import GammaPoisson, LinearGaussian, NormalNormal
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.infer.util import log_density

jax.config.update("jax_enable_x64", True)

QUICK = {"seed": 0, "num_warmup": 100, "num_samples": 100, "summary": False,
         "mcmc_kwargs": {"progress_bar": False}}


def model_log_density(model, args, params):
    with handlers.block(), handlers.seed(rng_seed=0):
        return float(log_density(model, args, {}, params)[0])


def log_integral(log_f, lo, hi, n=400_001):
    grid = np.linspace(lo, hi, n)
    values = log_f(grid)
    top = values.max()
    return top + np.log(np.trapezoid(np.exp(values - top), grid))


# ============================================================
# Types and distributions shared by the tests
# ============================================================

@parameter_dataclass(name="True value", latex="x")
class TrueValue:
    x: float = parameter(latex="x")


@parameter_dataclass(name="Measurement", latex=r"\hat x")
class Measurement:
    y: float = parameter(latex=r"\hat x")
    error: float = parameter(latex=r"\epsilon", support=constraints.positive)


class MeasuredWithError(ai.ConditionalDistribution):
    """No sample_single: an observed site takes its shape from the data."""

    data_type = Measurement
    parameter_type = TrueValue
    conjugate_likelihood: ClassVar[dict] = {"x": NormalNormal}

    def logpdf_single(self, d, value):
        return dist.Normal(value.x, d.error).log_prob(d.y)

    def conjugate_likelihood_single(self, path, d, value):
        return d.y, d.error


@parameter_dataclass(name="Population", latex=r"\Lambda")
class Population:
    mu: float = parameter(latex=r"\mu")
    sigma: float = parameter(latex=r"\sigma", support=constraints.positive)


class NormalPopulation(ai.ConditionalDistribution):
    data_type = TrueValue
    parameter_type = Population
    conjugate_prior = NormalNormal

    def sample_single(self, pop, key):
        return TrueValue(x=pop.mu + pop.sigma * jax.random.normal(key))

    def logpdf_single(self, value, pop):
        return dist.Normal(pop.mu, pop.sigma).log_prob(value.x)

    def conjugate_prior_single(self, pop):
        return pop.mu, pop.sigma


measured = MeasuredWithError()
DATA = Measurement(y=jnp.array([1.0, 2.5, -0.3, 0.8, 1.9]),
                   error=jnp.array([0.5, 1.0, 0.7, 0.3, 1.5]))


def numpyro_prior_model(data, marginalize=True):
    mu = ai.sample("mu", dist.Normal(0.0, 5.0))
    sigma = ai.sample("sigma", dist.HalfNormal(2.0))
    with numpyro.plate("points", data.batch_shape[0]):
        x = ai.sample("x", dist.Normal(mu, sigma), marginalize=marginalize)
        ai.sample("y", measured, TrueValue(x=x), obs=data)


def class_prior_model(data):
    pop = Population(mu=ai.sample("mu", dist.Normal(0.0, 5.0)),
                     sigma=ai.sample("sigma", dist.HalfNormal(2.0)))
    with numpyro.plate("points", data.batch_shape[0]):
        value = ai.sample("value", NormalPopulation(), pop, marginalize=True)
        ai.sample("y", measured, value, obs=data)


def expected_log_density(params, data):
    out = float(dist.Normal(0.0, 5.0).log_prob(params["mu"])
                + dist.HalfNormal(2.0).log_prob(params["sigma"]))
    scale = np.sqrt(params["sigma"] ** 2 + np.asarray(data.error) ** 2)
    return out + float(np.sum(st.norm(params["mu"], scale).logpdf(np.asarray(data.y))))


# ============================================================
# Families
# ============================================================

def test_normal_normal_matches_numerical_integration():
    prior = (0.4, 1.3)
    likelihood = (jnp.array([1.0, 0.2, 2.0]), jnp.array([0.5, 0.9, 1.4]))
    got = float(NormalNormal().log_marginal(prior, likelihood))
    expected = log_integral(
        lambda x: st.norm(0.4, 1.3).logpdf(x)
        + sum(st.norm(x, e).logpdf(y) for y, e in zip(*map(np.asarray, likelihood))),
        -20, 20,
    )
    assert got == pytest.approx(expected, rel=1e-8)
    # A NumPyro Normal works on either side.
    again = NormalNormal().log_marginal(dist.Normal(0.4, 1.3), dist.Normal(*likelihood))
    assert float(again) == pytest.approx(got, rel=1e-12)


def test_gamma_poisson_matches_numerical_integration():
    k, t = jnp.array([3.0, 0.0, 7.0]), jnp.array([1.2, 0.5, 2.0])
    got = float(GammaPoisson().log_marginal((2.5, 0.4), (k, t)))
    expected = log_integral(
        lambda r: st.gamma(2.5, scale=1 / 0.4).logpdf(r)
        + sum(st.poisson(r * ti).logpmf(ki) for ki, ti in zip(np.asarray(k), np.asarray(t))),
        1e-9, 60,
    )
    assert got == pytest.approx(expected, rel=1e-7)


@parameter_dataclass(name="Pair", latex="a")
class Pair:
    a: float = parameter(index=("component",))
    b: float = parameter()


def test_linear_gaussian_matches_dense_gaussian_with_mask_and_typed_design():
    rng = np.random.default_rng(0)
    K, n = 3, 4
    mean, A = rng.normal(size=K), rng.normal(size=(K, K))
    cov_prior = A @ A.T + np.eye(K)
    M = rng.normal(size=(n, K))
    B = rng.normal(size=(n, n))
    cov = B @ B.T + np.eye(n)
    y = rng.normal(size=n)
    mask = np.array([1.0, 1.0, 0.0, 1.0])
    # The design as Pair objects, one per row: a holds the first two columns, b the third.
    design = Pair(a=jnp.asarray(M[:, :2]), b=jnp.asarray(M[:, 2]))
    likelihood = {"observed": jnp.asarray(y), "design": design, "noise_cov": jnp.asarray(cov),
                  "mask": jnp.asarray(mask)}
    prior = (jnp.asarray(mean), jnp.linalg.cholesky(jnp.asarray(cov_prior)))
    got = float(LinearGaussian().log_marginal(prior, likelihood))
    keep = mask > 0
    Ms = M[keep]
    expected = st.multivariate_normal(Ms @ mean, Ms @ cov_prior @ Ms.T
                                      + cov[np.ix_(keep, keep)]).logpdf(y[keep])
    assert got == pytest.approx(expected, rel=1e-10)


def test_linear_gaussian_draws_have_the_conditional_moments():
    prior = (jnp.zeros(2), jnp.eye(2) * 2.0)
    M = jnp.array([[1.0, 0.5], [0.0, 1.0], [1.0, -1.0]])
    likelihood = {"observed": jnp.array([1.0, 2.0, 0.5]), "design": M,
                  "noise_scale_tril": jnp.eye(3) * 0.7}
    draws = jax.vmap(lambda k: LinearGaussian().draw(k, prior, likelihood))(
        jax.random.split(jax.random.key(0), 40_000))
    precision = np.eye(2) / 4.0 + np.asarray(M.T @ M) / 0.49
    cov = np.linalg.inv(precision)
    mean = cov @ (np.asarray(M.T) @ np.array([1.0, 2.0, 0.5]) / 0.49)
    np.testing.assert_allclose(np.mean(draws, 0), mean, atol=0.02)
    np.testing.assert_allclose(np.cov(np.asarray(draws).T), cov, atol=0.02)


# ============================================================
# Models
# ============================================================

@pytest.mark.parametrize("model", [numpyro_prior_model, class_prior_model])
def test_marginal_model_density(model):
    params = {"mu": 0.7, "sigma": 1.1}
    assert model_log_density(model, (DATA,), params) == pytest.approx(
        expected_log_density(params, DATA), rel=1e-10)


def test_marginalized_value_is_not_a_sample_site():
    trace = handlers.trace(handlers.seed(numpyro_prior_model, rng_seed=0)).get_trace(DATA)
    latent = [k for k, s in trace.items() if s["type"] == "sample" and not s["is_observed"]]
    assert latent == ["mu", "sigma"]


def test_shared_value_outside_the_plate():
    class AroundMean(MeasuredWithError):
        pass

    def model(data):
        s = ai.sample("s", dist.HalfNormal(2.0))
        mu = ai.sample("mu_shared", dist.Normal(0.0, s), marginalize=True)
        with numpyro.plate("points", data.batch_shape[0]):
            ai.sample("y", AroundMean(), TrueValue(x=mu), obs=data)

    got = model_log_density(model, (DATA,), {"s": 1.3})
    y, e = np.asarray(DATA.y), np.asarray(DATA.error)
    expected = float(dist.HalfNormal(2.0).log_prob(1.3)) + log_integral(
        lambda m: st.norm(0, 1.3).logpdf(m) + sum(st.norm(m, ei).logpdf(yi) for yi, ei in zip(y, e)),
        -20, 20)
    assert got == pytest.approx(expected, rel=1e-8)


def test_infer_draws_the_marginalized_values():
    posterior = ai.infer(numpyro_prior_model, DATA, **QUICK)
    assert posterior.parameter_names == ["mu", "sigma", "x"]
    assert posterior.theta.x.shape == (1, 100, 5)
    assert posterior.metadata["marginalized"] == ("x",)
    assert bool(jnp.all(jnp.isfinite(posterior.theta.x)))

    typed = ai.infer(class_prior_model, DATA, **QUICK)
    # Like any root inside a plate, the value's type gains the plate as a core axis.
    assert type(typed.theta.value).__name__ == "TrueValue"
    assert typed.theta.value.x.shape == (1, 100, 5)


def test_draws_follow_the_conditional_given_the_hyperparameters():
    # With mu and sigma fixed, x_i given y_i is Normal with known moments.
    def model(data):
        with numpyro.plate("points", data.batch_shape[0]):
            x = ai.sample("x", dist.Normal(1.0, 0.8), marginalize=True)
            ai.sample("y", measured, TrueValue(x=x), obs=data)

    from astroinfer.marginal import drawing
    from numpyro.infer import Predictive

    with drawing():
        draws = Predictive(model, num_samples=20_000, return_sites=["x"])(
            jax.random.PRNGKey(0), DATA)["x"]
    precision = 1 / 0.8**2 + 1 / np.asarray(DATA.error) ** 2
    mean = (1.0 / 0.8**2 + np.asarray(DATA.y) / np.asarray(DATA.error) ** 2) / precision
    np.testing.assert_allclose(np.mean(draws, 0), mean, atol=0.02)
    np.testing.assert_allclose(np.std(draws, 0), 1 / np.sqrt(precision), atol=0.02)


# ============================================================
# Pair rules
# ============================================================

class PlainPopulation(NormalPopulation):
    conjugate_prior = None


@ai.marginal.pair(PlainPopulation, MeasuredWithError)
class PlainPopulationMeasured:
    def _sums(self, pop, d):
        P = 1 / pop.sigma**2 + 1 / d.error**2
        h = pop.mu / pop.sigma**2 + d.y / d.error**2
        Q = pop.mu**2 / pop.sigma**2 + d.y**2 / d.error**2
        return P, h, Q

    def log_marginal(self, prior, likelihood, data, prior_conditions, likelihood_conditions):
        P, h, Q = self._sums(prior_conditions, data)
        return (-0.5 * (Q - h**2 / P) - 0.5 * jnp.log(P) - jnp.log(prior_conditions.sigma)
                - jnp.log(data.error) - 0.5 * jnp.log(2 * jnp.pi))

    def draw(self, key, prior, likelihood, data, prior_conditions, likelihood_conditions):
        P, h, _ = self._sums(prior_conditions, data)
        return TrueValue(x=h / P + jax.random.normal(key) / jnp.sqrt(P))


def test_pair_rule_gives_the_same_density():
    def model(data):
        pop = Population(mu=ai.sample("mu", dist.Normal(0.0, 5.0)),
                         sigma=ai.sample("sigma", dist.HalfNormal(2.0)))
        with numpyro.plate("points", data.batch_shape[0]):
            value = ai.sample("value", PlainPopulation(), pop, marginalize=True)
            ai.sample("y", measured, value, obs=data)

    params = {"mu": 0.7, "sigma": 1.1}
    assert model_log_density(model, (DATA,), params) == pytest.approx(
        expected_log_density(params, DATA), rel=1e-10)
    posterior = ai.infer(model, DATA, **QUICK)
    assert type(posterior.theta.value).__name__ == "TrueValue"
    assert posterior.theta.value.x.shape == (1, 100, 5)


def test_registering_a_second_rule_for_a_pair_needs_overwrite():
    with pytest.raises(ValueError, match="already registered"):
        @ai.marginal.pair(PlainPopulation, MeasuredWithError)
        class Another:
            pass


# ============================================================
# Errors
# ============================================================

def run(model):
    with handlers.seed(rng_seed=0):
        model(DATA)


def test_value_used_by_two_observed_sites():
    def model(data):
        x = ai.sample("x", dist.Normal(0.0, 1.0), marginalize=True)
        with numpyro.plate("points", 5):
            ai.sample("y", measured, TrueValue(x=x), obs=data)
            ai.sample("y2", measured, TrueValue(x=x), obs=data)

    with pytest.raises(NotImplementedError, match="already used by the site 'y'"):
        run(model)


def test_value_used_by_an_undeclared_likelihood():
    class Undeclared(MeasuredWithError):
        conjugate_likelihood = None

    def model(data):
        x = ai.sample("x", dist.Normal(0.0, 1.0), marginalize=True)
        with numpyro.plate("points", 5):
            ai.sample("y", Undeclared(), TrueValue(x=x), obs=data)

    with pytest.raises(TypeError, match="no conjugate_likelihood entry"):
        run(model)


def test_value_used_by_a_latent_site():
    def model(data):
        x = ai.sample("x", dist.Normal(0.0, 1.0), marginalize=True)
        ai.sample("z", measured, TrueValue(x=x))

    with pytest.raises(TypeError, match="must be an observed site"):
        run(model)


def test_prior_of_the_wrong_family():
    def model(data):
        x = ai.sample("x", dist.Gamma(2.0, 1.0), marginalize=True)
        with numpyro.plate("points", 5):
            ai.sample("y", measured, TrueValue(x=x), obs=data)

    with pytest.raises(TypeError, match="does not take a NumPyro Gamma"):
        run(model)


def test_class_prior_must_declare_the_same_family():
    class GammaDeclared(NormalPopulation):
        conjugate_prior = GammaPoisson

    def model(data):
        pop = Population(mu=0.0, sigma=1.0)
        with numpyro.plate("points", 5):
            value = ai.sample("value", GammaDeclared(), pop, marginalize=True)
            ai.sample("y", measured, value, obs=data)

    with pytest.raises(TypeError, match="declares conjugate_prior = GammaPoisson"):
        run(model)


def test_numpyro_site_reading_the_value_is_refused():
    def model(data):
        x = ai.sample("x", dist.Normal(0.0, 1.0), marginalize=True)
        ai.sample("z", dist.Normal(x, 1.0, validate_args=False), obs=1.0)

    with pytest.raises(TypeError, match="uses the marginalized value 'x'"):
        run(model)


def test_structured_prior_is_not_supported_yet():
    prior = TrueValue.distribution(x=dist.Normal(0.0, 1.0))

    def model(data):
        with numpyro.plate("points", 5):
            value = ai.sample("value", prior, marginalize=True)
            ai.sample("y", measured, value, obs=data)

    with pytest.raises(NotImplementedError, match="structured distribution"):
        run(model)

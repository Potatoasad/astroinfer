"""PE, population and joint population inference all return PosteriorData.

Toy with an exact answer:
    population   x ~ N(mu, sigma)
    measurement  dx ~ N(x, SIGMA)
so each observation's marginal likelihood is N(dx; mu, sqrt(sigma^2 + SIGMA^2)),
and p(mu, sigma | data) can be computed on a grid.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astroinfer import (
    Catalog,
    MeasurementProcess,
    PopulationModel,
    PosteriorData,
    Proposal,
    parameter_dataclass,
)
from astroinfer.generative import GenerativeModel, TruncatedNormal
from jax.scipy.stats import norm

SIGMA = 0.5
E = 40
TRUTH = {"mu": 1.0, "sigma": 1.0}
HYPERPRIORS = {"mu": dist.Uniform(-2.0, 4.0), "sigma": dist.Uniform(0.2, 3.0)}
RUN = {
    "num_warmup": 300,
    "num_samples": 400,
    "summary": False,
    "mcmc_kwargs": {"num_chains": 2, "chain_method": "sequential", "progress_bar": False},
}


@parameter_dataclass
class Theta:
    x: float


@parameter_dataclass
class Obs:
    dx: float


@parameter_dataclass
class Hyper:
    mu: float
    sigma: float


class Noisy(MeasurementProcess):
    parameter_type = Theta
    data_type = Obs

    def sample_single(self, theta, key):
        return Obs(dx=theta.x + SIGMA * jax.random.normal(key))

    def logpdf_single(self, d, theta):
        return norm.logpdf(d.dx, theta.x, SIGMA)

    def derived_parameters_single(self, theta):
        return {"x_squared": theta.x**2}


class Gaussian(PopulationModel):
    parameter_type = Theta

    def sample_single(self, Lambda, key):
        return Theta(x=Lambda["mu"] + Lambda["sigma"] * jax.random.normal(key))

    def logpdf_single(self, theta, Lambda):
        return norm.logpdf(theta.x, Lambda["mu"], Lambda["sigma"])

    def derived_parameters_single(self, theta, Lambda):
        return {"pull": (theta.x - Lambda["mu"]) / Lambda["sigma"]}


class TypedGaussian(Gaussian):
    """Declares a hyperparameter type, so Lambda arrives typed."""
    hyperparameter_type = Hyper

    def logpdf_single(self, theta, Lambda):
        assert isinstance(Lambda, Hyper)
        return norm.logpdf(theta.x, Lambda.mu, Lambda.sigma)


class AroundData(Proposal):
    parameter_type = Obs
    data_type = Theta

    def sample_single(self, d, key, Lambda):
        return Theta(x=d.dx + SIGMA * jax.random.normal(key))

    def logpdf_single(self, theta, d, Lambda):
        return norm.logpdf(theta.x, d.dx, SIGMA)


@pytest.fixture(scope="module")
def data():
    theta = Gaussian().sample(TRUTH, key=1, shape=(E,))
    return Noisy().sample(theta, key=2)


@pytest.fixture(scope="module")
def exact(data):
    mu = np.linspace(-1.0, 3.0, 301)
    sigma = np.linspace(0.2, 3.0, 301)
    M, S = np.meshgrid(mu, sigma, indexing="ij")
    scale = np.sqrt(S[..., None] ** 2 + SIGMA**2)
    logp = norm.logpdf(np.asarray(data.dx), M[..., None], scale).sum(-1)
    w = np.exp(logp - logp.max())
    w /= w.sum()
    return {"mu": float((w * M).sum()), "sigma": float((w * S).sum())}


@pytest.fixture(scope="module")
def pe_posterior(data):
    return Noisy().PE(data[0], {"x": dist.Uniform(-10, 10)}, seed=0, as_posterior=True, **RUN)


@pytest.fixture(scope="module")
def joint_posterior(data):
    return Noisy().joint_population_inference(data, Gaussian(), HYPERPRIORS, seed=3, **RUN)


# ============================================================
# Parameter estimation
# ============================================================

def test_pe_keeps_chains(pe_posterior):
    post = pe_posterior
    assert post.sample_dims == ("chain", "draw")
    assert post.theta.x.shape == (2, 400)
    for name in ("diverging", "accept_prob", "log_likelihood", "prior", "log_prior[x]"):
        assert post.sample_stats[name].shape == (2, 400), name
    np.testing.assert_allclose(post.extras["x_squared"], post.theta.x**2, rtol=1e-5)
    np.testing.assert_allclose(post.sample_stats["prior"], 1 / 20, rtol=1e-5)


def test_pe_dataframe_is_the_stacked_posterior(data):
    df = Noisy().PE(data[0], {"x": dist.Uniform(-10, 10)}, seed=0, **RUN)
    assert len(df) == 800
    assert {"x", "x_squared", "prior", "log_prior[x]", "chain"} <= set(df.columns)


def test_catalog_stacks_chains(pe_posterior):
    catalog = Catalog({"e0": pe_posterior}, raw=["x"])
    stored = catalog.posterior("e0")
    assert stored.sample_dims == ("draw",)
    assert stored.theta.x.shape == (800,)
    np.testing.assert_array_equal(stored.sample_stats["chain"], np.repeat([0, 1], 400))


def test_derived_parameters_single_is_vmapped():
    theta = Theta(x=jnp.arange(6.0).reshape(2, 3))
    out = Noisy().derived_parameters(theta)
    np.testing.assert_allclose(out["x_squared"], theta.x**2)


# ============================================================
# Population inference (theta_i marginalized)
# ============================================================

def test_population_inference_returns_posterior(data, exact):
    post = Noisy().population_inference(data, Gaussian(), HYPERPRIORS, N=1000, seed=0, **RUN)
    assert isinstance(post, PosteriorData)
    assert post.sample_dims == ("chain", "draw")
    assert [f for f in post.theta.to_dict()] == ["mu", "sigma"]
    assert post.sample_stats["N_eff_min"].shape == (2, 400)
    assert post.metadata["inference"] == "population"
    assert abs(float(post.theta.mu.mean()) - exact["mu"]) < 0.08
    assert abs(float(post.theta.sigma.mean()) - exact["sigma"]) < 0.08


def test_population_inference_with_hyperparameter_type(data):
    post = Noisy().population_inference(
        data, TypedGaussian(), HYPERPRIORS, N=200, seed=0,
        **{**RUN, "num_warmup": 100, "num_samples": 100},
    )
    assert isinstance(post.theta, Hyper)


# ============================================================
# Joint population inference (theta_i sampled)
# ============================================================

def test_joint_matches_exact(joint_posterior, exact):
    post = joint_posterior
    assert post.sample_dims == ("chain", "draw")
    assert post.local.batch_shape == (2, 400, E)
    assert post.local_dims == ("chain", "draw", "observation")
    assert post.metadata["inference"] == "joint population"
    assert abs(float(post.theta.mu.mean()) - exact["mu"]) < 0.08
    assert abs(float(post.theta.sigma.mean()) - exact["sigma"]) < 0.08


def test_joint_derived_quantities(joint_posterior):
    post = joint_posterior
    assert post.extras["pull"].shape == (2, 400, E)
    expected = (post.local.x - post.theta.mu[..., None]) / post.theta.sigma[..., None]
    np.testing.assert_allclose(post.extras["pull"], expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(post.extras["x_squared"], post.local.x**2, rtol=1e-5)


def test_joint_with_local_priors_and_proposal_init(data, exact):
    post = Noisy().joint_population_inference(
        data, Gaussian(), HYPERPRIORS,
        local_priors={"x": dist.Normal(0.0, 1.5)},
        init=AroundData(),
        seed=4,
        **RUN,
    )
    assert post.metadata["init"] == "proposal"
    # The local prior's density is divided back out, so the target is
    # unchanged. Without the cancellation both means move by about 0.1.
    assert abs(float(post.theta.mu.mean()) - exact["mu"]) < 0.05
    assert abs(float(post.theta.sigma.mean()) - exact["sigma"]) < 0.05


def test_joint_rejects_unknown_local_prior(data):
    with pytest.raises(KeyError, match="not fields"):
        Noisy().joint_population_inference(
            data, Gaussian(), HYPERPRIORS, local_priors={"y": dist.Normal()}, **RUN
        )


# ============================================================
# Posterior predictive
# ============================================================

def test_posterior_predictive_from_pe(pe_posterior):
    noisy = Noisy()
    rep = noisy.posterior_predictive(pe_posterior, key=0)
    assert isinstance(rep, Obs)
    assert rep.batch_shape == (2, 400)
    assert noisy.posterior_predictive(pe_posterior, draws=7, key=0).batch_shape == (7,)


def test_posterior_predictive_from_joint(joint_posterior):
    noisy = Noisy()
    assert noisy.posterior_predictive(joint_posterior, key=0).batch_shape == (2, 400, E)
    assert noisy.posterior_predictive(joint_posterior, draws=5, key=0).batch_shape == (5, E)


def test_posterior_predictive_type_check(joint_posterior):
    hyper_only = PosteriorData(theta=joint_posterior.theta, sample_dims=("chain", "draw"))
    with pytest.raises(TypeError, match="Hyperparameters"):
        Noisy().posterior_predictive(hyper_only)


# ============================================================
# Batched Lambda
# ============================================================

LAMBDAS = Hyper(mu=jnp.array([-1.0, 0.0, 2.0]), sigma=jnp.array([0.5, 1.0, 2.0]))


@pytest.mark.parametrize("population", [Gaussian(), TypedGaussian()])
def test_batched_lambda(population):
    theta = population.sample(LAMBDAS, key=0, shape=(2000,))
    assert theta.batch_shape == (3, 2000)
    np.testing.assert_allclose(theta.x.mean(axis=1), LAMBDAS.mu, atol=0.15)
    np.testing.assert_allclose(theta.x.std(axis=1), LAMBDAS.sigma, rtol=0.1)

    logp = population.logpdf(theta, LAMBDAS)
    expected = norm.logpdf(theta.x, LAMBDAS.mu[:, None], LAMBDAS.sigma[:, None])
    np.testing.assert_allclose(logp, expected, rtol=1e-5)

    with pytest.raises(ValueError, match="must start with"):
        population.logpdf(Theta(x=jnp.zeros((4, 2))), LAMBDAS)


def test_dict_lambda_is_one_point():
    theta = Gaussian().sample({"mu": 0.0, "sigma": 1.0}, key=0, shape=(5,))
    assert theta.batch_shape == (5,)


def test_batched_lambda_reaches_generative_populations():
    model = GenerativeModel(TruncatedNormal("x", "mu", "sigma", -50.0, 50.0), order=["x"])
    population = PopulationModel.from_generative(model, Theta)
    theta = population.sample(LAMBDAS, key=0, shape=(2000,))
    assert theta.batch_shape == (3, 2000)
    np.testing.assert_allclose(theta.x.mean(axis=1), LAMBDAS.mu, atol=0.15)
    logp = population.logpdf(theta, LAMBDAS)
    assert logp.shape == (3, 2000)


def test_joint_reports_an_infeasible_start(data):
    outside = Theta(x=jnp.full(E, 50.0))
    with pytest.raises(ValueError, match="No valid starting point"):
        Noisy().joint_population_inference(
            data, Gaussian(), HYPERPRIORS,
            local_priors={"x": dist.Uniform(-10.0, 10.0)}, init=outside, **RUN,
        )

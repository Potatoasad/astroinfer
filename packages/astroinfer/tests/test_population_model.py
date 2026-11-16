"""PopulationModel written from unbatched sample/logpdf.

Same toy as test_proposal.py, with an exact answer:
    population  m ~ N(mu_m, sig_m),   s | m ~ N(a m + mu_s, sig_s)
    measurement dm ~ N(m, SIG_M),     ds ~ N(s, SIG_S)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astroinfer import (
    MeasurementProcess,
    PopulationLikelihood,
    PopulationModel,
    Proposal,
    parameter_dataclass,
)
from astroinfer.generative import GenerativeModel, TruncatedNormal, conditional, product
from astroinfer.population import GenerativePopulation
from jax.scipy.stats import multivariate_normal, norm

jax.config.update("jax_enable_x64", True)

SIG_M, SIG_S = 0.3, 0.3
WIDEN = 1.5
LAMBDA = {"mu_m": 0.2, "sig_m": 1.0, "a": 0.5, "mu_s": -0.1, "sig_s": 0.5}


@parameter_dataclass
class Theta:
    m: float
    s: float


@parameter_dataclass
class Obs:
    dm: float
    ds: float


class Noise(MeasurementProcess):
    parameter_type = Theta
    data_type = Obs

    def sample_single(self, theta, key):
        k1, k2 = jax.random.split(key)
        return Obs(dm=theta.m + SIG_M * jax.random.normal(k1),
                   ds=theta.s + SIG_S * jax.random.normal(k2))

    def logpdf_single(self, d, theta):
        return (norm.logpdf(d.dm, theta.m, SIG_M)
                + norm.logpdf(d.ds, theta.s, SIG_S))


class GaussianAroundData(Proposal):
    """q(m, s | d): independent Gaussians centred on the data (Lambda-free)."""
    parameter_type = Obs
    data_type = Theta

    def sample_single(self, d, key, Lambda):
        k1, k2 = jax.random.split(key)
        return Theta(m=d.dm + WIDEN * SIG_M * jax.random.normal(k1),
                     s=d.ds + WIDEN * SIG_S * jax.random.normal(k2))

    def logpdf_single(self, theta, d, Lambda):
        return (norm.logpdf(theta.m, d.dm, WIDEN * SIG_M)
                + norm.logpdf(theta.s, d.ds, WIDEN * SIG_S))


def exact_log_z(data, L):
    mean = jnp.array([L["mu_m"], L["a"] * L["mu_m"] + L["mu_s"]])
    vm = L["sig_m"] ** 2
    cov = jnp.array([[vm + SIG_M**2, L["a"] * vm],
                     [L["a"] * vm, L["a"] ** 2 * vm + L["sig_s"] ** 2 + SIG_S**2]])
    x = jnp.stack([data.dm, data.ds], axis=-1)
    return multivariate_normal.logpdf(x, mean, cov)


# The same population as a GenerativeModel, for the adapter tests.
m_leaf = TruncatedNormal("m", "mu_m", "sig_m", -50.0, 50.0)
s_leaf = TruncatedNormal(
    "s", conditional(lambda x, L: L["a"] * x["m"] + L["mu_s"], on="m"),
    "sig_s", -50.0, 50.0,
)
pop_model = GenerativeModel(product([m_leaf, s_leaf]), order=["m", "s"])
spin_model = GenerativeModel(s_leaf, order=["s"], external=["m"])


class Gaussian(PopulationModel):
    """The toy population, written by hand."""
    parameter_type = Theta

    def sample_single(self, Lambda, key):
        k1, k2 = jax.random.split(key)
        m = Lambda["mu_m"] + Lambda["sig_m"] * jax.random.normal(k1)
        s = Lambda["a"] * m + Lambda["mu_s"] + Lambda["sig_s"] * jax.random.normal(k2)
        return Theta(m=m, s=s)

    def logpdf_single(self, theta, Lambda):
        return (norm.logpdf(theta.m, Lambda["mu_m"], Lambda["sig_m"])
                + norm.logpdf(theta.s, Lambda["a"] * theta.m + Lambda["mu_s"],
                              Lambda["sig_s"]))


class HardChoice(Gaussian):
    """Draws are not smooth in Lambda: the likelihood must be given a proposal."""
    differentiable_sample = False


class BatchedOnly(PopulationModel):
    """Overrides the batched logpdf instead of logpdf_single."""
    parameter_type = Theta
    differentiable_sample = False

    def logpdf(self, theta, Lambda):
        return Gaussian().logpdf_single(theta, Lambda)


@pytest.fixture(scope="module")
def data():
    theta = Gaussian().sample(LAMBDA, key=1, shape=6)
    return Noise().sample(theta, key=jax.random.key(2))


def make(data, model, proposal=None, N=40_000, **kw):
    return PopulationLikelihood(Noise(), data, model, N=N, seed=3, proposal=proposal, **kw)


# --------------------------------------------------------------
# Batched API
# --------------------------------------------------------------

def test_logpdf_matches_generative_model():
    theta = Gaussian().sample(LAMBDA, key=0, shape=(3, 4))
    np.testing.assert_allclose(
        Gaussian().logpdf(theta, LAMBDA),
        pop_model.log_prob(theta.to_dict(), LAMBDA), rtol=1e-10)


def test_sample_shapes_and_keys():
    model = Gaussian()
    assert model.sample(LAMBDA, key=0).batch_shape == ()
    assert model.sample(LAMBDA, key=0, shape=5).batch_shape == (5,)
    assert model.sample(LAMBDA, key=0, shape=(2, 3)).batch_shape == (2, 3)

    a, b = model.sample(LAMBDA, key=7, shape=4), model.sample(LAMBDA, key=7, shape=4)
    np.testing.assert_array_equal(a.m, b.m)

    key_before = model.key
    model.sample(LAMBDA, key=7, shape=4)
    assert model.key is key_before
    c, d = model.sample(LAMBDA, shape=4), model.sample(LAMBDA, shape=4)
    assert not np.allclose(c.m, d.m)


def test_sample_moments():
    theta = Gaussian().sample(LAMBDA, key=0, shape=200_000)
    np.testing.assert_allclose(theta.m.mean(), LAMBDA["mu_m"], atol=0.01)
    np.testing.assert_allclose(theta.m.std(), LAMBDA["sig_m"], atol=0.01)
    resid = theta.s - LAMBDA["a"] * theta.m - LAMBDA["mu_s"]
    np.testing.assert_allclose(resid.std(), LAMBDA["sig_s"], atol=0.01)


def test_logpdf_accepts_dict():
    x = {"m": jnp.array([0.0, 1.0]), "s": jnp.array([0.0, 0.5])}
    assert Gaussian().logpdf(x, LAMBDA).shape == (2,)


def test_missing_single_methods_raise():
    with pytest.raises(NotImplementedError, match="sample_single"):
        PopulationModel().sample(LAMBDA, key=0)


# --------------------------------------------------------------
# In the population likelihood
# --------------------------------------------------------------

@pytest.mark.parametrize("model, proposal", [
    (Gaussian(), None),
    (Gaussian(), GaussianAroundData()),
    (HardChoice(), GaussianAroundData()),
    (BatchedOnly(), GaussianAroundData()),
])
def test_matches_exact_evidence(data, model, proposal):
    lik = make(data, model, proposal)
    np.testing.assert_allclose(
        lik.event_logpdf(LAMBDA), exact_log_z(data, LAMBDA), atol=0.03)


@pytest.mark.parametrize("proposal", [None, GaussianAroundData()])
def test_common_random_numbers_and_gradient(data, proposal):
    lik = make(data, Gaussian(), proposal)
    f = jax.jit(lik.logpdf)
    assert f(LAMBDA) == f(LAMBDA)
    g = jax.grad(f)(LAMBDA)
    g_exact = jax.grad(lambda L: exact_log_z(data, L).sum())(LAMBDA)
    for k in LAMBDA:
        np.testing.assert_allclose(g[k], g_exact[k], rtol=0.1, atol=0.1)


def test_non_differentiable_sample_needs_proposal(data):
    with pytest.raises(ValueError, match="needs a proposal"):
        make(data, HardChoice())


def test_parameter_type_mismatch_raises(data):
    class WrongType(Gaussian):
        parameter_type = Obs

    with pytest.raises(TypeError, match="parameter_type"):
        make(data, WrongType())


def test_conditioned_on_rejected_for_population_model(data):
    with pytest.raises(TypeError, match="conditioned_on"):
        make(data, Gaussian(), conditioned_on={"m": 0.0})


# --------------------------------------------------------------
# GenerativeModel adapter
# --------------------------------------------------------------

def test_generative_model_is_wrapped(data):
    lik = make(data, pop_model, N=10)
    assert lik.pop_model is pop_model
    assert isinstance(lik.population, GenerativePopulation)
    assert lik.population.model is pop_model


def test_from_generative_reproduces_seed_bank():
    """The wrapper draws the same uniforms the likelihood used to precompute."""
    key = jax.random.key(3)
    keys = jax.random.split(key, len(pop_model.order))
    seeds = {v: jax.random.uniform(k, (2, 5), minval=1e-7, maxval=1 - 1e-7)
             for v, k in zip(pop_model.order, keys)}
    old = pop_model.sample(LAMBDA, seeds=seeds)
    new = PopulationModel.from_generative(pop_model, Theta).sample(LAMBDA, key=key, shape=(2, 5))
    for v in pop_model.order:
        np.testing.assert_array_equal(new.to_dict()[v], old[v])


def test_from_generative_checks_fields_and_externals():
    m_only = GenerativeModel(m_leaf, order=["m"])
    with pytest.raises(ValueError, match="missing parameters"):
        PopulationModel.from_generative(m_only, Theta)
    with pytest.raises(ValueError, match="external"):
        PopulationModel.from_generative(spin_model, Theta)


def test_from_generative_logpdf_refuses_conditional_only():
    wrapped = PopulationModel.from_generative(spin_model, Theta, conditioned_on={"m": 0.0})
    with pytest.raises(ValueError, match="conditional density"):
        wrapped.logpdf({"m": 0.0, "s": 0.0}, LAMBDA)


def test_display():
    html = Gaussian()._repr_html_()
    assert r"\mid\Lambda" in html and ".sample" in html
    assert "use a proposal" in HardChoice()._repr_html_()
    assert "Order" in PopulationModel.from_generative(pop_model, Theta)._repr_html_()

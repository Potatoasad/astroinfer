"""Importance-sampled population likelihood with a Proposal.

Toy model with an exact answer:
    population  m ~ N(mu_m, sig_m),   s | m ~ N(a m + mu_s, sig_s)
    measurement dm ~ N(m, SIG_M),     ds ~ N(s, SIG_S)
so Z_i(Lambda) = p(dm_i, ds_i | Lambda) is a 2-D Gaussian density.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal, norm

from astroinfer import MeasurementProcess, PopulationLikelihood, Proposal, parameter_dataclass
from astroinfer.generative import GenerativeModel, TruncatedNormal, conditional, product

jax.config.update("jax_enable_x64", True)

SIG_M, SIG_S = 0.3, 0.3
WIDEN = 1.5                       # proposal width relative to the noise
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


# Population, built from shared leaves so the sub-models stay consistent.
m_leaf = TruncatedNormal("m", "mu_m", "sig_m", -50.0, 50.0)
s_leaf = TruncatedNormal(
    "s", conditional(lambda x, L: L["a"] * x["m"] + L["mu_s"], on="m"),
    "sig_s", -50.0, 50.0,
)
pop_model = GenerativeModel(product([m_leaf, s_leaf]), order=["m", "s"])
mass_model = GenerativeModel(m_leaf, order=["m"])
spin_model = GenerativeModel(s_leaf, order=["s"], external=["m"])


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


class MassFromDataSpinFromPopulation(Proposal):
    """q(m | d) from the data, then s ~ p(s | m, Lambda) from the population."""
    parameter_type = Obs
    data_type = Theta

    def sample_single(self, d, key, Lambda):
        k1, k2 = jax.random.split(key)
        m = d.dm + WIDEN * SIG_M * jax.random.normal(k1)
        s = spin_model.sample(Lambda, key=k2, shape=(), conditioned_on={"m": m})["s"]
        return Theta(m=m, s=s)

    def log_q_mass(self, theta, d):
        return norm.logpdf(theta.m, d.dm, WIDEN * SIG_M)

    def logpdf_single(self, theta, d, Lambda):
        return (self.log_q_mass(theta, d)
                + spin_model.log_prob({"m": theta.m, "s": theta.s}, Lambda))


class SpinCancels(PopulationLikelihood):
    """p(s | m, Lambda) appears in both p and q, so it is never evaluated."""

    def log_weights(self, Lambda):
        theta = self.proposal_draws(Lambda)
        d = self.data.reshape_batch((self.E, 1))
        log_p_mass = mass_model.log_prob(theta.to_dict(), Lambda)
        return (self.log_likelihood_at(theta) + log_p_mass
                - self.proposal.log_q_mass(theta, d))


def exact_log_z(data, L):
    mean = jnp.array([L["mu_m"], L["a"] * L["mu_m"] + L["mu_s"]])
    vm = L["sig_m"] ** 2
    cov = jnp.array([[vm + SIG_M**2, L["a"] * vm],
                     [L["a"] * vm, L["a"] ** 2 * vm + L["sig_s"] ** 2 + SIG_S**2]])
    x = jnp.stack([data.dm, data.ds], axis=-1)
    return multivariate_normal.logpdf(x, mean, cov)


@pytest.fixture(scope="module")
def data():
    theta = pop_model.sample(LAMBDA, key=jax.random.key(1), shape=(6,))
    return Noise().sample(Theta(m=theta["m"], s=theta["s"]), key=jax.random.key(2))


def make(data, proposal=None, cls=PopulationLikelihood, N=40_000):
    return cls(Noise(), data, pop_model, N=N, seed=3, proposal=proposal)


@pytest.mark.parametrize("proposal, cls", [
    (None, PopulationLikelihood),
    (GaussianAroundData(), PopulationLikelihood),
    (MassFromDataSpinFromPopulation(), PopulationLikelihood),
    (MassFromDataSpinFromPopulation(), SpinCancels),
])
def test_matches_exact_evidence(data, proposal, cls):
    lik = make(data, proposal, cls)
    np.testing.assert_allclose(
        lik.event_logpdf(LAMBDA), exact_log_z(data, LAMBDA), atol=0.03)


def test_proposal_beats_population_draws(data):
    _, neff_p = make(data).event_logpdf_and_neff(LAMBDA)
    _, neff_q = make(data, GaussianAroundData()).event_logpdf_and_neff(LAMBDA)
    assert jnp.all(neff_q > 2 * neff_p)


def test_common_random_numbers_and_gradient(data):
    lik = make(data, MassFromDataSpinFromPopulation(), SpinCancels)
    f = jax.jit(lik.logpdf)
    assert f(LAMBDA) == f(LAMBDA)
    g = jax.grad(f)(LAMBDA)
    g_exact = jax.grad(lambda L: exact_log_z(data, L).sum())(LAMBDA)
    for k in LAMBDA:
        np.testing.assert_allclose(g[k], g_exact[k], rtol=0.1, atol=0.1)


def test_type_mismatch_raises(data):
    class Wrong(Proposal):
        parameter_type = Theta
        data_type = Obs

    with pytest.raises(TypeError, match="mirror"):
        make(data, Wrong())


def test_proposal_display_and_disabled_methods():
    q = GaussianAroundData()
    assert r"q_{\Lambda}" in q._repr_html_()
    assert ".PE" not in q._repr_html_()
    with pytest.raises(TypeError):
        q.PE(None, {})


def test_log_prob_out_of_support_has_finite_gradient():
    """Draws outside the population support give -inf, not NaN gradients."""
    x = {"m": jnp.array([0.0, 60.0]), "s": jnp.array([0.0, 0.0])}

    def f(L):
        lp = pop_model.log_prob(x, L)
        return jax.scipy.special.logsumexp(lp)

    assert jnp.isneginf(pop_model.log_prob(x, LAMBDA)[1])
    g = jax.grad(f)(LAMBDA)
    assert all(jnp.isfinite(v) for v in g.values())

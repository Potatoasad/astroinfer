"""Fixed-bank transport (``prior.as_generative``) and population compatibility.

The structured priors here are converted to the generative backend and
compared with the native structured density, with hand-built legacy
``GenerativeModel`` populations and with finite differences.
"""
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax.scipy.stats import norm
from numpyro.distributions import constraints

from astroinfer import (
    MeasurementProcess,
    PopulationLikelihood,
    Proposal,
    parameter,
    parameter_dataclass,
)
from astroinfer import generative as gen
from astroinfer.distributions.algebra import mixture
from astroinfer.distributions.base import ConditionalDistribution
from astroinfer.distributions.structured import conditional
from astroinfer.distributions.transport import (
    Reparameterizable,
    StructuredGenerativePopulation,
)
from astroinfer.inference.likelihoods import JointPopulationLikelihood
from astroinfer.population import DistributionPopulation, as_population_model

jax.config.update("jax_enable_x64", True)


@parameter_dataclass
class Masses:
    m1: float
    q: float


@parameter_dataclass
class Spins:
    chi1: float
    chi2: float = 0.0


@parameter_dataclass
class BBH:
    masses: Masses
    spins: Spins


@parameter_dataclass
class Hyper:
    mu: float
    sigma: float
    w: float


L0 = Hyper(mu=30.0, sigma=5.0, w=0.3)
N = (3, 400)


def bbh_prior():
    return BBH.distribution(
        masses=Masses.distribution(
            m1=lambda L: dist.TruncatedNormal(L.mu, L.sigma, low=5.0, high=100.0),
            q=conditional(lambda x, L: dist.Uniform(5.0 / x["m1"], 1.0), on="m1"),
        ),
        spins=Spins.distribution(
            chi1=conditional(lambda x, L: dist.Uniform(0.0, x["masses.q"]), on="masses.q"),
        ),
    ).given(Hyper)


def fd(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)


# ============================================================
# Fixed bank
# ============================================================

def test_population_type_and_order():
    pop = bbh_prior().as_generative()
    assert isinstance(pop, StructuredGenerativePopulation)
    assert isinstance(pop, Reparameterizable)
    assert pop.parameter_type is BBH and pop.hyperparameter_type is Hyper
    assert pop.order == ["masses.m1", "masses.q", "spins.chi1"] == bbh_prior().order
    spec = pop.base_noise_spec()
    assert spec.kind == "uniform" and spec.variables == tuple(pop.order)
    html = pop._repr_html_()
    assert "masses.m1, masses.q, spins.chi1" in html


def test_same_bank_same_draws_and_lambda_moves_them():
    pop = bbh_prior().as_generative()
    seeds = pop.make_seeds(0, N)
    a = pop.sample(L0, seeds=seeds)
    b = pop.sample(L0, seeds=seeds)
    c = pop.sample(L0, key=0, N=N)               # the bank made from the same key
    for p in ("masses.m1", "masses.q", "spins.chi1"):
        assert np.array_equal(a[p], b[p]) and np.array_equal(a[p], c[p])
    moved = pop.sample(Hyper(mu=40.0, sigma=5.0, w=0.3), seeds=seeds)
    assert not np.allclose(moved.masses.m1, a.masses.m1)
    assert np.array_equal(seeds["masses.m1"], pop.make_seeds(0, N)["masses.m1"])
    # Dict Lambda, typed output with the fixed field broadcast in.
    d = pop.sample({"mu": 30.0, "sigma": 5.0, "w": 0.3}, seeds=seeds)
    assert isinstance(d, BBH) and d.batch_shape == N
    assert d.spins.chi2.shape == N and np.all(d.spins.chi2 == 0.0)
    assert np.array_equal(d.masses.m1, a.masses.m1)
    # The conditional leaves respect the drawn values.
    assert np.all(a.masses.q >= 5.0 / a.masses.m1 - 1e-12)
    assert np.all(a.spins.chi1 <= a.masses.q + 1e-12)


def test_logpdf_matches_structured_density():
    prior = bbh_prior()
    pop = prior.as_generative()
    theta = pop.sample(L0, key=1, N=50)
    np.testing.assert_allclose(pop.logpdf(theta, L0), prior.logpdf(theta, L0), rtol=1e-10)
    # A fixed field that differs from its declared value has zero density.
    off = BBH(masses=theta.masses, spins=Spins(chi1=theta.spins.chi1, chi2=jnp.full(50, 0.5)))
    assert np.all(np.isneginf(pop.logpdf(off, L0)))


def test_batched_typed_lambda():
    pop = bbh_prior().as_generative()
    Ls = Hyper(mu=jnp.array([20.0, 40.0]), sigma=jnp.array([5.0, 5.0]), w=jnp.array([0.3, 0.3]))
    theta = pop.sample(Ls, key=0, N=100)
    assert theta.batch_shape == (2, 100)
    assert theta.masses.m1[0].mean() < theta.masses.m1[1].mean()
    assert pop.logpdf(theta, Ls).shape == (2, 100)
    # seeds= passes through the batched-Lambda wrapper: one bank for both.
    seeds = pop.make_seeds(0, 100)
    shared = pop.sample(Ls, seeds=seeds)
    np.testing.assert_allclose(shared.masses.m1[1], pop.sample(Hyper(mu=40.0, sigma=5.0, w=0.3), seeds=seeds).masses.m1)


def test_gradient_closed_form_leaf():
    pop = bbh_prior().as_generative()
    seeds = pop.make_seeds(3, (1000,))

    def stat(mu):
        theta = pop.sample(Hyper(mu=mu, sigma=5.0, w=0.3), seeds=seeds)
        return jnp.mean(theta.masses.m1 * theta.spins.chi1)

    np.testing.assert_allclose(jax.grad(stat)(30.0), fd(stat, 30.0), rtol=1e-5)


@parameter_dataclass
class Mass:
    m: float


@parameter_dataclass
class Slope:
    alpha: float


def test_gradient_bisection_leaf():
    # The power law has no listed icdf, so it is drawn by bisection on its CDF.
    prior = Mass.distribution(m=lambda L: dist.DoublyTruncatedPowerLaw(L.alpha, 5.0, 100.0)).given(Slope)
    pop = prior.as_generative()
    seeds = pop.make_seeds(0, (2000,))
    assert gen.leaves.has_differentiable_icdf(dist.DoublyTruncatedPowerLaw(-2.0, 5.0, 100.0)) is False

    def stat(a):
        return jnp.mean(pop.sample(Slope(alpha=a), seeds=seeds).m)

    np.testing.assert_allclose(jax.grad(stat)(-2.3), fd(stat, -2.3), rtol=1e-4)
    np.testing.assert_allclose(
        pop.logpdf(pop.sample(Slope(alpha=-2.3), seeds=seeds), Slope(alpha=-2.3)),
        prior.logpdf(pop.sample(Slope(alpha=-2.3), seeds=seeds), Slope(alpha=-2.3)), rtol=1e-8,
    )


def test_beta_leaf_draws_by_bisection():
    # Beta.icdf needs tensorflow_probability, so Beta uses bisection on (0, 1).
    # JAX's betainc has no gradient in the shape parameters, so only the
    # draws are checked here.
    prior = Spins.distribution(chi1=lambda L: dist.Beta(L.w * 10, 2.0)).given(Hyper)
    pop = prior.as_generative()
    seeds = pop.make_seeds(0, (5000,))
    chi = pop.sample(L0, seeds=seeds).chi1
    u = seeds["chi1"]
    np.testing.assert_allclose(dist.Beta(3.0, 2.0).cdf(chi), u, atol=1e-6)


# ============================================================
# Mixtures
# ============================================================

def spin_mixture(**kwargs):
    components = [
        Spins.distribution(chi1=dist.Uniform(0.0, 0.4)),
        Spins.distribution(chi1=dist.Beta(5.0, 2.0)),
    ]
    return mixture(components, support={"chi1": constraints.unit_interval}, **kwargs)


def test_mixture_block_weights_from_lambda():
    prior = BBH.distribution(
        masses=Masses.distribution(m1=dist.Uniform(5.0, 50.0), q=dist.Uniform(0.2, 1.0)),
        spins=spin_mixture(weights=lambda L: jnp.array([L.w, 1 - L.w]), parameter_type=Hyper),
    ).given(Hyper)
    pop = prior.as_generative()
    assert isinstance(pop.model.root.models[-1], gen.WeightedMixture)
    seeds = pop.make_seeds(0, (4000,))
    theta = pop.sample(L0, seeds=seeds)
    np.testing.assert_allclose(pop.logpdf(theta, L0), prior.logpdf(theta, L0), rtol=1e-9)
    # About w of the draws come from the low-spin component.
    assert abs(np.mean(theta.spins.chi1 < 0.4) - (0.3 + 0.7 * dist.Beta(5.0, 2.0).cdf(0.4))) < 0.03

    def stat(w):
        return jnp.mean(pop.sample(Hyper(mu=0.0, sigma=1.0, w=w), seeds=seeds).spins.chi1)

    g = jax.grad(stat)(0.3)
    np.testing.assert_allclose(g, fd(stat, 0.3), rtol=1e-4)
    # d E[chi] / dw = E_low[chi] - E_high[chi] = 0.2 - 5/7.
    np.testing.assert_allclose(g, 0.2 - 5 / 7, atol=0.05)


def test_mixture_weights_from_an_earlier_field():
    def weights(masses, hyper):
        w = jax.nn.sigmoid((masses.m1 - hyper.mu) / hyper.sigma)
        return jnp.array([w, 1 - w])

    prior = BBH.distribution(
        masses=Masses.distribution(m1=dist.Uniform(5.0, 50.0), q=dist.Uniform(0.2, 1.0)),
        spins=spin_mixture(weights=weights, parameter_type={"masses": Masses, "hyper": Hyper}),
    )
    pop = prior.as_generative(hyperparameter_type=Hyper)
    assert pop.model.root.models[-1].deps("spins.chi1")[:2] == ["masses.m1", "masses.q"]
    seeds = pop.make_seeds(0, (4000,))
    theta = pop.sample(L0, seeds=seeds)
    # Heavy systems are mostly drawn from the low-spin component.
    heavy = theta.masses.m1 > 40
    assert np.mean(theta.spins.chi1[heavy] < 0.4) > np.mean(theta.spins.chi1[~heavy] < 0.4)
    # The density is the native one with the drawn masses as the condition.
    native = prior.logpdf(theta, masses=theta.masses, hyper=L0)
    np.testing.assert_allclose(pop.logpdf(theta, L0), native, rtol=1e-9)

    def stat(mu):
        return jnp.mean(pop.sample(Hyper(mu=mu, sigma=5.0, w=0.0), seeds=seeds).spins.chi1)

    np.testing.assert_allclose(jax.grad(stat)(30.0), fd(stat, 30.0), rtol=1e-4)


def test_mixture_weights_need_the_field_first():
    prior = BBH.distribution(
        masses=Masses.distribution(m1=dist.Uniform(5.0, 50.0), q=dist.Uniform(0.2, 1.0)),
        spins=spin_mixture(weights=lambda masses, hyper: jnp.array([0.5, 0.5]),
                           parameter_type={"masses": Masses, "hyper": Hyper}),
    ).with_order(["spins", "masses.m1", "masses.q"])
    with pytest.raises(ValueError, match="does not precede"):
        prior.as_generative(hyperparameter_type=Hyper)


def test_top_level_mixture():
    prior = spin_mixture(weights=jnp.array([0.25, 0.75]))
    pop = prior.as_generative()
    assert pop.parameter_type is Spins and pop.order == ["chi1"]
    theta = pop.sample(None, key=0, N=1000)
    assert theta.batch_shape == (1000,) and np.all(theta.chi2 == 0.0)
    np.testing.assert_allclose(pop.logpdf(theta, {}), prior.logpdf(theta), rtol=1e-9)


# ============================================================
# Ordering and externals
# ============================================================

def test_explicit_order_carries_over():
    prior = Masses.distribution(
        m1=dist.Uniform(5.0, 50.0),
        q=conditional(lambda x: dist.Uniform(5.0 / x["m1"], 1.0), on="m1"),
    )
    with pytest.raises(ValueError, match="does not precede"):
        prior.with_order(["q", "m1"])
    indep = Masses.distribution(m1=dist.Uniform(5.0, 50.0), q=dist.Uniform(0.2, 1.0)).with_order(["q", "m1"])
    assert indep.as_generative().order == ["q", "m1"]


@parameter_dataclass
class Spin:
    chi: float


def test_externals_required_and_used():
    prior = Spin.distribution(chi=conditional(lambda x: dist.Uniform(0.0, x["z"]), on="z"))
    with pytest.raises(ValueError, match="chi: conditioned on unknown path 'z'"):
        prior.as_generative()
    with pytest.raises(ValueError, match="supply them in conditioned_on"):
        prior.as_generative(external=["z"])
    z = jnp.linspace(0.1, 1.0, 50)
    pop = prior.as_generative(external=["z"], conditioned_on={"z": z})
    theta = pop.sample({}, key=0, N=50)
    assert np.all(theta.chi <= z) and np.all(theta.chi >= 0)
    with pytest.raises(ValueError, match="external"):
        pop.logpdf(theta, {})


# ============================================================
# Unsupported leaves
# ============================================================

@parameter_dataclass
class Odd:
    a: float
    b: float
    c: float
    v: float = parameter(index=("k",))
    d: float = 1.0


class Opaque(ConditionalDistribution):
    data_type = float

    def sample_single(self, key):
        return jax.random.normal(key)

    def logpdf_single(self, x):
        return norm.logpdf(x)


def test_unsupported_report_lists_every_path():
    prior = Odd.distribution(
        a=dist.Gamma(2.0, 1.0),
        b=dist.Poisson(3.0),
        c=Opaque(),
        v=dist.Normal(jnp.zeros(3), 1.0).to_event(1),
        d=dist.Normal(0.0, 1.0),
    )
    with pytest.raises(ValueError) as err:
        prior.as_generative()
    msg = str(err.value)
    assert "a: no differentiable inverse CDF and unbounded support" in msg
    assert "b: Poisson is discrete" in msg
    assert "c: a Opaque block has no fixed-bank transport" in msg
    assert "v: has core axes" in msg
    assert "\n  d:" not in msg


def test_unbounded_callable_leaf_is_reported():
    prior = Spin.distribution(chi=lambda L: dist.Gamma(L["k"], 1.0)).given(Any)
    with pytest.raises(ValueError, match="chi: no differentiable inverse CDF and unbounded support"):
        prior.as_generative()
    # With a declared bounded support, bisection can run.
    @parameter_dataclass
    class Bounded:
        chi: float = parameter(support=constraints.interval(0.0, 50.0))

    pop = Bounded.distribution(chi=lambda L: dist.Gamma(L["k"], 1.0)).given(Any).as_generative()
    chi = pop.sample({"k": 3.0}, key=0, N=500).chi
    np.testing.assert_allclose(np.mean(chi), 3.0, atol=0.3)


# ============================================================
# Agreement with hand-built legacy models
# ============================================================

def test_bitwise_equal_to_legacy_uniform_model():
    prior = BBH.distribution(
        masses=Masses.distribution(m1=dist.Uniform(5.0, 50.0), q=dist.Uniform(0.2, 1.0)),
        spins=Spins.distribution(chi1=dist.Uniform(0.0, 1.0)),
    )
    pop = prior.as_generative()
    legacy = gen.GenerativeModel(
        gen.product([gen.Uniform("masses.m1", 5.0, 50.0), gen.Uniform("masses.q", 0.2, 1.0),
                     gen.Uniform("spins.chi1", 0.0, 1.0)]),
        order=["masses.m1", "masses.q", "spins.chi1"],
    )
    seeds = pop.make_seeds(0, N)
    new, old = pop.model.sample({}, seeds=seeds), legacy.sample({}, seeds=seeds)
    for v in legacy.order:
        assert np.array_equal(new[v], old[v])
    np.testing.assert_allclose(pop.model.log_prob(new, {}), legacy.log_prob(old, {}), rtol=1e-12)


def test_legacy_truncated_normal_and_mixture_within_tolerance():
    prior = BBH.distribution(
        masses=Masses.distribution(
            m1=lambda L: dist.TruncatedNormal(L["mu"], L["sigma"], low=5.0, high=100.0),
            q=conditional(lambda x, L: dist.TruncatedNormal(0.01 * x["m1"], 0.3, low=0.1, high=1.0), on="m1"),
        ),
        # The legacy Mixture bisects within its first component's bounds,
        # so the wider component comes first.
        spins=mixture([Spins.distribution(chi1=dist.Uniform(0.0, 1.0)),
                       Spins.distribution(chi1=dist.Uniform(0.0, 0.5))],
                      weights=lambda L: jnp.array([L["w"], 1 - L["w"]]),
                      parameter_type=Any, support={"chi1": constraints.unit_interval}),
    ).given(Any)
    pop = prior.as_generative()
    legacy = gen.GenerativeModel(
        gen.product([
            gen.TruncatedNormal("masses.m1", "mu", "sigma", 5.0, 100.0),
            gen.TruncatedNormal("masses.q", gen.conditional(lambda x, L: 0.01 * x["masses.m1"], on="masses.m1"),
                                0.3, 0.1, 1.0),
            gen.mixture([gen.Uniform("spins.chi1", 0.0, 1.0), gen.Uniform("spins.chi1", 0.0, 0.5)],
                        weights=["w"]),
        ]),
        order=["masses.m1", "masses.q", "spins.chi1"],
    )
    L = {"mu": 30.0, "sigma": 5.0, "w": 0.3}
    seeds = pop.make_seeds(1, N)
    new, old = pop.model.sample(L, seeds=seeds), legacy.sample(L, seeds=seeds)
    for v in legacy.order:
        np.testing.assert_allclose(new[v], old[v], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(pop.model.log_prob(new, L), legacy.log_prob(new, L), rtol=1e-8)


# ============================================================
# Phase 7: population likelihoods
# ============================================================

@parameter_dataclass
class Obs:
    dm1: float
    dchi1: float


class Noise(MeasurementProcess):
    parameter_type = BBH
    data_type = Obs

    def sample_single(self, theta, key):
        k1, k2 = jax.random.split(key)
        return Obs(dm1=theta.masses.m1 + 2.0 * jax.random.normal(k1),
                   dchi1=theta.spins.chi1 + 0.1 * jax.random.normal(k2))

    def logpdf_single(self, d, theta):
        return norm.logpdf(d.dm1, theta.masses.m1, 2.0) + norm.logpdf(d.dchi1, theta.spins.chi1, 0.1)


def uniform_bbh():
    return BBH.distribution(
        masses=Masses.distribution(m1=lambda L: dist.Uniform(5.0, L["mmax"]), q=dist.Uniform(0.2, 1.0)),
        spins=Spins.distribution(chi1=dist.Uniform(0.0, 1.0)),
    ).given(Any)


def legacy_uniform_bbh():
    class UniformMax(gen.Distribution):
        def __init__(self, var):
            self.var_names = [var]

        def cond_cdf(self, var, val, x, L):
            return jnp.clip((val - 5.0) / (L["mmax"] - 5.0), 0.0, 1.0)

        def cond_pdf(self, var, val, x, L):
            return jnp.where((val >= 5.0) & (val <= L["mmax"]), 1.0 / (L["mmax"] - 5.0), 0.0)

        def cond_ppf(self, var, u, x, L):
            return 5.0 + u * (L["mmax"] - 5.0)

        def bounds(self, var):
            return 5.0, 200.0

    return gen.GenerativeModel(
        gen.product([UniformMax("masses.m1"), gen.Uniform("masses.q", 0.2, 1.0),
                     gen.Uniform("spins.chi1", 0.0, 1.0)]),
        order=["masses.m1", "masses.q", "spins.chi1"],
    )


@pytest.fixture(scope="module")
def observations():
    theta = uniform_bbh().as_generative().sample({"mmax": 60.0}, key=5, N=8)
    return Noise().sample(theta, key=6)


def test_population_likelihood_identical_to_legacy(observations):
    new = PopulationLikelihood(Noise(), observations, uniform_bbh().as_generative(), N=500, seed=3)
    old = PopulationLikelihood(Noise(), observations, legacy_uniform_bbh(), N=500, seed=3,
                               conditioned_on={"spins.chi2": 0.0})
    L = {"mmax": 60.0}
    lp_new, ess_new = new.event_logpdf_and_neff(L)
    lp_old, ess_old = old.event_logpdf_and_neff(L)
    assert np.array_equal(lp_new, lp_old) and np.array_equal(ess_new, ess_old)
    g_new = jax.grad(lambda m: new.logpdf({"mmax": m}))(60.0)
    g_old = jax.grad(lambda m: old.logpdf({"mmax": m}))(60.0)
    np.testing.assert_allclose(g_new, g_old, rtol=1e-10)


def test_structured_distribution_without_transport_needs_a_proposal(observations):
    prior = uniform_bbh()
    population = as_population_model(prior, BBH)
    assert isinstance(population, DistributionPopulation)
    assert population.differentiable_sample is False
    assert population.sample({"mmax": 60.0}, key=0, N=4).batch_shape == (4,)
    with pytest.raises(ValueError, match="needs a proposal"):
        PopulationLikelihood(Noise(), observations, prior, N=100)

    class AroundData(Proposal):
        parameter_type = Obs
        data_type = BBH

        def sample_single(self, d, key, Lambda):
            k1, k2, k3 = jax.random.split(key, 3)
            return BBH(masses=Masses(m1=d.dm1 + 3.0 * jax.random.normal(k1),
                                     q=jax.random.uniform(k2, minval=0.2, maxval=1.0)),
                       spins=Spins(chi1=jax.random.uniform(k3), chi2=0.0))

        def logpdf_single(self, theta, d, Lambda):
            return norm.logpdf(theta.masses.m1, d.dm1, 3.0) - jnp.log(0.8)

    like = PopulationLikelihood(Noise(), observations, prior, N=4000, proposal=AroundData())
    ref = PopulationLikelihood(Noise(), observations, prior.as_generative(), N=4000, seed=1)
    np.testing.assert_allclose(like.logpdf({"mmax": 60.0}), ref.logpdf({"mmax": 60.0}), rtol=0.05)


def test_wrong_output_type_is_rejected():
    with pytest.raises(TypeError, match="output type"):
        as_population_model(Spins.distribution(chi1=dist.Uniform(0.0, 1.0)), BBH)
    with pytest.raises(TypeError, match="conditioned_on"):
        as_population_model(uniform_bbh(), BBH, conditioned_on={"z": 1.0})


def test_joint_likelihood_with_nested_type(observations):
    prior = uniform_bbh()
    pop = prior.as_generative()
    like = JointPopulationLikelihood(Noise(), observations, as_population_model(pop, BBH))
    theta = pop.sample({"mmax": 60.0}, key=0, N=8)
    x = {"mmax": 60.0, **{f"theta.{k}": v for k, v in theta.to_flat_dict().items()}}
    expected = (jnp.sum(Noise().logpdf(observations, theta))
                + jnp.sum(prior.logpdf(theta, {"mmax": 60.0})))
    np.testing.assert_allclose(like.logpdf(x), expected, rtol=1e-10)


@parameter_dataclass
class Loc:
    x: float


@parameter_dataclass
class Src:
    loc: Loc
    y: float


@parameter_dataclass
class SrcObs:
    dx: float
    dy: float


class SrcNoise(MeasurementProcess):
    parameter_type = Src
    data_type = SrcObs

    def sample_single(self, theta, key):
        k1, k2 = jax.random.split(key)
        return SrcObs(dx=theta.loc.x + 0.3 * jax.random.normal(k1),
                      dy=theta.y + 0.3 * jax.random.normal(k2))

    def logpdf_single(self, d, theta):
        return norm.logpdf(d.dx, theta.loc.x, 0.3) + norm.logpdf(d.dy, theta.y, 0.3)


def test_joint_population_inference_with_structured_population():
    prior = Src.distribution(
        loc=Loc.distribution(x=lambda L: dist.Normal(L["mu"], 1.0)),
        y=conditional(lambda x, L: dist.Normal(x["loc.x"], 0.5), on="loc.x"),
    ).given(Any)
    pop = prior.as_generative()
    data = SrcNoise().sample(pop.sample({"mu": 1.0}, key=0, N=20), key=1)
    post = SrcNoise().joint_population_inference(
        data, pop, {"mu": dist.Uniform(-5.0, 5.0)}, seed=0,
        num_warmup=150, num_samples=150, summary=False,
        mcmc_kwargs={"num_chains": 1, "progress_bar": False},
    )
    assert post.local.batch_shape[-1] == 20 and isinstance(post.local, Src)
    assert abs(float(jnp.mean(post.theta.mu)) - float(jnp.mean(data.dx))) < 0.8

"""StructuredDistribution: fields, validation, ordering, densities and display."""
import re

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astroinfer import ConditionalDistribution, parameter, parameter_dataclass
from astroinfer.distributions.structured import (
    Block,
    CallableLeaf,
    ConditionalLeaf,
    Fixed,
    Nested,
    NumPyroLeaf,
    StructuredDistribution,
    conditional,
    format_distribution,
)
from astroinfer.parameters import Parameters
from jax.scipy.stats import norm

jax.config.update("jax_enable_x64", True)


@parameter_dataclass(latex="m")
class Masses:
    m1: float = parameter(latex="m_1")
    q: float = parameter(latex="q")


@parameter_dataclass(latex="s")
class Spins:
    chi1: float = parameter(latex=r"\chi_1")
    chi2: float = parameter(latex=r"\chi_2")


@parameter_dataclass(latex=r"\theta")
class BBH:
    masses: Masses
    spins: Spins


@parameter_dataclass(latex=r"\Lambda")
class Hyper:
    mu: float
    sigma: float


@parameter_dataclass
class Strain:
    amp: float = parameter(index=("ifo",), support=dist.constraints.positive)
    phase: float


@parameter_dataclass
class WithDefault:
    a: float
    b: float = 2.0
    c: float = parameter(default=3.0)


KEY = jax.random.key(0)


def mass_prior():
    return Masses.distribution(m1=dist.Uniform(5.0, 100.0), q=dist.Uniform(0.1, 1.0))


def bbh_prior(chi2=0.0):
    return BBH.distribution(
        masses=mass_prior(),
        spins=Spins.distribution(chi1=dist.Beta(2.0, 5.0), chi2=chi2),
    )


def bbh(m1=30.0, q=0.5, chi1=0.3, chi2=0.0):
    return BBH(masses=Masses(m1=m1, q=q), spins=Spins(chi1=chi1, chi2=chi2))


# ============================================================
# Fields, sampling and densities
# ============================================================

def test_factory_returns_structured_distribution():
    prior = bbh_prior()
    assert isinstance(prior, StructuredDistribution)
    assert prior.data_type is BBH
    assert "distribution" not in BBH.__dict__
    assert isinstance(prior.fields["masses"], Nested)
    spins = prior.fields["spins"].distribution
    assert isinstance(spins.fields["chi1"], NumPyroLeaf)
    assert isinstance(spins.fields["chi2"], Fixed)


def test_scalar_leaves_sample_and_density():
    prior = mass_prior()
    m = prior.sample(KEY)
    assert isinstance(m, Masses) and m.batch_shape == ()
    assert 5 <= m.m1 <= 100 and 0.1 <= m.q <= 1
    expected = dist.Uniform(5.0, 100.0).log_prob(m.m1) + dist.Uniform(0.1, 1.0).log_prob(m.q)
    np.testing.assert_allclose(prior.logpdf(m), expected)


def test_deterministic_keys():
    prior = bbh_prior()
    a, b = prior.sample(KEY), prior.sample(KEY)
    assert a.masses.m1 == b.masses.m1 and a.spins.chi1 == b.spins.chi1
    c = prior.sample(jax.random.key(1))
    assert c.masses.m1 != a.masses.m1
    # The internal key advances between calls.
    assert prior.sample().masses.m1 != prior.sample().masses.m1


def test_nested_fixed_and_random_fields():
    prior = bbh_prior(chi2=0.25)
    theta = prior.sample(KEY)
    assert isinstance(theta, BBH) and isinstance(theta.spins, Spins)
    assert theta.spins.chi2 == 0.25
    assert prior.random_paths() == ["masses.m1", "masses.q", "spins.chi1"]
    assert prior.fixed_paths() == ["spins.chi2"]
    x = bbh(chi2=0.25)
    expected = (
        dist.Uniform(5.0, 100.0).log_prob(30.0)
        + dist.Uniform(0.1, 1.0).log_prob(0.5)
        + dist.Beta(2.0, 5.0).log_prob(0.3)
    )
    np.testing.assert_allclose(prior.logpdf(x), expected)


def test_fixed_nested_object():
    spins = Spins(chi1=0.1, chi2=0.2)
    prior = BBH.distribution(masses=mass_prior(), spins=spins)
    theta = prior.sample(KEY, N=3)
    np.testing.assert_allclose(theta.spins.chi1, [0.1] * 3)
    assert prior.fixed_paths() == ["spins.chi1", "spins.chi2"]
    assert np.isfinite(prior.logpdf(bbh(chi1=0.1, chi2=0.2)))
    assert prior.logpdf(bbh(chi1=0.15, chi2=0.2)) == -np.inf


def test_vector_leaf_with_to_event():
    prior = Strain.distribution(
        amp=dist.LogNormal(jnp.zeros(2), 1.0).to_event(1), phase=dist.Uniform(0.0, 6.0)
    )
    s = prior.sample(KEY, N=5)
    assert s.amp.shape == (5, 2) and s.phase.shape == (5,)
    lp = prior.logpdf(s)
    assert lp.shape == (5,)
    expected = (
        dist.LogNormal(jnp.zeros(2), 1.0).to_event(1).log_prob(s.amp)
        + dist.Uniform(0.0, 6.0).log_prob(s.phase)
    )
    np.testing.assert_allclose(lp, expected)


def test_batch_shapes():
    prior = bbh_prior()
    assert prior.sample(KEY, N=7).batch_shape == (7,)
    theta = prior.sample(KEY, N=(2, 3))
    assert theta.batch_shape == (2, 3)
    assert theta.spins.chi2.shape == (2, 3)
    assert prior.logpdf(theta).shape == (2, 3)


def test_fixed_mismatch_is_minus_inf_with_finite_gradient():
    prior = bbh_prior()
    assert np.isfinite(prior.logpdf(bbh(chi2=0.0)))
    assert prior.logpdf(bbh(chi2=0.1)) == -np.inf
    g = jax.grad(lambda th: prior.logpdf(th))(bbh(chi2=0.1))
    assert all(np.isfinite(v) for v in jax.tree_util.tree_leaves(g))
    loose = BBH.distribution(
        masses=mass_prior(),
        spins=StructuredDistribution(
            Spins, {"chi1": dist.Beta(2.0, 5.0), "chi2": 0.0}, check_fixed=False
        ),
    )
    assert np.isfinite(loose.logpdf(bbh(chi2=0.1)))


def test_jit_and_grad_through_nested_object():
    prior = BBH.distribution(
        masses=Masses.distribution(m1=dist.Normal(30.0, 5.0), q=dist.Uniform(0.1, 1.0)),
        spins=Spins.distribution(chi1=dist.Normal(0.0, 0.2), chi2=0.0),
    )
    theta = bbh(m1=32.0, chi1=0.1)
    lp = jax.jit(prior.logpdf)(theta)
    np.testing.assert_allclose(lp, prior.logpdf(theta))
    g = jax.jit(jax.grad(prior.logpdf))(theta)
    assert isinstance(g, BBH)
    np.testing.assert_allclose(g.masses.m1, -(32.0 - 30.0) / 25.0)
    np.testing.assert_allclose(g.spins.chi1, -0.1 / 0.04)


def test_field_defaults_are_fixed():
    prior = WithDefault.distribution(a=dist.Normal(0.0, 1.0))
    s = prior.sample(KEY)
    assert s.b == 2.0 and s.c == 3.0
    assert isinstance(prior.fields["b"], Fixed)
    assert prior.fixed_paths() == ["b", "c"]
    prior = WithDefault.distribution(a=dist.Normal(0.0, 1.0), b=dist.Normal(0.0, 1.0))
    assert prior.random_paths() == ["a", "b"]


def test_fold_in_is_stable_when_a_sibling_becomes_fixed():
    random = BBH.distribution(
        masses=mass_prior(),
        spins=Spins.distribution(chi1=dist.Beta(2.0, 5.0), chi2=dist.Beta(2.0, 5.0)),
    )
    fixed = bbh_prior(chi2=0.0)
    a, b = random.sample(KEY, N=4), fixed.sample(KEY, N=4)
    np.testing.assert_array_equal(a.masses.m1, b.masses.m1)
    np.testing.assert_array_equal(a.masses.q, b.masses.q)
    np.testing.assert_array_equal(a.spins.chi1, b.spins.chi1)
    assert random.unit_index("spins.chi1") == 2


# ============================================================
# Validation
# ============================================================

def test_unknown_field():
    with pytest.raises(TypeError, match=r"BBH\.spinz"):
        BBH.distribution(masses=mass_prior(), spins=Spins(chi1=0.1, chi2=0.0), spinz=1.0)


def test_missing_field():
    with pytest.raises(TypeError, match=r"Masses\.q: missing field"):
        Masses.distribution(m1=dist.Uniform(5.0, 100.0))
    with pytest.raises(TypeError, match=r"BBH\.spins: missing field"):
        BBH.distribution(masses=mass_prior())


def test_wrong_nested_structured_type():
    with pytest.raises(TypeError, match=r"BBH\.spins: expected Spins.*Masses"):
        BBH.distribution(masses=mass_prior(), spins=mass_prior())


def test_wrong_fixed_nested_type():
    with pytest.raises(TypeError, match=r"BBH\.spins: expected Spins"):
        BBH.distribution(masses=mass_prior(), spins=Masses(m1=1.0, q=0.5))


def test_numpyro_on_nested_field():
    with pytest.raises(TypeError, match=r"BBH\.masses: a NumPyro distribution"):
        BBH.distribution(masses=dist.Normal(0.0, 1.0), spins=Spins(chi1=0.1, chi2=0.0))


def test_structured_on_array_field():
    with pytest.raises(TypeError, match=r"Masses\.m1: expected an array field"):
        Masses.distribution(m1=bbh_prior(), q=0.5)


def test_block_with_wrong_output_type():
    with pytest.raises(TypeError, match=r"BBH\.spins: expected a distribution over Spins"):
        BBH.distribution(masses=mass_prior(), spins=MassBlock())
    with pytest.raises(TypeError, match=r"Masses\.m1: an array field"):
        Masses.distribution(m1=MassBlock(), q=0.5)


def test_leaf_shape_errors():
    with pytest.raises(ValueError, match=r"Strain\.amp.*\.to_event\(1\)"):
        Strain.distribution(amp=dist.Normal(jnp.zeros(2), 1.0), phase=0.0)
    with pytest.raises(ValueError, match=r"Masses\.m1.*batch_shape \(3,\)"):
        Masses.distribution(m1=dist.Normal(jnp.zeros(3), 1.0), q=0.5)
    with pytest.raises(ValueError, match=r"Strain\.amp.*event_shape \(\)"):
        Strain.distribution(amp=dist.Normal(0.0, 1.0), phase=0.0)


def test_fixed_value_shape_error():
    with pytest.raises(ValueError, match=r"Strain\.amp: a fixed value"):
        Strain.distribution(amp=1.0, phase=0.0)
    assert Strain.distribution(amp=jnp.ones(2), phase=0.0).sample(KEY).amp.shape == (2,)


def test_callable_leaf_shape_checked_when_evaluated():
    prior = Masses.distribution(m1=lambda: dist.Normal(jnp.zeros(3), 1.0), q=0.5)
    with pytest.raises(ValueError, match=r"Masses\.m1"):
        prior.sample(KEY)
    prior = Masses.distribution(m1=lambda: 3.0, q=0.5)
    with pytest.raises(TypeError, match=r"Masses\.m1: expected a NumPyro"):
        prior.sample(KEY)


# ============================================================
# Callable leaves and conditions
# ============================================================

def hyper_prior():
    return BBH.distribution(
        masses=Masses.distribution(
            m1=lambda L: dist.Normal(L.mu, L.sigma), q=dist.Uniform(0.1, 1.0)
        ),
        spins=Spins.distribution(chi1=dist.Beta(2.0, 5.0), chi2=0.0),
    ).given(Hyper)


def test_callable_leaf_with_given():
    prior = hyper_prior()
    assert isinstance(prior.fields["masses"].distribution.fields["m1"], CallableLeaf)
    assert prior.fields["masses"].distribution.parameter_type is Hyper
    h = Hyper(mu=30.0, sigma=2.0)
    theta = prior.sample(h, KEY)
    expected = (
        norm.logpdf(theta.masses.m1, 30.0, 2.0)
        + dist.Uniform(0.1, 1.0).log_prob(theta.masses.q)
        + dist.Beta(2.0, 5.0).log_prob(theta.spins.chi1)
    )
    np.testing.assert_allclose(prior.logpdf(theta, h), expected, rtol=1e-6)
    # A dict is coerced to the declared condition type.
    np.testing.assert_allclose(prior.logpdf(theta, {"mu": 30.0, "sigma": 2.0}), expected)


def test_callable_leaf_with_batched_condition():
    prior = hyper_prior()
    h = Hyper(mu=jnp.array([10.0, 50.0, 90.0]), sigma=jnp.ones(3))
    theta = prior.sample(h, KEY, N=200)
    assert theta.batch_shape == (3, 200)
    np.testing.assert_allclose(theta.masses.m1.mean(axis=1), [10.0, 50.0, 90.0], atol=0.3)
    lp = prior.logpdf(theta, h)
    assert lp.shape == (3, 200)
    expected = (
        norm.logpdf(theta.masses.m1, h.mu[:, None], 1.0)
        + dist.Uniform(0.1, 1.0).log_prob(theta.masses.q)
        + dist.Beta(2.0, 5.0).log_prob(theta.spins.chi1)
    )
    np.testing.assert_allclose(lp, expected, rtol=1e-6)


def test_mapping_condition_schema():
    prior = Masses.distribution(
        m1=lambda hyper: dist.Normal(hyper.mu, hyper.sigma), q=0.5
    ).given({"hyper": Hyper})
    h = Hyper(mu=20.0, sigma=1.0)
    m = prior.sample(KEY, hyper=h)
    np.testing.assert_allclose(prior.logpdf(m, hyper=h), norm.logpdf(m.m1, 20.0, 1.0))


def test_condition_type_adopted_from_child():
    masses = Masses.distribution(
        m1=lambda L: dist.Normal(L.mu, L.sigma), q=dist.Uniform(0.1, 1.0)
    ).given(Hyper)
    prior = BBH.distribution(masses=masses, spins=Spins(chi1=0.1, chi2=0.0))
    assert prior._condition_type() is Hyper
    assert prior.sample(Hyper(mu=30.0, sigma=1.0), KEY).masses.m1.shape == ()


def test_given_conflicts_with_child_condition_type():
    masses = Masses.distribution(m1=lambda L: dist.Normal(L.mu, L.sigma), q=0.5).given(Hyper)
    with pytest.raises(TypeError, match=r"BBH\.masses\.m1"):
        BBH.distribution(masses=masses, spins=Spins(chi1=0.1, chi2=0.0)).given(Spins)


# ============================================================
# Conditional fields and ordering
# ============================================================

def test_conditional_within_block():
    prior = Masses.distribution(
        m1=dist.Uniform(5.0, 100.0),
        q=conditional(lambda x: dist.Uniform(5.0 / x["m1"], 1.0), on="m1"),
    )
    assert isinstance(prior.fields["q"], ConditionalLeaf)
    m = prior.sample(KEY, N=500)
    assert bool(jnp.all(m.q >= 5.0 / m.m1))
    expected = dist.Uniform(5.0, 100.0).log_prob(m.m1) + dist.Uniform(5.0 / m.m1, 1.0).log_prob(m.q)
    np.testing.assert_allclose(prior.logpdf(m), expected)


def test_conditional_inside_nested_block_keeps_relative_paths():
    masses = Masses.distribution(
        m1=dist.Uniform(5.0, 100.0),
        q=conditional(lambda x: dist.Uniform(5.0 / x["m1"], 1.0), on="m1"),
    )
    prior = BBH.distribution(masses=masses, spins=Spins(chi1=0.1, chi2=0.0))
    assert prior.order == ["masses.m1", "masses.q"]
    theta = bbh(m1=20.0, q=0.6, chi1=0.1)
    expected = dist.Uniform(5.0, 100.0).log_prob(20.0) + dist.Uniform(0.25, 1.0).log_prob(0.6)
    np.testing.assert_allclose(prior.logpdf(theta), expected)


def cross_prior():
    return BBH.distribution(
        masses=mass_prior(),
        spins=Spins.distribution(
            chi1=conditional(
                lambda x, L: dist.Normal(L.mu * x["masses.m1"], L.sigma), on="masses.m1"
            ),
            chi2=conditional(lambda x, L: dist.Normal(x["chi1"], 1.0), on="chi1"),
        ),
    ).given(Hyper)


def test_conditional_across_nested_blocks():
    prior = cross_prior()
    assert prior.order == ["masses.m1", "masses.q", "spins.chi1", "spins.chi2"]
    h = Hyper(mu=0.01, sigma=0.1)
    theta = prior.sample(h, KEY, N=1000)
    np.testing.assert_allclose(
        jnp.mean(theta.spins.chi1 - 0.01 * theta.masses.m1), 0.0, atol=0.02
    )
    x = bbh(m1=40.0, q=0.5, chi1=0.3, chi2=-0.2)
    expected = (
        dist.Uniform(5.0, 100.0).log_prob(40.0)
        + dist.Uniform(0.1, 1.0).log_prob(0.5)
        + norm.logpdf(0.3, 0.4, 0.1)
        + norm.logpdf(-0.2, 0.3, 1.0)
    )
    np.testing.assert_allclose(prior.logpdf(x, h), expected)


def test_dependency_on_a_fixed_field():
    prior = Spins.distribution(
        chi1=0.5, chi2=conditional(lambda x: dist.Normal(x["chi1"], 0.1), on="chi1")
    )
    assert prior.order == ["chi2"]
    np.testing.assert_allclose(prior.logpdf(Spins(chi1=0.5, chi2=0.4)), norm.logpdf(0.4, 0.5, 0.1))


def test_default_order_moves_dependencies_first():
    @parameter_dataclass
    class Reversed:
        spins: Spins
        masses: Masses

    prior = Reversed.distribution(
        spins=Spins.distribution(
            chi1=conditional(lambda x: dist.Normal(x["masses.m1"] / 100, 0.1), on="masses.m1"),
            chi2=0.0,
        ),
        masses=mass_prior(),
    )
    assert prior.order == ["masses.m1", "spins.chi1", "masses.q"]
    assert prior.sample(KEY).spins.chi1.shape == ()


def test_unknown_dependency_raises_at_construction():
    spins = Spins.distribution(
        chi1=conditional(lambda x: dist.Normal(x["masses.m9"], 1.0), on="masses.m9"),
        chi2=0.0,
    )
    with pytest.raises(ValueError, match=r"BBH\.spins\.chi1: conditioned on unknown path"):
        BBH.distribution(masses=mass_prior(), spins=spins)


def test_unresolved_dependency_raises_when_evaluated():
    spins = Spins.distribution(
        chi1=conditional(lambda x: dist.Normal(x["masses.m1"], 1.0), on="masses.m1"),
        chi2=0.0,
    )
    with pytest.raises(ValueError, match=r"Spins\.chi1: conditioned on unknown path 'masses\.m1'"):
        spins.sample(KEY)
    # A name that is not a field here might be one of an enclosing type.
    masses = Masses.distribution(
        m1=dist.Uniform(5.0, 100.0),
        q=conditional(lambda x: dist.Uniform(0.1, 1.0), on="m2"),
    )
    with pytest.raises(ValueError, match=r"Masses\.q: conditioned on unknown path 'm2'"):
        masses.logpdf(Masses(m1=10.0, q=0.5))


def test_order_errors_and_with_order():
    prior = cross_prior()
    with pytest.raises(ValueError, match=r"BBH\.spins\.chi1 is conditioned on 'masses\.m1'"):
        prior.with_order(["spins.chi1", "masses.m1", "masses.q", "spins.chi2"])
    with pytest.raises(ValueError, match="not random fields"):
        prior.with_order(["masses.m1", "masses.q", "spins.chi1", "spins.chi3"])
    with pytest.raises(ValueError, match="missing the random fields"):
        prior.with_order(["masses.m1", "spins.chi1", "spins.chi2"])
    with pytest.raises(ValueError, match="more than once"):
        prior.with_order(["masses.m1", "masses.m1", "masses.q", "spins.chi1", "spins.chi2"])

    reordered = prior.with_order(["masses.m1", "spins.chi1", "spins.chi2", "masses.q"])
    assert reordered.order == ["masses.m1", "spins.chi1", "spins.chi2", "masses.q"]
    assert prior.order == ["masses.m1", "masses.q", "spins.chi1", "spins.chi2"]
    h = Hyper(mu=0.01, sigma=0.1)
    # Fold-in keys follow the field, not the position in the order.
    a, b = prior.sample(h, KEY), reordered.sample(h, KEY)
    np.testing.assert_allclose(a.spins.chi2, b.spins.chi2)
    x = bbh(m1=40.0, q=0.5, chi1=0.3, chi2=-0.2)
    np.testing.assert_allclose(reordered.logpdf(x, h), prior.logpdf(x, h))
    # The constructor takes the order too.
    built = StructuredDistribution(
        Masses, {"m1": dist.Uniform(5.0, 100.0), "q": dist.Uniform(0.1, 1.0)}, order=["q", "m1"]
    )
    assert built.order == ["q", "m1"]


def test_dependency_cycle():
    with pytest.raises(ValueError, match="cycle"):
        Spins.distribution(
            chi1=conditional(lambda x: dist.Normal(x["chi2"], 1.0), on="chi2"),
            chi2=conditional(lambda x: dist.Normal(x["chi1"], 1.0), on="chi1"),
        )


# ============================================================
# Custom ConditionalDistribution blocks
# ============================================================

class MassBlock(ConditionalDistribution):
    """Correlated masses: m1 ~ N(mu, 1), q = 0.5 + 0.1 * N(0, 1)."""
    data_type = Masses

    def sample_single(self, key):
        k1, k2 = jax.random.split(key)
        return Masses(m1=30.0 + jax.random.normal(k1), q=0.5 + 0.1 * jax.random.normal(k2))

    def logpdf_single(self, m):
        return norm.logpdf(m.m1, 30.0, 1.0) + norm.logpdf(m.q, 0.5, 0.1)


class SpinBlock(ConditionalDistribution):
    """chi1 = chi2 ~ N(mu, sigma), conditioned on Hyper."""
    data_type = Spins
    parameter_type = Hyper

    def sample_single(self, h, key):
        chi = h.mu + h.sigma * jax.random.normal(key)
        return Spins(chi1=chi, chi2=chi)

    def logpdf_single(self, s, h):
        return norm.logpdf(s.chi1, h.mu, h.sigma)


class ScalarBlock(ConditionalDistribution):
    """A distribution over one untyped array."""

    def sample_single(self, key):
        return 2.0 + jax.random.normal(key)

    def logpdf_single(self, v):
        return norm.logpdf(v, 2.0, 1.0)


def test_custom_block_on_nested_field():
    prior = BBH.distribution(masses=MassBlock(), spins=SpinBlock())
    assert isinstance(prior.fields["masses"], Block)
    assert prior.order == ["masses", "spins"]
    assert prior._condition_type() is Hyper
    h = Hyper(mu=0.2, sigma=0.05)
    theta = prior.sample(h, KEY, N=4)
    np.testing.assert_array_equal(theta.spins.chi1, theta.spins.chi2)
    expected = MassBlock().logpdf(theta.masses) + SpinBlock().logpdf(theta.spins, h)
    np.testing.assert_allclose(prior.logpdf(theta, h), expected)
    assert prior.block_conditions("spins", (h,)) == (h,)
    assert prior.block_conditions("masses", (h,)) == ()
    supports = prior.leaf_supports()
    assert set(supports) == {"masses.m1", "masses.q", "spins.chi1", "spins.chi2"}


def test_conditional_leaf_depends_on_block():
    prior = BBH.distribution(
        masses=MassBlock(),
        spins=Spins.distribution(
            chi1=conditional(lambda x: dist.Normal(x["masses.q"], 0.1), on="masses.q"),
            chi2=0.0,
        ),
    )
    assert prior.order == ["masses", "spins.chi1"]
    x = bbh(m1=31.0, q=0.4, chi1=0.5)
    expected = MassBlock().logpdf(x.masses) + norm.logpdf(0.5, 0.4, 0.1)
    np.testing.assert_allclose(prior.logpdf(x), expected)


def test_block_on_array_field_and_bound_block():
    prior = Masses.distribution(m1=ScalarBlock(), q=0.5)
    m = prior.sample(KEY, N=3)
    assert m.m1.shape == (3,)
    np.testing.assert_allclose(prior.logpdf(m), norm.logpdf(m.m1, 2.0, 1.0))
    assert prior.leaf_supports() == {"m1": dist.constraints.real}

    h = Hyper(mu=0.3, sigma=0.1)
    bound = BBH.distribution(masses=mass_prior(), spins=SpinBlock().condition(h))
    assert bound._condition_schema().kind == "none"
    theta = bound.sample(KEY)
    expected = (
        mass_prior().logpdf(theta.masses) + norm.logpdf(theta.spins.chi1, 0.3, 0.1)
    )
    np.testing.assert_allclose(bound.logpdf(theta), expected)


# ============================================================
# Adapter-facing API
# ============================================================

def test_adapter_api():
    prior = cross_prior()
    h = Hyper(mu=0.01, sigma=0.1)
    x = prior.fixed_values()
    assert x == {}
    x["masses.m1"] = jnp.asarray(40.0)
    d = prior.leaf_distribution("spins.chi1", x, (h,))
    np.testing.assert_allclose(d.loc, 0.4)
    assert prior.leaf_distribution("masses.m1", {}, (h,)) is prior.field_at("masses.m1").distribution
    x.update({"masses.q": 0.5, "spins.chi1": 0.3, "spins.chi2": -0.2})
    theta = prior.assemble(x)
    assert isinstance(theta, BBH) and theta.spins.chi2 == -0.2

    supports = bbh_prior().leaf_supports()
    assert set(supports) == {"masses.m1", "masses.q", "spins.chi1"}
    assert supports["spins.chi1"] == dist.Beta(2.0, 5.0).support
    vector = Strain.distribution(amp=lambda: dist.LogNormal(jnp.zeros(2), 1.0).to_event(1), phase=0.0)
    assert vector.leaf_supports()["amp"].event_dim == 1


# ============================================================
# Display
# ============================================================

def _math(html):
    return re.findall(r"\\\((.*?)\\\)", html)


def test_display():
    prior = cross_prior()
    html = prior._repr_html_()
    math = _math(html)
    assert r"\pi\!\left(\theta\mid \Lambda\right)" in math
    assert r"m_1 \sim \mathrm{Uniform}(5, 100)" in math
    assert r"\chi_1 \sim p\!\left(\chi_1\mid m_1, \Lambda\right)" in math
    assert "<strong>masses</strong>" in html and "<strong>spins</strong>" in html
    assert "masses.m1, masses.q, spins.chi1, spins.chi2" in html

    fixed = bbh_prior(chi2=0.0)._repr_html_()
    assert r"\chi_2 = 0" in _math(fixed)
    assert r"\chi_1 \sim \mathrm{Beta}(2, 5)" in _math(fixed)
    assert r"m_1 \sim p\!\left(m_1\mid \Lambda\right)" in _math(hyper_prior()._repr_html_())
    blocks = BBH.distribution(masses=MassBlock(), spins=SpinBlock())._repr_html_()
    assert "MassBlock" in blocks
    assert bbh_prior()._repr_markdown_()


def test_format_distribution():
    assert format_distribution(dist.Uniform(5, 100)) == r"\mathrm{Uniform}(5, 100)"
    assert format_distribution(dist.Normal(0.0, 1.5)) == r"\mathrm{Normal}(0, 1.5)"
    assert format_distribution(dist.TruncatedNormal(0.0, 1.0, low=0.0, high=2.0)) == (
        r"\mathrm{TruncatedNormal}(0, 1, 0, 2)"
    )
    assert format_distribution(dist.TruncatedNormal(0.0, 1.0, low=0.0)) == (
        r"\mathrm{TruncatedNormal}(0, 1, 0, \infty)"
    )
    assert format_distribution(dist.Beta(2.0, 5.0)) == r"\mathrm{Beta}(2, 5)"
    assert format_distribution(dist.LogNormal(0.0, 1.0)) == r"\mathrm{LogNormal}(0, 1)"
    assert format_distribution(dist.Exponential(2.0)) == r"\mathrm{Exponential}(2)"
    assert format_distribution(dist.Gamma(2.0, 3.0)) == r"\mathrm{Gamma}(2, 3)"
    assert format_distribution(dist.HalfNormal(1.0)) == r"\mathrm{HalfNormal}(1)"
    assert format_distribution(dist.Normal(jnp.zeros(2), 1.0).to_event(1)) == (
        r"\mathrm{Normal}([0, 0], 1)"
    )
    assert format_distribution(dist.Cauchy(0.0, 1.0)) == r"\mathrm{Cauchy}"
    assert isinstance(Parameters.__dict__["distribution"], classmethod)

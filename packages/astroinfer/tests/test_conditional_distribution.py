"""ConditionalDistribution: schemas, batching, binding, keys and display."""
import warnings
from math import prod
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astroinfer import (
    BoundConditionalDistribution,
    ConditionalDistribution,
    MeasurementProcess,
    PopulationLikelihood,
    PopulationModel,
    parameter_dataclass,
)
from astroinfer.generative import GenerativeModel, TruncatedNormal
from jax.scipy.stats import norm

jax.config.update("jax_enable_x64", True)


@parameter_dataclass(latex="m")
class Masses:
    m1: float
    m2: float


@parameter_dataclass(latex="s")
class Spins:
    chi: float


@parameter_dataclass(latex=r"\Lambda")
class Hyper:
    mu: float
    sigma: float


@parameter_dataclass
class Theta:
    x: float


@parameter_dataclass
class Obs:
    y: float


def _masses(h, key):
    k1, k2 = jax.random.split(key)
    return Masses(m1=h.mu + h.sigma * jax.random.normal(k1),
                  m2=h.mu + h.sigma * jax.random.normal(k2))


def _masses_logpdf(m, h):
    return norm.logpdf(m.m1, h.mu, h.sigma) + norm.logpdf(m.m2, h.mu, h.sigma)


class Prior(ConditionalDistribution):
    """No conditions."""
    data_type = Masses

    def sample_single(self, key):
        return _masses(Hyper(mu=0.0, sigma=1.0), key)

    def logpdf_single(self, m):
        return _masses_logpdf(m, Hyper(mu=0.0, sigma=1.0))


class MassModel(ConditionalDistribution):
    """One condition."""
    data_type = Masses
    parameter_type = Hyper

    def sample_single(self, h, key, scale=1.0):
        return _masses(Hyper(mu=h.mu, sigma=scale * h.sigma), key)

    def logpdf_single(self, m, h, scale=1.0):
        return _masses_logpdf(m, Hyper(mu=h.mu, sigma=scale * h.sigma))


class Joint(ConditionalDistribution):
    """A tuple output."""
    data_type = (Masses, Spins)
    parameter_type = Hyper

    def sample_single(self, h, key):
        k1, k2 = jax.random.split(key)
        m = _masses(h, k1)
        return m, Spins(chi=0.1 * m.m1 + jax.random.normal(k2))

    def logpdf_single(self, m, s, h):
        return _masses_logpdf(m, h) + norm.logpdf(s.chi, 0.1 * m.m1, 1.0)


class SpinGivenBoth(ConditionalDistribution):
    """A tuple of conditions."""
    data_type = Spins
    parameter_type = (Masses, Hyper)

    def sample_single(self, m, h, key):
        return Spins(chi=0.1 * m.m1 + h.mu + h.sigma * jax.random.normal(key))

    def logpdf_single(self, s, m, h):
        return norm.logpdf(s.chi, 0.1 * m.m1 + h.mu, h.sigma)


class SpinMapping(ConditionalDistribution):
    """A mapping of conditions."""
    data_type = Spins
    parameter_type: ClassVar = {"masses": Masses, "hyperparameters": Hyper}

    def sample_single(self, masses, hyperparameters, key):
        return SpinGivenBoth().sample_single(masses, hyperparameters, key)

    def logpdf_single(self, s, masses, hyperparameters):
        return SpinGivenBoth().logpdf_single(s, masses, hyperparameters)


class Shift(ConditionalDistribution):
    """An untyped condition: plain arrays reach the single methods whole."""
    data_type = Obs
    parameter_type = Any

    def sample_single(self, c, key):
        return Obs(y=jnp.sum(c) + jax.random.normal(key))

    def logpdf_single(self, d, c):
        return norm.logpdf(d.y, jnp.sum(c), 1.0)


class Noise(MeasurementProcess):
    parameter_type = Theta
    data_type = Obs

    def sample_single(self, theta, key):
        return Obs(y=theta.x + 0.3 * jax.random.normal(key))

    def logpdf_single(self, d, theta):
        return norm.logpdf(d.y, theta.x, 0.3)


class Gaussian(PopulationModel):
    parameter_type = Theta

    def sample_single(self, Lambda, key):
        return Theta(x=Lambda["mu"] + Lambda["sigma"] * jax.random.normal(key))

    def logpdf_single(self, theta, Lambda):
        return norm.logpdf(theta.x, Lambda["mu"], Lambda["sigma"])


H = Hyper(mu=jnp.array([0.0, 1.0, 2.0]), sigma=jnp.array([1.0, 2.0, 3.0]))
H0 = {"mu": 0.5, "sigma": 2.0}
LAMBDA = {"mu": 0.2, "sigma": 1.3}


def masses(shape):
    n = prod(shape)
    return Masses(m1=jnp.linspace(-1, 1, n).reshape(shape),
                  m2=jnp.linspace(0, 2, n).reshape(shape))


# --------------------------------------------------------------
# Schemas
# --------------------------------------------------------------

def test_no_conditions():
    prior = Prior()
    assert prior.sample(key=0).batch_shape == ()
    assert prior.sample(0, N=5).batch_shape == (5,)
    m = prior.sample(key=0, N=(2, 3))
    assert m.batch_shape == (2, 3)
    assert prior.logpdf(m).shape == (2, 3)
    np.testing.assert_allclose(prior.log_prob(m), prior.logpdf(m))


def test_single_condition():
    model = MassModel()
    m = model.sample(H, 0, N=4)
    assert m.batch_shape == (3, 4)
    assert model.logpdf(m, H).shape == (3, 4)
    # A dict is coerced to the declared type and is one point.
    assert model.sample(H0, key=0, N=4).batch_shape == (4,)
    np.testing.assert_allclose(
        model.logpdf(masses((4,)), H0),
        _masses_logpdf(masses((4,)), Hyper(**H0)))


def test_tuple_output():
    joint = Joint()
    m, s = joint.sample(H0, key=0, N=5)
    assert m.batch_shape == s.batch_shape == (5,)
    np.testing.assert_allclose(
        joint.logpdf(m, s, H0),
        _masses_logpdf(m, Hyper(**H0)) + norm.logpdf(s.chi, 0.1 * m.m1, 1.0))


def test_tuple_conditions():
    model = SpinGivenBoth()
    s = model.sample(masses((3,)), H, 0)
    assert s.batch_shape == (3,)
    assert model.sample(masses((3,)), H, key=0, N=2).batch_shape == (3, 2)
    assert model.logpdf(s, masses((3,)), H).shape == (3,)


def test_mapping_conditions():
    model, m = SpinMapping(), masses((3,))
    a = model.sample(key=0, N=2, masses=m, hyperparameters=H)
    b = model.sample(0, N=2, masses=m, hyperparameters=H)
    c = model.sample({"masses": m, "hyperparameters": H}, 0, N=2)
    assert a.batch_shape == (3, 2)
    np.testing.assert_array_equal(a.chi, b.chi)
    np.testing.assert_array_equal(a.chi, c.chi)
    np.testing.assert_array_equal(a.chi, SpinGivenBoth().sample(m, H, 0, N=2).chi)
    np.testing.assert_allclose(
        model.logpdf(a, masses=m, hyperparameters=H),
        SpinGivenBoth().logpdf(a, m, H))
    with pytest.raises(TypeError, match="Missing"):
        model.sample(key=0, masses=m)


def test_plain_arrays_are_not_batched():
    c = jnp.arange(3.0)
    d = Shift().sample(c, key=0)
    assert d.batch_shape == ()
    assert Shift().sample(c, key=0, N=4).batch_shape == (4,)
    np.testing.assert_allclose(Shift().logpdf(Obs(y=3.0), c), norm.logpdf(0.0))


def test_condition_lines_up_with_leading_data_axes():
    model, m = MassModel(), masses((3, 5))
    expected = _masses_logpdf(
        m, Hyper(mu=H.mu[:, None], sigma=H.sigma[:, None]))
    np.testing.assert_allclose(model.logpdf(m, H), expected)
    # Trailing alignment would accept this; leading alignment does not.
    with pytest.raises(ValueError, match="must start with"):
        model.logpdf(masses((2, 3)), H)


def test_round_trip_measurement():
    noise, theta = Noise(), Theta(x=jnp.arange(3.0))
    d = noise.sample(theta, key=0, N=4)
    assert d.batch_shape == (3, 4)
    np.testing.assert_allclose(
        noise.logpdf(d, theta), norm.logpdf(d.y, theta.x[:, None], 0.3))
    assert noise.sample(theta, key=0, N=(2, 5)).batch_shape == (3, 2, 5)


def test_round_trip_population():
    Lam = Hyper(mu=jnp.array([-1.0, 2.0]), sigma=jnp.array([0.5, 1.5]))
    theta = Gaussian().sample(Lam, key=0, N=6)
    assert theta.batch_shape == (2, 6)
    np.testing.assert_allclose(
        Gaussian().logpdf(theta, Lam),
        norm.logpdf(theta.x, Lam.mu[:, None], Lam.sigma[:, None]))


def test_outer_with_several_conditions():
    model = SpinGivenBoth()
    s, m = Spins(chi=jnp.linspace(-1, 1, 4)), masses((3,))
    out = model.logpdf(s, m, H, outer=True)
    assert out.shape == (3, 4)
    expected = norm.logpdf(
        s.chi[None, :], 0.1 * m.m1[:, None] + H.mu[:, None], H.sigma[:, None])
    np.testing.assert_allclose(out, expected)


def test_context_is_forwarded():
    model = MassModel()
    wide = model.sample(H0, key=0, N=4000, scale=10.0)
    assert float(jnp.std(wide.m1)) > 10.0
    np.testing.assert_allclose(
        model.logpdf(masses((2,)), H0, scale=10.0),
        _masses_logpdf(masses((2,)), Hyper(mu=0.5, sigma=20.0)))


def test_names_must_differ_between_mapping_types():
    with pytest.raises(TypeError, match="share the names"):
        class Clash(ConditionalDistribution):
            data_type: ClassVar = {"masses": Masses}
            parameter_type: ClassVar = {"masses": Masses, "hyperparameters": Hyper}


# --------------------------------------------------------------
# Binding conditions
# --------------------------------------------------------------

def test_condition_binds_everything():
    model = MassModel()
    bound = model.condition(H)
    assert isinstance(bound, BoundConditionalDistribution)
    np.testing.assert_array_equal(
        bound.sample(key=0, N=5).m1, model.sample(H, 0, N=5).m1)
    m = masses((3,))
    np.testing.assert_allclose(bound.logpdf(m), model.logpdf(m, H))
    np.testing.assert_allclose(bound.log_prob(m), model.logpdf(m, H))


def test_condition_binds_part_of_a_mapping():
    model, m = SpinMapping(), masses((3,))
    partial = model.condition(masses=m)
    assert partial._condition_schema().names == ("hyperparameters",)
    expected = model.sample(key=0, masses=m, hyperparameters=H)
    np.testing.assert_array_equal(
        partial.sample(key=0, hyperparameters=H).chi, expected.chi)
    full = partial.condition(hyperparameters=H)
    np.testing.assert_array_equal(full.sample(key=0).chi, expected.chi)
    np.testing.assert_allclose(
        partial.logpdf(expected, hyperparameters=H),
        model.logpdf(expected, masses=m, hyperparameters=H))


def test_bound_distribution_has_its_own_key():
    model = MassModel()
    bound = model.condition(H0)
    parent_key, k0 = model.key, bound.key
    a, b = bound.sample(N=3), bound.sample(N=3)
    assert not np.allclose(a.m1, b.m1)
    assert not np.array_equal(jax.random.key_data(k0), jax.random.key_data(bound.key))
    assert model.key is parent_key


def test_bound_population_uses_overridden_sample():
    model = GenerativeModel(TruncatedNormal("x", "mu", "sigma", -50.0, 50.0), order=["x"])
    population = PopulationModel.from_generative(model, Theta)
    np.testing.assert_array_equal(
        population.condition(LAMBDA).sample(key=3, N=4).x,
        population.sample(LAMBDA, key=3, N=4).x)


# --------------------------------------------------------------
# Display
# --------------------------------------------------------------

@pytest.mark.parametrize("dist, density", [
    (Prior(), r"p\!\left(m\right)"),
    (MassModel(), r"p\!\left(m\mid \Lambda\right)"),
    (Joint(), r"p\!\left(m, s\mid \Lambda\right)"),
    (SpinGivenBoth(), r"p\!\left(s\mid m, \Lambda\right)"),
    (SpinMapping(), r"p\!\left(s\mid m, \Lambda\right)"),
    (Shift(), r"p\!\left(Obs\mid \theta\right)"),
])
def test_display(dist, density):
    html = dist._repr_html_()
    assert density in html and ".sample" in html and ".logpdf" in html
    assert (".condition" in html) == (dist._condition_schema().kind != "none")


def test_display_of_bound_and_subclasses():
    bound = SpinMapping().condition(masses=masses((2,)))
    html = bound._repr_html_()
    assert r"p\!\left(s\mid m, \Lambda\right)" in html and "masses (bound)" in html
    assert r"\mathcal{L}\!\left(Obs\mid Theta\right)" in Noise()._repr_html_()
    assert ".PE" in Noise()._repr_html_()
    assert r"\pi\!\left(Theta\mid \Lambda\right)" in Gaussian()._repr_html_()


# --------------------------------------------------------------
# PopulationModel: N replaces shape
# --------------------------------------------------------------

def test_shape_is_deprecated_and_gives_identical_draws():
    generative = PopulationModel.from_generative(
        GenerativeModel(TruncatedNormal("x", "mu", "sigma", -50.0, 50.0), order=["x"]),
        Theta)
    for population in (Gaussian(), generative):
        with pytest.warns(DeprecationWarning, match="N="):
            old = population.sample(LAMBDA, key=1, shape=(2, 3))
        new = population.sample(LAMBDA, key=1, N=(2, 3))
        assert new.batch_shape == (2, 3)
        np.testing.assert_array_equal(old.x, new.x)


class OldSignature(Gaussian):
    """Overrides the batched sample with the pre-N signature."""
    calls: ClassVar[list] = []

    def sample(self, Lambda, key=None, *, shape=()):
        OldSignature.calls.append(shape)
        return Gaussian().sample(Lambda, key, N=shape or None)


def test_old_signature_override_in_population_likelihood():
    data = Noise().sample(Gaussian().sample(LAMBDA, key=1, N=6), key=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        old = PopulationLikelihood(Noise(), data, OldSignature(), N=500, seed=3)
        new = PopulationLikelihood(Noise(), data, Gaussian(), N=500, seed=3)
        np.testing.assert_array_equal(
            old.event_logpdf(LAMBDA), new.event_logpdf(LAMBDA))
        assert OldSignature().sample(LAMBDA, key=0, N=4).batch_shape == (4,)
    assert (6, 500) in OldSignature.calls


# ============================================================
# Regressions found in review
# ============================================================

@parameter_dataclass
class _T:
    x: float


class _RngPopulation(PopulationModel):
    parameter_type = _T

    def sample_single(self, Lambda, rng):  # key argument not named "key"
        return _T(x=Lambda["mu"] + jax.random.normal(rng))

    def logpdf_single(self, theta, Lambda):
        return jax.scipy.stats.norm.logpdf(theta.x, Lambda["mu"])


class _RngMeasurement(MeasurementProcess):
    parameter_type = _T

    def sample_single(self, theta, rng):
        return theta.x + jax.random.normal(rng)

    def logpdf_single(self, d, theta):
        return 0.0


def test_key_argument_may_have_any_name():
    assert _RngPopulation().sample({"mu": 1.0}, key=0, N=2).batch_shape == (2,)
    assert _RngMeasurement().sample(_T(x=jnp.zeros(2)), key=0, N=3).shape == (2, 3)


def test_bound_population_logpdf():
    pop = _RngPopulation()
    theta = _T(x=jnp.zeros(3))
    np.testing.assert_allclose(
        pop.condition({"mu": 1.0}).logpdf(theta), pop.logpdf(theta, {"mu": 1.0})
    )


def test_old_style_override_called_with_n_does_not_warn():
    class Old(_RngPopulation):
        def sample(self, Lambda, key=None, *, shape=()):
            return super().sample(Lambda, key, shape=shape)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert Old().sample({"mu": 1.0}, key=0, N=3).batch_shape == (3,)


def test_untyped_measurement_data_is_rejected_by_logpdf():
    with pytest.raises(TypeError, match="parameter_dataclass"):
        _RngMeasurement().logpdf(jnp.zeros((2, 3)), _T(x=jnp.zeros(2)))

"""Products and mixtures of conditional distributions."""
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astroinfer import ConditionalDistribution, parameter_dataclass
from astroinfer.distributions.algebra import (
    MixtureDistribution,
    ProductDistribution,
    mixture,
    product,
)
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from numpyro.distributions import constraints

jax.config.update("jax_enable_x64", True)


@parameter_dataclass(latex=r"\chi")
class Spin:
    chi: float


@parameter_dataclass(latex="m")
class Masses:
    m1: float
    q: float


@parameter_dataclass(latex=r"\Lambda")
class Hyper:
    mu: float
    sigma: float


@parameter_dataclass
class Pair:
    a: float
    b: float


@dataclass(eq=False)
class Leaf(ConditionalDistribution):
    """One NumPyro distribution for ``Spin.chi``."""
    d: Any
    data_type = Spin

    def sample_single(self, key):
        return Spin(chi=self.d.sample(key))

    def logpdf_single(self, s):
        return self.d.log_prob(s.chi)

    def leaf_supports(self):
        return {"chi": self.d.support}


@dataclass(eq=False)
class SpinGivenMasses(ConditionalDistribution):
    """chi ~ Normal(mu * q, sigma), conditional on the masses."""
    mu: float
    sigma: float = 0.5
    data_type = Spin
    parameter_type: ClassVar = {"masses": Masses}

    def sample_single(self, masses, key):
        return Spin(chi=self.mu * masses.q + self.sigma * jax.random.normal(key))

    def logpdf_single(self, s, masses):
        return norm.logpdf(s.chi, self.mu * masses.q, self.sigma)


class SpinGivenBoth(ConditionalDistribution):
    data_type = Spin
    parameter_type: ClassVar = {"masses": Masses, "hyperparameters": Hyper}

    def sample_single(self, masses, hyperparameters, key):
        h = hyperparameters
        return Spin(chi=h.mu * masses.q + h.sigma * jax.random.normal(key))

    def logpdf_single(self, s, masses, hyperparameters):
        h = hyperparameters
        return norm.logpdf(s.chi, h.mu * masses.q, h.sigma)


class MassModel(ConditionalDistribution):
    data_type = Masses
    parameter_type = Hyper

    def sample_single(self, h, key):
        k1, k2 = jax.random.split(key)
        return Masses(m1=h.mu + h.sigma * jax.random.normal(k1),
                      q=jax.random.uniform(k2))

    def logpdf_single(self, m, h):
        return norm.logpdf(m.m1, h.mu, h.sigma) + jnp.where(
            (m.q >= 0) & (m.q <= 1), 0.0, -jnp.inf
        )


class SpinGivenHyper(ConditionalDistribution):
    data_type = Spin
    parameter_type = Hyper

    def sample_single(self, h, key, scale=1.0):
        return Spin(chi=h.mu + scale * h.sigma * jax.random.normal(key))

    def logpdf_single(self, s, h, scale=1.0):
        return norm.logpdf(s.chi, h.mu, scale * h.sigma)


class SpinGivenMassesOnly(ConditionalDistribution):
    data_type = Spin
    parameter_type = Masses

    def sample_single(self, m, key):
        return Spin(chi=m.q + jax.random.normal(key))

    def logpdf_single(self, s, m):
        return norm.logpdf(s.chi, m.q, 1.0)


@dataclass(eq=False)
class Correlated(ConditionalDistribution):
    """Pair(u, u) or Pair(10 + u, -10 - u): the fields are tied."""
    flip: bool
    data_type = Pair

    def sample_single(self, key):
        u = jax.random.uniform(key)
        return Pair(a=10 + u, b=-10 - u) if self.flip else Pair(a=u, b=u)

    def logpdf_single(self, p):
        return jnp.zeros(())


class WideSpin(ConditionalDistribution):
    """Spin with a vector-valued chi: a different leaf shape."""
    data_type = Spin

    def sample_single(self, key):
        return Spin(chi=jax.random.normal(key, (2,)))

    def logpdf_single(self, s):
        return jnp.sum(norm.logpdf(s.chi))


H = Hyper(mu=jnp.array([0.0, 1.0, 2.0]), sigma=jnp.array([1.0, 2.0, 3.0]))
M = Masses(m1=jnp.array([10.0, 20.0, 40.0, 80.0]), q=jnp.array([0.2, 0.4, 0.6, 0.8]))
NORMALS = [Leaf(dist.Normal(-10.0, 1.0)), Leaf(dist.Normal(0.0, 1.0)), Leaf(dist.Normal(10.0, 1.0))]
W = jnp.array([0.2, 0.5, 0.3])


# --------------------------------------------------------------
# Products
# --------------------------------------------------------------

def test_product_with_tuple_output():
    joint = product(Leaf(dist.Normal(0.0, 1.0)), MassModel())
    assert isinstance(joint, ProductDistribution)
    assert joint._output_type() == (Spin, Masses)
    assert joint._condition_type() is Hyper

    s, m = joint.sample(H, jax.random.key(1), N=5)
    assert s.batch_shape == m.batch_shape == (3, 5)
    lp = joint.logpdf(s, m, H)
    ref = joint.components[0].logpdf(s) + MassModel().logpdf(m, H)
    assert lp.shape == (3, 5)
    np.testing.assert_allclose(lp, ref)


def test_product_samples_components_with_folded_keys():
    a, b = Leaf(dist.Normal(0.0, 1.0)), Leaf(dist.Normal(0.0, 1.0))
    x, y = product(a, b).sample_single(key=jax.random.key(3))
    assert x.chi == a.sample_single(jax.random.fold_in(jax.random.key(3), 0)).chi
    assert y.chi == b.sample_single(jax.random.fold_in(jax.random.key(3), 1)).chi
    assert x.chi != y.chi


def test_product_with_mapping_output():
    joint = product(spins=SpinGivenMasses(mu=1.0), other=Leaf(dist.Normal(0.0, 2.0)))
    assert joint._output_schema().names == ("spins", "other")
    assert joint._condition_schema().names == ("masses",)

    draw = joint.sample(key=jax.random.key(2), N=6, masses=M)
    assert set(draw) == {"spins", "other"}
    assert draw["spins"].batch_shape == (4, 6)
    lp = joint.logpdf(draw, masses=M)
    ref = (SpinGivenMasses(mu=1.0).logpdf(draw["spins"], masses=M)
           + Leaf(dist.Normal(0.0, 2.0)).logpdf(draw["other"]))
    np.testing.assert_allclose(lp, ref)
    np.testing.assert_allclose(joint.logpdf(**draw, masses=M), ref)


def test_product_merges_mapping_schemas():
    joint = product(SpinGivenMasses(mu=1.0), SpinGivenBoth(), Leaf(dist.Normal(0.0, 1.0)))
    schema = joint._condition_schema()
    assert schema.names == ("masses", "hyperparameters")
    assert schema.types == (Masses, Hyper)

    h = Hyper(mu=0.5, sigma=2.0)
    x = joint.sample(key=jax.random.key(0), masses=M, hyperparameters=h)
    ref = (SpinGivenMasses(mu=1.0).logpdf(x[0], masses=M)
           + SpinGivenBoth().logpdf(x[1], masses=M, hyperparameters=h)
           + Leaf(dist.Normal(0.0, 1.0)).logpdf(x[2]))
    np.testing.assert_allclose(joint.logpdf(*x, masses=M, hyperparameters=h), ref)


def test_unconditional_components_join_positional_schemas():
    joint = product(Leaf(dist.Normal(0.0, 1.0)), SpinGivenHyper())
    assert joint._condition_type() is Hyper


class OtherMasses(SpinGivenMasses):
    parameter_type: ClassVar = {"masses": Hyper}


def test_context_reaches_every_component():
    joint = product(SpinGivenHyper(), SpinGivenHyper())
    x, y = joint.sample(H, jax.random.key(0), N=2, scale=2.0)
    ref = SpinGivenHyper().logpdf(x, H, scale=2.0) + SpinGivenHyper().logpdf(y, H, scale=2.0)
    np.testing.assert_allclose(joint.logpdf(x, y, H, scale=2.0), ref)


def test_product_schema_conflicts():
    with pytest.raises(TypeError, match="'masses' is Masses"):
        product(SpinGivenMasses(mu=1.0), OtherMasses(mu=1.0))
    with pytest.raises(TypeError, match="Positional conditions must be equal"):
        product(SpinGivenHyper(), SpinGivenMassesOnly())
    with pytest.raises(TypeError, match="Cannot join a mapping"):
        product(SpinGivenMasses(mu=1.0), SpinGivenHyper())
    with pytest.raises(TypeError, match="either positionally or as keywords"):
        product(SpinGivenHyper(), spins=SpinGivenHyper())
    with pytest.raises(TypeError, match="share the names"):
        product(masses=SpinGivenMasses(mu=1.0))
    with pytest.raises(TypeError, match="single output type"):
        product(product(SpinGivenHyper(), SpinGivenHyper()), SpinGivenHyper())


def test_product_leaf_supports():
    a, b = Leaf(dist.Normal(0.0, 1.0)), Leaf(dist.HalfNormal(1.0))
    assert product(a, b).leaf_supports() == {"0.chi": constraints.real, "1.chi": constraints.positive}
    assert set(product(x=a, y=b).leaf_supports()) == {"x.chi", "y.chi"}
    assert product(a, SpinGivenHyper()).leaf_supports() is None


# --------------------------------------------------------------
# Mixtures
# --------------------------------------------------------------

def test_mixture_component_frequencies():
    mix = mixture(NORMALS, weights=W)
    assert isinstance(mix, MixtureDistribution)
    chi = mix.sample(key=jax.random.key(0), N=40_000).chi
    freq = [jnp.mean(chi < -5), jnp.mean(jnp.abs(chi) < 5), jnp.mean(chi > 5)]
    np.testing.assert_allclose(freq, W, atol=0.01)


def test_mixture_density_is_logsumexp():
    s = Spin(chi=jnp.linspace(-12, 12, 7))
    ref = logsumexp(
        jnp.log(W)[:, None]
        + jnp.stack([norm.logpdf(s.chi, mu, 1.0) for mu in (-10, 0, 10)]),
        axis=0,
    )
    np.testing.assert_allclose(mixture(NORMALS, weights=W).logpdf(s), ref)
    logits = jnp.log(W) + 3.0
    np.testing.assert_allclose(mixture(NORMALS, logits=logits).logpdf(s), ref)


def test_gradient_with_respect_to_weights():
    s = Spin(chi=jnp.array([-9.0, 0.5, 3.0, 11.0]))

    def f(w):
        return mixture(NORMALS, weights=w).logpdf(s).sum()

    g = jax.grad(f)(W)
    assert np.all(np.isfinite(g))
    # Finite differences along directions that stay on the simplex.
    eps = 1e-6
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        d = jnp.zeros(3).at[i].set(1.0).at[j].set(-1.0)
        fd = (f(W + eps * d) - f(W - eps * d)) / (2 * eps)
        np.testing.assert_allclose(g @ d, fd, rtol=1e-5)


def test_gradient_with_respect_to_logits():
    s = Spin(chi=jnp.array([-9.0, 0.5, 3.0, 11.0]))
    logits = jnp.array([0.3, -1.0, 0.7])

    def f(l):
        return mixture(NORMALS, logits=l).logpdf(s).sum()

    g = jax.grad(f)(logits)
    assert np.all(np.isfinite(g))
    eps = 1e-6
    fd = [(f(logits.at[k].add(eps)) - f(logits.at[k].add(-eps))) / (2 * eps) for k in range(3)]
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def mass_fractions(m1, alpha):
    w = jax.nn.sigmoid(alpha * (m1 - 30.0))
    return jnp.stack([1 - w, w])


def mass_dependent():
    return mixture(
        [SpinGivenMasses(mu=0.0), SpinGivenMasses(mu=2.0)],
        weights=lambda masses, alpha: mass_fractions(masses.m1, alpha),
    )


def test_condition_dependent_weights():
    mix = mass_dependent()
    schema = mix._condition_schema()
    assert schema.names == ("masses", "alpha")
    assert schema.types == (Masses, Any)

    s = mix.sample(key=jax.random.key(0), N=3, masses=M, alpha=0.2)
    assert s.batch_shape == (4, 3)
    lp = mix.logpdf(s, masses=M, alpha=0.2)
    assert lp.shape == (4, 3)
    w = mass_fractions(M.m1, 0.2)
    ref = jnp.log(
        w[0][:, None] * norm.pdf(s.chi, 0.0, 0.5)
        + w[1][:, None] * norm.pdf(s.chi, 2.0 * M.q[:, None], 0.5)
    )
    np.testing.assert_allclose(lp, ref)

    # Light objects come from the first component, heavy ones from the second.
    chi = mix.sample(key=jax.random.key(1), N=4000, masses=M, alpha=1.0).chi
    np.testing.assert_allclose(jnp.mean(chi, axis=-1), [0.0, 0.0, 1.2, 1.6], atol=0.05)


def test_condition_gradients_are_finite():
    mix = mass_dependent()
    s = Spin(chi=jnp.array([0.1, 0.9, 1.0, 1.5]))

    def f(m1, alpha):
        return mix.logpdf(s, masses=Masses(m1=m1, q=M.q), alpha=alpha).sum()

    g_m1, g_alpha = jax.grad(f, argnums=(0, 1))(M.m1, 0.2)
    assert np.all(np.isfinite(g_m1)) and np.isfinite(g_alpha)
    eps = 1e-6
    fd = (f(M.m1, 0.2 + eps) - f(M.m1, 0.2 - eps)) / (2 * eps)
    np.testing.assert_allclose(g_alpha, fd, rtol=1e-5)


def test_whole_object_is_selected():
    mix = mixture([Correlated(flip=False), Correlated(flip=True)], weights=jnp.array([0.5, 0.5]))
    p = mix.sample(key=jax.random.key(0), N=2000)
    low = p.a < 5
    assert 0.4 < jnp.mean(low) < 0.6
    assert jnp.all(jnp.where(low, p.b == p.a, p.b == -p.a))


def test_bound_components():
    a, b = Hyper(mu=0.0, sigma=1.0), Hyper(mu=10.0, sigma=2.0)
    mix = mixture([SpinGivenHyper().condition(a), SpinGivenHyper().condition(b)],
                  weights=jnp.array([0.4, 0.6]))
    s = Spin(chi=jnp.array([0.0, 5.0, 10.0]))
    ref = jnp.log(0.4 * norm.pdf(s.chi, 0.0, 1.0) + 0.6 * norm.pdf(s.chi, 10.0, 2.0))
    np.testing.assert_allclose(mix.logpdf(s), ref)
    with pytest.raises(ValueError, match="batched"):
        mixture([SpinGivenHyper().condition(H)], weights=jnp.array([1.0]))


def test_positional_conditions_and_weights():
    mix = mixture([SpinGivenHyper(), Leaf(dist.Normal(0.0, 1.0))],
                  weights=lambda h: jnp.stack([h.sigma / 4, 1 - h.sigma / 4]))
    assert mix._condition_type() is Hyper
    s = mix.sample(H, jax.random.key(0), N=2)
    assert s.batch_shape == (3, 2)
    assert mix.logpdf(s, H).shape == (3, 2)


def test_explicit_parameter_type():
    mix = mixture([SpinGivenMasses(mu=0.0), SpinGivenMasses(mu=2.0)],
                  logits=lambda masses, hyperparameters: jnp.stack([masses.m1 * 0, hyperparameters.mu]),
                  parameter_type={"masses": Masses, "hyperparameters": Hyper})
    assert mix._condition_schema().types == (Masses, Hyper)
    lp = mix.logpdf(Spin(chi=jnp.zeros(4)), masses=M, hyperparameters=Hyper(mu=0.0, sigma=1.0))
    assert lp.shape == (4,)
    with pytest.raises(TypeError, match="parameter_type must name"):
        mixture([SpinGivenMasses(mu=0.0)], weights=jnp.array([1.0]),
                parameter_type={"hyperparameters": Hyper})
    with pytest.raises(TypeError, match="weight function needs"):
        mixture([SpinGivenMasses(mu=0.0)], weights=lambda alpha: jnp.ones(1),
                parameter_type={"masses": Masses})


@pytest.mark.parametrize("kwargs, match", [
    ({"weights": jnp.array([0.5, 0.6, -0.1])}, "nonnegative"),
    ({"weights": jnp.array([0.5, jnp.nan, 0.5])}, "finite"),
    ({"weights": jnp.array([0.2, 0.2, 0.2])}, "sum to one"),
    ({"weights": jnp.array([0.5, 0.5])}, r"shape \(3,\)"),
    ({"weights": jnp.full((2, 3), 1 / 3)}, r"shape \(3,\)"),
    ({"logits": jnp.zeros(2)}, r"shape \(3,\)"),
    ({}, "exactly one"),
    ({"weights": W, "logits": jnp.zeros(3)}, "exactly one"),
])
def test_weight_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mixture(NORMALS, **kwargs)


def test_callable_weights_with_wrong_shape():
    mix = mixture(NORMALS, weights=lambda: jnp.ones(2) / 2)
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        mix.logpdf(Spin(chi=0.0))


def test_output_type_and_shape_mismatches():
    with pytest.raises(TypeError, match="same output type"):
        mixture([Leaf(dist.Normal(0.0, 1.0)), MassModel()], weights=jnp.array([0.5, 0.5]))
    with pytest.raises(TypeError, match="same structure and leaf shapes"):
        mixture([Leaf(dist.Normal(0.0, 1.0)), WideSpin()], weights=jnp.array([0.5, 0.5]))


def test_leaf_supports():
    unit = mixture([Leaf(dist.Uniform(0.0, 1.0)), Leaf(dist.Beta(2.0, 3.0))], weights=jnp.array([0.5, 0.5]))
    assert set(unit.leaf_supports()) == {"chi"}
    pos = mixture([Leaf(dist.HalfNormal(1.0)), Leaf(dist.Exponential(1.0))], weights=jnp.array([0.5, 0.5]))
    assert pos.leaf_supports()["chi"] is constraints.positive

    with pytest.raises(ValueError, match="different supports at 'chi'"):
        mixture([Leaf(dist.Normal(0.0, 1.0)), Leaf(dist.HalfNormal(1.0))], weights=jnp.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="different supports"):
        mixture([Leaf(dist.Uniform(0.0, 1.0)), Leaf(dist.Uniform(0.0, 2.0))], weights=jnp.array([0.5, 0.5]))

    explicit = mixture([Leaf(dist.Normal(0.0, 1.0)), Leaf(dist.HalfNormal(1.0))],
                       weights=jnp.array([0.5, 0.5]), support={"chi": constraints.real})
    assert explicit.leaf_supports() == {"chi": constraints.real}
    assert mixture([Leaf(dist.Normal(0.0, 1.0)), SpinGivenMasses(mu=1.0)],
                   weights=jnp.array([0.5, 0.5])).leaf_supports() is None


def test_mixture_inside_product():
    joint = product(mixture(NORMALS, weights=W), SpinGivenHyper())
    x, y = joint.sample(H, jax.random.key(0))
    ref = mixture(NORMALS, weights=W).logpdf(x) + SpinGivenHyper().logpdf(y, H)
    np.testing.assert_allclose(joint.logpdf(x, y, H), ref)
    assert joint.leaf_supports() is None


# --------------------------------------------------------------
# Display
# --------------------------------------------------------------

def test_display():
    html = mixture(NORMALS, weights=W)._repr_html_()
    assert "Mixture" in html and r"\sum_{k=1}^{3}" in html
    assert "Component 1: Leaf" in html and "Component 3: Leaf" in html
    assert "w = (0.2, 0.5, 0.3)" in html

    html = mass_dependent()._repr_html_()
    assert r"w_k(m, \mathrm{alpha})" in html and r"w = f(m, \mathrm{alpha})" in html
    html = mixture(NORMALS, weights=lambda: W)._repr_html_()
    assert "w = f()" in html

    html = mixture(NORMALS, logits=jnp.zeros(3))._repr_html_()
    assert r"\mathrm{softmax}" in html

    html = product(Leaf(dist.Normal(0.0, 1.0)), MassModel())._repr_html_()
    assert "Product" in html and "Component 0: Leaf" in html and "Component 1: MassModel" in html
    assert r"p\!\left(\chi, m\mid \Lambda\right) = p\!\left(\chi\right)\, p\!\left(m\mid \Lambda\right)" in html

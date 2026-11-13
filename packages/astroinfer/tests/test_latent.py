import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.distributions import constraints

from astroinfer import LatentSpace, parameter, parameter_dataclass


@parameter_dataclass
class D:
    a: float                                                     # real line
    b: float = parameter(support=constraints.positive)
    c: float = parameter(support=constraints.interval(-1.0, 1.0))


@parameter_dataclass
class Other:
    a: float


@parameter_dataclass
class Vec:
    h: float = parameter(index=("ifo",), support=constraints.interval(0.0, 1.0))


def test_round_trip_and_real_valued_latent():
    d = D(a=jnp.array([-3.0, 0.5]), b=jnp.array([0.1, 20.0]), c=jnp.array([-0.9, 0.3]))
    z = d.to_unconstrained()
    assert isinstance(z, D)
    np.testing.assert_allclose(z.a, d.a)                       # identity on R
    np.testing.assert_allclose(z.b, np.log(d.b), rtol=1e-6)    # exp bijection
    back = D.from_unconstrained(z)
    for k in "abc":
        np.testing.assert_allclose(getattr(back, k), getattr(d, k), rtol=1e-6, atol=1e-6)


def test_any_latent_point_maps_into_supports():
    z = D(a=jnp.linspace(-50, 50, 7), b=jnp.linspace(-50, 50, 7), c=jnp.linspace(-50, 50, 7))
    d = D.from_unconstrained(z)
    assert np.all(np.asarray(d.b) >= 0)
    assert np.all((np.asarray(d.c) >= -1) & (np.asarray(d.c) <= 1))


def test_boundary_maps_to_infinity():
    z = D(a=0.0, b=0.0, c=jnp.array(1.0)).to_unconstrained()
    assert float(z.b) == -np.inf and float(z.c) == np.inf


def test_call_time_supports_override_with_constraints_or_distributions():
    latent = LatentSpace(D, {"a": dist.Uniform(0.0, 2.0), "b": constraints.real})
    d = D(a=1.0, b=-5.0, c=0.0)
    z = d.to_unconstrained(latent)
    assert float(z.a) == pytest.approx(0.0, abs=1e-6)    # midpoint of (0, 2)
    assert float(z.b) == -5.0                            # now unconstrained
    back = D.from_unconstrained(z, latent)
    assert float(back.a) == pytest.approx(1.0, abs=1e-6)


def test_bad_supports_and_types_raise():
    with pytest.raises(KeyError):
        LatentSpace(D, {"nope": constraints.real})
    with pytest.raises(TypeError):
        LatentSpace(D, {"a": (0, 1)})
    with pytest.raises(TypeError):
        Other.from_unconstrained(Other(a=1.0), LatentSpace(D))


def test_gradient_ascent_stays_in_support():
    # Maximize -(c - 2)^2 with c restricted to (-1, 1): optimum pushes to the edge.
    def f(d):
        return -(d.c - 2.0) ** 2 - (d.a - 3.0) ** 2 - (d.b - 0.5) ** 2

    z = D(a=0.0, b=1.0, c=0.0).to_unconstrained()
    g = jax.jit(jax.grad(lambda z: f(D.from_unconstrained(z))))
    for _ in range(500):
        z = z + 0.1 * g(z)
    d = D.from_unconstrained(z)
    assert float(d.a) == pytest.approx(3.0, abs=1e-3)
    assert float(d.b) == pytest.approx(0.5, abs=1e-3)
    assert 0.99 < float(d.c) <= 1.0


def test_log_abs_det_jacobian_matches_autodiff():
    latent = LatentSpace.default(D)
    z = D(a=jnp.array([0.3, -1.0]), b=jnp.array([0.2, 1.5]), c=jnp.array([-0.4, 2.0]))
    ladj = latent.log_abs_det_jacobian(z)
    assert ladj.shape == (2,)
    for i in range(2):
        zi = z[i]
        flat, unravel = jax.flatten_util.ravel_pytree(zi)
        J = jax.jacfwd(lambda v: jax.flatten_util.ravel_pytree(
            latent.to_parameter(unravel(v)))[0])(flat)
        assert float(ladj[i]) == pytest.approx(float(jnp.linalg.slogdet(J)[1]), rel=1e-5)


def test_core_axes_are_summed_in_jacobian():
    z = Vec(h=jnp.zeros((4, 3)))
    ladj = LatentSpace.default(Vec).log_abs_det_jacobian(z)
    assert ladj.shape == (4,)
    np.testing.assert_allclose(ladj, 3 * np.log(0.25), rtol=1e-6)   # sigmoid'(0) = 1/4


def test_display():
    html = LatentSpace.default(D)._repr_html_()
    assert "Interval" in html and "Positive" in html

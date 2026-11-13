import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astroinfer import parameter, parameter_dataclass


@parameter_dataclass
class Theta:
    x: float
    y: float


@parameter_dataclass
class Other:
    x: float
    y: float


@parameter_dataclass
class Strain:
    a: float
    h: float = parameter(index=("ifo",))


def test_add_sub_neg_fieldwise():
    a, b = Theta(x=1.0, y=2.0), Theta(x=0.5, y=-1.0)
    s, d, n = a + b, a - b, -a
    assert isinstance(s, Theta) and isinstance(d, Theta) and isinstance(n, Theta)
    assert (float(s.x), float(s.y)) == (1.5, 1.0)
    assert (float(d.x), float(d.y)) == (0.5, 3.0)
    assert (float(n.x), float(n.y)) == (-1.0, -2.0)


def test_grad_step_returns_same_type():
    def f(t):
        return t.x**2 + 3 * t.y

    theta = Theta(x=2.0, y=1.0)
    g = jax.grad(f)(theta)
    new = theta - g
    assert isinstance(new, Theta)
    assert (float(new.x), float(new.y)) == (-2.0, -2.0)
    # Works under jit too.
    assert jax.jit(lambda t: t - jax.grad(f)(t))(theta).x == new.x


def test_batch_and_core_axes_broadcast():
    batch = Strain(a=jnp.arange(5.0), h=jnp.ones((5, 3)))
    shift = Strain(a=1.0, h=jnp.array([0.0, 1.0, 2.0]))
    out = batch + shift
    assert out.batch_shape == (5,)
    np.testing.assert_allclose(out.a, np.arange(5.0) + 1)
    np.testing.assert_allclose(out.h, np.ones((5, 3)) + [0, 1, 2])


def test_mismatched_types_raise():
    with pytest.raises(TypeError):
        Theta(x=1.0, y=1.0) + Other(x=1.0, y=1.0)
    with pytest.raises(TypeError):
        Theta(x=1.0, y=1.0) + 1.0


@pytest.mark.parametrize(
    "lr", [0.5, 2, np.float64(0.5), np.array(0.5), jnp.asarray(0.5)]
)
def test_scalar_multiplication_both_sides(lr):
    t = Theta(x=2.0, y=-4.0)
    for out in (lr * t, t * lr):
        assert isinstance(out, Theta)
        assert (float(out.x), float(out.y)) == (2.0 * lr, -4.0 * lr)
    half = t / 2
    assert isinstance(half, Theta) and (float(half.x), float(half.y)) == (1.0, -2.0)


def test_gradient_step_with_traced_learning_rate():
    def f(t):
        return t.x**2 + t.y**2

    @jax.jit
    def step(t, lr):
        return t - lr * jax.grad(f)(t)

    out = step(Theta(x=1.0, y=-2.0), 0.25)
    assert isinstance(out, Theta)
    assert (float(out.x), float(out.y)) == (0.5, -1.0)


def test_scaling_batched_object():
    t = Strain(a=jnp.arange(4.0), h=jnp.ones((4, 3)))
    out = 3 * t
    assert out.batch_shape == (4,)
    np.testing.assert_allclose(out.h, 3 * np.ones((4, 3)))


def test_non_scalar_multiplication_raises():
    t = Theta(x=1.0, y=1.0)
    for bad in (np.ones(3), jnp.ones(2), t, "a"):
        with pytest.raises(TypeError):
            t * bad
    with pytest.raises(TypeError):
        np.ones(3) * t
    with pytest.raises(TypeError):
        2 / t


def test_parameters_lists_field_names_in_definition_order():
    @parameter_dataclass
    class Thing:
        y: float = parameter(name="the y value")
        x: float = parameter(name="the x value")
        a: float = parameter(index=("ifo",))

    assert Thing(y=0.1, x=0.9, a=jnp.zeros(3)).parameters == ["y", "x", "a"]
    assert Thing.parameters == ["y", "x", "a"]
    assert Theta(x=1.0, y=2.0).parameters == ["x", "y"]

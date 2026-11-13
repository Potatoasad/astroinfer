import jax
import jax.numpy as jnp
import numpy as np

from astroinfer import parameter, parameter_dataclass


@parameter_dataclass
class Theta:
    x: float
    y: float


@parameter_dataclass
class Strain:
    a: float
    h: float = parameter(index=("ifo",))


def test_copy_numpy_fields_are_independent():
    t = Theta.from_dict({"x": np.arange(3.0), "y": np.ones(3)})
    c = t.copy()
    assert isinstance(c, Theta) and c is not t
    c.x[0] = 100.0
    c.y[:] = -1.0
    np.testing.assert_array_equal(t.x, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(t.y, [1.0, 1.0, 1.0])
    assert c.x is not t.x


def test_copy_preserves_values_types_and_shapes():
    t = Strain(a=jnp.arange(4.0), h=np.ones((4, 3)))
    c = t.copy()
    assert c.batch_shape == t.batch_shape == (4,)
    assert isinstance(c.a, jax.Array) and isinstance(c.h, np.ndarray)
    np.testing.assert_array_equal(c.a, t.a)
    np.testing.assert_array_equal(c.h, t.h)


def test_copy_python_scalars_and_lists():
    t = Theta(x=1.0, y=[1.0, 2.0])
    c = t.copy()
    assert c.x == 1.0 and c.y == [1.0, 2.0] and c.y is not t.y


def test_copy_under_jit():
    out = jax.jit(lambda t: t.copy() + t)(Theta(x=1.0, y=2.0))
    assert (float(out.x), float(out.y)) == (2.0, 4.0)

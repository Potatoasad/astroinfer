from dataclasses import dataclass

import jax
import numpy as np

from astroinfer import MeasurementProcess, parameter_dataclass


@parameter_dataclass
class Theta:
    x: float


@parameter_dataclass
class Obs:
    y: float


class Toy(MeasurementProcess):
    parameter_type = Theta
    data_type = Obs

    def sample_single(self, theta, key):
        return Obs(y=theta.x + jax.random.normal(key))


@dataclass
class ToyWithFields(Toy):
    sigma: float  # non-default field after the base's kw-only key


class ToyCustomInit(Toy):
    def __init__(self, sigma):  # does not call super().__init__()
        self.sigma = sigma


def keys_equal(a, b):
    return np.array_equal(jax.random.key_data(a), jax.random.key_data(b))


def test_default_key_on_every_subclass():
    for m in (Toy(), ToyWithFields(sigma=1.0)):
        assert keys_equal(m.key, jax.random.key(0))
    assert keys_equal(Toy(key=jax.random.key(7)).key, jax.random.key(7))


def test_sample_without_key_advances_internal_key():
    m = Toy()
    theta = Theta(x=np.zeros(5))
    k0 = m.key
    d1 = m.sample(theta)
    k1 = m.key
    d2 = m.sample(theta)
    assert not keys_equal(k0, k1) and not keys_equal(k1, m.key)
    assert not np.allclose(d1.y, d2.y)
    # Reproducible: a fresh instance gives the same sequence.
    m2 = Toy()
    np.testing.assert_array_equal(m2.sample(theta).y, d1.y)
    np.testing.assert_array_equal(m2.sample(theta).y, d2.y)


def test_sample_with_key_leaves_internal_key():
    m = Toy()
    theta = Theta(x=np.zeros(5))
    k0 = m.key
    a = m.sample(theta, jax.random.key(3))
    b = m.sample(theta, key=jax.random.key(3))
    m.sample(theta, 3)
    m.sample(theta, np.random.default_rng(0))
    assert keys_equal(m.key, k0)
    np.testing.assert_array_equal(a.y, b.y)


def test_seed_resets_key():
    m = Toy()
    theta = Theta(x=np.zeros(3))
    first = m.sample(theta)
    m.seed(0)
    np.testing.assert_array_equal(m.sample(theta).y, first.y)


def test_custom_init_without_super_still_samples():
    m = ToyCustomInit(sigma=1.0)
    d1 = m.sample(Theta(x=np.zeros(3)))
    d2 = m.sample(Theta(x=np.zeros(3)))
    assert not np.allclose(d1.y, d2.y)


def test_instances_stay_hashable():
    hash(Toy())

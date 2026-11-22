# Definitions for the spin subpopulations example on the landing page (docs/index.md). The page
# shows them piece by piece between the "# --- name" markers and runs the whole file in a hidden
# cell.
# --- imports
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

import astroinfer as ai
from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)
# --- theta
@parameter_dataclass(name="Masses", latex="m")
class Masses:
    m1: float = parameter(name="Primary mass [solar masses]", latex="m_1",
                          support=constraints.positive)
    q: float = parameter(name="Mass ratio", latex="q",
                         support=constraints.unit_interval)

@parameter_dataclass(name="Spins", latex=r"\chi")
class Spins:
    chi: float = parameter(name="Spin magnitude", latex=r"\chi",
                           support=constraints.unit_interval)
    cos_tilt: float = parameter(name="Cosine of the spin tilt", latex=r"\cos t",
                                support=constraints.interval(-1, 1))

@parameter_dataclass(name="Binary", latex=r"\theta")
class Binary:
    masses: Masses
    spins: Spins
# --- hyper
@parameter_dataclass(name="Hyperparameters", latex=r"\Lambda")
class Hyper:
    m_t: float = parameter(name="Transition mass", latex="m_t")
# --- masses
mass_prior = Masses.distribution(
    m1=dist.Uniform(5.0, 80.0),
    q=ai.conditional(lambda x: dist.Uniform(5.0 / x["m1"], 1.0), on="m1"),
)
# --- spins
aligned = Spins.distribution(
    chi=dist.Beta(2.0, 6.0),
    cos_tilt=dist.TruncatedNormal(1.0, 0.15, low=-1.0, high=1.0),
)
isotropic = Spins.distribution(
    chi=dist.Beta(2.0, 3.0),
    cos_tilt=dist.Uniform(-1.0, 1.0),
)

def dynamical_fraction(m1, m_t):
    """Fraction of binaries in the isotropic subpopulation at primary mass m1."""
    return jax.nn.sigmoid((m1 - m_t) / 5.0)

spin_population = ai.mixture(
    [aligned, isotropic],
    weights=lambda masses, hyper: jnp.stack(
        [1 - dynamical_fraction(masses.m1, hyper.m_t),
         dynamical_fraction(masses.m1, hyper.m_t)]),
    parameter_type={"masses": Masses, "hyper": Hyper},
)
# --- data
@parameter_dataclass(name="Event", latex="d")
class Event:
    m1: float = parameter(name="Measured primary mass", latex=r"\hat m_1")
    q: float = parameter(name="Measured mass ratio", latex=r"\hat q")
    chi: float = parameter(name="Measured spin magnitude", latex=r"\hat\chi")
    cos_tilt: float = parameter(name="Measured cosine of tilt", latex=r"\widehat{\cos t}")
# --- measurement
@dataclass
class Detector(MeasurementProcess):
    mass_error: float = 0.1     # relative error of m1
    q_error: float = 0.15
    chi_error: float = 0.1
    tilt_error: float = 0.3

    parameter_type = Binary     # theta is a Binary
    data_type = Event           # d is an Event

    def _parts(self, theta: Binary):
        mean = (theta.masses.m1, theta.masses.q, theta.spins.chi,
                theta.spins.cos_tilt)
        scale = (self.mass_error * theta.masses.m1, self.q_error,
                 self.chi_error, self.tilt_error)
        return mean, scale

    def sample_single(self, theta: Binary, key: jax.Array) -> Event:
        mean, scale = self._parts(theta)
        eps = jax.random.normal(key, (4,))
        return Event(*(m + s * e for m, s, e in zip(mean, scale, eps)))

    def logpdf_single(self, d: Event, theta: Binary) -> jax.Array:
        mean, scale = self._parts(theta)
        values = (d.m1, d.q, d.chi, d.cos_tilt)
        return sum(dist.Normal(m, s).log_prob(v)
                   for m, s, v in zip(mean, scale, values))
# --- model
def spin_model(data, detector):
    hyper = ai.sample("Lambda", Hyper.distribution(m_t=dist.Uniform(10.0, 70.0)))
    with numpyro.plate("events", data.batch_shape[0]):
        masses = ai.sample("masses", mass_prior)
        spins = ai.sample("spins", spin_population, masses=masses, hyper=hyper)
        ai.sample("d", detector, Binary(masses=masses, spins=spins), obs=data)
# --- end

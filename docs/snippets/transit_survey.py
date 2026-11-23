# Definitions for the transit survey example on the landing page (docs/index.md). The page shows
# them piece by piece between the "# --- name" markers and runs the whole file in a hidden cell.
# --- imports
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.stats import norm
from numpyro.distributions import constraints

import astroinfer as ai
from astroinfer import MeasurementProcess, parameter, parameter_dataclass
from astroinfer.detection import ThresholdTrigger

jax.config.update("jax_enable_x64", True)
# --- theta
@parameter_dataclass(name="Star", latex=r"\star")
class Star:
    radius: float = parameter(name="Stellar radius [solar radii]", latex="R_\\star",
                              support=constraints.positive)
    noise: float = parameter(name="Photometric noise", latex=r"\sigma_\star",
                             support=constraints.positive)

@parameter_dataclass(name="Planet", latex="p")
class Planet:
    period: float = parameter(name="Orbital period [days]", latex="P",
                              support=constraints.positive)
    t0: float = parameter(name="Time of first transit [days]", latex="T_0",
                          support=constraints.positive)
    radius: float = parameter(name="Planet radius [Earth radii]", latex="R_p",
                              support=constraints.positive)

@parameter_dataclass(name="System", latex=r"\theta")
class System:
    star: Star
    planet: Planet
# --- hyper
@parameter_dataclass(name="Hyperparameters", latex=r"\Lambda")
class Hyper:
    mu_R: float = parameter(name="Mean planet radius", latex=r"\mu_R")
    sigma_R: float = parameter(name="Width of planet radii", latex=r"\sigma_R",
                               support=constraints.positive)
# --- population
population = System.distribution(
    star=Star.distribution(
        radius=1.0,                                  # fixed: Sun-like stars
        noise=dist.Uniform(2e-4, 1e-3),
    ),
    planet=Planet.distribution(
        period=dist.Uniform(1.0, 30.0),
        t0=ai.conditional(lambda x, L: dist.Uniform(0.0, x["period"]),
                          on="period"),
        radius=lambda L: dist.TruncatedNormal(L.mu_R, L.sigma_R,
                                              low=0.5, high=6.0),
    ),
).given(Hyper)
# --- data
@parameter_dataclass(name="Detection", latex="d")
class Detection:
    snr: float = parameter(name="Transit SNR", latex=r"\hat\rho")
    period: float = parameter(name="Measured period [days]", latex=r"\hat P")
    t0: float = parameter(name="Measured first transit [days]", latex=r"\hat T_0")
    noise: float = parameter(name="Measured noise", latex=r"\hat\sigma_\star")
# --- measurement
@dataclass
class TransitSurvey(MeasurementProcess):
    days: float = 90.0         # length of the survey
    cadence: float = 1 / 48    # one exposure every 30 minutes
    duration: float = 0.2      # transit duration [days]

    parameter_type = System    # theta is a System
    data_type = Detection      # d is a Detection

    def snr(self, system: System) -> jax.Array:
        """Expected SNR of all transits: depth over the noise of the mean in-transit flux."""
        depth = (0.009158 * system.planet.radius / system.star.radius) ** 2
        n_in = self.days / system.planet.period * self.duration / self.cadence
        return depth * jnp.sqrt(n_in) / system.star.noise

    def _errors(self, system):
        return (1.0, 1e-3 * system.planet.period, 0.01, 0.05 * system.star.noise)

    def sample_single(self, system: System, key: jax.Array) -> Detection:
        mean = (self.snr(system), system.planet.period, system.planet.t0,
                system.star.noise)
        eps = jax.random.normal(key, (4,))
        return Detection(*(m + s * e for m, s, e in
                           zip(mean, self._errors(system), eps)))

    def logpdf_single(self, d: Detection, system: System) -> jax.Array:
        mean = (self.snr(system), system.planet.period, system.planet.t0,
                system.star.noise)
        values = (d.snr, d.period, d.t0, d.noise)
        return sum(dist.Normal(m, s).log_prob(v) for m, s, v in
                   zip(mean, self._errors(system), values))
# --- trigger
trigger = ThresholdTrigger(7.0, statistic="snr")
# --- model
hyperprior = Hyper.distribution(mu_R=dist.Uniform(0.5, 5.0),
                                sigma_R=dist.Uniform(0.1, 3.0))

def survey_model(data, survey, bank=None, seeds=None):
    L = ai.sample("Lambda", hyperprior)
    with numpyro.plate("planets", data.batch_shape[0]):
        system = ai.sample("system", population, L)
        ai.sample("d", survey, system, obs=data)
    if bank is not None:
        # alpha(Lambda): the detected fraction of a fixed bank of draws
        draws = bank.sample(L, seeds=seeds)
        p_det = norm.cdf(survey.snr(draws) - trigger.threshold)
        ai.factor("selection", -data.batch_shape[0] * jnp.log(p_det.mean()))
# --- end

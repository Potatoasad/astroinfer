# Class definitions for the star population example on the landing page (docs/index.md). The page
# shows them piece by piece between the "# --- name" markers and runs the whole file in a hidden
# cell.
# --- imports
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from numpyro.distributions import constraints

from astroinfer import MeasurementProcess, parameter, parameter_dataclass
from astroinfer.generative import (
    GenerativeModel, TruncatedNormal, Uniform, product)

jax.config.update("jax_enable_x64", True)
# --- population
population = GenerativeModel(
    product([
        TruncatedNormal("log_L", mu="mu_L", sigma="sigma_L",
                        a=-3.0, b=5.0),
        Uniform("distance", 0.5, 2.0),
    ]),
    order=["log_L", "distance"],
)
# --- theta
@parameter_dataclass(name="Star", latex=r"\theta")
class Star:
    log_L: float = parameter(name="Log luminosity",
                             latex=r"\log_{10}L")
    distance: float = parameter(name="Distance [kpc]", latex="r",
                                support=constraints.positive)
# --- data
@parameter_dataclass(name="Star observation", latex="d")
class StarObservation:
    log_flux: float = parameter(name="Measured log flux",
                                latex=r"\log_{10}\hat F")
    parallax: float = parameter(name="Measured parallax [mas]",
                                latex=r"\hat\varpi")
# --- measurement
@dataclass
class Telescope(MeasurementProcess):
    sigma_log_flux: float = 0.2
    sigma_parallax: float = 0.3

    parameter_type = Star        # theta is a Star
    data_type = StarObservation  # d is a StarObservation

    def sample_single(self, theta: Star,
                      key: jax.Array) -> StarObservation:
        k_flux, k_parallax = jax.random.split(key)
        log_flux = theta.log_L - 2 * jnp.log10(theta.distance)
        parallax = 1 / theta.distance
        noise_flux = self.sigma_log_flux * jax.random.normal(k_flux)
        noise_parallax = self.sigma_parallax * jax.random.normal(k_parallax)
        return StarObservation(log_flux=log_flux + noise_flux,
                               parallax=parallax + noise_parallax)

    def logpdf_single(self, d: StarObservation,
                      theta: Star) -> jax.Array:
        log_flux = theta.log_L - 2 * jnp.log10(theta.distance)
        parallax = 1 / theta.distance
        flux_dist = dist.Normal(log_flux, self.sigma_log_flux)
        parallax_dist = dist.Normal(parallax, self.sigma_parallax)
        return (flux_dist.log_prob(d.log_flux)
                + parallax_dist.log_prob(d.parallax))
# --- end

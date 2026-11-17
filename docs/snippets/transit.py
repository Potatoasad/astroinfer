# Class definitions for the transit example on the landing page (docs/index.md). The page shows
# them piece by piece between the "# --- name" markers and runs the whole file in a hidden cell.
# --- imports
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from numpyro.distributions import constraints

from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)

# --- theta
@parameter_dataclass(name="Planet", latex=r"\theta")
class Planet:
    period: float = parameter(name="Orbital period [days]", latex="P",
                              support=constraints.positive)
    t0: float = parameter(name="Time of first transit [days]",
                          latex="T_0")
    radius: float = parameter(name="Planet radius [stellar radii]",
                              latex="R_p",
                              support=constraints.unit_interval)
# --- data
@parameter_dataclass(name="Light curve", latex="d")
class LightCurve:
    time: jax.Array = parameter(name="Time [days]", latex="t",
                            index=("time",), index_latex=("i",))
    flux: jax.Array = parameter(name="Relative flux", latex=r"\hat F",
                            index=("time",), index_latex=("i",))
    flux_err: jax.Array = parameter(name="Flux error", latex=r"\sigma",
                                index=("time",), index_latex=("i",))

    def plot(self, ax, noise=True, **kwargs):
        """Draw the light curve on ax, with error bars if noise."""
        if noise:
            ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".",
                        ms=2, lw=0.5, **kwargs)
        else:  # a batch of light curves is drawn as one line each
            ax.plot(self.time.T, self.flux.T, lw=1, **kwargs)
        ax.set(xlabel="time [days]", ylabel="relative flux")
        return ax.figure, ax
# --- measurement
@dataclass
class Photometry(MeasurementProcess):
    days: float = 20.0         # length of the observation
    cadence: float = 1 / 48    # one exposure every 30 minutes
    duration: float = 0.2      # transit duration [days]
    sigma: float = 0.002       # white noise per exposure

    parameter_type = Planet  # theta is a Planet
    data_type = LightCurve   # d is a LightCurve

    def light_curve(self, theta: Planet, t: jax.Array) -> jax.Array:
        """Flux of the star at times t: a box with smoothed edges."""
        P = theta.period
        phase = (t - theta.t0 + P / 2) % P - P / 2
        edge = jnp.abs(phase) - self.duration / 2
        return 1 - theta.radius**2 * jax.nn.sigmoid(-edge / 0.01)

    def sample_single(self, theta: Planet, key: jax.Array) -> LightCurve:
        t = jnp.arange(0, self.days, self.cadence)
        noise = self.sigma * jax.random.normal(key, t.shape)
        return LightCurve(time=t, flux=self.light_curve(theta, t) + noise,
                          flux_err=jnp.full_like(t, self.sigma))

    def logpdf_single(self, d: LightCurve, theta: Planet) -> jax.Array:
        model = self.light_curve(theta, d.time)
        return dist.Normal(model, d.flux_err).log_prob(d.flux).sum()
# --- end

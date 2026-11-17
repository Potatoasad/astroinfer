# Class definitions for the coin flip example on the landing page (docs/index.md). The page shows
# them piece by piece between the "# --- name" markers and runs the whole file in a hidden cell.
# --- imports
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist

from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)
# --- theta
@parameter_dataclass(name="Coin")
class Coin:
    p: float = parameter(name="Probability of heads")
# --- data
@parameter_dataclass(name="Flips")
class Flips:
    heads: bool = parameter(name="Landed heads", index=("flip",))
# --- measurement
@dataclass
class FlipCoin(MeasurementProcess):
    n_flips: int = 50

    display_name = "Flip a coin"
    parameter_type = Coin  # theta is a Coin
    data_type = Flips      # d is a Flips

    def sample_single(self, theta: Coin, key: jax.Array) -> Flips:
        h = jax.random.bernoulli(key, theta.p, (self.n_flips,))
        return Flips(heads=h)

    def logpdf_single(self, d: Flips, theta: Coin) -> jax.Array:
        p = theta.p
        return jnp.where(d.heads, jnp.log(p), jnp.log1p(-p)).sum()
# --- end

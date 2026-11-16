# The example on the landing page (docs/index.md), which includes it piece by piece between the
# "# --- name" markers. Run it with `uv run python docs/snippets/coin_flip.py` after editing.
# --- imports
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from astroinfer import MeasurementProcess, parameter, parameter_dataclass
# --- theta
@parameter_dataclass
class Coin:
    p: float = parameter()
# --- data
@parameter_dataclass
class Flips:
    heads: bool = parameter(index=("flip",), index_latex=("i",))
# --- measurement
@dataclass
class FlipCoin(MeasurementProcess):
    n_flips: int = 50

    parameter_type = Coin
    data_type = Flips

    def sample_single(self, theta, key):
        h = jax.random.bernoulli(key, theta.p, (self.n_flips,))
        return Flips(heads=h)

    def logpdf_single(self, d, theta):
        p = theta.p
        return jnp.where(d.heads, jnp.log(p), jnp.log1p(-p)).sum()
# --- inference
experiment = FlipCoin(n_flips=50)
coin = Coin(p=0.7)

d = experiment.sample(coin, N=1000, key=0)   # 1000 experiments
d.batch_shape                                # (1000,)
experiment.logpdf(d, coin)                   # log L(d | theta), shape (1000,)

priors = {"p": dist.Uniform(0, 1)}
posterior = experiment.PE(d[0], priors)      # NUTS on the first experiment
posterior["p"].mean()
# --- end

if __name__ == "__main__":
    assert d.batch_shape == (1000,)
    assert experiment.logpdf(d, coin).shape == (1000,)
    print("posterior mean of p:", posterior["p"].mean())

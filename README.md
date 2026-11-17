<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/astroinfer-logo-dark.svg">
    <img src="assets/astroinfer-logo.svg" alt="astroinfer" width="320">
  </picture>
</h1>

[![Docs](https://img.shields.io/badge/docs-website-418FC5)](https://potatoasad.github.io/astroinfer/)
[![CI](https://github.com/Potatoasad/astroinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/Potatoasad/astroinfer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-17243D)](https://github.com/Potatoasad/astroinfer/blob/main/pyproject.toml)

**astroinfer** is a JAX-based framework for astrophysical simulation and Bayesian inference.
Define a physical system and its measurement process once, then use the same model to simulate
observations, estimate individual parameters, and model and infer population distributions.

[Documentation](https://potatoasad.github.io/astroinfer/) | [Getting started](https://potatoasad.github.io/astroinfer/getting-started/) |
[Quickstart](https://potatoasad.github.io/astroinfer/quickstart/) | [Installation](https://potatoasad.github.io/astroinfer/installation/) | [Examples](https://potatoasad.github.io/astroinfer/examples/)

## Example

Imagine an experiment in which you flip a coin 50 times and write down the whole list of heads
and tails as your data. The parameters $\theta$ go into a measurement process
$\mathcal{L}(d\mid\theta)$, which produces the data $d$, and each of the three is a class you
write. The code below runs from top to bottom, starting with these imports:

```python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)
```

### Parameters $\theta$

A coin has a probability $p$ of landing heads.

```python
@parameter_dataclass(name="Coin")
class Coin:
    p: float = parameter(name="Probability of heads")
```

### Data $d$

The data are the 50 flips, `True` for heads. `index` makes `heads` an array along an axis named
`flip`.

```python
@parameter_dataclass(name="Flips")
class Flips:
    heads: bool = parameter(name="Landed heads", index=("flip",))
```

### Measurement process $\mathcal{L}(d\mid\theta)$

Flipping the coin 50 times gives a list of heads and tails. `sample_single` simulates the 50
flips, and `logpdf_single` returns the log probability of a given list for a coin with
probability $p$.

```python
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
```

### Simulation and inference

We ask the measurement process to run the 50-flip experiment 1000 times for a coin with
$p = 0.7$, and get back a batch of data:

```python
experiment = FlipCoin(n_flips=50)
coin = Coin(p=0.7)

d = experiment.sample(coin, N=1000, key=0)
```

`d` is one `Flips` object that holds all 1000 experiments. Its `heads` field is a 1000 by 50
array of `True` and `False`, with one row per experiment and one column per flip.

The likelihood works on the whole batch, and returns one value of $\log\mathcal{L}(d\mid\theta)$
for each of the 1000 experiments:

```python
log_like = experiment.logpdf(d, coin)   # shape (1000,)
```

`sample_single` and `logpdf_single` were written for one experiment. astroinfer applies them to
batches of any shape with `jax.vmap` and manages the random keys.

Parameter estimation samples the posterior of $p$ with NUTS. Here we give it the first
experiment, `d[0]`, and a uniform prior on $p$:

```python
priors = {"p": dist.Uniform(0, 1)}
posterior = experiment.PE(d[0], priors, seed=0)
posterior["p"].mean()
```

The [landing page of the documentation](https://potatoasad.github.io/astroinfer/) shows the
outputs, along with a planetary transit, a population of stars, and a template for your own model.

## Beyond one object

The same measurement process is the starting point for population inference. A population
model $\pi(\theta\mid\Lambda)$ describes how $\theta$ is spread over many objects, and
astroinfer samples the hyperparameters $\Lambda$ from a batch of measurements, with a selection
term when only some objects are detected. The posteriors of single objects can be kept in a
`Catalog`, which saves them to disk and exports them as arrays for a population analysis.

## Where to go next

- [Getting started](https://potatoasad.github.io/astroinfer/getting-started/): the coin flip example step by step, with outputs.
- [Examples](https://potatoasad.github.io/astroinfer/examples/): ten worked problems, from a coin flip to selection effects and custom
  catalog representations.
- [User guide](https://potatoasad.github.io/astroinfer/terminology/): parameter types, measurement processes,
  population models, population inference and catalogs, one page each.
- [API reference](https://potatoasad.github.io/astroinfer/api/): every public class and function.

## Installation

astroinfer is not on PyPI yet. Install it from git, together with the gravitational-wave models:

```bash
uv add "astroinfer @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer" \
       "astroinfer-gw @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer-gw"
```

This also installs gravpop from git. See
[Installation](https://potatoasad.github.io/astroinfer/installation/) for pip and for the
optional `astroinfer-tgmm` extension.

## What is in this repository

| path | contents |
|---|---|
| [`packages/astroinfer`](packages/astroinfer) | the core library |
| [`packages/astroinfer-tgmm`](packages/astroinfer-tgmm) | truncated Gaussian mixture (TGMM) catalog representations; uses Julia |
| [`packages/astroinfer-gw`](packages/astroinfer-gw) | gravitational-wave models, including a GWMockCat-style measurement process for mock catalogs |
| [`docs/`](docs) | the documentation |
| [`tutorials/`](tutorials) | ten marimo notebooks, from a coin flip to selection effects and custom representations |
| [`package_tour/`](package_tour) | a notebook that goes through every type and its methods |
| [`notebooks/gw/`](notebooks/gw) | gravitational-wave examples: a mock catalog, TGMM fits, and population inference with a proposal |

## Developing

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): one
environment with every package installed in editable mode, plus marimo, matplotlib and pytest.

```bash
uv sync
uv run marimo edit tutorials/01_coin_flips.py
uv run pytest -m "not julia"   # tests that do not start Julia
uv run pytest                  # all tests; the first TGMM fit installs Julia
scripts/run_tutorials.sh       # run every tutorial as a script
```

To add a package, create `packages/<name>/` with its own `pyproject.toml`, depend on
`astroinfer` by name, and add `<name> = { workspace = true }` to the root `[tool.uv.sources]`.
A new catalog representation registers itself through the `astroinfer.representations` entry
point; see `packages/astroinfer-tgmm/pyproject.toml`.

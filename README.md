# astroinfer

Typed forward models and hierarchical inference in JAX.

You describe your parameters $\theta$ and data $d$ as typed, batched dataclasses, write the
measurement process $K(d\mid\theta)$ for **one** $\theta$ and **one** observation, and astroinfer
does the rest: batching, random keys, parameter estimation, differentiable population models,
population inference with selection effects, and catalogs of posteriors with pluggable,
saveable representations.

```python
from dataclasses import dataclass
import jax, jax.numpy as jnp
import numpyro.distributions as dist
from astroinfer import MeasurementProcess, parameter, parameter_dataclass

@parameter_dataclass(latex=r"\mathrm{coin}")
class Coin:
    p: float = parameter(name="Probability of heads", latex="p")

@parameter_dataclass(latex=r"\hat h")
class Flips:
    heads: bool = parameter(latex=r"\hat h", index=("flip",))

@dataclass
class FlipMany(MeasurementProcess):
    n_flips: int = 50
    parameter_type, data_type = Coin, Flips

    def sample_single(self, theta, key):
        return Flips(heads=jax.random.bernoulli(key, theta.p, (self.n_flips,)))

    def logpdf_single(self, d, theta):
        return jnp.sum(jnp.where(d.heads, jnp.log(theta.p), jnp.log1p(-theta.p)))

experiment = FlipMany()
d = experiment.sample(Coin(p=0.7))
posterior = experiment.PE(d, {"p": dist.Uniform(0, 1)})
```

## What's in this repository

| path | what |
|---|---|
| [`packages/astroinfer`](packages/astroinfer) | the core: parameters, measurement processes, `generative` population models, inference, detection, catalogs and representations |
| [`packages/astroinfer-tgmm`](packages/astroinfer-tgmm) | extension: truncated Gaussian mixture (TGMM) catalog representations (uses Julia) |
| [`packages/astroinfer-gwmockcat`](packages/astroinfer-gwmockcat) | extension: GWMockCat-style mock gravitational-wave measurement process |
| [`tutorials/`](tutorials) | nine marimo notebooks, from a coin flip to selection effects and custom representations: **start here** |
| [`notebooks/gw/`](notebooks/gw) | the gravitational-wave track: mock catalog, TGMM fits, proposal-based population inference |

## Installing

The packages are not on PyPI yet. Install from git, naming each package's subdirectory:

```bash
uv add "astroinfer @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer"

# extensions: list astroinfer too, so it also comes from git
uv add "astroinfer @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer" \
       "astroinfer-tgmm @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer-tgmm"
```

Some dependencies live only in git ([gravpop](https://git.ligo.org/asad.hussain/gravpop),
[truncatedgaussianmixtures](https://github.com/Potatoasad/truncatedgaussianmixtures),
[plot_helper](https://github.com/Potatoasad/plot_helper)). This repository's `uv.lock` pins them,
but a project that installs astroinfer must point at them too, e.g. in its own `pyproject.toml`:

```toml
[tool.uv.sources]
gravpop = { git = "https://git.ligo.org/asad.hussain/gravpop", rev = "simplify-composite" }
truncatedgaussianmixtures = { git = "https://github.com/Potatoasad/truncatedgaussianmixtures", rev = "jax-backend" }
```

## Developing

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): one
environment with every package installed editable, plus marimo, matplotlib and pytest.

```bash
uv sync
uv run marimo edit tutorials/01_coin_flips.py   # the tutorials
uv run pytest -m "not julia"                    # fast tests
uv run pytest                                   # everything (TGMM tests start Julia)
scripts/run_tutorials.sh                        # run every tutorial as a script
```

The first TGMM fit installs Julia into `.venv` (one-time, a few minutes).

To add a package, create `packages/<name>/` with its own `pyproject.toml`, depend on
`astroinfer` by name, and add `<name> = { workspace = true }` to the root `[tool.uv.sources]`.
A new catalog representation registers itself through the `astroinfer.representations` entry
point (see `packages/astroinfer-tgmm/pyproject.toml`).

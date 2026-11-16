---
kernelspec:
  name: python3
  display_name: Python 3
---

# Sharp bits

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

This page collects behaviour of astroinfer that tends to surprise new users. Each section shows
the code that goes wrong, the result, the reason, and what to do instead.

The examples use the source and camera from the [quickstart](quickstart.md):

```{code-cell} python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import astroinfer as ai
from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)

@parameter_dataclass(name="Source", latex=r"\theta")
class Source:
    x: float = parameter(name="Position [arcsec]", latex="x")
    flux: float = parameter(name="Flux", latex="f")

@parameter_dataclass(name="Measurement", latex="d")
class Measurement:
    x_hat: float = parameter(name="Measured position [arcsec]", latex=r"\hat x")
    flux_hat: float = parameter(name="Measured flux", latex=r"\hat f")

@dataclass
class Camera(MeasurementProcess):
    sigma_x: float = 0.1
    sigma_flux: float = 0.3

    parameter_type = Source
    data_type = Measurement

    def sample_single(self, theta, key):
        key_x, key_flux = jax.random.split(key)
        return Measurement(x_hat=theta.x + self.sigma_x * jax.random.normal(key_x),
                           flux_hat=theta.flux + self.sigma_flux * jax.random.normal(key_flux))

    def logpdf_single(self, d, theta):
        norm = jax.scipy.stats.norm
        return (norm.logpdf(d.x_hat, theta.x, self.sigma_x)
                + norm.logpdf(d.flux_hat, theta.flux, self.sigma_flux))

camera = Camera()
src = Source(x=0.2, flux=3.0)
sources = Source(x=jnp.array([-0.5, 0.0, 0.5]), flux=2.0)
```

## Parameter objects are frozen

Assigning to a field raises an error:

```{code-cell} python
:tags: [raises-exception]

src.x = 1.0
```

`@parameter_dataclass` makes frozen dataclasses unless you pass `frozen=False`. To change a field
of a frozen object, make a new object with `dataclasses.replace`:

```{code-cell} python
import dataclasses

dataclasses.replace(src, x=1.0)
```

## A broadcast field is stored as given

A field given as a single number stays a single number, even when the other fields make the
object a batch:

```{code-cell} python
sources.batch_shape, sources.flux, jnp.shape(sources.flux)
```

`sample` and `logpdf` broadcast the fields when they need to, so the object does not store
copies. When you need every field at the full batch shape, call `broadcast()`:

```{code-cell} python
sources.broadcast().flux
```

## Ordinary methods do not broadcast over a batch

A method written for one object usually fails on a batch, because the batch axes of its fields
meet the axes of its arguments:

```{code-cell} python
:tags: [raises-exception]

@parameter_dataclass
class Peak:
    centre: float = parameter()
    width: float = parameter()

    def profile(self, grid):
        return jnp.exp(-0.5 * ((grid - self.centre) / self.width) ** 2)

grid = jnp.linspace(-1.0, 1.0, 5)
Peak(centre=jnp.array([-0.5, 0.0, 0.5]), width=0.3).profile(grid)
```

`self.centre` has shape `(3,)` and `grid` has shape `(5,)`. Decorate the method with
`@batched_method`, which applies `jax.vmap` over every batch axis so that the body only sees one
object:

```{code-cell} python
@parameter_dataclass
class BatchedPeak:
    centre: float = parameter()
    width: float = parameter()

    @ai.batched_method
    def profile(self, grid):
        return jnp.exp(-0.5 * ((grid - self.centre) / self.width) ** 2)

BatchedPeak(centre=jnp.array([-0.5, 0.0, 0.5]), width=0.3).profile(grid).shape
```

## `sample` without a key gives different noise every time

Calling `sample` twice with the same parameters gives two different measurements:

```{code-cell} python
float(camera.sample(src).x_hat), float(camera.sample(src).x_hat)
```

Without `key=`, a measurement process uses a JAX key that it stores in `camera.key` and advances
it on every call. Pass `key=` for a reproducible draw, which leaves the stored key alone, or reset
the stored key with `seed`:

```{code-cell} python
float(camera.sample(src, key=3).x_hat), float(camera.sample(src, key=3).x_hat)
```

```{code-cell} python
camera.seed(0)
first = float(camera.sample(src).x_hat)
camera.seed(0)
first == float(camera.sample(src).x_hat)
```

## `logpdf` broadcasts batch shapes; it does not pair every combination

Evaluating four measurements against three sources fails:

```{code-cell} python
:tags: [raises-exception]

four = camera.sample(Source(x=jnp.zeros(4), flux=1.0), key=0)
camera.logpdf(four, sources)
```

`logpdf` broadcasts the batch shapes of the data and the parameters against each other, as NumPy
does, and `(4,)` and `(3,)` do not broadcast. To evaluate every source against every measurement,
pass `outer=True`. The parameter axes come first in the result:

```{code-cell} python
camera.logpdf(four, sources, outer=True).shape
```

## Population models check the order of their variables

Every variable a model draws must appear in `order`:

```{code-cell} python
:tags: [raises-exception]

from astroinfer.generative import GenerativeModel, TruncatedNormal, Uniform, conditional, product

GenerativeModel(product([Uniform("x", -1.0, 1.0), TruncatedNormal("flux", 3.0, 1.0, 0.0, 20.0)]),
                order=["x"])
```

A variable that a `conditional` reads must be drawn before it:

```{code-cell} python
:tags: [raises-exception]

fainter_outside = TruncatedNormal("flux", conditional(lambda x, L: 3.0 - x["x"] ** 2, on="x"),
                                  1.0, 0.0, 20.0)
GenerativeModel(product([Uniform("x", -1.0, 1.0), fainter_outside]), order=["flux", "x"])
```

The model is built once and checked then, so these mistakes raise immediately instead of producing
wrong draws inside a sampler. Put `x` first, or declare it with `external=` and pass its values
with `conditioned_on=` when sampling.

## Leaves without a closed-form quantile are slow

Every draw from a population model is a uniform number pushed through the inverse CDF of a leaf.
`Uniform` and `TruncatedNormal` have a closed-form inverse. `Beta` and `mixture` do not, so their
draws are found by bisection, which takes many more operations:

```{code-cell} python
import time

from astroinfer.generative import Beta

def time_ms(model, Lambda, n=100_000):
    draw = jax.jit(lambda key: model.sample(Lambda, key=key, shape=(n,)))
    jax.block_until_ready(draw(jax.random.key(0)))
    start = time.perf_counter()
    jax.block_until_ready(draw(jax.random.key(1)))
    return (time.perf_counter() - start) * 1e3

closed = time_ms(GenerativeModel(TruncatedNormal("p", "mu", "sigma", 0.0, 1.0), order=["p"]),
                 {"mu": 0.5, "sigma": 0.1})
bisection = time_ms(GenerativeModel(Beta("p", "a", "b"), order=["p"]), {"a": 2.0, "b": 3.0})
f"closed form: {closed:.1f} ms, bisection: {bisection:.1f} ms"
```

The draws are still differentiable in the hyperparameters. Prefer leaves with a closed-form
quantile in code that runs many times, such as a population likelihood. A leaf you write yourself
can provide one through `cond_ppf`.

## A small `N` biases the population likelihood low

`PopulationLikelihood` estimates each source's evidence by averaging $\mathcal{L}(d_i\mid\theta)$ over `N`
draws from the population, and its `logpdf` is the sum of the logs of those averages. The log of
an average is biased low when the average is noisy. Here the same 50 measurements are evaluated
with five seeds at each `N`:

```{code-cell} python
population = GenerativeModel(
    product([Uniform("x", -1.0, 1.0), TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0)]),
    order=["x", "flux"],
)
Lambda = {"mu": 3.0, "sigma": 1.0}
data = camera.sample(Source.from_dict(population.sample(Lambda, key=jax.random.key(1),
                                                         shape=(50,))), key=2)

for N in [10, 100, 1000, 20000]:
    values = [float(ai.PopulationLikelihood(camera, data, population, N=N, seed=s).logpdf(Lambda))
              for s in range(5)]
    print(f"N = {N:>5}: mean {np.mean(values):8.2f}, spread {np.std(values):6.2f}")
```

The estimate at `N = 10` is far below the converged value, and the size of the bias depends on
$\Lambda$, so a sampler run on it converges to the wrong hyperparameters. Check the effective
number of draws per source before sampling. `event_logpdf_and_neff` returns it, and a population
run records its minimum as `N_eff_min`:

```{code-cell} python
_, neff = ai.PopulationLikelihood(camera, data, population, N=100, seed=0).event_logpdf_and_neff(Lambda)
float(neff.min()), float(jnp.median(neff))
```

When the minimum is close to 1, increase `N`, or draw from a `Proposal` close to each source's
likelihood (see [Population inference](user-guide/population-inference.md)).

## Proposal draws must be smooth in the key and the hyperparameters

A `PopulationLikelihood` with a proposal redraws $\theta$ from the same key at every value of
$\Lambda$, so that its estimate is a smooth function of $\Lambda$ and NUTS can use its gradient.
`sample_single` must therefore be a smooth function of `(key, Lambda)`. Location-scale transforms,
inverse CDFs of uniform numbers and `GenerativeModel.sample(key=...)` are smooth. Rejection
sampling is not, because a small change in $\Lambda$ can change which candidate is accepted:

```py
# Not smooth: the accepted draw jumps as Lambda changes.
def sample_single(self, d, key, Lambda):
    while True:
        key, sub = jax.random.split(key)
        candidate = ...
        if accept(candidate, Lambda):
            return candidate

# Smooth: a fixed standard normal draw, moved and scaled.
def sample_single(self, d, key, Lambda):
    return Source(x=d.x_hat + 0.15 * jax.random.normal(key), flux=...)
```

## Catalogs need `types=` to load classes defined in a notebook

A catalog records the import path of its classes, such as `__main__:Source`. If the classes
were defined in a notebook or a script, loading the catalog in another session fails. Here the
catalog is saved, then loaded in a new Python process:

```{code-cell} python
import tempfile
from pathlib import Path

priors = {"x": dist.Uniform(-1.0, 1.0), "flux": dist.Uniform(0.0, 10.0)}
posterior = camera.PE(camera.sample(src, key=0), priors, seed=0, as_posterior=True,
                      num_warmup=300, num_samples=1000, summary=False)
catalog = ai.Catalog({"src_0": posterior}, raw=["x", "flux"])

tmp = tempfile.TemporaryDirectory()
root = catalog.save(Path(tmp.name) / "catalog")

import subprocess
import sys

script = f"import astroinfer as ai; ai.Catalog.load({str(root)!r})"
result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
print(result.stderr.strip().splitlines()[-1])
```

Pass the classes with `types=`:

```{code-cell} python
ai.Catalog.load(root, types=[Source, Measurement])
```

Classes defined in an installed package load without `types=`.

## A catalog posterior must have one draw axis

A posterior whose draws have a chain axis and a draw axis cannot be added to a catalog:

```{code-cell} python
:tags: [raises-exception]

chains = Source(x=jnp.zeros((2, 500)), flux=jnp.ones((2, 500)))
catalog.add_event("two_chains", ai.PosteriorData(theta=chains))
```

The catalog resamples each event along a single draw axis. Flatten the chains before building the
`PosteriorData`:

```{code-cell} python
catalog.add_event("two_chains", ai.PosteriorData(theta=chains.reshape_batch((1000,))))
catalog.eventnames
```

## `RawSamples` columns must be a list

A single column name passed as a string is rejected:

```{code-cell} python
:tags: [raises-exception]

ai.RawSamples(columns="x")
```

A string is itself a sequence of characters, so it is refused rather than read as `["x"]`. Pass a
list, `ai.RawSamples(columns=["x"])`.

## Injections do not support data with core axes

`DetectionTrigger.injections` fails when the data type has a field with a core axis:

```{code-cell} python
:tags: [raises-exception]

from astroinfer.detection import DetectionTrigger

@parameter_dataclass
class Exposures:
    flux_hat: float = parameter(index=("exposure",))

@dataclass
class RepeatCamera(MeasurementProcess):
    parameter_type = Source
    data_type = Exposures

    def sample_single(self, theta, key):
        return Exposures(flux_hat=theta.flux + 0.3 * jax.random.normal(key, (5,)))

    def logpdf_single(self, d, theta):
        return jax.scipy.stats.norm.logpdf(d.flux_hat, theta.flux, 0.3).sum()

class MeanFluxTrigger(DetectionTrigger):
    def detection_probability(self, d):
        return (np.asarray(d["flux_hat"]).mean(axis=-1) > 2.5).astype(float)

import pandas as pd

draws = pd.DataFrame({"x": np.zeros(10), "flux": np.linspace(1.0, 4.0, 10), "prior": 1.0})
MeanFluxTrigger().injections(RepeatCamera(), draws, np.random.default_rng(0))
```

`injections` writes each data field into a column of the returned DataFrame, and a DataFrame
column cannot hold a two-dimensional array. `detect` works for such data, so measure and detect
the draws yourself and keep whatever summary of the data you need:

```{code-cell} python
d = RepeatCamera().sample(Source.from_dict(draws), key=0)
draws["mean_flux_hat"] = np.asarray(d.flux_hat).mean(axis=-1)
draws["detected"] = MeanFluxTrigger().detect(d, np.random.default_rng(0))
draws[["flux", "mean_flux_hat", "detected"]].head(4)
```

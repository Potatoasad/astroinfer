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

`logpdf` broadcasts the batch shapes of the data and the parameters against each other. When both
have the same number of batch axes this is NumPy broadcasting, and `(4,)` and `(3,)` do not
broadcast. To evaluate every source against every measurement,
pass `outer=True`. The parameter axes come first in the result:

```{code-cell} python
camera.logpdf(four, sources, outer=True).shape
```

## A condition with fewer batch axes lines up with the leading data axes

Four draws for each of three sources have batch shape `(3, 4)`, and evaluate against the sources
directly:

```{code-cell} python
four_each = camera.sample(sources, key=0, N=4)
camera.logpdf(four_each, sources).shape
```

The same data with the axes swapped, batch shape `(4, 3)`, fail against the same sources, although
NumPy would broadcast `(4, 3)` with `(3,)`:

```{code-cell} python
:tags: [raises-exception]

swapped = Measurement(x_hat=four_each.x_hat.T, flux_hat=four_each.flux_hat.T)
camera.logpdf(swapped, sources)
```

When the parameters have fewer batch axes than the data, astroinfer pads them with trailing
length-1 axes, so that they line up with the leading data axes. NumPy pads with leading axes
instead. The leading rule matches the layout of `sample`, which puts the sample axes `N` after
the batch axes of the parameters, so `logpdf(sample(theta, N=n), theta)` always lines up. Every
conditional distribution follows it, including population models and the distributions in
[Distributions over parameter types](user-guide/distributions.md). To pair the last data axis with
the sources, broadcast the sources to the full batch shape yourself:

```{code-cell} python
camera.logpdf(swapped, sources.broadcast((4, 3))).shape
```

## Only parameter objects are batched

A mixture whose weights depend on a fraction `f_faint` takes the fraction as a condition. Passing
one fraction per source as a plain array fails:

```{code-cell} python
:tags: [raises-exception]

faint = Source.distribution(x=dist.Uniform(-1.0, 1.0), flux=dist.TruncatedNormal(1.0, 0.2, low=0.0))
bright = Source.distribution(x=dist.Uniform(-1.0, 1.0), flux=dist.TruncatedNormal(3.0, 1.0, low=0.0))
groups = ai.mixture([faint, bright], weights=lambda f_faint: jnp.stack([f_faint, 1.0 - f_faint]))
groups.sample(key=0, f_faint=jnp.array([0.1, 0.5, 0.9]))
```

`sample` and `logpdf` batch only the arguments that are parameter objects (or dicts and
DataFrames given for an argument declared with a parameter type). A number, a plain array or an
untyped dict reaches every single call whole. Here the weight function received all three
fractions for one source and returned weights of shape `(2, 3)`. Wrap per-object values in a
parameter type to batch them:

```{code-cell} python
@parameter_dataclass
class Fraction:
    f_faint: float = parameter()

groups = ai.mixture([faint, bright], weights=lambda frac: jnp.stack([frac.f_faint, 1.0 - frac.f_faint]))
groups.sample(key=0, frac=Fraction(f_faint=jnp.array([0.1, 0.5, 0.9]))).batch_shape
```

A value shared by every object, such as a single number, needs no wrapping.

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

## Fixed fields of a structured distribution are not random

A structured distribution with a field given as a value gives $-\infty$ to any object whose field
differs from that value, even by a little:

```{code-cell} python
on_axis = Source.distribution(x=0.0, flux=dist.Uniform(0.0, 10.0))
on_axis.logpdf(Source(x=jnp.array([0.0, 0.01]), flux=3.0))
```

A field given a value is a fixed field. It is neither drawn nor given a density, and
`logpdf` checks it against the declared value with a relative tolerance of $10^{-6}$. A field
left out of `distribution(...)` that has a default in the dataclass is fixed at its default, which
is easy to miss:

```{code-cell} python
@parameter_dataclass
class Star:
    flux: float = parameter()
    distance: float = parameter(default=10.0)

Star.distribution(flux=dist.Uniform(0.0, 10.0)).fixed_paths()
```

Give the field a distribution to make it random.

## NumPyro distributions for fields with core axes need `.to_event`

A field with a core axis of four channels rejects a NumPyro distribution with four parameters:

```{code-cell} python
:tags: [raises-exception]

@parameter_dataclass
class Spectrum:
    flux: float = parameter(index=("channel",))

Spectrum.distribution(flux=dist.LogNormal(jnp.zeros(4), 1.0))
```

`dist.LogNormal(jnp.zeros(4), 1.0)` has batch shape `(4,)`: it is four independent distributions,
and its `log_prob` returns four values. A structured distribution needs one scalar log density
per field value, so a field with $k$ core axes needs a distribution with an empty batch shape and
$k$ event dimensions. `.to_event(1)` turns the batch axis into an event axis:

```{code-cell} python
spectra = Spectrum.distribution(flux=dist.LogNormal(jnp.zeros(4), 1.0).to_event(1))
spectra.sample(key=0, N=3).flux.shape
```

## A mixture picks one component for the whole object

Two groups of sources, one faint on the left and one bright on the right, mixed with equal
weights:

```{code-cell} python
left_faint = Source.distribution(x=dist.Normal(-0.5, 0.2), flux=dist.Normal(1.0, 0.2))
right_bright = Source.distribution(x=dist.Normal(0.5, 0.2), flux=dist.Normal(3.0, 0.5))
two_groups = ai.mixture([left_faint, right_bright], weights=jnp.array([0.5, 0.5]))
drawn = two_groups.sample(key=0, N=10_000)
float(jnp.corrcoef(drawn.x, drawn.flux)[0, 1])
```

Within each component, `x` and `flux` are independent, but in the mixture their correlation
coefficient is 0.87. `ai.mixture` draws one component label per object and takes every field of
that object from the same component, so a mixture of products is not a product of mixtures. When
each field should be mixed on its own, put a NumPyro mixture on the field instead:

```{code-cell} python
field_by_field = Source.distribution(
    x=dist.MixtureSameFamily(dist.Categorical(probs=jnp.array([0.5, 0.5])),
                             dist.Normal(jnp.array([-0.5, 0.5]), 0.2)),
    flux=dist.MixtureSameFamily(dist.Categorical(probs=jnp.array([0.5, 0.5])),
                                dist.Normal(jnp.array([1.0, 3.0]), jnp.array([0.2, 0.5]))),
)
drawn = field_by_field.sample(key=0, N=10_000)
float(jnp.corrcoef(drawn.x, drawn.flux)[0, 1])
```

The correlation is now close to zero, $-0.008$. The label is not a variable of the mixture: `logpdf` sums over it with `logsumexp`.

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

## `as_generative` needs a closed-form quantile or finite bounds

A structured distribution whose flux follows a Gamma distribution with a shape hyperparameter `k`
cannot be converted to the generative backend:

```{code-cell} python
:tags: [raises-exception]

from typing import Any

from numpyro.distributions import constraints

gamma_flux = Source.distribution(
    x=dist.Uniform(-1.0, 1.0), flux=lambda L: dist.Gamma(L["k"], 1.0)
).given(Any)
gamma_flux.as_generative()
```

`as_generative` draws each field by pushing a uniform number through the inverse of its CDF. It
uses NumPyro's closed-form `icdf` for a short list of distributions (Uniform, Normal, LogNormal,
Exponential, Cauchy, Logistic and the truncations of Normal, Cauchy and Logistic) and bisection on
the CDF for the rest. Bisection needs a finite interval, and the support of a Gamma distribution
is unbounded. Declare a bounded support on the field that holds nearly all of the distribution's
mass, or use a distribution from the list. The declared support is used for fields given a
function or `ai.conditional`; for a field given a NumPyro distribution directly, only the
distribution's own support counts:

```{code-cell} python
@parameter_dataclass
class BoundedSource:
    x: float = parameter()
    flux: float = parameter(support=constraints.interval(0.0, 50.0))

converted = BoundedSource.distribution(
    x=dist.Uniform(-1.0, 1.0), flux=lambda L: dist.Gamma(L["k"], 1.0)
).given(Any).as_generative()
float(converted.sample({"k": 3.0}, key=0, N=10_000).flux.mean())
```

The mean is 3.0, that of a Gamma distribution with shape 3 and rate 1. Bisection inside an interval that cuts off part
of the distribution truncates the draws without renormalizing them.

## A converted Beta field has no gradient in its shape parameters

A Beta field converts, but differentiating its draws with respect to a hyperparameter that sets
its shape fails:

```{code-cell} python
:tags: [raises-exception]

spin_like = Source.distribution(
    x=dist.Uniform(-1.0, 1.0), flux=lambda L: dist.Beta(L["a"], 2.0)
).given(Any).as_generative()
seeds = spin_like.make_seeds(0, 1000)
jax.grad(lambda a: spin_like.sample({"a": a}, seeds=seeds).flux.mean())(3.0)
```

A Beta field is drawn by bisection on its CDF, the regularized incomplete beta function, and the
gradient of the draw with respect to $\Lambda$ needs the derivative of that function with respect
to its shape parameters, which JAX does not provide. The `Beta` leaf of `astroinfer.generative`
has the same limitation. Fixed shape parameters are fine. When $\Lambda$ must set the shape, use a
distribution with a closed-form quantile, such as a `TruncatedNormal` on $[0, 1]$, or pass the
unconverted distribution to `PopulationLikelihood` with a proposal, which needs only its `logpdf`.

## Mixture weights in `as_generative` may read a field drawn earlier

A mixture field whose weights depend on another field of the same object names that field as a
condition. Here the fraction of bright sources rises with the position, but sampling the
structured distribution fails:

```{code-cell} python
:tags: [raises-exception]

@parameter_dataclass
class Position:
    x: float = parameter()

@parameter_dataclass
class Brightness:
    flux: float = parameter()

@parameter_dataclass
class Located:
    position: Position
    brightness: Brightness

def bright_fraction(position, hyper):
    w = jax.nn.sigmoid(position.x / hyper["width"])
    return jnp.stack([1.0 - w, w])

located_prior = Located.distribution(
    position=Position.distribution(x=dist.Uniform(-1.0, 1.0)),
    brightness=ai.mixture(
        [Brightness.distribution(flux=dist.Uniform(0.0, 2.0)),
         Brightness.distribution(flux=dist.Uniform(2.0, 6.0))],
        weights=bright_fraction,
        support={"flux": constraints.interval(0.0, 6.0)},
    ),
)
located_prior.sample(key=0, N=3, hyper={"width": 0.2})
```

The structured distribution is conditioned on `position` and `hyper`, the arguments of the weight
function. Its own `sample` and `logpdf` treat both as conditions you supply, and a `position`
passed this way is not the position drawn for the same object. `as_generative` instead fills a
condition whose name is a top-level field with the value drawn for that field, and passes
$\Lambda$ as the other condition:

```{code-cell} python
located = located_prior.as_generative()
draws = located.sample({"width": 0.2}, key=0, N=10_000)
right, left = draws.position.x > 0.5, draws.position.x < -0.5
float(draws.brightness.flux[right].mean()), float(draws.brightness.flux[left].mean())
```

Sources on the right have a mean flux of 4.0 and sources on the left 1.1. The density of the
converted population is the native one with the drawn field passed as the condition:

```{code-cell} python
L = {"width": 0.2}
float(jnp.max(jnp.abs(located.logpdf(draws, L)
                      - located_prior.logpdf(draws, position=draws.position, hyper=L))))
```

The field must come before the mixture in the draw order. With the mixture first, the conversion
fails:

```{code-cell} python
:tags: [raises-exception]

located_prior.with_order(["brightness", "position.x"]).as_generative()
```

## `ai.infer` traces the model's arguments

A model that sets the size of a plate from an integer argument fails under `ai.infer`:

```{code-cell} python
:tags: [raises-exception]

import numpyro

def model(num_events):
    with numpyro.plate("events", num_events):
        ai.sample("x", dist.Normal(0.0, 1.0))

ai.infer(model, 3, seed=0, num_warmup=100, num_samples=100, summary=False)
```

`ai.infer` runs NumPyro's `MCMC` with `jit_model_args=True`, the default of gravpop's sampler, so
every model argument is traced by `jax.jit`, and a traced integer cannot set a shape. Take sizes
from the shapes of array arguments, as in the [spin mixture](user-guide/numpyro-models.md#a-mass-dependent-spin-mixture),
or turn the tracing off; the model is then compiled again for every new argument value:

```{code-cell} python
post = ai.infer(model, 3, seed=0, num_warmup=100, num_samples=100, summary=False,
                mcmc_kwargs={"jit_model_args": False})
post.theta.x.shape
```

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

---
kernelspec:
  name: python3
  display_name: Python 3
---

# Quickstart

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```


[Getting started](getting-started.md) fitted a model with one parameter to one experiment. This
page goes further, to populations of objects and catalogs of posteriors, with a model that has
two parameters. The objects are point sources on the sky, such as stars. Each source has a
position $x$ along one axis of an image, in arcseconds, and a flux $f$. A camera takes one image
of the source and measures both, with Gaussian noise of standard deviation 0.1 arcsec in
position and 0.3 in flux. The parameters $\theta = (x, f)$ are a `Source`, the data
$d = (\hat x, \hat f)$ are a `Measurement`, and the measurement process is a `Camera`. The
other pages of the user guide use the same three classes.

The page assumes you know Bayesian inference. JAX is explained where it is needed.

To begin, import the libraries used on this page:

```{code-cell} python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import astroinfer as ai
from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)
```

## Parameter types

Parameters and data are described by *parameter types*: dataclasses whose fields are declared
with `parameter`. The `name` and `latex` arguments are used when the object is displayed in a
notebook.

```{code-cell} python
@parameter_dataclass(name="Source", latex=r"\theta")
class Source:
    x: float = parameter(name="Position [arcsec]", latex="x")
    flux: float = parameter(name="Flux", latex="f")

src = Source(x=0.2, flux=3.0)
src
```

A field can hold an array. The object is then a *batch* of sources, and fields broadcast against
each other like NumPy arrays:

```{code-cell} python
sources = Source(x=jnp.array([-0.5, 0.0, 0.5]), flux=2.0)
sources
```

The display marks the batch with the title $\{\theta_n\}$ and the *Batch* line
$n \in 1\ldots 3$, so `sources.batch_shape` is `(3,)`. The three sources have different positions
and share the flux 2.0, which has shape `()` and broadcasts.

Indexing selects from the batch, and a field name selects a field:

```{code-cell} python
sources[1]
```

```{code-cell} python
sources["x"]
```

Parameter types are JAX pytrees, so JAX transformations accept them and return them. The
gradient of a function of a `Source` is a `Source`:

```{code-cell} python
def brightness(s):
    return s.flux * jnp.exp(-s.x**2)

jax.grad(brightness)(src)
```

For more, see [Parameter types](user-guide/parameters.md).

## A measurement process

A measurement process describes how data $d$ are produced from parameters $\theta$, written
$\mathcal{L}(d\mid\theta)$. The data are a second parameter type:

```{code-cell} python
@parameter_dataclass(name="Measurement", latex="d")
class Measurement:
    x_hat: float = parameter(name="Measured position [arcsec]", latex=r"\hat x")
    flux_hat: float = parameter(name="Measured flux", latex=r"\hat f")
```

You subclass `MeasurementProcess`, name the two types, and write two methods for a single
$\theta$ and a single $d$: `sample_single` draws a measurement using a JAX random key, and
`logpdf_single` returns $\log \mathcal{L}(d\mid\theta)$.

```{code-cell} python
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
```

`sample` and `logpdf` apply these methods to whole batches. An integer `key` makes a draw
reproducible:

```{code-cell} python
d = camera.sample(src, key=0)
d
```

Given the batch of three sources, `sample` measures each one. With `N=4` it measures each source
four times, and the result has two batch axes, $n_0$ over the three sources and $n_1$ over the four
repeats:

```{code-cell} python
camera.sample(sources, key=1, N=4)
```

`logpdf` broadcasts in the same way. The single measurement `d` against the batch of three
sources gives one log likelihood per source:

```{code-cell} python
camera.logpdf(d, sources)
```

For more, see [Measurement processes](user-guide/measurement-processes.md).

## Parameter estimation

`PE` samples the posterior $p(\theta\mid d)$ with NUTS. It needs a prior for every field, as
numpyro distributions; other keyword arguments go to the sampler.

```{code-cell} python
priors = {"x": dist.Uniform(-1.0, 1.0), "flux": dist.Uniform(0.0, 10.0)}
posterior = camera.PE(d, priors, seed=0, as_posterior=True,
                      num_warmup=300, num_samples=1000, summary=False)
posterior
```

The result is a `PosteriorData`. Its display shows the 1000 draws as a batch of `Source` objects,
$\{\theta_{n_0,n_1}\}$, which `posterior.theta` returns. The two batch axes are the sample axes
`("chain", "draw")`, here one chain of 1000 draws. Below them are the log likelihood, the prior
density and the sampler's statistics for every draw under *Sample Statistics*, the run's metadata,
and the measurement $d$ the posterior is conditioned on. Indexing with a field name returns the draws of
that field. Their means are:

```{code-cell} python
float(posterior["x"].mean()), float(posterior["flux"].mean())
```

Without `as_posterior=True`, `PE` returns the same draws as a pandas DataFrame. For more, see
[Measurement processes](user-guide/measurement-processes.md#parameter-estimation).

## A population

A population model $\pi(\theta\mid\Lambda)$ describes how parameters are spread over many
sources, given hyperparameters $\Lambda$. It is built from one-dimensional distributions in
`astroinfer.generative`. Leaf parameters given as strings are looked up in $\Lambda$:

```{code-cell} python
from astroinfer.generative import GenerativeModel, TruncatedNormal, Uniform, product

population = GenerativeModel(
    product([Uniform("x", -1.0, 1.0), TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0)]),
    order=["x", "flux"],
)
Lambda = {"mu": 3.0, "sigma": 1.0}
draws = population.sample(Lambda, key=jax.random.key(1), shape=(200,))
pop_sources = Source.from_dict(draws)
pop_sources
```

`pop_sources` is a batch of 200 sources, one for each draw from the population.

Measuring every source simulates a survey:

```{code-cell} python
data = camera.sample(pop_sources, key=2)
data
```

For more, see [Population models](user-guide/population-models.md).

## Population inference

`population_inference` samples the hyperparameters $\Lambda$ given a batch of measurements of
independent sources. For each source it averages $\mathcal{L}(d_i\mid\theta)$ over `N` draws from the
population. The result is a `PosteriorData` whose `theta` holds the draws of $\Lambda$, and
`summary()` gives their means, standard deviations and sampling diagnostics:

```{code-cell} python
hyper_priors = {"mu": dist.Uniform(0.0, 6.0), "sigma": dist.Uniform(0.2, 3.0)}
hyper_posterior = camera.population_inference(data, population, hyper_priors, N=500, seed=0,
                                              num_warmup=300, num_samples=1000, summary=False)
hyper_posterior.summary()
```

The true values were $\mu = 3$ and $\sigma = 1$. Real surveys only detect some sources, and
the likelihood then needs a selection term. For more, see
[Population inference](user-guide/population-inference.md).

## Catalogs

A `Catalog` stores one posterior per event and exports them as arrays for population analyses.
`RawSamples` sets which columns are exported and how many draws per event:

```{code-cell} python
catalog = ai.Catalog(raw=ai.RawSamples(columns=["x", "flux"], N=500))
for i in range(3):
    post = camera.PE(data[i], priors, seed=i, as_posterior=True,
                     num_warmup=300, num_samples=1000, summary=False)
    catalog.add_event(f"src_{i}", post)

{name: array.shape for name, array in catalog.dataproduct().items()}
```

`catalog.save(path)` writes the catalog to a folder and `ai.Catalog.load(path)` reads it back.
For more, see [Catalogs](user-guide/catalogs.md).

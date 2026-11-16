---
kernelspec:
  name: python3
  display_name: Python 3
---

# How astroinfer works

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

This page explains the mechanisms behind the parts of astroinfer you use every day: how methods
written for one object run on batches, why parameter objects work with JAX transformations, how
population draws are made differentiable, and what the population likelihood computes. Each
section shows the mechanism directly, using the example from the [quickstart](quickstart.md).

```{code-cell} python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import astroinfer as ai
from astroinfer import MeasurementProcess, parameter, parameter_dataclass
from astroinfer.generative import GenerativeModel, TruncatedNormal, Uniform, product

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

## Parameter objects are pytrees

JAX transformations such as `jax.jit`, `jax.grad` and `jax.vmap` work on *pytrees*: nested
containers whose leaves are arrays. `@parameter_dataclass` registers each class as a pytree whose
leaves are its fields, in the order they are declared:

```{code-cell} python
leaves, treedef = jax.tree_util.tree_flatten(src)
leaves
```

Rebuilding from the leaves gives an object of the original type, which is why a transformation
that takes a `Source` can return one:

```{code-cell} python
rebuilt = jax.tree_util.tree_unflatten(treedef, [jnp.array(1.0), jnp.array(5.0)])
rebuilt
```

The same mechanism lets `jax.tree_util.tree_map` apply a function to every field:

```{code-cell} python
jax.tree_util.tree_map(lambda value: 10 * value, src)
```

## Batching is repeated `vmap`

`sample_single` and `logpdf_single` are written for one object. `sample` and `logpdf` first
broadcast their inputs to a common batch shape, then wrap the single-object method in
`jax.vmap` once for each batch axis. Inside the method, every field therefore has the shape of
a single object. You can see this by printing a shape inside `logpdf_single`:

```{code-cell} python
@dataclass
class LoudCamera(Camera):
    def logpdf_single(self, d, theta):
        print("inside logpdf_single: theta.x has shape", jnp.shape(theta.x))
        return super().logpdf_single(d, theta)

loud = LoudCamera()
d = loud.sample(src, key=0)
loud.logpdf(d, sources.reshape_batch((3, 1)).broadcast((3, 2))).shape
```

The batch had shape `(3, 2)`, but the method saw a scalar, and it ran once rather than six
times. `vmap` traces the function once with abstract values and turns the result into a
vectorised computation.

`@batched_method` uses the same mechanism for methods of parameter types: it wraps the method in
one `vmap` per batch axis of `self`, and passes the other arguments unchanged to every element.

## Random keys

JAX random functions are deterministic functions of a *key*. `sample` turns its key into one
independent key per batch element with `jax.random.split`, so each source gets its own noise
and the result does not depend on how the work is scheduled. An integer `key=` is converted
with `jax.random.key`, so the same integer always gives the same draws:

```{code-cell} python
first = camera.sample(sources, key=7).x_hat
second = camera.sample(sources, key=7).x_hat
first, second
```

When no key is given, the measurement process splits its own key, stored in `camera.key`, keeps
one half for next time and uses the other. Repeated calls therefore give different noise:

```{code-cell} python
camera.seed(0)
camera.sample(src).x_hat, camera.sample(src).x_hat
```

## Population draws are transforms of uniform numbers

`GenerativeModel.sample` draws every variable by inverse transform sampling. It starts from
uniform numbers $u\in(0,1)$, one array per variable, and passes each through the inverse CDF of
that variable's distribution, $\theta = F^{-1}(u;\Lambda)$. With `key=` the uniforms are
generated for you; with `seeds=` you supply them. For a uniform leaf on $(-1, 1)$ the inverse
CDF is $-1 + 2u$:

```{code-cell} python
population = GenerativeModel(
    product([Uniform("x", -1.0, 1.0), TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0)]),
    order=["x", "flux"],
)
u = {"x": jnp.array([0.1, 0.5, 0.9]), "flux": jnp.array([0.1, 0.5, 0.9])}
draws = population.sample({"mu": 3.0, "sigma": 1.0}, seeds=u)
draws["x"]
```

With the uniforms held fixed, each draw is a smooth function of the hyperparameters. Raising
$\mu$ from 3.0 to 3.1 moves every flux draw by a small amount rather than producing new random
values:

```{code-cell} python
(population.sample({"mu": 3.0, "sigma": 1.0}, seeds=u)["flux"],
 population.sample({"mu": 3.1, "sigma": 1.0}, seeds=u)["flux"])
```

This is what makes Monte Carlo estimates built from population draws differentiable, so that
NUTS can use their gradients. The technique is known as *common random numbers*.

`TruncatedNormal` and `Uniform` have closed-form inverse CDFs. For a distribution without one,
such as `Beta` or a `mixture`, `sample` finds $F^{-1}(u)$ by bisection. Bisection is not
differentiated step by step: the gradient of the root of $F(\theta;\Lambda) = u$ comes from the
implicit function theorem, $\partial\theta/\partial\Lambda =
-(\partial F/\partial\Lambda)/\pi(\theta\mid\Lambda)$.

## The population likelihood is an average of weights

For events $i = 1\ldots E$ with data $d_i$, the population likelihood is

$$
\mathcal L(\Lambda) = \prod_{i=1}^{E} Z_i(\Lambda),\qquad
Z_i(\Lambda) = \int \mathcal{L}(d_i\mid\theta)\,\pi(\theta\mid\Lambda)\,d\theta .
$$

`PopulationLikelihood` estimates each $Z_i$ as the mean of $N$ weights. Without a proposal the
weights are $w_{ij} = \mathcal{L}(d_i\mid\theta_{ij})$, with $\theta_{ij}$ drawn from the population using
uniforms fixed by `seed`. `log_weights(Lambda)` returns $\log w_{ij}$ with shape `(E, N)`:

```{code-cell} python
Lambda = {"mu": 3.0, "sigma": 1.0}
data = camera.sample(Source.from_dict(population.sample(Lambda, key=jax.random.key(1),
                                                        shape=(20,))), key=2)
likelihood = ai.PopulationLikelihood(camera, data, population, N=400, seed=0)
log_w = likelihood.log_weights(Lambda)
log_w.shape
```

The per-event estimate is the log of the mean weight, computed stably with `logsumexp`, and
`logpdf` is the sum over events:

```{code-cell} python
by_hand = jax.scipy.special.logsumexp(log_w, axis=1) - jnp.log(400)
jnp.allclose(by_hand, likelihood.event_logpdf(Lambda)), jnp.allclose(by_hand.sum(),
                                                                      likelihood.logpdf(Lambda))
```

The effective sample size of event $i$ is $(\sum_j w_{ij})^2 / \sum_j w_{ij}^2$. It equals $N$
when all weights are equal and approaches 1 when one draw dominates:

```{code-cell} python
w = jnp.exp(log_w)
neff_by_hand = w.sum(axis=1) ** 2 / (w**2).sum(axis=1)
jnp.allclose(neff_by_hand, likelihood.event_logpdf_and_neff(Lambda)[1])
```

With a proposal $q(\theta\mid d_i)$, the draws come from $q$ and the weights become
$\mathcal{L}\,\pi/q$. Subclasses can replace `log_weights` to use a different weight.

## What `PE` computes

`PE` wraps the measurement process and the data in a `FixedDataLikelihood`, whose `logpdf` is
the measurement's `logpdf` with the data held fixed, and samples it with NUTS through gravpop.
It then adds columns to the draws: the log prior of each field, evaluated with the numpyro
distributions you passed, their sum as the `prior` density, and any derived parameters.

```{code-cell} python
posterior = camera.PE(d, {"x": dist.Uniform(-1.0, 1.0), "flux": dist.Uniform(0.0, 10.0)},
                      seed=0, num_warmup=300, num_samples=1000, summary=False)
float(posterior["prior"].iloc[0]), 0.5 * 0.1
```

Both priors are uniform, with densities 1/2 and 1/10, so every draw has prior density 0.05.

## Catalog data products are resampled per event

`Catalog.dataproduct()` builds each event's arrays separately and stacks them. For the raw
representation, an event's draws are resampled to `N` with a random generator whose seed is
computed from the catalog's seed and the event's name. The `prior` array holds
`posterior.marginal_prior(columns)` at the same indices. Because the seed depends only on the
name, an event's draws do not change when other events are added or requested:

```{code-cell} python
catalog = ai.Catalog(raw=ai.RawSamples(columns=["x", "flux"], N=200))
priors = {"x": dist.Uniform(-1.0, 1.0), "flux": dist.Uniform(0.0, 10.0)}
for i in range(2):
    catalog.add_event(f"src_{i}", camera.PE(data[i], priors, seed=i, as_posterior=True,
                                            num_warmup=300, num_samples=1000, summary=False))

both = catalog.dataproduct()
alone = catalog.dataproduct(eventnames=["src_1"])
jnp.array_equal(both["x"][1], alone["x"][0])
```

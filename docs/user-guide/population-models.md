---
kernelspec:
  name: python3
  display_name: Python 3
---

# Population models

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

A population model is a distribution $\pi(\theta\mid\Lambda)$ over the parameters of many
sources, controlled by hyperparameters $\Lambda$. astroinfer builds population models from the
pieces in `astroinfer.generative`, and every model can both draw samples and evaluate its log
density.

This page uses the imports and the `Source` type from the [quickstart](../quickstart.md):

```{code-cell} python
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import astroinfer as ai
from astroinfer import parameter, parameter_dataclass
from astroinfer.generative import (
    Beta,
    Distribution,
    GenerativeModel,
    TruncatedNormal,
    Uniform,
    conditional,
    mixture,
    product,
)

jax.config.update("jax_enable_x64", True)

@parameter_dataclass(name="Source", latex=r"\theta")
class Source:
    x: float = parameter(name="Position [arcsec]", latex="x")
    flux: float = parameter(name="Flux", latex="f")
```

## Leaves

A *leaf* is a one-dimensional distribution for one named variable. Three leaves are built in:

| leaf | distribution | closed-form quantile |
|---|---|---|
| `Uniform(var, a, b)` | uniform on $[a, b]$ | yes |
| `TruncatedNormal(var, mu, sigma, a, b)` | normal with mean `mu` and width `sigma`, truncated to $[a, b]$ | yes |
| `Beta(var, alpha, beta)` | beta distribution on $[0, 1]$ | no |

The parameters of a leaf (`mu`, `sigma`, `alpha`, `beta`, and for `TruncatedNormal` also the
bounds) are *parameter specs*. A spec is one of three things:

| spec | example | value |
|---|---|---|
| a number | `sigma=0.5` | fixed |
| a string | `mu="mu"` | the hyperparameter `Lambda["mu"]` |
| `conditional(fn, on=)` | see [Conditional parameters](#conditional-parameters) | `fn(x, Lambda)`, computed from variables drawn earlier |

## Building a model

`GenerativeModel(root, order)` wraps a distribution and fixes the order in which its variables
are drawn. Here the flux follows a normal distribution truncated to $[0, 20]$, with its mean and
width given by hyperparameters:

```{code-cell} python
flux_model = GenerativeModel(TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0), order=["flux"])
Lambda = {"mu": 3.0, "sigma": 1.0}
```

`sample(Lambda, key=, shape=)` returns a plain dict with one array per variable:

```{code-cell} python
flux_draws = flux_model.sample(Lambda, key=jax.random.key(0), shape=(4,))
flux_draws
```

`log_prob(values, Lambda)` evaluates the log density of such a dict, one value per draw:

```{code-cell} python
flux_model.log_prob(flux_draws, Lambda)
```

Values outside a leaf's bounds get a log density of $-\infty$:

```{code-cell} python
flux_model.log_prob({"flux": jnp.array([-1.0, 3.0])}, Lambda)
```

## Several variables

`product(models)` combines distributions of different variables. A uniform position and a
truncated-normal flux give a model for a whole `Source`:

```{code-cell} python
population = GenerativeModel(
    product([Uniform("x", -1.0, 1.0), TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0)]),
    order=["x", "flux"],
)
draws = population.sample(Lambda, key=jax.random.key(1), shape=(1000,))
sorted(draws)
```

The variable names match the fields of `Source`, so the dict converts directly to a batch of
1000 sources:

```{code-cell} python
Source.from_dict(draws)
```

`log_prob` of a product is the sum of the log densities of its variables:

```{code-cell} python
first = {k: v[:3] for k, v in draws.items()}
population.log_prob(first, Lambda) - flux_model.log_prob({"flux": first["flux"]}, Lambda)
```

The difference is $\log(1/2)$, the log density of $U(-1, 1)$.

## Mixtures

`mixture(models, weights)` combines distributions of the same variables. `weights` names one
hyperparameter for each component except the last, whose weight is one minus the others. Here
30% of sources come from a faint group:

```{code-cell} python
two_groups = GenerativeModel(
    mixture([TruncatedNormal("flux", 1.0, 0.2, 0.0, 20.0),
             TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0)], weights=["f_faint"]),
    order=["flux"],
)
mixed = two_groups.sample({**Lambda, "f_faint": 0.3}, key=jax.random.key(2), shape=(10_000,))
float(jnp.mean(mixed["flux"] < 1.6))
```

Almost all of the faint group lies below 1.6, together with a few sources from the bright group.

Give exactly one weight name fewer than there are components. Otherwise the mixture fails when
it is built:

```{code-cell} python
:tags: [raises-exception]

mixture([TruncatedNormal("flux", 1.0, 0.2, 0.0, 20.0),
         TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0)], weights=["f_faint", "f_bright"])
```

## Conditional parameters

`conditional(fn, on=)` makes a spec depend on variables drawn earlier. `fn(x, L)` receives the
dict of variables drawn so far and the hyperparameters; `on` names the variables it reads. Here
sources further from the centre are fainter:

```{code-cell} python
faint_edges = GenerativeModel(
    product([
        Uniform("x", -1.0, 1.0),
        TruncatedNormal("flux", conditional(lambda x, L: L["mu"] - x["x"] ** 2, on="x"),
                        "sigma", 0.0, 20.0),
    ]),
    order=["x", "flux"],
)
edge_draws = faint_edges.sample(Lambda, key=jax.random.key(3), shape=(10_000,))
centre = jnp.abs(edge_draws["x"]) < 0.3
float(edge_draws["flux"][centre].mean()), float(edge_draws["flux"][~centre].mean())
```

A conditional may only read variables that come before it in `order`. `GenerativeModel` checks
this when it is built:

```{code-cell} python
:tags: [raises-exception]

GenerativeModel(faint_edges.root, order=["flux", "x"])
```

Every variable of the model has to appear in `order`:

```{code-cell} python
:tags: [raises-exception]

GenerativeModel(faint_edges.root, order=["x"])
```

## External variables

A variable that the model reads but does not draw is declared in `external=`. Its values are
passed at sampling time with `conditioned_on=`, and they appear in the returned dict alongside
the drawn variables:

```{code-cell} python
flux_given_x = GenerativeModel(
    TruncatedNormal("flux", conditional(lambda x, L: L["mu"] - x["x"] ** 2, on="x"),
                    "sigma", 0.0, 20.0),
    order=["flux"], external=["x"],
)
flux_given_x.sample(Lambda, key=jax.random.key(4), shape=(3,),
                    conditioned_on={"x": jnp.array([-1.0, 0.0, 1.0])})
```

Forgetting the external values raises an error:

```{code-cell} python
:tags: [raises-exception]

flux_given_x.sample(Lambda, key=jax.random.key(4), shape=(3,))
```

`log_prob` reads external variables from the same dict as the drawn ones.

## Ways to sample

`sample` accepts the randomness in three forms:

| call | uniform numbers used |
|---|---|
| `sample(Lambda, key=key, shape=shape)` | drawn from `key`, one array of shape `shape` per variable |
| `sample(Lambda, shape=shape)` | drawn from a fresh random key on every call |
| `sample(Lambda, seeds={var: u})` | the arrays `u` you pass, each in $(0, 1)$ |

Each variable is drawn by passing a uniform number through the inverse of its conditional
cumulative distribution function, $\theta = F^{-1}(u;\Lambda)$. The same `key`, or the same
`seeds`, therefore gives the same uniform numbers whatever $\Lambda$ is.

## Differentiating through samples

With fixed seeds, the draws are a smooth function of $\Lambda$, so quantities computed from
samples can be differentiated. The mean flux of a population moves almost one-for-one with
`mu`:

```{code-cell} python
seeds = {"x": jax.random.uniform(jax.random.key(5), (5000,)),
         "flux": jax.random.uniform(jax.random.key(6), (5000,))}

def mean_flux(L):
    return population.sample(L, seeds=seeds)["flux"].mean()

jax.grad(mean_flux)(Lambda)
```

Leaves without a closed-form quantile, such as `Beta`, and every `mixture`, are inverted
numerically by bisection. The gradient is still exact: astroinfer differentiates the root of
$F(\theta;\Lambda) = u$ with the implicit function theorem rather than the bisection loop. The
mean flux of the two-group mixture above is $f_\mathrm{faint}\cdot 1 + (1-f_\mathrm{faint})\cdot
3.0$, so its derivative with respect to $f_\mathrm{faint}$ is close to $-2$:

```{code-cell} python
mix_seeds = {"flux": jax.random.uniform(jax.random.key(7), (20_000,))}

def mixture_mean(L):
    return two_groups.sample(L, seeds=mix_seeds)["flux"].mean()

jax.grad(mixture_mean)({**Lambda, "f_faint": 0.3})
```

`Beta` is the exception: JAX cannot differentiate the incomplete beta function with respect to
its shape parameters, so do not differentiate a `Beta` leaf whose shape parameters are
hyperparameters:

```{code-cell} python
:tags: [raises-exception]

beta_model = GenerativeModel(Beta("p", "a", "b"), order=["p"])
beta_seeds = {"p": jax.random.uniform(jax.random.key(7), (5000,))}
jax.grad(lambda L: beta_model.sample(L, seeds=beta_seeds)["p"].mean())({"a": 2.0, "b": 5.0})
```

Bisection is slower than a closed-form quantile. Prefer leaves with a closed-form quantile in
code that runs many times, such as a population likelihood:

```{code-cell} python
tn_model = GenerativeModel(TruncatedNormal("p", "mu", "sigma", 0.0, 1.0), order=["p"])

def time_sampling(model, L, repeats=5):
    run = jax.jit(lambda L: model.sample(L, key=jax.random.key(0), shape=(100_000,)))
    jax.block_until_ready(run(L))
    start = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(run(L))
    return (time.perf_counter() - start) / repeats * 1e3

t_tn = time_sampling(tn_model, {"mu": 0.3, "sigma": 0.1})
t_beta = time_sampling(beta_model, {"a": 2.0, "b": 5.0})
print(f"TruncatedNormal: {t_tn:.2f} ms, Beta: {t_beta:.2f} ms per 100,000 draws")
```

## Writing your own distribution

Subclass `Distribution` to add a leaf. Every method receives the variable name `var`, the dict
`x` of variables drawn so far, and the hyperparameters `L`, so any leaf can be conditional.

| member | returns |
|---|---|
| `var_names` | the variables this distribution describes, e.g. `["distance"]` |
| `cond_pdf(var, val, x, L)` | the density at `val`, zero outside the support |
| `cond_cdf(var, val, x, L)` | the cumulative distribution function at `val` |
| `bounds(var)` | the support `(a, b)`, used by bisection and by `log_prob` |
| `cond_ppf(var, u, x, L)` | optional: the closed-form quantile. The default returns `None`, which falls back to bisection |
| `deps(var)` | optional: the names of earlier variables this distribution reads, used to check `order` |

Write the methods with `jax.numpy` so that they can be differentiated. This power law,
$\pi(r)\propto r^{\gamma}$ on $[r_\mathrm{lo}, r_\mathrm{hi}]$, takes $\gamma$ as a number or a
hyperparameter name:

```{code-cell} python
class PowerLaw(Distribution):
    def __init__(self, var, index, lo, hi):
        self.var_names, self.var = [var], var
        self.index, self.lo, self.hi = index, lo, hi

    def _k(self, L):
        k = (L[self.index] if isinstance(self.index, str) else self.index) + 1.0
        return jnp.where(jnp.abs(k) < 1e-6, 1e-6, k)

    def cond_pdf(self, var, val, x, L):
        k = self._k(L)
        p = k * val ** (k - 1.0) / (self.hi ** k - self.lo ** k)
        return jnp.where((val >= self.lo) & (val <= self.hi), p, 0.0)

    def cond_cdf(self, var, val, x, L):
        k = self._k(L)
        v = jnp.clip(val, self.lo, self.hi)
        return (v ** k - self.lo ** k) / (self.hi ** k - self.lo ** k)

    def cond_ppf(self, var, u, x, L):
        k = self._k(L)
        return (self.lo ** k + u * (self.hi ** k - self.lo ** k)) ** (1.0 / k)

    def bounds(self, var):
        return self.lo, self.hi
```

With $\gamma = 2$ the distances are spread uniformly in volume between 0.5 and 2:

```{code-cell} python
volume = GenerativeModel(PowerLaw("distance", 2.0, 0.5, 2.0), order=["distance"])
r = volume.sample({}, key=jax.random.key(8), shape=(100_000,))["distance"]
float(r.min()), float(r.max()), float(jnp.mean(r < 1.0))
```

The fraction below 1 is $(1^3 - 0.5^3)/(2^3 - 0.5^3) = 0.111$.

Without `cond_ppf`, the same leaf is sampled by bisection. The draws agree to the precision of
the bisection:

```{code-cell} python
class PowerLawNoQuantile(PowerLaw):
    def cond_ppf(self, var, u, x, L):
        return None

volume_slow = GenerativeModel(PowerLawNoQuantile("distance", 2.0, 0.5, 2.0), order=["distance"])
r_slow = volume_slow.sample({}, key=jax.random.key(8), shape=(100_000,))["distance"]
float(jnp.abs(r - r_slow).max())
```

## Population models as conditional distributions

`PopulationLikelihood` wraps a `GenerativeModel` in a `PopulationModel`, which you can also make
yourself with `ai.PopulationModel.from_generative(model, parameter_type)`. A `PopulationModel` is
a conditional distribution with $\theta$ as the output and $\Lambda$ as the condition (see
[Distributions over parameter types](distributions.md)). Its `sample` returns parameter objects
and takes the number of draws as `N=`, like every other conditional distribution, while
`GenerativeModel.sample` returns a dict and keeps `shape=`:

```{code-cell} python
source_population = ai.PopulationModel.from_generative(population, Source)
source_population.sample(Lambda, key=0, N=3)
```

`PopulationModel.sample` still accepts `shape=`, with a `DeprecationWarning`.

## Converting a structured distribution

A [structured distribution](distributions.md#structured-distributions) describes a population
field by field, with NumPyro distributions. It draws through NumPyro, and astroinfer does not treat
those draws as differentiable in $\Lambda$. `prior.as_generative(hyperparameter_type=None,
external=(), conditioned_on=None)` converts it to the generative backend of this page. The result
is a `StructuredGenerativePopulation`, a `PopulationModel` whose draws are made from seeds by
inverse transform sampling, as for a `GenerativeModel`, and are therefore differentiable in
$\Lambda$. This is the population above written as a structured distribution, with a type for its
hyperparameters:

```{code-cell} python
import numpyro.distributions as dist
from numpyro.distributions import constraints

@parameter_dataclass(name="Flux hyperparameters", latex=r"\Lambda")
class FluxHyper:
    mu: float = parameter(latex=r"\mu")
    sigma: float = parameter(latex=r"\sigma")

source_prior = Source.distribution(
    x=dist.Uniform(-1.0, 1.0),
    flux=lambda L: dist.TruncatedNormal(L.mu, L.sigma, low=0.0, high=20.0),
).given(FluxHyper)
converted = source_prior.as_generative()
converted
```

Each random field becomes one generative variable, named by its path, in the distribution's draw
order, and `model` is the underlying `GenerativeModel`:

```{code-cell} python
converted.order, type(converted.model).__name__
```

The hyperparameter type comes from `.given`; pass `hyperparameter_type=` for a distribution that
has none.

These parts of a structured distribution convert:

- a field given a NumPyro distribution, a function of the conditions, or `ai.conditional(fn,
  on=...)`, provided the distribution is scalar and continuous. The paths in `on` become the
  variables the leaf reads, so `GenerativeModel` checks the order as usual;
- nested fields, whose fields keep their dotted paths;
- fixed fields, which are not variables and are inserted into every draw;
- a mixture of structured distributions that draw the same random fields and fix the same values,
  as the whole distribution or as a nested field. It becomes a `generative.WeightedMixture`, which
  draws no component label, like `mixture` above.

Fields with core axes, discrete distributions, and nested fields given any other conditional
distribution do not convert. `external=` and `conditioned_on=` work as for `GenerativeModel` (see
[External variables](#external-variables)); a population with external variables has no `logpdf`.

### Seeds and gradients

`make_seeds(key, N)` returns the seeds, one array of uniform numbers per variable.
`sample(Lambda, seeds=seeds)` draws from them and returns parameter objects, with fixed fields
broadcast to the sample shape; `sample(Lambda, key=key, N=N)` makes the seeds from `key`. The same
seeds and the same $\Lambda$ give the same draws. Given the same seeds, the converted population
draws the same sources as the hand-built `population` from [Several variables](#several-variables):

```{code-cell} python
seeds = converted.make_seeds(0, 5000)
converted_draws = converted.sample(Lambda, seeds=seeds)
hand_built = population.sample(Lambda, seeds=seeds)
float(jnp.max(jnp.abs(converted_draws.flux - hand_built["flux"])))
```

The largest difference is of order $10^{-14}$, from two implementations of the truncated normal
quantile. A dict $\Lambda$ is converted to `FluxHyper`. The draws are differentiable in $\Lambda$,
so common random numbers work as before; the mean flux moves almost one-for-one with $\mu$:

```{code-cell} python
def converted_mean_flux(mu):
    return converted.sample(FluxHyper(mu=mu, sigma=1.0), seeds=seeds).flux.mean()

jax.grad(converted_mean_flux)(3.0)
```

`logpdf` is `GenerativeModel.log_prob` of the converted model, and agrees with the structured
distribution's own `logpdf`:

```{code-cell} python
float(jnp.max(jnp.abs(converted.logpdf(converted_draws, Lambda)
                      - source_prior.logpdf(converted_draws, Lambda))))
```

### How each field is drawn

A field is drawn with NumPyro's closed-form inverse CDF when astroinfer can differentiate it. The
list is Uniform, Normal, LogNormal, Exponential, Cauchy and Logistic, and the truncations of
Normal, Cauchy and Logistic, such as TruncatedNormal. Every other distribution, including Beta and
Gamma, is drawn by bisection on its CDF, which is slower (see
[Differentiating through samples](#differentiating-through-samples)) and needs finite bounds. The
bounds come from the distribution's support when it is a fixed interval, as for Beta. For a field
given a function or `ai.conditional`, they are intersected with the field's declared
`parameter(support=...)`; a field given a NumPyro distribution directly uses only the
distribution's own support.

`as_generative` checks every field before it converts any, and raises one error that lists each
path that cannot be converted and the reason:

```{code-cell} python
:tags: [raises-exception]

@parameter_dataclass
class Galaxy:
    flux: float = parameter()
    n_stars: int = parameter()
    spectrum: float = parameter(index=("channel",))

Galaxy.distribution(
    flux=lambda L: dist.Gamma(L.mu, L.sigma),
    n_stars=dist.Poisson(100.0),
    spectrum=dist.Normal(jnp.zeros(3), 1.0).to_event(1),
).given(FluxHyper).as_generative()
```

The Gamma distribution has no listed inverse CDF and an unbounded support. Declaring a bounded
support for the field, which is given a function here, gives bisection its bounds:

```{code-cell} python
@parameter_dataclass
class BoundedFlux:
    flux: float = parameter(support=constraints.interval(0.0, 50.0))

gamma_population = BoundedFlux.distribution(
    flux=lambda L: dist.Gamma(L.mu, L.sigma)
).given(FluxHyper).as_generative()
float(gamma_population.sample(Lambda, key=0, N=5000).flux.mean())
```

The mean of 3.0 is that of a Gamma distribution with shape 3 and rate 1. The declared support
should hold nearly all of the distribution's mass, because bisection inside a narrower interval
truncates the draws without renormalizing them.

A Beta field is drawn by bisection on $[0, 1]$. As for the `Beta` leaf above, JAX cannot
differentiate the incomplete beta function with respect to the shape parameters, so a Beta field
whose shape parameters depend on $\Lambda$ has no gradient in $\Lambda$:

```{code-cell} python
:tags: [raises-exception]

beta_population = Source.distribution(
    x=dist.Uniform(-1.0, 1.0), flux=lambda L: dist.Beta(L.mu, L.sigma)
).given(FluxHyper).as_generative()
jax.grad(lambda mu: beta_population.sample(FluxHyper(mu=mu, sigma=2.0), seeds=seeds).flux.mean())(3.0)
```

### Mixtures

The two-group population of [Mixtures](#mixtures), written with structured components, converts
to a `WeightedMixture`. Each component needs `.given` when its fields read the conditions, and the
components need a common `support=` because the support of a function field is not known before
it is called:

```{code-cell} python
@parameter_dataclass(name="Two-group hyperparameters", latex=r"\Lambda")
class GroupHyper:
    mu: float
    sigma: float
    f_faint: float

faint = Source.distribution(x=dist.Uniform(-1.0, 1.0),
                            flux=dist.TruncatedNormal(1.0, 0.2, low=0.0, high=20.0))
bright = Source.distribution(
    x=dist.Uniform(-1.0, 1.0),
    flux=lambda L: dist.TruncatedNormal(L.mu, L.sigma, low=0.0, high=20.0),
).given(GroupHyper)
two_group_prior = ai.mixture(
    [faint, bright],
    weights=lambda L: jnp.stack([L.f_faint, 1.0 - L.f_faint]),
    parameter_type=GroupHyper,
    support={"flux": constraints.interval(0.0, 20.0)},
)
two_group_population = two_group_prior.as_generative()
type(two_group_population.model.root).__name__
```

No component label is drawn, so the draws are smooth in the weights. As for `two_groups` above,
the derivative of the mean flux with respect to $f_\mathrm{faint}$ is close to $-2$:

```{code-cell} python
group_seeds = two_group_population.make_seeds(1, 20_000)

def group_mean_flux(f_faint):
    L = GroupHyper(mu=3.0, sigma=1.0, f_faint=f_faint)
    return two_group_population.sample(L, seeds=group_seeds).flux.mean()

jax.grad(group_mean_flux)(0.3)
```

A mixture is drawn by bisection on its CDF, so every field of the components needs finite bounds.
Inside a structured distribution, the weights of a mixture field may also read a field drawn
before it; [Sharp bits](../sharp-bits.md) explains how the converted population fills it in.

For more, see [Population inference](population-inference.md), which uses these models, and
tutorial [05](https://github.com/Potatoasad/astroinfer/blob/main/tutorials/05_coin_mint.py).

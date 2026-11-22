---
kernelspec:
  name: python3
  display_name: Python 3
---

# Measurement processes

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

A *measurement process* is the model $\mathcal{L}(d\mid\theta)$ of how data $d$ are produced from
parameters $\theta$. You write it for a single $\theta$ and a single $d$, and astroinfer applies
it to batches, simulates data, evaluates the likelihood and runs parameter estimation. This page
uses the imports and types from the [quickstart](../quickstart.md):

```{code-cell} python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd

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
```

## Writing a measurement process

Subclass `MeasurementProcess` and fill in these members:

| member | required | what it is |
|---|---|---|
| `parameter_type` | yes | the parameter type of $\theta$ |
| `data_type` | yes | the parameter type of $d$ |
| `sample_single(self, theta, key)` | yes | one draw $d\sim \mathcal{L}(\cdot\mid\theta)$ for a single $\theta$, using the JAX random key `key` |
| `logpdf_single(self, d, theta)` | yes | the scalar $\log \mathcal{L}(d\mid\theta)$ for a single $d$ and $\theta$ |
| `derived_parameters(self, theta)` | no | extra quantities computed from $\theta$, added to the output of `PE` |
| `display_name` | no | title of the notebook display (default: the class name) |
| `_display_equations(self)` | no | LaTeX lines shown under *Model* in the display |
| `_display_config(self)` | no | `{label: value}` shown under *Configuration* (default: the dataclass fields) |

Write `sample_single` and `logpdf_single` with `jax.numpy` and `jax.random`, because they run
under `jax.vmap` and are differentiated by the sampler. Settings, such as noise levels, are
ordinary dataclass fields:

```{code-cell} python
@dataclass
class Camera(MeasurementProcess):
    sigma_x: float = 0.1
    sigma_flux: float = 0.3

    display_name = "Camera"
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

    def derived_parameters(self, theta):
        return {"magnitude": -2.5 * np.log10(theta["flux"])}

    def _display_equations(self):
        return [r"\hat x \sim \mathcal{N}(x, \sigma_x^2)",
                r"\hat f \sim \mathcal{N}(f, \sigma_f^2)"]

camera = Camera()
camera
```

The repr shows the settings. In a notebook, the display also shows the model equations and the
available methods.

`parameter_names` lists the fields of the parameter type:

```{code-cell} python
camera.parameter_names
```

`MeasurementProcess` is a subclass of `ConditionalDistribution`, the base class of every
distribution over parameter types in astroinfer, with $d$ as the output and $\theta$ as the
condition. Batching, keys, `.condition` and the display come from that class, and
`MeasurementProcess` adds parameter estimation, population inference and posterior predictive
draws. [Distributions over parameter types](distributions.md) describes the base class.

## Simulating data

`sample(theta)` calls `sample_single` once for every element of the batch of `theta`, with a
separate key for each:

```{code-cell} python
src = Source(x=0.2, flux=3.0)
sources = Source(x=jnp.array([-0.5, 0.0, 0.5]), flux=2.0)
d = camera.sample(src, key=0)
d
```

The batch shape of the result is the batch shape of `theta`. `N=` adds a trailing batch axis of
`N` independent draws for each element:

```{code-cell} python
(camera.sample(sources, key=1).batch_shape,
 camera.sample(src, key=1, N=4).batch_shape,
 camera.sample(sources, key=1, N=4).batch_shape)
```

`theta` may also be a dict or a DataFrame with one column per field:

```{code-cell} python
camera.sample({"x": np.array([0.0, 0.1]), "flux": np.array([1.0, 2.0])}, key=1).batch_shape
```

The `key` argument decides where the randomness comes from:

| `key=` | randomness | changes `camera.key` |
|---|---|---|
| `None` (default) | the camera's own key, which is then advanced | yes |
| an integer | `jax.random.key(key)` | no |
| a JAX key | that key | no |
| a NumPy `Generator` | a JAX key seeded from the generator | no (the generator advances) |

Every measurement process carries a JAX key in `key`. Without `key=`, repeated calls give
different noise. `seed(n)` resets the camera's key to `jax.random.key(n)`:

```{code-cell} python
camera.seed(5)
first, second = camera.sample(src), camera.sample(src)
camera.seed(5)
again = camera.sample(src)
float(first.x_hat), float(second.x_hat), float(again.x_hat)
```

An integer key gives the same draw every time:

```{code-cell} python
float(camera.sample(src, key=7).x_hat), float(camera.sample(src, key=7).x_hat)
```

## Evaluating the likelihood

`logpdf(d, theta)` returns $\log \mathcal{L}(d\mid\theta)$. The batch shapes of `d` and `theta` broadcast
against each other (with the rule for unequal numbers of batch axes described below), so one
measurement against three sources gives three values:

```{code-cell} python
camera.logpdf(d, src), camera.logpdf(d, sources)
```

With equal batch shapes, element $i$ of `d` is paired with element $i$ of `theta`:

```{code-cell} python
three_measurements = camera.sample(sources, key=2)
camera.logpdf(three_measurements, sources)
```

When `d` has more batch axes than `theta`, the axes of `theta` line up with the leading axes of
`d`. Data drawn with `N=` can therefore be evaluated against the parameters they were drawn from:

```{code-cell} python
camera.logpdf(camera.sample(sources, key=4, N=5), sources).shape
```

Row $i$ of the `(3, 5)` result holds the five draws for source $i$.

With `outer=True`, every element of `theta` is evaluated against every element of `d`. The axes
of `theta` come first in the result:

```{code-cell} python
two_measurements = camera.sample(sources[:2], key=3)
camera.logpdf(two_measurements, sources, outer=True).shape
```

`d` and `theta` may also be dicts or DataFrames.

## Derived parameters

`derived_parameters(theta)` receives a dict of NumPy arrays, one per field, and returns a dict of
extra quantities. It is applied to tables of draws, so write it with array operations rather
than methods of the parameter type. `with_derived(table)` adds the derived columns to any table
that has the parameter columns:

```{code-cell} python
camera.with_derived(pd.DataFrame({"x": [0.0, 0.1], "flux": [1.0, 10.0]}))
```

Columns that are already in the table are not overwritten.

## Parameter estimation

`PE(d, priors)` samples the posterior $p(\theta\mid d)\propto \mathcal{L}(d\mid\theta)\,\pi(\theta)$ with
NUTS. `priors` holds a numpyro distribution for every field of the parameter type:

```{code-cell} python
priors = {"x": dist.Uniform(-1.0, 1.0), "flux": dist.Uniform(0.0, 10.0)}
table = camera.PE(d, priors, seed=0, num_warmup=300, num_samples=1000, summary=False)
table.head(3)
```

A missing prior raises an error:

```{code-cell} python
:tags: [raises-exception]

camera.PE(d, {"x": dist.Uniform(-1.0, 1.0)})
```

Each row of the result is one draw, with these columns:

| column | contents |
|---|---|
| one per field | the draw of $\theta$ |
| `log_likelihood` | $\log \mathcal{L}(d\mid\theta)$ at the draw |
| `log_prior[<field>]` | the log prior density of that field |
| `prior` | the prior density of the draw, the product of the fields' prior densities |
| `accept_prob`, `diverging`, `energy`, `num_steps`, `potential_energy`, `adapt_state.step_size` | the NUTS statistics of the draw |
| `chain` | the chain the draw came from |
| one per derived parameter | the output of `derived_parameters` |

A prior can also be a gravpop prior block. A block with `var_names` and a `log_prob(theta)` method
gives the joint density of those fields, stored as one column `log_prior[<field>, <field>, ...]`;
`marginal_prior` can then use it only for all of those fields together. Blocks without `log_prob`,
such as `gravpop.Deterministic`, set variables but do not count as a prior for any field.

Keyword arguments other than `seed` and `as_posterior` are passed to the sampler. They override
the class attribute `sampler_kwargs`:

```{code-cell} python
MeasurementProcess.sampler_kwargs
```

After a run, the sampler itself is stored as `camera.sampler`:

```{code-cell} python
type(camera.sampler).__name__
```

`ai.parameter_estimation(measurement, d, priors, ...)` is the same function as the method.

## Posterior data

With `as_posterior=True`, `PE` returns a `PosteriorData` instead of a DataFrame:

```{code-cell} python
posterior = camera.PE(d, priors, seed=0, as_posterior=True,
                      num_warmup=300, num_samples=1000, summary=False)
posterior
```

A `PosteriorData` has these parts:

| attribute | contents |
|---|---|
| `theta` | the draws, as the parameter type with one batch axis per sample axis |
| `sample_dims` | the names of the sample axes, `("chain", "draw")` for NUTS |
| `sample_stats` | per-draw inference quantities: `log_likelihood`, `prior`, `log_prior[...]` and the sampler's statistics |
| `extras` | other per-draw columns, such as derived parameters |
| `metadata` | information about the run |
| `observed_data` | the data $d$ the posterior is conditioned on |
| `local` | for joint population inference only, the parameters of every observation (see [Population inference](population-inference.md#joint-inference)) |

```{code-cell} python
posterior.sizes, list(posterior.sample_stats), list(posterior.extras), posterior.metadata
```

```{code-cell} python
posterior.observed_data
```

`parameter_type` is the type of `theta`, and `variable_names` lists the fields followed by the
extras:

```{code-cell} python
posterior.parameter_type.__name__, posterior.variable_names
```

### Sample axes

The sample axes are named by `sample_dims`, and `sizes` maps each name to its length. NUTS gives a
chain axis and a draw axis. The keyword `mcmc_kwargs` is passed on to numpyro's `MCMC`, so it sets
the number of chains:

```{code-cell} python
chains = camera.PE(d, priors, seed=0, as_posterior=True,
                   num_warmup=300, num_samples=1000, summary=False,
                   mcmc_kwargs={"num_chains": 4, "chain_method": "sequential",
                                "progress_bar": False})
chains.sizes
```

`summary()` gives the mean, standard deviation and 90% central interval of every field, with its
effective sample size and split $\hat R$. $\hat R$ compares chains, so it is NaN for a posterior
without a chain axis. The number of divergent transitions is in `attrs`:

```{code-cell} python
table = chains.summary()
table.attrs["divergences"], table
```

A field with core axes, such as one value per star, takes one row in the summary, with its
smallest effective sample size and largest $\hat R$; `summary(expand=True)` gives a row per element.

`stack_samples()` merges the sample axes into one axis of draws, chain after chain, and keeps the
chain of every draw in `sample_stats["chain"]`. Plots, tables and catalogs use this form:

```{code-cell} python
stacked = chains.stack_samples()
stacked.sizes, stacked.sample_stats["chain"][[0, 999, 1000, 3999]]
```

With `arviz` installed, `to_arviz()` converts a posterior for ArviZ's plots and diagnostics. The
dims of each variable are the sample dims followed by the field's `index`.

### Indexing

A string returns that column from whichever part holds it. Integers, slices and 1-D boolean or
integer arrays index the sample axes in order, and every per-draw array is indexed the same way.
An integer removes its axis from `sample_dims`; the other indices keep it. `isel` indexes by name:

```{code-cell} python
posterior["flux"].shape, posterior["magnitude"].shape, posterior["log_likelihood"].shape
```

```{code-cell} python
posterior[0].sample_dims, posterior[:, :100].sizes, posterior.isel(draw=slice(None, 100)).sizes
```

A boolean mask indexes one axis, so a selection of draws by value is made on the stacked
posterior:

```{code-cell} python
stacked = posterior.stack_samples()
stacked[stacked["flux"] > 3.0].sizes
```

`variables(names)` returns several columns as a dict:

```{code-cell} python
list(posterior.variables(["x", "magnitude", "prior"]))
```

### Priors of a subset of fields

`marginal_prior(columns)` returns the prior density of every draw for the given fields only,
built from the `log_prior[...]` columns. For all fields it returns the `prior` column:

```{code-cell} python
stacked.marginal_prior(["x"])[:3], stacked.marginal_prior(["flux"])[:3]
```

A derived column has no stored prior. Asking for it without all the fields raises an error:

```{code-cell} python
:tags: [raises-exception]

posterior.marginal_prior(["magnitude"])
```

### Content hash

`content_hash()` is a SHA-256 hash of the parameter type and every per-draw array. Any change to
the draws changes it:

```{code-cell} python
posterior.content_hash()[:16], posterior[:, :500].content_hash()[:16]
```

[Catalogs](catalogs.md) use the hash to detect when a posterior has been replaced.

### Converting to and from DataFrames

`to_dataframe()` returns the table that `PE` returns by default. It requires one draw axis and
one value per draw in every column:

```{code-cell} python
posterior.to_dataframe().columns.tolist()
```

`ai.PosteriorData.from_dataframe(table, parameter_type)` turns a table into a `PosteriorData`.
Columns named after fields go to `theta`. `prior`, `log_prior[...]`, `log_likelihood` and a few
other recognised names go to `sample_stats`, and every other column goes to `extras`. This also
works for draws from another sampler:

```{code-cell} python
round_trip = ai.PosteriorData.from_dataframe(posterior.to_dataframe(), Source)
round_trip.content_hash() == posterior.content_hash()
```

`metadata=` and `observed_data=` can be passed to `from_dataframe` as keyword arguments.

### Creating one directly

Only `theta` is required. Every per-draw array must start with the batch shape of `theta`:

```{code-cell} python
by_hand = ai.PosteriorData(
    theta=Source(x=jnp.zeros(5), flux=jnp.linspace(1.0, 2.0, 5)),
    sample_stats={"prior": jnp.full(5, 0.05)},
    metadata={"origin": "written by hand"},
)
by_hand
```

```{code-cell} python
:tags: [raises-exception]

ai.PosteriorData(theta=Source(x=jnp.zeros(5), flux=1.0), extras={"score": jnp.ones(4)})
```

Names may not appear in more than one part. For examples of `PosteriorData` in use, see
[the doctor's office tutorial](https://github.com/Potatoasad/astroinfer/blob/main/tutorials/04_doctors_office.py).

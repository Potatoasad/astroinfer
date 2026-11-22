---
kernelspec:
  name: python3
  display_name: Python 3
---

# NumPyro models

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

The distributions of [Distributions over parameter types](distributions.md) can be used inside a
NumPyro model. `ai.sample` creates the sample sites for one typed object and returns the object,
`ai.factor` adds a term to the log density, and `ai.infer` samples the posterior of the model with
NUTS and returns a `PosteriorData`. You write the model function yourself, so it can have any
structure NumPyro allows: shared hyperparameters, plates over events, observed data and
deterministic quantities.

`MeasurementProcess.PE`, `population_inference` and `joint_population_inference` do not use these
functions. They build their model from a dict of priors and run it with gravpop's sampler, as
described in [Measurement processes](measurement-processes.md#parameter-estimation) and
[Population inference](population-inference.md). `ai.infer` runs NumPyro through astroinfer's own
runner, `run_mcmc`, which takes the same keyword names and has the same defaults as gravpop's
sampler (see [Sampler settings](#sampler-settings)). Use `ai.infer` when the model you need is not
one of those three.

The examples use the binaries of the distributions page, with two hyperparameters for the
distribution of the primary mass:

```{code-cell} python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.infer import Predictive

import astroinfer as ai
from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)

@parameter_dataclass(name="Masses", latex="m")
class Masses:
    m1: float = parameter(name="Primary mass", latex="m_1", support=constraints.positive)
    q: float = parameter(name="Mass ratio", latex="q", support=constraints.unit_interval)

@parameter_dataclass(name="Spins", latex=r"\chi")
class Spins:
    chi1: float = parameter(name="Primary spin", latex=r"\chi_1", support=constraints.unit_interval)
    chi2: float = parameter(name="Secondary spin", latex=r"\chi_2", support=constraints.unit_interval)

@parameter_dataclass(name="Binary", latex=r"\theta")
class Binary:
    masses: Masses
    spins: Spins

@parameter_dataclass(name="Hyperparameters", latex=r"\Lambda")
class Hyper:
    mu_m: float = parameter(latex=r"\mu_m")
    sigma_m: float = parameter(latex=r"\sigma_m")
```

## Sample sites

`ai.sample(name, distribution, *conditions, obs=None, **conditions)` takes the place of
`numpyro.sample`. The sites it creates depend on the distribution:

| distribution | sites | returns |
|---|---|---|
| a NumPyro distribution | one site, `name` | the value of the site |
| a structured distribution | one site per random field, named `name.<path>` | an object of the output type |
| any other conditional distribution | one site, `name`, holding a whole object | an object of the output type |

A NumPyro distribution passes straight through to `numpyro.sample` and takes no conditions.

### Structured distributions

A prior over `Binary` with the secondary spin fixed at zero:

```{code-cell} python
binary_prior = Binary.distribution(
    masses=Masses.distribution(m1=dist.Uniform(5.0, 100.0), q=dist.Uniform(0.1, 1.0)),
    spins=Spins.distribution(chi1=dist.Beta(2.0, 5.0), chi2=0.0),
)

def prior_model():
    return ai.sample("bbh", binary_prior)
```

NumPyro's `handlers.trace` records the sites of one run of the model:

```{code-cell} python
trace = handlers.trace(handlers.seed(prior_model, rng_seed=0)).get_trace()
{name: site["type"] for name, site in trace.items()}
```

The three random fields are sample sites, named by their paths under `bbh` and created in the
distribution's draw order, `binary_prior.order`. A conditional field reads the values drawn at
earlier sites. The fixed field $\chi_2$ is not sampled; it is recorded as a deterministic site so
that the object can be rebuilt from the draws. The model returns a `Binary`:

```{code-cell} python
with handlers.seed(rng_seed=1):
    bbh = prior_model()
bbh
```

### Joint sites

A conditional distribution that is not structured is sampled as one unit. Here the secondary spin
depends on the primary spin through a density written as a class:

```{code-cell} python
class CorrelatedSpins(ai.ConditionalDistribution):
    data_type = Spins

    def secondary(self, chi1):
        return dist.Beta(1.0 + 10.0 * chi1, 5.0)

    def sample_single(self, key):
        key1, key2 = jax.random.split(key)
        chi1 = dist.Beta(2.0, 5.0).sample(key1)
        return Spins(chi1=chi1, chi2=self.secondary(chi1).sample(key2))

    def logpdf_single(self, spins):
        return dist.Beta(2.0, 5.0).log_prob(spins.chi1) + self.secondary(spins.chi1).log_prob(spins.chi2)

def spin_model():
    return ai.sample("spins", CorrelatedSpins())
```

`ai.sample` gives it a *joint site*, one sample site whose value is the whole object packed into a
vector with one entry per scalar field:

```{code-cell} python
site = handlers.trace(handlers.seed(spin_model, rng_seed=0)).get_trace()["spins"]
site["value"].shape, site["fn"].support
```

The vector has length 2, for $\chi_1$ and $\chi_2$. The log density of the site is the
distribution's `logpdf`, so a density that does not split into one factor per field stays whole.

NUTS works in unconstrained coordinates, so a latent joint site needs the support of every field.
The supports come from the output type's `parameter(support=...)`, here the unit interval for
both spins. A field without a declared support is taken to be real, so declare one for every
bounded field; otherwise the sampler proposes values where the density is zero. A distribution
whose output is not a parameter type, such as `typing.Any`, needs a `support` attribute instead: a
NumPyro constraint for every scalar, or a dict from path to constraint. The fields must be
continuous; a distribution over integers can only be observed.

## Conditions

Conditions follow the distribution's schema, as they do for `sample` and `logpdf`: positionally
for a single type or a tuple, and as keywords for a mapping. Here the primary mass of the
population depends on the hyperparameters:

```{code-cell} python
population = Binary.distribution(
    masses=Masses.distribution(
        m1=lambda L: dist.TruncatedNormal(L.mu_m, L.sigma_m, low=5.0, high=100.0),
        q=dist.Uniform(0.1, 1.0),
    ),
    spins=Spins.distribution(chi1=dist.Beta(2.0, 5.0), chi2=dist.Beta(2.0, 5.0)),
).given(Hyper)

def population_model():
    mu_m = ai.sample("mu_m", dist.Uniform(10.0, 80.0))
    sigma_m = ai.sample("sigma_m", dist.Uniform(1.0, 20.0))
    return ai.sample("bbh", population, Hyper(mu_m=mu_m, sigma_m=sigma_m))
```

The hyperparameters are drawn at plain NumPyro sites and wrapped in `Hyper`, the condition type of
`population`. The random key comes from NumPyro, so a positional argument after the conditions
raises an error:

```{code-cell} python
:tags: [raises-exception]

with handlers.seed(rng_seed=0):
    ai.sample("bbh", population, Hyper(mu_m=30.0, sigma_m=5.0), jax.random.key(0))
```

A mapping schema takes its conditions as keywords, as in the [spin mixture](#a-mass-dependent-spin-mixture)
below.

## Observed sites

With `obs=`, the sites are observed. They add the distribution's log density at `obs` to the model
and are not sampled. A measurement process is a conditional distribution, so its likelihood enters
a model as an observed site. This detector measures the primary mass and the primary spin of each
binary:

```{code-cell} python
@parameter_dataclass(name="Observation", latex="d")
class Observation:
    m1_hat: float = parameter(latex=r"\hat m_1")
    chi1_hat: float = parameter(latex=r"\hat\chi_1")

class Detector(MeasurementProcess):
    parameter_type = Binary
    data_type = Observation

    def sample_single(self, theta, key):
        key1, key2 = jax.random.split(key)
        return Observation(m1_hat=theta.masses.m1 + 3.0 * jax.random.normal(key1),
                           chi1_hat=theta.spins.chi1 + 0.1 * jax.random.normal(key2))

    def logpdf_single(self, d, theta):
        return (dist.Normal(theta.masses.m1, 3.0).log_prob(d.m1_hat)
                + dist.Normal(theta.spins.chi1, 0.1).log_prob(d.chi1_hat))

detector = Detector()
```

We simulate 30 binaries from the population and measure each one:

```{code-cell} python
truth = Hyper(mu_m=35.0, sigma_m=5.0)
data = detector.sample(population.sample(truth, key=0, N=30), key=1)
data.batch_shape
```

The hierarchical model draws the hyperparameters once and every binary inside `numpyro.plate`. The
measurement then compares each binary with its own data:

```{code-cell} python
def hierarchical_model(data):
    mu_m = ai.sample("mu_m", dist.Uniform(10.0, 80.0))
    sigma_m = ai.sample("sigma_m", dist.Uniform(1.0, 20.0))
    with numpyro.plate("events", data.batch_shape[0]):
        bbh = ai.sample("bbh", population, Hyper(mu_m=mu_m, sigma_m=sigma_m))
        ai.sample("d", detector, bbh, obs=data)
    numpyro.deterministic("m2", bbh.masses.m1 * bbh.masses.q)

trace = handlers.trace(handlers.seed(hierarchical_model, rng_seed=0)).get_trace(data)
{name: (site["type"], jnp.shape(site["value"]), site.get("is_observed"))
 for name, site in trace.items() if site["type"] != "plate"}
```

Every field of `bbh` has one value per event, shape `(30,)`. The scalar hyperparameters reach
every event whole, by the [batching rules](distributions.md#batching). The measurement is a joint
site, `d`, observed at the packed data, with one row of two values per event. For a structured
distribution, each field's site is observed at the matching field of `obs`.

## Extra terms in the log density

`ai.factor(name, value)` adds `value` to the log density of the model, through `numpyro.factor`.
Inside a plate, NumPyro sums a batched value over the plate. One use is the product of two
densities over the same object (see
[Products of densities over the same object](distributions.md#products-of-densities-over-the-same-object)).
Here a uniform prior on the masses is combined with a second constraint on the same masses:

```{code-cell} python
mass_prior = Masses.distribution(m1=dist.Uniform(5.0, 100.0), q=dist.Uniform(0.1, 1.0))
constraint = Masses.distribution(m1=dist.Normal(35.0, 10.0), q=dist.Beta(5.0, 2.0))

def constrained_model():
    masses = ai.sample("masses", mass_prior)
    ai.factor("mass_constraint", constraint.logpdf(masses))

constrained = ai.infer(constrained_model, seed=0, num_warmup=300, num_samples=500, summary=False)
float(constrained.theta.masses.m1.mean()), float(constrained.theta.masses.q.mean())
```

The posterior means are 35.6 and 0.72, close to the means of the constraint, 35 and 0.71, because
the uniform prior is flat where the constraint has most of its mass.

## Running inference

`ai.infer(model, *args, method="nuts", seed=None, **kwargs)` samples the posterior of the model.
Keywords that are [sampler settings](#sampler-settings) go to the runner; all other arguments go
to the model:

```{code-cell} python
posterior = ai.infer(hierarchical_model, data, seed=0, num_warmup=300, num_samples=300,
                     summary=False)
posterior
```

The `theta` of the result has a type built for this model, with one field per *root*, the name
given to `ai.sample` for a latent site. A typed object is a nested field of its own type, and a
plain NumPyro site is an array field:

```{code-cell} python
posterior.parameter_names
```

The sample axes come first, as for `PE`. Axes that a root has beyond the sample axes, such as the
plate, become core axes named after the plate:

```{code-cell} python
posterior.sample_dims, posterior.theta.bbh.masses.m1.shape, type(posterior.theta.bbh.masses).field_specs()["m1"].index
```

Here that is one chain of 300 draws for 30 events. The hyperparameters are recovered within their
uncertainty; the true values are $\mu_m = 35$ and $\sigma_m = 5$:

```{code-cell} python
posterior.summary(["mu_m", "sigma_m"])
```

The posterior mean of $\mu_m$ is 33.0 with a standard deviation of 1.2, and that of $\sigma_m$ is
5.5. Observed sites are left out of `theta`. Deterministic sites go to `extras`, except names of
sampler statistics such as `log_likelihood`, which go to `sample_stats` together with the NUTS
statistics:

```{code-cell} python
list(posterior.extras), list(posterior.sample_stats)
```

The result of the runner is kept as `posterior.metadata["run"]`. Its `get_samples()` returns the
draws of every site, keyed by site name, which is the form `Predictive` takes (see
[Predictive draws](#predictive-draws)).

Every root must be a valid Python identifier, since it becomes a field name. The sites of a
structured root, such as `bbh.masses.m1`, are not roots themselves.

## A mass-dependent spin mixture

The spins of a binary may depend on its masses. In this population, a binary has small spins or
large spins, and the fraction with large spins rises with the primary mass, with a steepness
$\alpha$. The weights of the mixture are a function of the masses and of $\alpha$, as in
[Weights that depend on the conditions](distributions.md#weights-that-depend-on-the-conditions):

```{code-cell} python
small = Spins.distribution(chi1=dist.Beta(2.0, 8.0), chi2=dist.Beta(2.0, 8.0))
large = Spins.distribution(chi1=dist.Beta(8.0, 2.0), chi2=dist.Beta(8.0, 2.0))

def large_spin_fraction(masses, alpha):
    w = jax.nn.sigmoid(alpha * (masses.m1 - 40.0))
    return jnp.stack([1.0 - w, w])

spins_by_mass = ai.mixture([small, large], weights=large_spin_fraction)
```

The data are 40 binaries simulated with $\alpha = 0.3$ and measured by the detector above:

```{code-cell} python
masses_true = mass_prior.sample(key=2, N=40)
spins_true = spins_by_mass.sample(key=3, masses=masses_true, alpha=0.3)
spin_data = detector.sample(Binary(masses=masses_true, spins=spins_true), key=4)
```

The model passes the masses and $\alpha$ to the mixture as keywords, the names of its conditions:

```{code-cell} python
def mixture_model(data=None, num_events=None):
    E = data.batch_shape[0] if data is not None else num_events
    alpha = ai.sample("alpha", dist.HalfNormal(1.0))
    with numpyro.plate("events", E):
        masses = ai.sample("masses", mass_prior)
        spins = ai.sample("spins", spins_by_mass, masses=masses, alpha=alpha)
        ai.sample("d", detector, Binary(masses=masses, spins=spins), obs=data)

trace = handlers.trace(handlers.seed(mixture_model, rng_seed=0)).get_trace(spin_data)
[name for name, site in trace.items() if site["type"] == "sample"]
```

The label that says which component a binary came from is not a variable of the model. The mixture
is one joint site per event, `spins`, whose log density sums over the label with `logsumexp`. NUTS
therefore samples only continuous variables, and the gradient with respect to $\alpha$ passes
through the weights. The model takes the number of events from the data, for a reason given in
[Predictive draws](#predictive-draws).

```{code-cell} python
spin_posterior = ai.infer(mixture_model, spin_data, seed=0, num_warmup=300, num_samples=300,
                          summary=False)
spin_posterior.summary(["alpha"])
```

The posterior of $\alpha$ has a mean of 1.08 and a 90% interval from 0.24 to 1.90, which contains
the true value of 0.3; 40 events constrain the steepness only loosely. The run has no divergent
transitions:

```{code-cell} python
int(spin_posterior.sample_stats["diverging"].sum())
```

## Predictive draws

`numpyro.infer.Predictive` runs a model forward. For a joint site, it draws from the distribution's
own `sample`, so the draws of the mixture pick one component for the whole object. It returns the
draws keyed by site name, with joint sites packed; `unpack` turns them back into typed objects,
one per root:

```{code-cell} python
from astroinfer.modeling import unpack

prior_draws = Predictive(mixture_model, num_samples=500)(jax.random.PRNGKey(0), num_events=40)
prior_values = unpack(mixture_model, prior_draws, num_events=40)
{name: type(value).__name__ for name, value in prior_values.items()}
```

`unpack(model, samples, *args, **kwargs)` traces the model once with the given arguments to find its
roots, and puts the sample axes first. With `data=None`, the site `d` is latent, so it holds
simulated data, an `Observation` with batch shape `(500, 40)`:

```{code-cell} python
prior_values["d"].batch_shape
```

Posterior predictive draws condition `Predictive` on the posterior samples:

```{code-cell} python
posterior_samples = spin_posterior.metadata["run"].get_samples()
replicated = Predictive(mixture_model, posterior_samples=posterior_samples,
                        return_sites=["d"])(jax.random.PRNGKey(1), num_events=40)
d_rep = unpack(mixture_model, replicated, num_events=40)["d"]
float(jnp.mean(d_rep.chi1_hat > 0.5)), float(jnp.mean(spin_data.chi1_hat > 0.5))
```

The replicated data have a measured primary spin above 0.5 in 54% of binaries, close to the 55% of
the observed data.

`ai.infer` passes the model arguments through `jax.jit` (NumPyro's `jit_model_args=True`), so an
integer argument becomes a traced value and cannot set the size of a plate. `mixture_model`
therefore takes the number of events from the shape of the data, and uses `num_events` only for
`Predictive`, which does not trace its arguments. To pass static arguments to `ai.infer`, set
`mcmc_kwargs={"jit_model_args": False}`; the model is then compiled again for every new value.

## Sampler settings

`ai.infer` and `run_mcmc` take these settings:

| setting | default | meaning |
|---|---|---|
| `method` | `"nuts"` | `"nuts"`, or `"nested"` for NumPyro's nested sampler |
| `seed` | `None` | an integer seed; `None` picks a random one |
| `num_warmup` | 1000 | NUTS warmup steps per chain |
| `num_samples` | 2000 | draws per chain |
| `num_chains` | 1 | number of chains |
| `target_accept_prob` | 0.7 | NUTS target acceptance probability |
| `max_tree_depth` | 6 | largest NUTS tree depth |
| `summary` | `True` | print NumPyro's summary table after the run |
| `sampler_kwargs` | `None` | further arguments of `NUTS` (`dense_mass`, `init_strategy`, ...), or of `NestedSampler` |
| `mcmc_kwargs` | `None` | further arguments of `MCMC` (`chain_method`, `progress_bar`, ...) |

The defaults of `target_accept_prob` and `max_tree_depth` are gravpop's, not NumPyro's (0.8 and
10). With several chains on one CPU device, pass `mcmc_kwargs={"chain_method": "sequential"}` or
NumPyro will warn that it cannot run them in parallel.

`method="nested"` needs the `jaxns` package. It ignores the NUTS and `MCMC` settings, takes
`num_samples` draws from the nested sampling result, and gives a posterior with one sample axis,
`draw`.

`run_mcmc` runs a model without building a `PosteriorData`. It is imported from
`astroinfer.inference.runner`, takes the same arguments as `ai.infer`, and returns a `RunResult`
with the draws of every site:

```{code-cell} python
from astroinfer.inference.runner import run_mcmc

run = run_mcmc(spin_model, seed=0, num_warmup=300, num_samples=300, summary=False)
run, run.sites["spins"].shape
```

`sites` keeps the chain and draw axes, here `(1, 300, 2)`, with the joint site still packed;
`get_samples()` merges the chains, and `unpack` gives the typed objects. `sample_stats` holds the
NUTS statistics and `sampler` the NumPyro `MCMC` object.

---
kernelspec:
  name: python3
  display_name: Python 3
---

# Population inference

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

Population inference samples the hyperparameters $\Lambda$ of a population model given
measurements of many independent sources. astroinfer computes the likelihood of $\Lambda$ with
`PopulationLikelihood`, corrects it for detection by subclassing, and can importance-sample the
per-source integrals with a `Proposal`.

This page uses the `Source`, `Measurement` and `Camera` classes and the `population` model from
the [quickstart](../quickstart.md):

```{code-cell} python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd

import astroinfer as ai
from astroinfer import MeasurementProcess, parameter, parameter_dataclass
from astroinfer.detection import DetectionTrigger, ThresholdTrigger
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
population = GenerativeModel(
    product([Uniform("x", -1.0, 1.0), TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0)]),
    order=["x", "flux"],
)
Lambda_true = {"mu": 3.0, "sigma": 1.0}
```

We simulate 1000 sources from the population and measure each one:

```{code-cell} python
sources = Source.from_dict(population.sample(Lambda_true, key=jax.random.key(0), shape=(1000,)))
data = camera.sample(sources, key=1)
data
```

## The population likelihood

The sources are independent, so the likelihood of $\Lambda$ is a product over sources. Each
factor is the source's *evidence* under the population, with its own parameters integrated out:

$$
\mathcal L(\Lambda) = \prod_{i=1}^{E} Z_i(\Lambda), \qquad
Z_i(\Lambda) = \int \mathcal{L}(d_i\mid\theta)\,\pi(\theta\mid\Lambda)\,\mathrm d\theta
\approx \frac{1}{N}\sum_{j=1}^{N} \mathcal{L}(d_i\mid\theta_{ij}), \quad \theta_{ij}\sim\pi(\theta\mid\Lambda).
$$

`PopulationLikelihood` computes this Monte Carlo estimate. Its arguments are:

| argument | meaning |
|---|---|
| `measurement` | the `MeasurementProcess`; its `logpdf` gives $\mathcal{L}(d_i\mid\theta_{ij})$ |
| `data` | all measurements, one object of the measurement's `data_type` with batch shape `(E,)` |
| `pop_model` | a `GenerativeModel` that draws every field of the measurement's `parameter_type`, or a `PopulationModel` or other conditional distribution over that type (see [Structured populations](#structured-populations)) |
| `N` | the number of Monte Carlo draws per source (default 2000) |
| `seed` | the seed of the fixed uniform numbers behind the draws (default 42) |
| `conditioned_on` | values of external variables of `pop_model`, if it has any |
| `proposal` | an optional `Proposal`; see [Proposals](#proposals) |

The uniform numbers behind the draws $\theta_{ij}$ are drawn once, when the likelihood is built,
and reused for every $\Lambda$. The estimate is therefore a smooth, differentiable function of
$\Lambda$, which NUTS needs.

Here we use the first 200 measurements:

```{code-cell} python
likelihood = ai.PopulationLikelihood(camera, data[:200], population, N=500, seed=0)
likelihood.logpdf(Lambda_true)
```

`logpdf(Lambda)` returns $\log\hat{\mathcal L}(\Lambda)$. It is differentiable:

```{code-cell} python
jax.grad(likelihood.logpdf)(Lambda_true)
```

The other methods expose the pieces of the estimate:

| method | returns |
|---|---|
| `event_logpdf(Lambda)` | $\log\hat Z_i$, shape `(E,)` |
| `event_logpdf_and_neff(Lambda)` | $\log\hat Z_i$ and the effective number of draws per source, both shape `(E,)` |
| `log_weights(Lambda)` | the log weights $\log w_{ij}$, shape `(E, N)`; $\hat Z_i$ is their mean |
| `population_draws(Lambda)` | the draws $\theta_{ij}$, a parameter object with batch shape `(E, N)` |
| `log_likelihood_draws(Lambda)` | $\log \mathcal{L}(d_i\mid\theta_{ij})$ at the population draws, shape `(E, N)` |

```{code-cell} python
likelihood.population_draws(Lambda_true).batch_shape, likelihood.log_weights(Lambda_true).shape
```

## Checking the effective number of draws

Each draw contributes with weight $w_{ij} = \mathcal{L}(d_i\mid\theta_{ij})$. If a few draws carry most of
the weight, $\hat Z_i$ is noisy. The effective number of draws,
$N_\mathrm{eff} = (\sum_j w_{ij})^2 / \sum_j w_{ij}^2$, measures this:

```{code-cell} python
log_z, neff = likelihood.event_logpdf_and_neff(Lambda_true)
float(jnp.median(neff)), float(neff.min())
```

The median source has about 27 effective draws out of 500, but the worst has about one.

Check $N_\mathrm{eff}$ before sampling. When it is close to 1 for some sources, $\log\hat Z_i$ is
biased low, and the bias depends on $\Lambda$. Increase `N` or use a [proposal](#proposals).
Every call to `logpdf` also stores a summary in `diagnostics`, which the sampler records for
each draw:

```{code-cell} python
likelihood.logpdf(Lambda_true)
likelihood.diagnostics
```

## Sampling the hyperparameters

`ai.run_sampler(likelihood, priors, **kwargs)` samples $\Lambda$ with NUTS and returns the
sampler. Any object with a `logpdf(Lambda)` method can be passed. The draws are in the sampler's
`samples` DataFrame, together with the recorded diagnostics:

```{code-cell} python
hyper_priors = {"mu": dist.Uniform(0.0, 6.0), "sigma": dist.Uniform(0.2, 3.0)}
sampler = ai.run_sampler(likelihood, hyper_priors, seed=0,
                         num_warmup=300, num_samples=1000, summary=False)
sampler.samples[["mu", "sigma", "N_eff_min"]].describe().loc[["mean", "std", "min"]]
```

`measurement.population_inference(data, pop_model, priors, N=, seed=, ...)` builds the
`PopulationLikelihood` and calls `run_sampler` in one step. It also accepts `proposal=` and
`likelihood_cls=`, a subclass of `PopulationLikelihood` to build instead. The same operation is
available as the function `ai.population_inference(measurement, ...)`, just as `PE` is available
as `ai.parameter_estimation(measurement, ...)`.

It returns a `PosteriorData` whose `theta` holds the draws of $\Lambda$, with the Monte Carlo
diagnostics among its sample stats. A population model that declares a `hyperparameter_type`
gets $\Lambda$ as that type; otherwise the type is built from the sampled names.

```{code-cell} python
hyper_posterior = camera.population_inference(data[:200], population, hyper_priors, N=500,
                                              seed=0, num_warmup=300, num_samples=1000,
                                              summary=False)
hyper_posterior.summary()
```

## Detection triggers

Real surveys only record some sources. A *detection trigger* gives the probability that a
measurement is detected, $P(\mathrm{det}\mid d)$. It reads the data, not the parameters.

`ThresholdTrigger(threshold, statistic=)` detects a measurement when one data field is above the
threshold. `detect(d, rng)` draws the outcome for every measurement in a batch and returns a
boolean mask, which can index the data:

```{code-cell} python
trigger = ThresholdTrigger(2.5, statistic="flux_hat")
detected = trigger.detect(data, np.random.default_rng(0))
detected_data = data[detected]
int(detected.sum()), detected_data.batch_shape
```

To write another rule, subclass `DetectionTrigger`:

| method | returns |
|---|---|
| `detection_probability(self, d)` | the detection probability of each measurement in the batch |
| `selection_statistic(self, d)` | optional: the number detection is based on, recorded in injection tables |

A trigger works on a whole batch at once and may use plain NumPy, because detection is never
differentiated. `d` can be a parameter object, a dict or a DataFrame; `d["flux_hat"]` works for
all three. This trigger raises the detection probability smoothly around the threshold:

```{code-cell} python
class SoftFluxLimit(DetectionTrigger):
    def __init__(self, threshold, width):
        self.threshold, self.width = float(threshold), float(width)

    def selection_statistic(self, d):
        return np.asarray(d["flux_hat"], dtype=np.float64)

    def detection_probability(self, d):
        return 1.0 / (1.0 + np.exp(-(self.selection_statistic(d) - self.threshold) / self.width))

soft = SoftFluxLimit(2.5, 0.2)
soft.detection_probability(data[:4]).round(3)
```

## Injections

`injections(measurement, draws, rng)` simulates a set of parameter draws through the
measurement process and the trigger. `draws` is a DataFrame with one row per draw and a
`prior` column holding the density $g(\theta)$ the draws were taken from. Here $g$ is a broader
version of the population:

```{code-cell} python
Lambda_g = {"mu": 3.0, "sigma": 2.0}
g_draws = {k: np.asarray(v) for k, v in
           population.sample(Lambda_g, key=jax.random.key(2), shape=(20_000,)).items()}
draw_table = pd.DataFrame(g_draws)
draw_table["prior"] = np.exp(np.asarray(population.log_prob(g_draws, Lambda_g)))

injections = trigger.injections(camera, draw_table, np.random.default_rng(3))
list(injections.columns)
```

The table adds the measured data, one `observed_<field>` column per data field, and the
selection statistic, detection probability and outcome of every draw:

```{code-cell} python
float(injections["detected"].mean())
```

The data type must have scalar fields. A data field with a core axis cannot be stored as a
column, and `injections` fails with `ValueError: Expected a 1D array, got an array with shape
(n, k)`.

## Correcting for detection

Analysing only the detected sources as if they were a fair sample biases the result towards
bright sources. The likelihood of detected data divides each factor by the fraction of the
population that would be detected:

$$
\mathcal L(\Lambda) = \prod_{i=1}^{E}\frac{Z_i(\Lambda)}{\alpha(\Lambda)}, \qquad
\hat\alpha(\Lambda) = \frac{1}{N_\mathrm{inj}}\sum_{j\in\text{found}}
\frac{\pi(\theta_j\mid\Lambda)}{g(\theta_j)} .
$$

To add the term, subclass `PopulationLikelihood`, override `logpdf(Lambda)`, call
`super().logpdf(Lambda)` and subtract $E\log\hat\alpha(\Lambda)$. Extra constructor arguments
carry the found injections:

```{code-cell} python
class SelectedLikelihood(ai.PopulationLikelihood):
    def __init__(self, *args, injections, log_g, n_injections, **kwargs):
        super().__init__(*args, **kwargs)
        self.injections, self.log_g, self.n_injections = injections, log_g, n_injections

    def log_alpha(self, Lambda):
        log_r = self.pop_model.log_prob(self.injections, Lambda) - self.log_g
        return jax.scipy.special.logsumexp(log_r) - jnp.log(self.n_injections)

    def logpdf(self, Lambda):
        return super().logpdf(Lambda) - self.E * self.log_alpha(Lambda)

found = injections[injections["detected"]]
selection = dict(
    injections={k: jnp.asarray(found[k].to_numpy()) for k in ["x", "flux"]},
    log_g=jnp.log(jnp.asarray(found["prior"].to_numpy())),
    n_injections=len(injections),
)
selected = SelectedLikelihood(camera, detected_data[:200], population, N=500, seed=0,
                              **selection)
float(jnp.exp(selected.log_alpha(Lambda_true)))
```

$\hat\alpha$ at the true $\Lambda$ is close to the detected fraction of the simulation. We
sample the hyperparameters twice from the same 200 detected sources, once with the plain
likelihood and once with the corrected one:

```{code-cell} python
naive = ai.PopulationLikelihood(camera, detected_data[:200], population, N=500, seed=0)
results = {}
for label, lik in [("naive", naive), ("corrected", selected)]:
    s = ai.run_sampler(lik, hyper_priors, seed=0, num_warmup=300, num_samples=1000,
                       summary=False)
    results[label] = s.samples[["mu", "sigma"]].agg(["mean", "std"])
pd.concat(results, axis=1).round(3)
```

The true values are $\mu = 3$ and $\sigma = 1$. The naive analysis describes the detected
sources, which are brighter and less spread out than the population.

To use such a subclass with `population_inference`, pass it as `likelihood_cls=`, binding the
extra arguments with `functools.partial`. Tutorial
[07](https://github.com/Potatoasad/astroinfer/blob/main/tutorials/07_flux_limited_survey.py) works through this correction on a flux-limited
survey.

## Proposals

When the measurements are much narrower than the population, few population draws land where
$\mathcal{L}(d_i\mid\theta)$ is large and $N_\mathrm{eff}$ collapses. A *proposal* $q(\theta\mid d_i)$
draws the parameters of each source near its data instead, and the draws are reweighted:

$$
\hat Z_i(\Lambda) = \frac{1}{N}\sum_{j=1}^{N}
\frac{\mathcal{L}(d_i\mid\theta_{ij})\,\pi(\theta_{ij}\mid\Lambda)}{q(\theta_{ij}\mid d_i)}, \qquad
\theta_{ij}\sim q(\cdot\mid d_i).
$$

A proposal reads data and draws parameters, so it is written as a measurement process with the
two types swapped. Subclass `ai.Proposal` and fill in:

| member | for a proposal |
|---|---|
| `parameter_type` | the measurement's data type (what $q$ reads) |
| `data_type` | the measurement's parameter type (what $q$ draws) |
| `sample_single(self, d, key, Lambda)` | one draw $\theta\sim q_\Lambda(\cdot\mid d)$ |
| `logpdf_single(self, theta, d, Lambda)` | $\log q_\Lambda(\theta\mid d)$ |

`Lambda` is the hyperparameter dict; a proposal that does not depend on it ignores it. The
likelihood redraws from the proposal with the same key at every $\Lambda$, so the draws must be
a smooth function of `(key, Lambda)`. Location-scale transforms and inverse CDFs are fine;
rejection sampling is not.

This proposal is a Gaussian centred on the measurement, 1.5 times wider than the camera's
noise:

```{code-cell} python
class CameraProposal(ai.Proposal):
    parameter_type = Measurement
    data_type = Source

    def __init__(self, camera, widen=1.5, **kwargs):
        super().__init__(**kwargs)
        self.camera, self.widen = camera, widen

    def sample_single(self, d, key, Lambda):
        key_x, key_flux = jax.random.split(key)
        sx, sf = self.widen * self.camera.sigma_x, self.widen * self.camera.sigma_flux
        return Source(x=d.x_hat + sx * jax.random.normal(key_x),
                      flux=d.flux_hat + sf * jax.random.normal(key_flux))

    def logpdf_single(self, theta, d, Lambda):
        norm = jax.scipy.stats.norm
        return (norm.logpdf(theta.x, d.x_hat, self.widen * self.camera.sigma_x)
                + norm.logpdf(theta.flux, d.flux_hat, self.widen * self.camera.sigma_flux))
```

The likelihood checks that the proposal's types mirror the measurement's. With the proposal,
far fewer draws are needed for the same effective number:

```{code-cell} python
with_proposal = ai.PopulationLikelihood(camera, data[:200], population, N=100, seed=0,
                                        proposal=CameraProposal(camera))
_, neff_q = with_proposal.event_logpdf_and_neff(Lambda_true)
_, neff_p = likelihood.event_logpdf_and_neff(Lambda_true)
{"population, N=500": float(jnp.median(neff_p)), "proposal, N=100": float(jnp.median(neff_q))}
```

Both estimate the same likelihood:

```{code-cell} python
float(likelihood.logpdf(Lambda_true)), float(with_proposal.logpdf(Lambda_true))
```

They differ by about 1.4 over 200 sources. The plain estimate is the lower one, as expected
when some sources have few effective draws.

With a proposal, the default `log_weights(Lambda)` is
$\log \mathcal{L} + \log\pi(\theta\mid\Lambda) - \log q(\theta\mid d)$. Override it for a different
weight, for example when $q$ draws some parameters from the population itself
and that factor cancels between $\pi$ and $q$. The default weight cannot be used with a
population model that has external variables; in that case `log_weights` raises an error and
has to be overridden. Tutorial [08](https://github.com/Potatoasad/astroinfer/blob/main/tutorials/08_sharper_telescopes.py) builds a proposal
for a survey whose data are much more precise than the population is wide.

## Joint inference

`population_inference` integrates out the parameters of every source inside each evaluation of
the likelihood. `joint_population_inference` samples them instead, together with $\Lambda$, from
the joint posterior

$$
p(\Lambda, \theta_1, \ldots, \theta_E \mid d) \propto
\pi(\Lambda) \prod_{i=1}^{E} \mathcal{L}(d_i\mid\theta_i)\,\pi(\theta_i\mid\Lambda).
$$

Ignoring the $\theta_i$ draws leaves draws from the same $p(\Lambda\mid d)$, with no Monte Carlo
estimate of $Z_i$ and so no effective number of draws to check. The cost moves to the sampler,
which explores every $\theta_i$ as well as $\Lambda$: here $2 \times 200 + 2 = 402$ dimensions.

The two methods suit opposite regimes. When the measurements are narrow compared with the
population, as here, few population draws land near the data and $N_\mathrm{eff}$ collapses,
while the joint posterior splits into narrow, nearly independent pieces that NUTS samples
easily. When the measurements are broad, the Monte Carlo estimate is accurate, but $\Lambda$ and
the $\theta_i$ become strongly correlated (a funnel), and NUTS slows down or reports divergent
transitions.

The population model must provide a density $\pi(\theta\mid\Lambda)$ normalized over $\theta$
for every $\Lambda$; its `sample` is not used. Each field of $\theta_i$ is sampled on the support
declared with `parameter(support=...)`, with a flat density there. `local_priors` takes PE
priors instead, in the form `PE` accepts, and applies them to every source. Their density is
divided back out, so the posterior is unchanged, but they set the support of each $\theta_i$,
which must contain the population's. `Source` declares no supports, so we pass PE priors that
cover the population:

```{code-cell} python
pe_priors = {"x": dist.Uniform(-1.0, 1.0), "flux": dist.Uniform(0.0, 20.0)}
joint = camera.joint_population_inference(
    data[:200], population, hyper_priors, local_priors=pe_priors,
    init=CameraProposal(camera), seed=0, num_warmup=300, num_samples=1000, summary=False,
)
joint
```

`init` sets the starting point. With a proposal, each $\theta_i$ starts at a draw from
$q(\theta\mid d_i)$ at the median of the hyperpriors, which is also where $\Lambda$ starts for
hyperpriors given as distributions. Sources whose starting density is zero are redrawn. A
parameter object with batch shape `(E,)` can be passed instead of a proposal. Without `init`,
numpyro starts every site uniformly in $(-2, 2)$ on the unconstrained scale, which is enough for
small problems.

The result is a `PosteriorData` whose `theta` holds $\Lambda$ and whose `local` holds the
$\theta_i$, with the sample axes followed by an `observation` axis:

```{code-cell} python
joint.local.batch_shape, joint.local_dims
```

The three estimates of $\Lambda$ from the same 200 sources are the plain Monte Carlo one from
[above](#sampling-the-hyperparameters), one with the proposal, and the joint one:

```{code-cell} python
with_q = camera.population_inference(data[:200], population, hyper_priors, N=500, seed=0,
                                     proposal=CameraProposal(camera), num_warmup=300,
                                     num_samples=1000, summary=False)
pd.concat({"population draws": hyper_posterior.summary()[["mean", "sd"]],
           "proposal": with_q.summary()[["mean", "sd"]],
           "joint": joint.summary(["mu", "sigma"])[["mean", "sd"]]}, axis=1).round(3)
```

The joint estimate agrees with the one that uses a proposal. The plain Monte Carlo estimate is
about 0.05 higher in $\mu$ and 0.03 higher in $\sigma$, the bias from sources with few effective
draws described in [Checking the effective number of draws](#checking-the-effective-number-of-draws).

`summary()` shows each field of `local` as one row, with its smallest effective sample size and
largest $\hat R$ over the 200 sources:

```{code-cell} python
joint.summary()
```

Each source's $\theta_i$ draws are its posterior given the population as well as its own data.
`posterior_predictive` replicates the data from them, here for 100 random draws:

```{code-cell} python
replicated = camera.posterior_predictive(joint, draws=100, key=0)
replicated.batch_shape
```

A `PopulationModel` can define `derived_parameters_single(theta, Lambda)`, for example the
probability that a source belongs to each component of a mixture. Joint inference evaluates it
for every draw and source and stores the result in `extras`, next to the measurement's derived
parameters.

## Structured populations

`pop_model` may also be a [structured distribution](distributions.md#structured-distributions)
over the measurement's parameter type. This is the population above, field by field, with a type
for its hyperparameters:

```{code-cell} python
@parameter_dataclass(name="Flux hyperparameters", latex=r"\Lambda")
class FluxHyper:
    mu: float = parameter(latex=r"\mu")
    sigma: float = parameter(latex=r"\sigma")

source_prior = Source.distribution(
    x=dist.Uniform(-1.0, 1.0),
    flux=lambda L: dist.TruncatedNormal(L.mu, L.sigma, low=0.0, high=20.0),
).given(FluxHyper)
```

Converted with `as_generative` (see
[Converting a structured distribution](population-models.md#converting-a-structured-distribution)),
it draws from seeds like a `GenerativeModel` and needs no proposal. Its draws for a given `seed`
are those of the hand-built `population`, so the two likelihoods agree:

```{code-cell} python
converted = ai.PopulationLikelihood(camera, data[:200], source_prior.as_generative(), N=500, seed=0)
float(converted.logpdf(Lambda_true)), float(likelihood.logpdf(Lambda_true))
```

Both are $-441.72$.

Passed as it is, the structured distribution is wrapped in a `DistributionPopulation`, a
`PopulationModel` that calls the distribution's own `sample` and `logpdf`. Its draws are not
treated as differentiable in $\Lambda$, so `PopulationLikelihood` refuses to run without a
proposal:

```{code-cell} python
:tags: [raises-exception]

ai.PopulationLikelihood(camera, data[:200], source_prior, N=500, seed=0)
```

With a proposal, only the population's `logpdf` is needed:

```{code-cell} python
wrapped = ai.PopulationLikelihood(camera, data[:200], source_prior, N=100, seed=0,
                                  proposal=CameraProposal(camera))
float(wrapped.logpdf(Lambda_true)), float(with_proposal.logpdf(Lambda_true))
```

The result, $-440.31$, is the same as with the hand-built population and the same proposal.

The same holds for any other conditional distribution over the parameter type with one condition
or none: it is wrapped, and needs a proposal unless it implements the `Reparameterizable` protocol
of `astroinfer.distributions.transport`, as converted populations do. `joint_population_inference`
uses only the population's `logpdf`, so it accepts a structured distribution in either form.

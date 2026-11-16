# Terminology

astroinfer uses a few terms in a narrower sense than general statistics writing does, and adds
some of its own. This page defines them. The definitions refer to the objects from the
[quickstart](../quickstart.md): `Source` and `Measurement` are parameter types, `src` is a
`Source`, `d` is a `Measurement`, `camera` is a measurement process, `posterior` is the result
of `camera.PE(d, priors, as_posterior=True)`, `population` is a population model, and `catalog`
is a `Catalog`.

## Parameters and data

### Parameter type

A class decorated with `@parameter_dataclass` whose fields are declared with `parameter`.
`Source` and `Measurement` are parameter types. Both the parameters $\theta$ and the data $d$
are described by parameter types.

### Field

One named quantity of a parameter type, such as `Source.flux`. `Source.parameters` lists the
fields in the order they were declared.

### Batch axes

The leading axes of a parameter object's fields, which index separate objects. They are shared
by all fields and broadcast like NumPy axes. `Source(x=jnp.zeros(3), flux=1.0)` has one batch
axis of length 3.

### Batch shape

The shape of the batch axes, `src.batch_shape`. A single object has batch shape `()`.

### Core axes

Trailing axes of a field that belong to a single object, declared with `index=` in
`parameter`. A spectrum with one value per channel has a core axis. Core axes are not part of
the batch shape.

### Support

The set of values a field may take, given as a numpyro constraint with `support=` in
`parameter`. A field without a support may take any real value.

### Latent space

The unconstrained coordinates of a parameter type, in which every field may take any real
value. `LatentSpace` maps between a parameter type and its latent space, field by field.

## Measurement

### Measurement process

A subclass of `MeasurementProcess` that defines how data are produced from parameters. `camera`
is a measurement process. It can simulate data and evaluate their probability.

### Likelihood

The probability density of the data given the parameters, $\mathcal{L}(d\mid\theta)$, which a
measurement process defines. `camera.logpdf(d, src)` returns $\log \mathcal{L}(d\mid\theta)$. As a
density over $d$ it is what `camera.sample` draws from; as a function of $\theta$ it is what
parameter estimation uses.

### Derived parameter

A quantity computed from the parameters by `derived_parameters`, such as a magnitude computed
from a flux. Derived parameters are added to the output of parameter estimation but are not
fields.

## Single-event inference

### Event

One observed object and its data. A catalog holds one posterior per event, under a name such
as `"src_0"`.

### PE

Parameter estimation: sampling the posterior $p(\theta\mid d)$ of one event. `camera.PE(d,
priors)` runs it.

### PE prior

The prior $\pi(\theta)$ used for parameter estimation. Its density at every draw is stored
with the draws, in total as `prior` and per field as `log_prior[x]`, `log_prior[flux]`, so
that a population analysis can divide it out later.

### PosteriorData

The container for one event's posterior. `posterior.theta` holds the draws as a batch of
`Source` objects, `sample_stats` holds per-draw inference quantities such as the log likelihood
and the PE prior, `extras` holds other per-draw columns such as derived parameters, `metadata`
describes the run, and `observed_data` holds `d`.

### Marginal prior

The PE prior density of a subset of the fields, rebuilt from the per-field `log_prior[...]`
columns. `posterior.marginal_prior(["x"])` is the prior density of `x` alone.

## Populations

### Population model

A distribution $\pi(\theta\mid\Lambda)$ describing how parameters are spread over many
objects. In astroinfer it is a `GenerativeModel`.

### Hyperparameters

The parameters $\Lambda$ of a population model, passed as a dict such as
`{"mu": 3.0, "sigma": 1.0}`.

### Leaf

A one-dimensional distribution for one named variable, such as `Uniform("x", -1.0, 1.0)`.
Population models are built from leaves with `product` and `mixture`.

### Seeds

The uniform random numbers, one array per variable, from which a population model makes its
draws by inverse transform sampling. Passing fixed seeds to `population.sample` makes the
draws a smooth function of $\Lambda$.

### Common random numbers

Using the same seeds at every value of $\Lambda$, so that Monte Carlo estimates change smoothly
with $\Lambda$ and can be differentiated.

### External variable

A variable that a population model conditions on but does not draw itself. It is declared
with `external=` and passed in with `conditioned_on=`.

## Population inference

### Population likelihood

The likelihood of the hyperparameters given the data of all events,
$\prod_i \int \mathcal{L}(d_i\mid\theta)\,\pi(\theta\mid\Lambda)\,d\theta$. `PopulationLikelihood`
estimates each integral by Monte Carlo.

### Weights

The terms $w_{ij}$ whose mean estimates the integral for event $i$. Without a proposal,
$w_{ij} = \mathcal{L}(d_i\mid\theta_{ij})$ with $\theta_{ij}$ drawn from the population.

### Effective sample size

$N_\mathrm{eff} = (\sum_j w_{ij})^2/\sum_j w_{ij}^2$, the number of equally weighted draws that
would give an estimate of the same quality. It is $N$ when all weights are equal and close to
1 when one draw dominates.

### Proposal

A distribution $q(\theta\mid d)$ that the population likelihood draws from instead of the
population, for events whose data are much narrower than the population. A `Proposal` is
written like a measurement process with the parameter and data types swapped.

### Detection trigger

A rule giving the probability $P(\mathrm{det}\mid d)$ that a measurement is detected, written
as a subclass of `DetectionTrigger`.

### Selection effect

The bias that arises when only detected events are analysed. It is corrected by dividing each
event's term in the population likelihood by $\alpha(\Lambda)$, the fraction of the population
that would be detected.

### Injections

Parameter draws from a known distribution that are passed through the measurement process
and the detection trigger once each, used to estimate $\alpha(\Lambda)$.

## Catalogs

### Catalog

A collection of posteriors, one per event, with the settings needed to export them and the
representations fitted to them.

### Data product

The arrays a catalog exports for a population analysis, with events along the first axis.
For raw draws it is `{column: (E, N), "prior": (E, N)}`.

### Representation

A way of passing posteriors to a population analysis. The built-in representation, `raw`, is
the draws themselves; others are compressed fits, such as a Gaussian per event.

### Fit

The result of fitting a representation to one event's posterior.

### Fitter

The object that holds a representation's settings and makes one fit per event.

### Source key

A hash of an event's posterior draws, `catalog.source_key(name)`. It changes when the
posterior is replaced.

### Fit key

A hash of everything that determines a fit: the fitter's settings for that event and the
event's source key.

### Stale

Describes a fit whose fit key no longer matches the one its fitter would compute now, because
the posterior or a setting has changed. `catalog.fit(name)` refits stale and missing fits.

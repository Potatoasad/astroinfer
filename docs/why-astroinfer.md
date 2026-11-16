# Why astroinfer?

astroinfer lets you write a forward model once, for a single object and a single observation,
and then use that same model to simulate data, estimate parameters, and infer the properties of
a population of objects, including the effect of which objects were detected.

## What it takes care of

A typical analysis of a population of astronomical objects needs the same model in several
places: to simulate a mock survey, to compute each object's posterior, and to evaluate the
likelihood of population hyperparameters. Written directly in NumPy or JAX, each of these needs
its own code for array shapes, random number keys and bookkeeping, and the versions tend to
drift apart. The examples below use the coin from [Getting started](getting-started.md), with
the parameter type `Coin`, the data type `Flips` and the measurement process `experiment`, and
a batch of three coins, `coins = Coin(p=jnp.array([0.3, 0.5, 0.8]))`. With astroinfer you:

- Write the likelihood for one object and get it for any batch: `logpdf_single(d, theta)` is
  written for one coin and one experiment, and `experiment.logpdf(d, coins)` evaluates it for
  any batch shapes of `d` and `coins` that broadcast, through `jax.vmap`.
- Keep the parameters of one object together, with names: a parameter type can be indexed like
  an array of records, as in `coins[coins.p > 0.5]`, and can be passed to `jax.grad`, which
  returns an object of the same type holding the derivatives.
- Separate the axes that count objects from the axes inside one object: the 50 flips of one
  experiment are a core axis declared with `index=("flip",)`, and `batch_shape` never includes
  it.
- Draw random numbers without managing keys: `experiment.sample(coins)` gives every coin its
  own key, and `key=0` makes a draw reproducible.
- Store each posterior with its prior: `PE` records the prior density of every draw, field by
  field, so that a population analysis can divide it out for any subset of fields with
  `marginal_prior`.
- Differentiate Monte Carlo estimates with respect to hyperparameters: population draws are
  inverse-CDF transforms of fixed uniform numbers, so the population likelihood is smooth in
  $\Lambda$ and can be sampled with NUTS.
- Model detection explicitly: a `DetectionTrigger` gives $P(\mathrm{det}\mid d)$, and
  `trigger.injections(experiment, draws, rng)` produces the table needed to estimate the
  fraction of a population that would be detected.
- Keep posteriors and their compressed fits consistent: a `Catalog` saves and loads posteriors,
  and marks a fit as stale when the posterior or the fit settings change.

## Core data structures

astroinfer has four main types. The first two describe a model, the last two hold results.

A *parameter type*, made with `@parameter_dataclass`, describes either the parameters $\theta$
or the data $d$. An instance works like a NumPy array of records: each field is an array, the
leading axes are shared by all fields and broadcast as NumPy axes do, and indexing selects
records. It is also a JAX pytree, so JAX transformations accept it.

A *measurement process*, a subclass of `MeasurementProcess`, defines the likelihood
$\mathcal{L}(d\mid\theta)$. You write how to draw one measurement and how to evaluate its log
density. The base class provides sampling and evaluation over batches, parameter estimation, and
population inference.

A `PosteriorData` holds one event's posterior draws as an instance of your parameter type,
together with per-draw statistics, the run's metadata and the observed data. Its group names,
`sample_stats` and `observed_data`, are the same as the corresponding groups of ArviZ's
`InferenceData`.

A `Catalog` holds one `PosteriorData` per event and exports them as arrays with events along
the first axis. It can also hold other representations of each posterior, such as a fitted
density, and saves everything to a folder.

Population models, detection triggers and likelihoods are built from these four types.

## Scope

astroinfer is about the models and the data structures around them. It relies on other packages
for everything else. Priors are numpyro distributions, sampling is done with NUTS through the
gravpop package, and plotting is left to you. Models specific to one kind of source, such as
gravitational-wave measurements, live in separate extension packages, for example
`astroinfer-gw`, and so do representations that need extra dependencies, such as the
truncated Gaussian mixtures in `astroinfer-tgmm`.

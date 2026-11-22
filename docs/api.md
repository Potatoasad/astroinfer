# API reference

This page lists the public API with a one-line summary for each name. The docstrings, available
with `help(name)`, give the full details. The user guide pages explain how the pieces fit
together.

## Parameter types (`astroinfer`)

| name | summary |
|---|---|
| `parameter_dataclass(cls=None, *, name=None, latex=None, frozen=True)` | Decorator that turns a class into a parameter type: a dataclass and a JAX pytree with batch and core axes. |
| `parameter(*, name=None, latex=None, index=(), index_latex=(), support=None, default=..., default_factory=...)` | Declares a field of a parameter type. |
| `ParameterSpec` | The metadata of one field, as returned by `cls.field_specs()`. |
| `Parameters` | Base class of every parameter type. |
| `batched_method(fn)` | Decorator that applies a method to every element of a batch with `jax.vmap`. |
| `coerce(obj, cls)` | Converts a dict or DataFrame to `cls`; instances of `cls` are returned unchanged. |
| `LatentSpace(parameter_type, supports=None)` | Maps a parameter type to unconstrained coordinates and back, field by field. |

Methods and properties of every parameter object:

| name | summary |
|---|---|
| `batch_shape`, `is_batched()` | Shape of the batch axes; whether there are any. |
| `broadcast(batch_shape=None)` | Copy with every field broadcast to the full batch shape. |
| `reshape_batch(shape)` | Copy with a different batch shape. |
| `obj[i]`, `obj[mask]`, `obj["field"]`, `obj["field.subfield"]` | Select from the batch, or return a field; a dotted path reaches into nested fields. |
| `to_dict()`, `cls.from_dict(values)` | Convert to and from a dict or DataFrame; nested fields become nested dicts, and `from_dict` also accepts dotted keys. |
| `to_flat_dict()` | The array fields as a dict keyed by dotted path. |
| `copy()` | Copy with fresh arrays. |
| `to_unconstrained(latent=None)`, `cls.from_unconstrained(z, latent=None)` | Map to and from the latent space. |
| `cls.parameters`, `cls.field_specs()`, `parameter_specs` | Field names and their metadata. |
| `cls.nested_types()`, `cls.flat_field_specs()` | The nested fields and their types; the metadata of every array field by dotted path. |
| `cls.distribution(**fields)` | A `StructuredDistribution` over the type, given field by field. |
| `name`, `latex` | Display name and symbol. |

## Distributions (`astroinfer`)

| name | summary |
|---|---|
| `ConditionalDistribution` | Base class for $p(x\mid c)$. Subclasses set `data_type` and `parameter_type` (none, a type, a tuple or a dict) and write `sample_single(*conditions, key)` and `logpdf_single(*data, *conditions)`. |
| `ConditionalDistribution.sample(*conditions, key=None, N=None, **conditions)` | Batched draws; `N` (an int or a tuple) adds sample axes after the batch axes of the conditions. |
| `ConditionalDistribution.logpdf(*data, *conditions, outer=False, **conditions)`, `log_prob(...)` | Log density over the batch shapes; a condition with fewer batch axes lines up with the leading data axes. |
| `ConditionalDistribution.condition(*conditions, **conditions)` | A `BoundConditionalDistribution` with some or all conditions fixed. |
| `ConditionalDistribution.seed(seed)` | Reset the internal random key. |
| `BoundConditionalDistribution` | A conditional distribution with fixed conditions; the remaining names of a mapping schema stay conditions. |
| `StructuredDistribution(data_type, fields, *, parameter_type=None, order=None, check_fixed=True)` | Distribution over a parameter type given field by field; usually made with `cls.distribution(**fields)`. |
| `StructuredDistribution.given(parameter_type)` | Copy with a condition type, passed to function and conditional fields. |
| `StructuredDistribution.with_order(paths)`, `order` | Copy with a different draw order; the random fields in draw order. |
| `StructuredDistribution.random_paths()`, `fixed_paths()` | Dotted paths of the random and the fixed fields. |
| `conditional(fn, on)` | A field whose NumPyro distribution `fn(x, *conditions)` depends on the fields named in `on`. |
| `product(*components, **named_components)` | Independent product with a tuple or dict output; returns a `ProductDistribution`. |
| `mixture(components, weights=None, logits=None, parameter_type=None, *, support=None)` | Mixture over one output type, with fixed or condition-dependent weights; returns a `MixtureDistribution`. |
| `StructuredDistribution.as_generative(hyperparameter_type=None, external=(), *, conditioned_on=None)` | The distribution on the generative backend, as a `StructuredGenerativePopulation`; also defined on a `MixtureDistribution` of structured distributions. Raises one error listing every field that cannot be converted. |

## Measurement (`astroinfer`)

| name | summary |
|---|---|
| `MeasurementProcess` | Base class for a measurement model $\mathcal{L}(d\mid\theta)$, a `ConditionalDistribution` with inference methods. Subclasses set `parameter_type` and `data_type` and write `sample_single(theta, key)` and `logpdf_single(d, theta)`. |
| `MeasurementProcess.sample(theta, key=None, *, N=None)` | Draw data for a batch of parameters. |
| `MeasurementProcess.logpdf(d, theta, *, outer=False)` | $\log \mathcal{L}(d\mid\theta)$, broadcast over the batch shapes. |
| `MeasurementProcess.seed(seed)` | Reset the internal random key. |
| `MeasurementProcess.derived_parameters_single(theta)` | Optional: extra quantities computed from one parameter object. |
| `MeasurementProcess.derived_parameters(theta)` | The derived quantities over a batch; by default `derived_parameters_single` with `jax.vmap`. |
| `MeasurementProcess.with_derived(df)` | Add the derived parameters to a table. |
| `MeasurementProcess.PE(d, pe_priors, seed=None, *, as_posterior=False, **sampler_kwargs)` | Sample the posterior of one observation. |
| `MeasurementProcess.population_inference(data, pop_model, pop_priors, **kwargs)` | Sample the population hyperparameters given a batch of observations, with each source's parameters integrated out. |
| `MeasurementProcess.joint_population_inference(data, pop_model, pop_priors, **kwargs)` | Sample the hyperparameters and every source's parameters together. |
| `MeasurementProcess.posterior_predictive(posterior, *, draws=None, key=None)` | Replicated data from posterior draws. |
| `Proposal` | Base class for an importance-sampling distribution $q_\Lambda(\theta\mid d)$, a `ConditionalDistribution` written like a measurement process with the two types swapped. |

## NumPyro models (`astroinfer`)

| name | summary |
|---|---|
| `sample(name, distribution, *conditions, obs=None, **conditions)` | Sample sites for one object inside a NumPyro model: one site for a NumPyro distribution, one per random field (`name.<path>`) for a structured distribution, one joint site for any other conditional distribution. Returns the value. |
| `factor(name, value)` | Add `value` to the model's log density, through `numpyro.factor`. |
| `infer(model, *args, method="nuts", seed=None, **kwargs)` | Sample the posterior of a model written with `sample` and return a `PosteriorData` with one `theta` field per root; `run_mcmc` settings among `kwargs` go to the runner. |
| `modeling.unpack(model, samples, *args, **kwargs)` | Typed objects, one per root, from a dict of site draws such as `Predictive` returns. Import from `astroinfer.modeling`. |
| `inference.runner.run_mcmc(model, *args, method="nuts", seed=None, num_warmup=1000, num_samples=2000, num_chains=1, target_accept_prob=0.7, max_tree_depth=6, summary=True, sampler_kwargs=None, mcmc_kwargs=None, **model_kwargs)` | Run NUTS or, with `method="nested"` (needs `jaxns`), nested sampling on a NumPyro model. Returns a `RunResult`. Import from `astroinfer.inference.runner`. |
| `inference.runner.RunResult` | The draws of every site (`sites`), the NUTS statistics (`sample_stats`), `sample_dims`, `seed`, the NumPyro `sampler`, and `get_samples(group_by_chain=False)`. |

## Posteriors (`astroinfer`)

| name | summary |
|---|---|
| `PosteriorData(theta, sample_stats={}, extras={}, metadata={}, observed_data=None, local=None, sample_dims=None)` | Posterior draws, per-draw statistics, run metadata and observed data. |
| `sample_dims`, `sizes`, `batch_shape`, `num_samples` | Names and lengths of the sample axes, e.g. `("chain", "draw")`. |
| `parameter_type`, `variable_names` | Parameter type, and the exportable column names. |
| `local`, `local_dims` | Joint population inference only: every source's parameters, with an `observation` axis after the sample axes. |
| `posterior["column"]`, `posterior[i]`, `posterior[:, s]`, `isel(**by_name)` | A column, or a subset of the draws indexed by position or by sample dim. |
| `stack_samples()` | Merge the sample axes into one `draw` axis, keeping each draw's chain in `sample_stats["chain"]`. |
| `summary(var_names=None, *, prob=0.9, expand=False)` | Mean, sd, central interval, effective sample size and split $\hat R$ per field. |
| `to_arviz()` | Convert to an ArviZ object (needs `arviz`). |
| `variables(names)` | Several columns as a dict. |
| `marginal_prior(columns)` | PE prior density of a subset of the fields. |
| `content_hash()` | Hash of all per-draw arrays. |
| `to_dataframe()`, `PosteriorData.from_dataframe(df, parameter_type, *, metadata=None, observed_data=None)` | Convert to and from a DataFrame. |

## Population models (`astroinfer`)

| name | summary |
|---|---|
| `PopulationModel` | Base class for $\pi(\theta\mid\Lambda)$, a `ConditionalDistribution`. Subclasses set `parameter_type` and write `sample_single(Lambda, key)` and `logpdf_single(theta, Lambda)`. |
| `PopulationModel.hyperparameter_type` | Optional parameter type for $\Lambda$; the `_single` methods then receive $\Lambda$ as that type. |
| `PopulationModel.sample(Lambda, key=None, *, N=None)`, `logpdf(theta, Lambda)` | Batched draws and log density; a typed $\Lambda$ with batch axes is mapped over. `shape=` is a deprecated name for `N`. |
| `PopulationModel.from_generative(model, parameter_type, *, conditioned_on=None)` | A `GenerativeModel` as a `PopulationModel`. |
| `PopulationModel.derived_parameters_single(theta, Lambda)` | Optional: quantities of one source under one $\Lambda$, such as mixture membership. |
| `distributions.transport.StructuredGenerativePopulation` | The `PopulationModel` made by `as_generative`; `order`, `model` (the `GenerativeModel`), `make_seeds(key=None, N=None)`, and `sample(Lambda, key=None, *, N=None, seeds=None)` returning parameter objects. |
| `distributions.transport.Reparameterizable` | Protocol of a distribution whose draws are a differentiable map of uniform numbers: `base_noise_spec()` and `sample_from_base(noise, *conditions)`. |
| `population.DistributionPopulation(distribution, parameter_type=None)` | A conditional distribution over $\theta$, with one condition or none, as a `PopulationModel`; it needs a proposal in `PopulationLikelihood` unless the distribution is `Reparameterizable`. Made by `PopulationLikelihood` when it is given such a distribution. |

## Generative population models (`astroinfer.generative`)

| name | summary |
|---|---|
| `GenerativeModel(root, order, external=())` | A population model $\pi(\theta\mid\Lambda)$ with a fixed drawing order. |
| `GenerativeModel.sample(params, seeds=None, key=None, shape=None, conditioned_on=None)` | Draw variables as a dict of arrays. |
| `GenerativeModel.log_prob(data, params)` | Log density of a dict of variables. |
| `Uniform(var, a, b)` | Uniform leaf. |
| `TruncatedNormal(var, mu, sigma, a, b)` | Truncated normal leaf. |
| `Beta(var, alpha, beta)` | Beta leaf on $[0, 1]$; sampled by bisection. |
| `product(models)` | Combine models of different variables. |
| `mixture(models, weights)` | Mixture of models of the same variables, with $K-1$ weight hyperparameters. |
| `conditional(fn, on)` | A distribution parameter computed from earlier draws and the hyperparameters. |
| `NumPyroLeaf(var, make_dist, deps=(), bounds=None)` | Leaf for a scalar NumPyro distribution `make_dist(x, L)`; drawn with its inverse CDF when that is differentiable in closed form, otherwise by bisection within `bounds`. |
| `WeightedMixture(models, weight_fn, deps=(), *, logits=False)` | Mixture whose weights `weight_fn(x, L)` may depend on earlier draws (named in `deps`) and the hyperparameters. |
| `Distribution` | Base class for a leaf: `cond_cdf`, `cond_pdf`, `bounds`, and optionally `cond_ppf` and `deps`. |

## Detection (`astroinfer.detection`)

| name | summary |
|---|---|
| `DetectionTrigger` | Base class for $P(\mathrm{det}\mid d)$. Subclasses write `detection_probability(d)` and optionally `selection_statistic(d)`. |
| `DetectionTrigger.detect(d, rng)` | Boolean mask of detected measurements. |
| `DetectionTrigger.injections(measurement, draws, rng)` | Measure and trigger a table of parameter draws once each. |
| `ThresholdTrigger(threshold=8.0, statistic="snr")` | Detected when one data field exceeds a threshold. |

## Likelihoods and sampling (`astroinfer`)

| name | summary |
|---|---|
| `FixedDataLikelihood(measurement, d)` | $\mathcal{L}(d\mid\theta)$ for fixed data; used by `PE`. |
| `PopulationLikelihood(measurement, data, pop_model, *, N=2000, seed=42, conditioned_on=None, proposal=None)` | Monte Carlo population likelihood of $\Lambda$ for independent events. |
| `PopulationLikelihood.logpdf(Lambda)` | Scalar log likelihood. |
| `PopulationLikelihood.event_logpdf(Lambda)`, `event_logpdf_and_neff(Lambda)` | Per-event log likelihoods, and their effective sample sizes. |
| `PopulationLikelihood.log_weights(Lambda)` | Log weights with shape `(E, N)`; the override point for custom weights. |
| `JointPopulationLikelihood(measurement, data, population, *, prefix="theta.")` | $\log\prod_i \mathcal{L}(d_i\mid\theta_i)\,\pi(\theta_i\mid\Lambda)$ as a function of $\Lambda$ and every $\theta_i$. |
| `run_sampler(likelihood, priors, *, seed=None, **sampler_kwargs)` | Run gravpop's sampler on any object with a `logpdf` method and return the sampler. |
| `parameter_estimation(measurement, d, pe_priors, seed=None, *, as_posterior=False, **sampler_kwargs)` | Function form of `MeasurementProcess.PE`. |
| `population_inference(measurement, data, pop_model, pop_priors, *, N=2000, seed=None, conditioned_on=None, proposal=None, likelihood_cls=PopulationLikelihood, **sampler_kwargs)` | Function form of `MeasurementProcess.population_inference`. |
| `joint_population_inference(measurement, data, pop_model, pop_priors, *, local_priors=None, init=None, init_Lambda=None, seed=None, conditioned_on=None, **sampler_kwargs)` | Function form of `MeasurementProcess.joint_population_inference`. |

## Catalogs and representations (`astroinfer`, `astroinfer.representations`)

| name | summary |
|---|---|
| `Catalog(posteriors=None, *, raw, seed=42)` | One posterior per event, plus representations fitted to them. |
| `Catalog.add_event(eventname, posterior, *, metadata=None, overwrite=False)`, `remove_event(eventname)` | Add, replace or remove an event. |
| `Catalog.eventnames`, `posteriors`, `len(catalog)`, `name in catalog`, `catalog[name]` | Access events. |
| `Catalog.source_key(eventname)` | Hash of an event's posterior. |
| `Catalog.dataproduct(*, representation="raw", N=None, seed=None, eventnames=None, allow_stale=False)` | Arrays for a population analysis, events first. |
| `Catalog.register_representation(name, fitter, *, overwrite=False)`, `remove_representation(name)` | Attach or remove a representation. |
| `Catalog.fit(name, eventnames=None, *, refit=False, errors="raise")` | Fit missing and stale events. |
| `Catalog.missing(name)`, `stale(name)`, `is_stale(eventname, name)` | Which fits need to be made. |
| `Catalog.get_fit(eventname, name)`, `fits(name)`, `fitter(name)` | Access fits and fitters. |
| `Catalog.save(root)`, `Catalog.load(root, *, types=())` | Write to and read from a folder. |
| `RawSamples(columns, N=2000)` | Settings of the built-in raw representation. |
| `RawFit` | One event's raw draws. |
| `BaseFit`, `BaseFitter` | Base classes for a representation's fit and fitter. |
| `register(fitter_cls=None, *, overwrite=False)` | Decorator that records a fitter's `kind` so `Catalog.load` can rebuild it. |
| `config_hash(value)` | Stable hash of a JSON-serialisable value. |
| `available_kinds()`, `get_fitter_class(kind)` | Look up registered representation kinds. |

## Displays (`astroinfer`, `astroinfer.display`)

| name | summary |
|---|---|
| `RichDisplay` | Base class for the collapsible notebook display. Subclasses write `_display_title`, `_display_math` and `_display_body`. |
| `display.kv_table(rows)`, `display.html_table(header, rows)` | HTML tables for a display body. |

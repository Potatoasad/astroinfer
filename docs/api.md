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
| `obj[i]`, `obj[mask]`, `obj["field"]` | Select from the batch, or return a field. |
| `to_dict()`, `cls.from_dict(values)` | Convert to and from a dict or DataFrame. |
| `copy()` | Copy with fresh arrays. |
| `to_unconstrained(latent=None)`, `cls.from_unconstrained(z, latent=None)` | Map to and from the latent space. |
| `cls.parameters`, `cls.field_specs()`, `parameter_specs` | Field names and their metadata. |
| `name`, `latex` | Display name and symbol. |

## Measurement (`astroinfer`)

| name | summary |
|---|---|
| `MeasurementProcess` | Base class for a measurement model $\mathcal{L}(d\mid\theta)$. Subclasses set `parameter_type` and `data_type` and write `sample_single(theta, key)` and `logpdf_single(d, theta)`. |
| `MeasurementProcess.sample(theta, key=None, *, N=None)` | Draw data for a batch of parameters. |
| `MeasurementProcess.logpdf(d, theta, *, outer=False)` | $\log \mathcal{L}(d\mid\theta)$, broadcast over the batch shapes. |
| `MeasurementProcess.seed(seed)` | Reset the internal random key. |
| `MeasurementProcess.derived_parameters(theta)` | Optional: extra quantities computed from a dict of parameter arrays. |
| `MeasurementProcess.with_derived(df)` | Add the derived parameters to a table. |
| `MeasurementProcess.PE(d, pe_priors, seed=None, *, as_posterior=False, **sampler_kwargs)` | Sample the posterior of one observation. |
| `MeasurementProcess.population_inference(data, pop_model, pop_priors, **kwargs)` | Sample the population hyperparameters given a batch of observations. |
| `Proposal` | Base class for an importance-sampling distribution $q_\Lambda(\theta\mid d)$, written like a measurement process with the two types swapped. |

## Posteriors (`astroinfer`)

| name | summary |
|---|---|
| `PosteriorData(theta, sample_stats={}, extras={}, metadata={}, observed_data=None)` | One event's posterior draws, per-draw statistics, run metadata and observed data. |
| `batch_shape`, `parameter_type`, `variable_names` | Draw shape, parameter type, and the exportable column names. |
| `posterior["column"]`, `posterior[i]`, `posterior[mask]` | A column, or a subset of the draws. |
| `variables(names)` | Several columns as a dict. |
| `marginal_prior(columns)` | PE prior density of a subset of the fields. |
| `content_hash()` | Hash of all per-draw arrays. |
| `to_dataframe()`, `PosteriorData.from_dataframe(df, parameter_type, *, metadata=None, observed_data=None)` | Convert to and from a DataFrame. |

## Population models (`astroinfer.generative`)

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
| `run_sampler(likelihood, priors, *, seed=None, **sampler_kwargs)` | Run gravpop's sampler on any object with a `logpdf` method and return the sampler. |
| `parameter_estimation(measurement, d, pe_priors, seed=None, *, as_posterior=False, **sampler_kwargs)` | Function form of `MeasurementProcess.PE`. |
| `population_inference(measurement, data, pop_model, pop_priors, *, N=2000, seed=None, conditioned_on=None, proposal=None, likelihood_cls=PopulationLikelihood, **sampler_kwargs)` | Function form of `MeasurementProcess.population_inference`. |

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

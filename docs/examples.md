# Examples

These examples are the marimo notebooks in
[`tutorials/`](https://github.com/Potatoasad/astroinfer/tree/main/tutorials), executed when the
documentation is built. Each one works through one problem, builds on the ones before it, and
ends with an exercise whose solution is hidden in a collapsed block. They assume you know
Bayesian inference, but not JAX or astroinfer.

| # | example | topics |
|---|---|---|
| 1 | [Is this coin rigged?](examples/01_coin_flips.ipynb) | parameter types, a first `MeasurementProcess`, `sample`, `logpdf` and `PE` |
| 2 | [The lighthouse](examples/02_lighthouse.ipynb) | batch and core axes, indexing, JAX gradients, supports and `LatentSpace` |
| 3 | [A transiting planet](examples/03_transit.ipynb) | methods on a data class, `as_posterior=True`, posterior predictive checks |
| 4 | [The doctor's office](examples/04_doctors_office.ipynb) | `PosteriorData`, derived parameters, custom displays |
| 5 | [The coin mint](examples/05_coin_mint.ipynb) | population models with `astroinfer.generative`, writing your own leaf |
| 6 | [Auditing the mint](examples/06_auditing_the_mint.ipynb) | hierarchical inference with `PopulationLikelihood` |
| 7 | [A flux-limited survey](examples/07_flux_limited_survey.ipynb) | `DetectionTrigger`, injections, selection effects |
| 8 | [Sharper telescopes](examples/08_sharper_telescopes.ipynb) | a NumPyro model with `ai.sample` and `ai.infer`, integrating values out with a pair rule, importance sampling with a `Proposal` |
| 9 | [The star catalog](examples/09_star_catalog.ipynb) | `Catalog`, `RawSamples`, data products, saving and loading |
| 10 | [Your own representation](examples/10_your_own_representation.ipynb) | the `Fit` and `Fitter` protocol |

The sliders and dropdowns in the notebooks are shown here at their default values. To use them,
run a notebook with marimo from the repository root:

```bash
uv sync
uv run marimo edit tutorials/01_coin_flips.py
```

## Gravitational waves

These notebooks, in
[`notebooks/gw/`](https://github.com/Potatoasad/astroinfer/tree/main/notebooks/gw), apply
astroinfer to mock gravitational-wave catalogs, using the measurement process in
`astroinfer_gw.gwmockcat` and the population distributions in `astroinfer_gw.populations`.
Notebook 1 runs parameter estimation for every detected event and notebook 3 runs population
inference, both of which take a long time, so unlike the examples above these notebooks are
executed ahead of time and the pages show their stored outputs. Notebook 2 also needs Julia.
Notebooks 4 to 8 fit the ringdowns of binary mergers, after the package
[ringdown](https://github.com/maxisi/ringdown); notebooks 6 to 8 download open LIGO-Virgo data
and need the `ringdown` dependency group. They assume you have worked through the examples above.

| # | example | topics |
|---|---|---|
| 1 | [A mock gravitational-wave catalog](examples/gw/01_gw_mock_catalog.ipynb) | the measurement process and detection trigger, a population as a structured distribution over `GWParams`, PE, the raw data product, injections |
| 2 | [Fitting the GW catalog with TGMMs](examples/gw/02_gw_tgmm_fitting.ipynb) | TGMM configurations, per-event overrides and stale fits, padded data products, transformed coordinates, saving and loading |
| 3 | [Population inference with an event-informed proposal](examples/gw/03_proposal_population.ipynb) | `as_generative`, an event-informed `Proposal`, the selection term, `ai.infer` with the Monte Carlo population likelihood and with a joint NumPyro model |
| 4 | [A damped sinusoid in white noise](examples/gw/04_ringdown_damped_sinusoid.ipynb) | ringdown's introductory example: a measurement process with core axes, a structured prior with `.given`, `ai.sample` and `ai.infer` |
| 5 | [Kerr quasinormal modes in LIGO noise](examples/gw/05_ringdown_kerr.ipynb) | the Kerr spectrum, a `DetectorConfig` with `ifo` and `time` axes, the Advanced LIGO design PSD, a prior over a nested type |
| 6 | [Kerr ringdown in real detector noise](examples/gw/06_ringdown_real_noise.ipynb) | open LIGO-Virgo data through gwpy, ringdown's conditioning and PSD estimates, checking the noise model |
| 7 | [The ringdown of GW150914](examples/gw/07_gw150914_ringdown.ipynb) | a fit of real data with ringdown's Kerr spectrum and angular factors, compared with ringdown's own fit |
| 8 | [Marginalizing over the amplitudes](examples/gw/08_ringdown_marginalized.ipynb) | integrating linear amplitudes out of a Gaussian likelihood, a marginal likelihood and its linear posterior as conditional distributions |

They run with marimo in the same way, for example
`uv run marimo edit notebooks/gw/01_gw_mock_catalog.py`.

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
| 8 | [Sharper telescopes](examples/08_sharper_telescopes.ipynb) | importance sampling with a `Proposal` |
| 9 | [The star catalog](examples/09_star_catalog.ipynb) | `Catalog`, `RawSamples`, data products, saving and loading |
| 10 | [Your own representation](examples/10_your_own_representation.ipynb) | the `Fit` and `Fitter` protocol |

The sliders and dropdowns in the notebooks are shown here at their default values. To use them,
run a notebook with marimo from the repository root:

```bash
uv sync
uv run marimo edit tutorials/01_coin_flips.py
```

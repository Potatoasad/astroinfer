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
| 3 | [The doctor's office](examples/03_doctors_office.ipynb) | `PosteriorData`, derived parameters, custom displays |
| 4 | [The coin mint](examples/04_coin_mint.ipynb) | population models with `astroinfer.generative`, writing your own leaf |
| 5 | [Auditing the mint](examples/05_auditing_the_mint.ipynb) | hierarchical inference with `PopulationLikelihood` |
| 6 | [A flux-limited survey](examples/06_flux_limited_survey.ipynb) | `DetectionTrigger`, injections, selection effects |
| 7 | [Sharper telescopes](examples/07_sharper_telescopes.ipynb) | importance sampling with a `Proposal` |
| 8 | [The star catalog](examples/08_star_catalog.ipynb) | `Catalog`, `RawSamples`, data products, saving and loading |
| 9 | [Your own representation](examples/09_your_own_representation.ipynb) | the `Fit` and `Fitter` protocol |

The sliders and dropdowns in the notebooks are shown here at their default values. To use them,
run a notebook with marimo from the repository root:

```bash
uv sync
uv run marimo edit tutorials/01_coin_flips.py
```

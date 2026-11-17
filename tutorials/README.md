# astroinfer tutorials

These marimo notebooks go from a single coin flip to population inference with selection
effects, catalogs and custom representations. Each notebook works through one problem, builds on
the previous ones, and ends with an exercise whose solution is hidden in a collapsed cell.

The notebooks assume you know Bayesian inference, but not JAX or astroinfer.

| # | notebook | problem | topics |
|---|---|---|---|
| 1 | `01_coin_flips.py` | is this coin rigged? | `@parameter_dataclass`, `parameter`, a first `MeasurementProcess`, `.sample` / `.logpdf` / `.PE` |
| 2 | `02_lighthouse.py` | Gull's lighthouse | batch vs core axes, indexing, JAX gradients, supports and `LatentSpace` |
| 3 | `03_transit.py` | a planet transiting its star | methods on a data class, `as_posterior=True`, posterior predictive checks |
| 4 | `04_doctors_office.py` | height and weight from BMI | `PosteriorData`, derived parameters, custom rich displays |
| 5 | `05_coin_mint.py` | the coin mint | population models with `astroinfer.generative`, writing your own leaf |
| 6 | `06_auditing_the_mint.py` | auditing the mint | hierarchical inference with `PopulationLikelihood` |
| 7 | `07_flux_limited_survey.py` | a flux-limited survey | `DetectionTrigger`, injections, selection effects |
| 8 | `08_sharper_telescopes.py` | sharper telescopes | importance sampling with a `Proposal` |
| 9 | `09_star_catalog.py` | the star catalog | `Catalog`, `RawSamples`, data products, save / load |
| 10 | `10_your_own_representation.py` | your own representation | the `Fit` / `Fitter` protocol |

The GW notebooks in `../notebooks/gw/` (mock catalog, TGMM fits, proposal-based population
inference) follow on from these.

## Running

From the repository root:

```bash
uv sync
uv run marimo edit tutorials/01_coin_flips.py
```

Every notebook also runs as a plain script (`uv run python tutorials/01_coin_flips.py`) and takes
about a minute or less on a laptop. Notebook 9 saves its catalog to `tutorials/data/star_catalog`
(notebook 10 reuses it).

`_style.py` holds the shared plot style. Colors keep the same meaning in every notebook: blue for
the posterior, orange for the prior or population, aqua for data, violet for a fourth series, and
a dashed navy line for the truth.

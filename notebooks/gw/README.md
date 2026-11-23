# Gravitational-wave notebooks

These notebooks apply astroinfer to mock gravitational-wave catalogs, using the measurement
process in `astroinfer_gw.gwmockcat` and the population distributions in
`astroinfer_gw.populations`. They assume you have worked through the tutorials in `tutorials/`.
Run them from the repository root:

```bash
uv run marimo edit notebooks/gw/01_gw_mock_catalog.py
```

1. `01_gw_mock_catalog.py` simulates a catalog: the measurement process and detection trigger, a
   true population written as a structured distribution over `GWParams`, simulated detections,
   PE for the detected events, the raw data product, and an injection table drawn from a second,
   broad structured distribution. It saves the catalog to `notebooks/data/gw_catalog`. Later runs
   reload it if it holds the same simulated events, and rerun PE otherwise or when you tick the
   box in the notebook.
2. `02_gw_tgmm_fitting.py` fits the catalog from notebook 1 with TGMMs: configurations, per-event
   overrides and stale fits, padded data products, a second representation for the spins,
   transformed coordinates, and saving and loading. It starts Julia the first time it runs.
3. `03_proposal_population.py` runs population inference on a catalog of 100 events that it
   simulates from the population of notebook 1. It converts the population with `as_generative`
   for draws from a fixed bank, and writes an event-informed `Proposal` that draws the spins from
   the population, so that their factor cancels in the importance weight. The selection term
   comes from the fixed bank and the analytic detection probability, and is compared with the
   estimate from injections. A `PopulationLikelihood` subclass with the proposal marginalizes over
   each event's parameters, and `ai.run_sampler` samples the hyperparameters. The samples are
   saved to `notebooks/data/proposal_population` and reloaded if the settings match; the sampler
   takes RUNTIME on a laptop CPU.

4. `04_ringdown_damped_sinusoid.py` reimplements the introductory example of
   [ringdown](https://github.com/maxisi/ringdown): one damped sinusoid in white noise. The
   signal parameters and the data are parameter types with core axes, the noise is a
   `MeasurementProcess`, and ringdown's prior is a structured distribution conditioned on the
   amplitude scale with `.given`. `ai.sample` and `ai.infer` fit it with NUTS in a few seconds.
5. `05_ringdown_kerr.py` fits the Kerr 220 and 221 modes, parameterized by the remnant mass and
   spin, to simulated data in two detectors with the analytic Advanced LIGO design PSD. The
   likelihood uses the Toeplitz covariance of the autocovariance, as ringdown does. The
   measurement process holds a `DetectorConfig` parameter object with core axes `"ifo"` and
   `"time"`, whose methods do the per-detector algebra, and the prior is a structured
   distribution over a nested type. NUTS takes about 30 s on a laptop CPU.
6. `06_ringdown_real_noise.py` repeats the fit of notebook 5 in real noise from H1, L1 and V1,
   256 s of public O2 data near GW170814. ringdown fetches the data through gwpy, conditions it,
   computes the start times and antenna patterns, and estimates each PSD from the conditioned
   data; `DetectorConfig.from_ringdown` turns the result into the configuration of the
   measurement process. The notebook checks the noise model on the whitened noise and fits a
   ringdown added to the real noise. It needs the `ringdown` dependency group:
   `uv run --group ringdown marimo edit notebooks/gw/06_ringdown_real_noise.py`. The first run
   downloads about 3 minutes of data from GWOSC; later runs take about a minute.
7. `07_gw150914_ringdown.py` fits the ringdown of GW150914 in the public H1 and L1 data, with the
   settings of ringdown's GW150914 example. The model uses ringdown's Kerr spectrum and angular
   factors, so it is the same model as ringdown's, and the notebook runs ringdown's own fit on the
   same conditioned data for comparison; the two posteriors agree. It also needs the `ringdown`
   group. The first run downloads two 4096 s files of 16 kHz data, about 10 minutes; later runs
   take about a minute.
8. `08_ringdown_marginalized.py` integrates the quadratures of notebook 7 out of the likelihood
   analytically. It writes the marginalization once, for any measurement that is linear in part
   of its parameters with Gaussian noise: `LinearGaussianMeasurement` gets its design matrix from
   the signal by autodiff, `marginalized()` gives the marginal likelihood and
   `linear_posterior()` the Gaussian posterior of the linear parameters, both conditional
   distributions. It checks the marginal likelihood against a dense Gaussian, samples the
   remnant, draws the amplitudes back in one batched call, and compares with ringdown. It also
   needs the `ringdown` group.

All of them also run as scripts from the repository root, for example
`uv run python notebooks/gw/01_gw_mock_catalog.py`. Notebook 2 reads the catalog that notebook 1
saves; notebooks 3 to 5 need no saved data, and notebooks 6 and 7 download theirs.

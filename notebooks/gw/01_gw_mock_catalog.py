import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    from pathlib import Path

    import astroinfer as ai
    import marimo as mo
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    import plot_helper as ph
    from astroinfer import Catalog, RawSamples, parameter, parameter_dataclass
    from astroinfer.detection import ThresholdTrigger

    from astroinfer_gw.gwmockcat import GWMockCatMeasurement, GWParams, ToyOptimalSNR
    from astroinfer_gw.physics import isotropic_projection_factor
    from astroinfer_gw.populations import PowerLaw, PowerLawRedshift, ProjectionFactor

    ph.restore_plot_defaults()
    ph.set_plot_defaults(latex_font_size=16, usetex=True)

    # The notebook's own folder (mo.notebook_dir() is the cwd when run as a script).
    try:
        NOTEBOOK_DIR = Path(__file__).resolve().parent
    except NameError:
        NOTEBOOK_DIR = Path(mo.notebook_dir())
    DATA_DIR = NOTEBOOK_DIR.parent / "data"
    CATALOG_DIR = DATA_DIR / "gw_catalog"

    N_TRUTHS = 20_000     # population draws to simulate
    SNR_THRESHOLD = 4.0   # detected if the observed SNR exceeds this
    N_EVENTS = 20         # detected events to run PE on

    M_MIN, M_MAX = 5.0, 80.0   # primary mass range of the population, in solar masses
    Z_MAX = 1.5                # largest redshift

    LABELS = dict(mass_1_source=r"$m_1$", mass_ratio=r"$q$", redshift=r"$z$",
                  chi_1=r"$\chi_1$", chi_2=r"$\chi_2$", cos_tilt_1=r"$\cos\theta_1$",
                  cos_tilt_2=r"$\cos\theta_2$", chi_eff=r"$\chi_\mathrm{eff}$", chi_p=r"$\chi_p$",
                  projection_factor=r"$\Theta$")

    def as_table(theta):
        """A parameter object with one batch axis as a DataFrame, one column per field."""
        return pd.DataFrame({k: np.asarray(v) for k, v in theta.to_dict().items()})


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. A mock gravitational-wave catalog

    This notebook builds the mock catalog that notebook 2 fits with TGMMs. It goes through these
    steps in order:

    1. a GWMockCat-style measurement process $\mathcal{L}(d\mid\theta)$ with effective spins;
    2. a detection trigger $P(\mathrm{det}\mid d)$;
    3. true source parameters $\theta_i \sim \pi(\theta\mid\Lambda_\mathrm{true})$, drawn from a
       structured distribution over `GWParams`;
    4. simulated data $d_i \sim \mathcal{L}(d\mid\theta_i)$, of which only the detected events are kept;
    5. parameter estimation for each detected event, with the posteriors collected in a `Catalog`
       and saved.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The measurement process and the trigger

    The measurement process uses a toy model of the optimal SNR. An event is detected when its
    observed SNR is above 4. Expanding the displays below shows the measurement's noise model and
    the trigger's detection rule.
    """)
    return


@app.cell
def _():
    measurement = GWMockCatMeasurement(
        snr_model=ToyOptimalSNR(bns_horizon=300.0, reference_snr=8.0, reference_chirp_mass=1.2, m_cut=250.0)
    )
    trigger = ThresholdTrigger(SNR_THRESHOLD, statistic="snr")
    mo.vstack([measurement, trigger])
    return measurement, trigger


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The true population

    The population is a structured distribution over `GWParams`, written one field at a time with
    `GWParams.distribution(...)`. Its hyperparameters $\Lambda$ are a parameter dataclass,
    `GWHyper`, which `.given(GWHyper)` declares as the distribution's condition. A field written
    as a function receives the hyperparameters as `L`, and a field written as a distribution has
    no hyperparameters. The fields are:

    - $p(m_1) \propto m_1^{\alpha}$ between $m_\mathrm{min} = 5\,M_\odot$ and
      $m_\mathrm{max} = 80\,M_\odot$;
    - $p(q \mid m_1) \propto q^{\beta}$ between $m_\mathrm{min}/m_1$ and 1, so that the secondary
      mass is at least $m_\mathrm{min}$. This is a conditional field, which reads the value of
      $m_1$ drawn before it;
    - $p(z) \propto \frac{dV_c}{dz}(1+z)^{\lambda-1}$ up to $z = 1.5$;
    - the isotropic projection factor $\Theta\sim\mathrm{Beta}(2,4)$;
    - truncated normals for the spin magnitudes $\chi_1, \chi_2$ on $[0,1]$ and the tilt cosines
      $\cos\theta_1, \cos\theta_2$ on $[-1,1]$, each with its own mean and width.

    `PowerLaw`, `PowerLawRedshift` and `ProjectionFactor` are NumPyro distributions from
    `astroinfer_gw.populations`. `ProjectionFactor()` is $\mathrm{Beta}(2,4)$ with a polynomial
    CDF, which is cheaper to invert when the population is converted with `as_generative`, as in
    notebook 3. The lower bound of $q$ depends on $m_1$, so its `PowerLaw` is given the fixed
    support $[0, 1]$ for that conversion; the density is zero below $m_\mathrm{min}/m_1$. Neither
    power law is smoothed, so the mass distribution has a sharp edge at $m_\mathrm{min}$.
    """)
    return


@app.cell
def _():
    @parameter_dataclass(name="GW population hyperparameters", latex=r"\Lambda")
    class GWHyper:
        alpha: float = parameter(name="Primary mass slope", latex=r"\alpha")
        beta: float = parameter(name="Mass ratio slope", latex=r"\beta")
        lamb: float = parameter(name="Redshift evolution", latex=r"\lambda")
        mu_chi_1: float = parameter(name="Mean primary spin magnitude", latex=r"\mu_{\chi_1}")
        sigma_chi_1: float = parameter(name="Width of primary spin magnitudes", latex=r"\sigma_{\chi_1}")
        mu_chi_2: float = parameter(name="Mean secondary spin magnitude", latex=r"\mu_{\chi_2}")
        sigma_chi_2: float = parameter(name="Width of secondary spin magnitudes", latex=r"\sigma_{\chi_2}")
        mu_cos_tilt_1: float = parameter(name="Mean primary tilt cosine", latex=r"\mu_{t_1}")
        sigma_cos_tilt_1: float = parameter(name="Width of primary tilt cosines", latex=r"\sigma_{t_1}")
        mu_cos_tilt_2: float = parameter(name="Mean secondary tilt cosine", latex=r"\mu_{t_2}")
        sigma_cos_tilt_2: float = parameter(name="Width of secondary tilt cosines", latex=r"\sigma_{t_2}")

    population = GWParams.distribution(
        mass_1_source=lambda L: PowerLaw(L.alpha, M_MIN, M_MAX),
        mass_ratio=ai.conditional(
            lambda x, L: PowerLaw(L.beta, M_MIN / x["mass_1_source"], 1.0, support=(0.0, 1.0)),
            on="mass_1_source"),
        redshift=lambda L: PowerLawRedshift(L.lamb, z_max=Z_MAX),
        projection_factor=ProjectionFactor(),
        chi_1=lambda L: dist.TruncatedNormal(L.mu_chi_1, L.sigma_chi_1, low=0.0, high=1.0),
        chi_2=lambda L: dist.TruncatedNormal(L.mu_chi_2, L.sigma_chi_2, low=0.0, high=1.0),
        cos_tilt_1=lambda L: dist.TruncatedNormal(L.mu_cos_tilt_1, L.sigma_cos_tilt_1, low=-1.0, high=1.0),
        cos_tilt_2=lambda L: dist.TruncatedNormal(L.mu_cos_tilt_2, L.sigma_cos_tilt_2, low=-1.0, high=1.0),
    ).given(GWHyper)

    Lambda_true = GWHyper(alpha=-2.35, beta=1.1, lamb=2.7,
                          mu_chi_1=0.4, sigma_chi_1=0.05, mu_chi_2=0.0, sigma_chi_2=0.05,
                          mu_cos_tilt_1=0.8, sigma_cos_tilt_1=0.1, mu_cos_tilt_2=0.8, sigma_cos_tilt_2=0.1)
    mo.vstack([population, Lambda_true])
    return Lambda_true, population


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulating detections

    `population.sample(Lambda, key=..., N=...)` returns the true parameters as a `GWParams` with
    batch shape `(N,)`. `measurement.sample` takes them directly and returns the simulated data as
    a batched `GWMockCatData`. `trigger.detect` returns a boolean mask of the detected events,
    which can index both the true parameters and the data:
    """)
    return


@app.cell
def _(Lambda_true, measurement, population, trigger):
    truths = population.sample(Lambda_true, key=1, N=N_TRUTHS)
    data = measurement.sample(truths, key=jax.random.key(2))
    detected = trigger.detect(data, np.random.default_rng(3))

    detected_truths = truths[detected]
    detected_data = data[detected]

    mo.md(f"{detected.sum()} of {N_TRUTHS:,} simulated sources are detected "
          f"(observed SNR > {SNR_THRESHOLD:g}).")
    return detected, detected_data, detected_truths, truths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For plots and for the catalog metadata, `measurement.with_derived` turns the true parameters
    into a table with the derived parameters ($\chi_\mathrm{eff}$, $\chi_p$, the detector-frame
    chirp mass, $d_L$ and $m_2$) added as columns:
    """)
    return


@app.cell
def _(detected, measurement, truths):
    truth_table = measurement.with_derived(as_table(truths))
    detected_table = truth_table[np.asarray(detected)].reset_index(drop=True)
    detected_table.head()
    return detected_table, truth_table


@app.cell(expand_output=True)
def _(detected_table, truth_table):
    _variables = ["mass_1_source", "mass_ratio", "redshift", "chi_eff"]
    _fig, _ = ph.make_corner_plot(
        [truth_table.sample(5000, random_state=0)[_variables], detected_table[_variables]],
        model_labels=["population", "detected"], variables=_variables,
        variable_labels=[LABELS[v] for v in _variables], scatter_kwargs=dict(s=3, alpha=0.5))
    _fig
    return


@app.cell
def _(detected_data):
    observed = as_table(detected_data)
    observed.head(N_EVENTS)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Parameter estimation

    The PE priors are uniform in $m_1$, $q$, $z$ and the spins, and isotropic in $\Theta$.
    `measurement.PE(d, priors, as_posterior=True)` runs NUTS on $\mathcal{L}(d\mid\theta)$ and returns a
    `PosteriorData`. It holds the draws of $\theta$ as a `GWParams`, the derived parameters, the PE
    prior density as `prior`, and the log prior of each parameter as `log_prior[...]`. The catalog
    and the TGMM fits use these per-parameter factors to compute the prior of only the columns they
    export.

    PE takes a few seconds per event. The catalog is saved to `notebooks/data/gw_catalog`. If that
    folder already holds a catalog of the same events, with the true parameters simulated above in
    its metadata, it is loaded instead, unless you tick the box. A saved catalog of other events,
    left over from a different population or seed, is replaced.
    """)
    return


@app.cell
def _():
    pe_priors = dict(
        mass_1_source=dist.Uniform(2.0, 120.0), mass_ratio=dist.Uniform(0.05, 1.0),
        redshift=dist.Uniform(0.0, 1.5), projection_factor=isotropic_projection_factor(),
        chi_1=dist.Uniform(0.0, 1.0), chi_2=dist.Uniform(0.0, 1.0),
        cos_tilt_1=dist.Uniform(-1.0, 1.0), cos_tilt_2=dist.Uniform(-1.0, 1.0))
    force_pe = mo.ui.checkbox(label="re-run PE (overwrites notebooks/data/gw_catalog)")
    force_pe
    return force_pe, pe_priors


@app.cell
def _(detected_data, detected_table, force_pe, measurement, pe_priors):
    _n = min(N_EVENTS, len(detected_table))

    def _same_events(saved):
        """Whether a saved catalog holds PE for the first _n detected events simulated above."""
        if len(saved.eventnames) != _n:
            return False
        stored = [saved.posterior(name).metadata["injected_true"] for name in saved.eventnames]
        return all(np.isclose(row[k], detected_table.iloc[i][k])
                   for i, row in enumerate(stored) for k in GWParams.field_specs())

    catalog = Catalog.load(CATALOG_DIR) if CATALOG_DIR.exists() and not force_pe.value else None
    if catalog is None or not _same_events(catalog):
        catalog = Catalog(raw=RawSamples(columns=["mass_1_source", "mass_ratio", "chi_1", "chi_2", "cos_tilt_1", "cos_tilt_2", "redshift"], N=2000))
        for _i in mo.status.progress_bar(range(_n), title="Parameter estimation"):
            _posterior = measurement.PE(detected_data[_i], pe_priors, seed=_i, as_posterior=True,
                                        num_warmup=500, num_samples=4000)
            catalog.add_event(f"event_{_i + 1:02d}", _posterior,
                              metadata={"injected_true": detected_table.iloc[_i].to_dict()})
        catalog.save(CATALOG_DIR)
    catalog
    return (catalog,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One event's posterior

    The plot marks the event's true parameters, which were stored in its metadata when it was added
    to the catalog.
    """)
    return


@app.cell
def _(catalog):
    event = mo.ui.dropdown(catalog.eventnames, value=catalog.eventnames[0], label="event")
    event
    return (event,)


@app.cell(expand_output=True)
def _(catalog, event):
    _posterior = catalog.posterior(event.value)
    _variables = ["mass_1_source", "mass_ratio", "chi_1", "chi_2", "cos_tilt_1", "cos_tilt_2", "redshift"]
    _frame = pd.DataFrame({k: np.asarray(v) for k, v in _posterior.variables(_variables).items()})
    _truth = _posterior.metadata["injected_true"]
    _fig, _ = ph.make_corner_plot([_frame], model_labels=[event.value], variables=_variables,
                                  variable_labels=[LABELS[v] for v in _variables],
                                  truth={v: _truth[v] for v in _variables},
                                  scatter_kwargs=dict(s=2, alpha=0.3))
    mo.vstack([_posterior, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The raw data product

    `catalog.dataproduct()` resamples every event to the same number of draws. It returns the
    exported columns and the PE prior density of those columns only, each with shape
    `(events, draws)`:
    """)
    return


@app.cell
def _(catalog):
    raw_product = catalog.dataproduct(N=1000)
    {k: v.shape for k, v in raw_product.items()}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Injections

    The selection function is estimated from injections. They are drawn from a broad distribution
    $g$, another structured distribution over `GWParams` with fixed parameters and no conditions.
    It is flatter than the true population in $m_1$ and $z$, uniform in $q$ and the spins, and
    covers the population's support. Each injection is measured once and passed through the same
    trigger. `trigger.injections` takes a table of draws and returns it with the measured data,
    the detection statistic and whether each injection was detected. The density of $g$ goes in
    the `prior` column.
    """)
    return


@app.cell
def _(measurement, trigger):
    injection_population = GWParams.distribution(
        mass_1_source=PowerLaw(-1.5, 4.0, 120.0),
        mass_ratio=dist.Uniform(0.05, 1.0),
        redshift=PowerLawRedshift(1.0, z_max=Z_MAX),
        projection_factor=ProjectionFactor(),
        chi_1=dist.Uniform(0.0, 1.0),
        chi_2=dist.Uniform(0.0, 1.0),
        cos_tilt_1=dist.Uniform(-1.0, 1.0),
        cos_tilt_2=dist.Uniform(-1.0, 1.0),
    )
    _draws = injection_population.sample(key=4, N=100_000)
    _table = as_table(_draws)
    _table["prior"] = np.exp(np.asarray(injection_population.logpdf(_draws)))
    injections = trigger.injections(measurement, _table, np.random.default_rng(5))
    mo.vstack([injection_population,
               mo.md(f"{injections.detected.sum():,} of {len(injections):,} injections are detected "
                     f"({injections.detected.mean():.2%})."),
               injections[injections.detected].head()])
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    from pathlib import Path

    import gravpop as gp
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    import plot_helper as ph
    from astroinfer import Catalog, RawSamples
    from astroinfer.detection import ThresholdTrigger

    from astroinfer_gw.gwmockcat import GWMockCatMeasurement, ToyOptimalSNR
    from astroinfer_gw.physics import isotropic_projection_factor, projection_factor_model

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

    LABELS = dict(mass_1_source=r"$m_1$", mass_ratio=r"$q$", redshift=r"$z$",
                  chi_1=r"$\chi_1$", chi_2=r"$\chi_2$", cos_tilt_1=r"$\cos\theta_1$",
                  cos_tilt_2=r"$\cos\theta_2$", chi_eff=r"$\chi_\mathrm{eff}$", chi_p=r"$\chi_p$",
                  projection_factor=r"$\Theta$")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. A mock gravitational-wave catalog

    This notebook builds the mock catalog that notebook 2 fits with TGMMs. It goes through these
    steps in order:

    1. a GWMockCat-style measurement process $\mathcal{L}(d\mid\theta)$ with effective spins;
    2. a detection trigger $P(\mathrm{det}\mid d)$;
    3. true source parameters $\theta_i \sim p(\theta\mid\Lambda_\mathrm{true})$, drawn from a gravpop
       population;
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

    The true population is a product of gravpop models: a smoothed power law in $m_1$ with
    $p(q\mid m_1)\propto q^\beta$, truncated Gaussians for the spin magnitudes and tilt cosines, the
    isotropic projection factor $\Theta\sim\mathrm{Beta}(2,4)$, and a redshift distribution
    $p(z)\propto \frac{dV_c}{dz}(1+z)^{\lambda-1}$.
    """)
    return


@app.cell
def _():
    spin_magnitudes = gp.fixed_parameters(
        gp.TruncatedGaussian2DAnalytic(a=[0.0, 0.0], b=[1.0, 1.0], var_names=["chi_1", "chi_2"],
                                       hyper_var_names=["mu_chi_1", "sigma_chi_1", "mu_chi_2", "sigma_chi_2", "rho_chi"]),
        dict(rho_chi=1e-5))
    spin_tilts = gp.fixed_parameters(
        gp.TruncatedGaussian2DAnalytic(a=[-1.0, -1.0], b=[1.0, 1.0], var_names=["cos_tilt_1", "cos_tilt_2"],
                                       hyper_var_names=["mu_cos_tilt_1", "sigma_cos_tilt_1", "mu_cos_tilt_2",
                                                        "sigma_cos_tilt_2", "rho_cos_tilt"]),
        dict(rho_cos_tilt=1e-5))
    redshift_model = gp.PowerLawRedshift(var_names=["redshift"], hyper_var_names=["lamb"], z_max=1.5)

    true_population = gp.product([
        gp.SmoothedPrimaryMassRatio(gp.TruncatedPowerLaw(var_names=["mass_1_source"],
                                                         hyper_var_names=["alpha", "mmin", "mmax"]),
                                    mmin_fixed=2, mmax_fixed=100),
        spin_magnitudes, spin_tilts, projection_factor_model(), redshift_model,
    ])
    true_hyper = dict(alpha=-2.35, mmin=5.0, mmax=80.0, delta_m=3.0, beta=1.1,
                      mu_chi_1=0.4, mu_chi_2=0.0, sigma_chi_1=0.05, sigma_chi_2=0.05,
                      mu_cos_tilt_1=0.8, mu_cos_tilt_2=0.8, sigma_cos_tilt_1=0.1, sigma_cos_tilt_2=0.1,
                      lamb=2.7)
    return redshift_model, true_hyper, true_population


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulating detections

    `measurement.sample` takes a DataFrame of true parameters, or a `GWParams` object, and returns
    the simulated data as a batched `GWMockCatData`. `trigger.detect` returns a boolean mask of the
    detected events, which can index both the table of true parameters and the data:
    """)
    return


@app.cell
def _(measurement, trigger, true_hyper, true_population):
    truths = true_population.sample(pd.DataFrame([true_hyper]), oversample=N_TRUTHS,
                                    random_state=np.random.default_rng(1)).reset_index(drop=True)
    truths = measurement.with_derived(truths)          # + chirp_mass_det, d_L, chi_eff, chi_p, ...

    data = measurement.sample(truths, np.random.default_rng(2))
    detected = trigger.detect(data, np.random.default_rng(3))

    detected_truths = truths[detected].reset_index(drop=True)
    detected_data = data[detected]

    mo.md(f"{detected.sum()} of {len(truths):,} simulated sources are detected "
          f"(observed SNR > {SNR_THRESHOLD:g}).")
    return detected_data, detected_truths, truths


@app.cell(expand_output=True)
def _(detected_truths, truths):
    _variables = ["mass_1_source", "mass_ratio", "redshift", "chi_eff"]
    _fig, _ = ph.make_corner_plot(
        [truths.sample(5000, random_state=0)[_variables], detected_truths[_variables]],
        model_labels=["population", "detected"], variables=_variables,
        variable_labels=[LABELS[v] for v in _variables], scatter_kwargs=dict(s=3, alpha=0.5))
    _fig
    return


@app.cell
def _(detected_data):
    observed = pd.DataFrame({k: np.asarray(v) for k, v in detected_data.to_dict().items()})
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
    folder already exists, the catalog is loaded from it instead, unless you tick the box.
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
def _(detected_data, detected_truths, force_pe, measurement, pe_priors):
    if CATALOG_DIR.exists() and not force_pe.value:
        catalog = Catalog.load(CATALOG_DIR)
    else:
        catalog = Catalog(raw=RawSamples(columns=["mass_1_source", "mass_ratio", "chi_1", "chi_2", "cos_tilt_1", "cos_tilt_2", "redshift"], N=2000))
        _n = min(N_EVENTS, len(detected_truths))
        for _i in mo.status.progress_bar(range(_n), title="Parameter estimation"):
            _posterior = measurement.PE(detected_data[_i], pe_priors, seed=_i, as_posterior=True,
                                        num_warmup=500, num_samples=4000)
            catalog.add_event(f"event_{_i + 1:02d}", _posterior,
                              metadata={"injected_true": detected_truths.iloc[_i].to_dict()})
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

    The selection function is estimated from injections. They are drawn from a broad population
    $g$, flatter than the true population in $m_1$, $q$ and redshift and uniform in the spins, then
    measured once each and passed through the same trigger. `trigger.injections` returns a table
    with the measured data, the detection statistic and whether each injection was detected. The
    density of $g$ is in the `prior` column.
    """)
    return


@app.cell
def _(measurement, redshift_model, trigger):
    injection_population = gp.product([
        gp.TruncatedPowerLaw(var_names=["mass_1_source"], hyper_var_names=["alpha", "mmin", "mmax"]),
        gp.TruncatedPowerLaw(var_names=["mass_ratio"], hyper_var_names=["beta", "qmin", "qmax"]),
        gp.Uniform2DAnalytic(a=[0, 0], b=[1, 1], var_names=["chi_1", "chi_2"]),
        gp.Uniform2DAnalytic(a=[-1, -1], b=[1, 1], var_names=["cos_tilt_1", "cos_tilt_2"]),
        projection_factor_model(), redshift_model,
    ])
    injection_hyper = dict(alpha=-1.5, mmin=4.0, mmax=120.0, beta=0.0, qmin=0.05, qmax=1.0, lamb=1.0)

    _rng = np.random.default_rng(4)
    injection_draws = injection_population.sample(pd.DataFrame([injection_hyper]), oversample=100_000,
                                                  random_state=_rng).reset_index(drop=True)
    injection_draws["prior"] = np.asarray(injection_population.evaluate(
        {k: jnp.asarray(injection_draws[k].to_numpy()) for k in injection_population.var_names},
        injection_hyper))
    injections = trigger.injections(measurement, injection_draws, _rng)
    mo.vstack([mo.md(f"{injections.detected.sum():,} of {len(injections):,} injections are detected "
                     f"({injections.detected.mean():.2%})."),
               injections[injections.detected].head()])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

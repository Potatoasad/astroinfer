import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    from dataclasses import dataclass
    from pathlib import Path
    from typing import ClassVar

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    from numpyro.distributions import constraints

    import astroinfer as ai
    from astroinfer import MeasurementProcess, parameter, parameter_dataclass
    from astroinfer.display import code, html_table, kv_table
    from astroinfer.generative import Distribution, GenerativeModel, TruncatedNormal, product
    from astroinfer.representations import BaseFit, BaseFitter, config_hash

    import _style as style

    style.apply()

    try:
        TUTORIAL_DIR = Path(__file__).resolve().parent
    except NameError:
        TUTORIAL_DIR = Path(mo.notebook_dir())
    CATALOG_DIR = TUTORIAL_DIR / "data" / "star_catalog"            # written by notebook 9
    GAUSS_DIR = TUTORIAL_DIR / "data" / "star_catalog_gauss"        # written here

    R_MIN, R_MAX = 0.5, 2.0
    LABELS = dict(log_L=r"$\log_{10} L$", distance=r"$r$ [kpc]",
                  mu_L=r"$\mu_L$", sigma_L=r"$\sigma_L$")



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 10. Your own representation

    In notebook 9 the catalog passed posteriors to the population analysis as raw draws. Raw draws
    are one *representation* of a posterior, and there are others that are often better. A
    Gaussian, or a mixture of Gaussians, is compact and smooth and can be integrated analytically.
    A truncated Gaussian mixture, provided by the `astroinfer_tgmm` extension, fits posteriors that
    pile up against hard edges. KDEs, normalising flows and histograms are further options.

    Besides the built-in `raw` representation, a catalog can carry any number of named
    representations. It fits them, notices when a fit is out of date, stacks the fits into data
    products, and saves and loads them, all without knowing what kind of representation it is
    handling. Your classes only have to follow a small protocol. In this notebook we write a
    moment-matched Gaussian representation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **What you implement: a `Fitter` and a `Fit`**

    The fitter holds the settings and makes one fit per event. Subclass `BaseFitter`.

    | member | description |
    |---|---|
    | `kind` (class attribute) | a unique name for the representation type, e.g. `"gaussian"` |
    | `fit_type` (class attribute) | your Fit class |
    | `fit(posterior, *, eventname, source_key)` | fit one event's `PosteriorData`, return a Fit |
    | `fit_key(eventname, source_key)` | a hash of everything that determines the fit (`config_hash` computes one) |
    | `dataproduct_options(eventname)` | settings that only shape the data product (e.g. `N`) |
    | `to_dict()` / `from_dict(d)` | the settings as JSON, so the fitter is saved with the catalog |

    The fit is the fitted object for one event. Subclass `BaseFit`.

    | member | description |
    |---|---|
    | `fit_key` | the key it was made with |
    | `dataproduct(*, N, seed, **options)` | this event's arrays, e.g. `{"mean": (D,), "cov": (D, D)}` |
    | `save(directory)` / `load(directory, ctx)` | write / rebuild yourself in a folder you own |
    | `stack(products)` (optional) | combine events into `(E, ...)` arrays; the default stacks equal shapes |

    A fit is *stale* when `fit.fit_key != fitter.fit_key(event, source_key)`. Since `source_key` is
    the hash of the posterior (notebook 9), a fit becomes stale when you change one of its settings
    or replace the posterior. Changing a data-product option does not make it stale.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The survey and its catalog

    The classes are the same as in notebooks 7 and 9. We load the catalog that notebook 9 saved,
    passing `types=` because the classes are defined in this notebook. If that catalog does not
    exist yet, a smaller one is built in memory.
    """)
    return


@app.class_definition
class PowerLaw(Distribution):
    """p(x) ∝ x**index on [lo, hi]. index: a number or a hyperparameter name."""

    def __init__(self, var, index, lo, hi):
        self.var_names, self.var = [var], var
        self.index, self.lo, self.hi = index, lo, hi

    def _k(self, L):
        k = (L[self.index] if isinstance(self.index, str) else self.index) + 1.0
        return jnp.where(jnp.abs(k) < 1e-6, 1e-6, k)

    def cond_pdf(self, var, val, x, L):
        k = self._k(L)
        p = k * val ** (k - 1.0) / (self.hi ** k - self.lo ** k)
        return jnp.where((val >= self.lo) & (val <= self.hi), p, 0.0)

    def cond_cdf(self, var, val, x, L):
        k = self._k(L)
        v = jnp.clip(val, self.lo, self.hi)
        return (v ** k - self.lo ** k) / (self.hi ** k - self.lo ** k)

    def cond_ppf(self, var, u, x, L):
        k = self._k(L)
        return (self.lo ** k + u * (self.hi ** k - self.lo ** k)) ** (1.0 / k)

    def bounds(self, var):
        return self.lo, self.hi


@app.class_definition
@parameter_dataclass(name="Star", latex=r"\theta")
class Star:
    log_L: float = parameter(name="Log luminosity", latex=r"\log_{10}L")
    distance: float = parameter(name="Distance [kpc]", latex="r", support=constraints.positive)


@app.class_definition
@parameter_dataclass(name="Star observation", latex="d")
class StarObservation:
    log_flux: float = parameter(name="Measured log flux", latex=r"\log_{10}\hat F")
    parallax: float = parameter(name="Measured parallax [mas]", latex=r"\hat\varpi")


@app.class_definition
@dataclass
class Telescope(MeasurementProcess):
    sigma_log_flux: float = 0.2
    sigma_parallax: float = 0.3

    display_name = "Telescope"
    parameter_type = Star
    data_type = StarObservation

    def sample_single(self, theta, key):
        kf, kp = jax.random.split(key)
        return StarObservation(
            log_flux=theta.log_L - 2 * jnp.log10(theta.distance) + self.sigma_log_flux * jax.random.normal(kf),
            parallax=1.0 / theta.distance + self.sigma_parallax * jax.random.normal(kp))

    def logpdf_single(self, d, theta):
        return (dist.Normal(theta.log_L - 2 * jnp.log10(theta.distance), self.sigma_log_flux).log_prob(d.log_flux)
                + dist.Normal(1.0 / theta.distance, self.sigma_parallax).log_prob(d.parallax))


@app.cell
def _():
    pop_model = GenerativeModel(product([
        TruncatedNormal("log_L", "mu_L", "sigma_L", -3.0, 5.0),
        PowerLaw("distance", 2.0, R_MIN, R_MAX)]), order=["log_L", "distance"])
    Lambda_true = dict(mu_L=0.0, sigma_L=0.7)
    telescope = Telescope()
    trigger = ai.detection.ThresholdTrigger(threshold=-0.3, statistic="log_flux")
    pe_priors = dict(log_L=dist.Uniform(-3.0, 5.0), distance=dist.Uniform(R_MIN, R_MAX))
    return Lambda_true, pe_priors, pop_model, telescope, trigger


@app.cell
def _(Lambda_true, pe_priors, pop_model, telescope, trigger):
    if CATALOG_DIR.exists():
        catalog = ai.Catalog.load(CATALOG_DIR, types=[Star, StarObservation])
        _source = f"Loaded notebook 9's catalog from `{CATALOG_DIR.relative_to(TUTORIAL_DIR)}`."
    else:
        _stars = Star.from_dict(pop_model.sample(Lambda_true, key=jax.random.key(0), shape=(5000,)))
        _obs = telescope.sample(_stars, key=1)
        _idx = np.flatnonzero(trigger.detect(_obs, np.random.default_rng(2)))[:12]
        catalog = ai.Catalog(raw=ai.RawSamples(columns=["log_L", "distance"], N=1000))
        for _i in mo.status.progress_bar(range(len(_idx)), title="Parameter estimation"):
            _post = telescope.PE(_obs[_idx[_i]], pe_priors, seed=_i, as_posterior=True,
                                 num_warmup=300, num_samples=1500)
            catalog.add_event(f"star_{_i:02d}", _post, metadata={
                "truth": {k: float(v) for k, v in _stars[_idx[_i]].to_dict().items()}})
        _source = ("Notebook 9's catalog was not found, so a 12-star catalog was built in memory. "
                   "Run notebook 9 first for the full 40 stars.")
    mo.vstack([mo.md(_source), catalog])
    return (catalog,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The fit

    A `GaussianFit` stores the columns, a mean vector $\mu$ and a covariance $\Sigma$. Its data
    product is $\mu$, $\Sigma$ and `N` draws from the Gaussian, which are useful for plotting. It
    saves itself as one small JSON file. Because it has a `_display_body`, it gets a rich display,
    including inside the catalog's display.
    """)
    return


@app.class_definition
class GaussianFit(BaseFit):
    """A multivariate normal N(mean, cov) over `columns`, fit to one event."""

    def __init__(self, columns, mean, cov, *, fit_key):
        self.columns = list(columns)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self._fit_key = fit_key

    @property
    def fit_key(self):
        return self._fit_key

    def dataproduct(self, *, N, seed):
        draws = np.random.default_rng(seed).multivariate_normal(self.mean, self.cov, size=N)
        return {"mean": jnp.asarray(self.mean), "cov": jnp.asarray(self.cov),
                **{c: jnp.asarray(draws[:, i]) for i, c in enumerate(self.columns)}}

    def save(self, directory):
        ai.io.write_json(directory / "gaussian.json", {
            "columns": self.columns, "mean": self.mean, "cov": self.cov, "fit_key": self.fit_key})

    @classmethod
    def load(cls, directory, ctx):
        d = ai.io.read_json(directory / "gaussian.json")
        return cls(d["columns"], d["mean"], d["cov"], fit_key=d["fit_key"])

    # ---- display ----
    def _display_title(self):
        return "Gaussian fit"

    def _display_math(self):
        return r"\mathcal{N}(\mu,\Sigma)"

    def _display_body(self):
        sd = np.sqrt(np.diag(self.cov))
        corr = self.cov / np.outer(sd, sd)
        rows = [[code(c), f"{m:.3f}", f"{s:.3f}"] + [f"{r:+.2f}" for r in corr[i]]
                for i, (c, m, s) in enumerate(zip(self.columns, self.mean, sd))]
        return (html_table(["column", "mean", "sd"] + [f"corr {c}" for c in self.columns], rows)
                + kv_table({"fit_key": self.fit_key[:12] + "..."}))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The fitter

    The fitter has three settings:

    * `columns`, the posterior variables to fit (affects the fit);
    * `inflate`, a factor that multiplies $\Sigma$, as a crude way to be conservative (affects the
      fit);
    * `N`, the number of draws per event in the data product (affects only the data product).

    Each event can override the fit settings, for example
    `fitter.overrides["star_03"] = {"inflate": 2.0}`. `fit_key` hashes the settings that affect
    this event's fit, together with the posterior's `source_key`. This is how the catalog decides
    whether a fit is stale.

    `@ai.register` adds the `"gaussian"` kind to astroinfer's registry, so `Catalog.load` knows which
    class rebuilds a saved `"gaussian"` representation.
    """)
    return


@app.class_definition
@ai.register
class GaussianFitter(BaseFitter):
    kind: ClassVar[str] = "gaussian"
    fit_type: ClassVar[type] = GaussianFit

    def __init__(self, columns, *, inflate=1.0, N=500, overrides=None):
        self.columns = list(columns)
        self.inflate = float(inflate)
        self.N = int(N)
        self.overrides = {k: dict(v) for k, v in (overrides or {}).items()}

    def settings_for(self, eventname):
        """The fit settings for one event: defaults, then that event's overrides."""
        return {"columns": self.columns, "inflate": self.inflate,
                **self.overrides.get(eventname, {})}

    def fit_key(self, eventname, source_key):
        return config_hash({"kind": self.kind, "settings": self.settings_for(eventname),
                            "source": source_key})

    def fit(self, posterior, *, eventname, source_key):
        s = self.settings_for(eventname)
        values = posterior.variables(s["columns"])
        x = np.stack([np.asarray(values[c]) for c in s["columns"]], axis=-1)
        return GaussianFit(s["columns"], x.mean(axis=0), s["inflate"] * np.cov(x, rowvar=False),
                           fit_key=self.fit_key(eventname, source_key))

    def dataproduct_options(self, eventname):
        return {"N": self.N}

    def to_dict(self):
        return {"columns": self.columns, "inflate": self.inflate, "N": self.N,
                "overrides": self.overrides}

    @classmethod
    def from_dict(cls, d):
        return cls(d["columns"], inflate=d["inflate"], N=d["N"], overrides=d["overrides"])

    def _display_title(self):
        return "Gaussian fitter"

    def _display_body(self):
        overrides = ", ".join(f"{e}: {o}" for e, o in self.overrides.items()) or "none"
        return kv_table({"kind": self.kind, "columns": ", ".join(self.columns),
                         "inflate": self.inflate, "N (data product)": self.N,
                         "per-event overrides": overrides})


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fitting the catalog

    `register_representation(name, fitter)` attaches the fitter to the catalog under a name of your
    choice, so you could have two Gaussian representations of different columns.
    `catalog.fit(name)` fits every event whose fit is missing or stale, and returns the names of the
    events it fitted:
    """)
    return


@app.cell
def _(catalog):
    catalog.register_representation("gauss", GaussianFitter(["log_L", "distance"]), overwrite=True)
    first_fit = catalog.fit("gauss")
    mo.vstack([
        mo.md(f"The first call fitted {len(first_fit)} events. Calling `catalog.fit('gauss')` "
              f"again fits {len(catalog.fit('gauss'))}, because no fit is missing or stale."),
        catalog,
    ])
    return (first_fit,)


@app.cell
def _(catalog):
    event = mo.ui.dropdown(catalog.eventnames, value=catalog.eventnames[3], label="event")
    event
    return (event,)


@app.cell(expand_output=True)
def _(catalog, event, first_fit):
    _ = first_fit
    _fit = catalog.get_fit(event.value, "gauss")
    _posterior = catalog[event.value]
    _fig, _ = style.corner(
        [_posterior.variables(_fit.columns), _fit.dataproduct(N=1500, seed=0)], _fit.columns,
        labels=[LABELS[c] for c in _fit.columns], colors=[style.POSTERIOR, style.EXTRA],
        truths=_posterior.metadata["truth"], ranges=dict(distance=(0.3, 2.4)),
        names=["posterior draws", "Gaussian fit"])
    mo.vstack([_fit, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The Gaussian matches the bulk of most posteriors. Near the hard edge at $r = 2\,\mathrm{kpc}$,
    which is the boundary of the PE prior, it spills over into a region the posterior does not
    reach. The last section of this notebook shows what this costs.

    ## The data product

    `catalog.dataproduct(representation="gauss")` calls the `dataproduct` method of every fit and
    stacks the results along a leading event axis `E`. Our fits all return arrays of the same shape,
    so the default `stack` is enough. (`astroinfer_tgmm` overrides `stack` to pad events that have
    different numbers of components.)
    """)
    return


@app.cell
def _(catalog, first_fit):
    _ = first_fit
    gauss_product = catalog.dataproduct(representation="gauss")
    pd.DataFrame({k: [str(tuple(v.shape))] for k, v in gauss_product.items()}, index=["shape"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Stale fits

    The cell below makes four changes in a row and calls `catalog.stale("gauss")` after each one to
    list the fits that are out of date:
    """)
    return


@app.cell
def _(catalog, first_fit, pe_priors, telescope):
    _ = first_fit
    _fitter = catalog.fitter("gauss")
    _e = catalog.eventnames
    _rows = []

    _fitter.N = 200                                       # 1. data-product option
    _rows.append(("set `N = 200` (data product only)", catalog.stale("gauss")))

    _fitter.overrides[_e[2]] = {"inflate": 2.0}           # 2. one event's fit setting
    _rows.append((f"override `inflate = 2` for `{_e[2]}`", catalog.stale("gauss")))

    _redo = telescope.PE(catalog[_e[5]].observed_data, pe_priors, seed=123, as_posterior=True,
                         num_warmup=300, num_samples=1500)
    catalog.add_event(_e[5], _redo, metadata=catalog[_e[5]].metadata, overwrite=True)  # 3. new posterior
    _rows.append((f"replace `{_e[5]}`'s posterior (re-run PE)", catalog.stale("gauss")))

    _snapshot = mo.Html(catalog._repr_html_())            # the catalog shows stale badges now

    _refit = catalog.fit("gauss")                         # 4. refit only what is stale
    _rows.append(("`catalog.fit('gauss')`", catalog.stale("gauss")))

    staleness_done = True
    mo.vstack([
        mo.md("| change | `catalog.stale('gauss')` |\n|---|---|\n"
              + "\n".join(f"| {c} | `{s}` |" for c, s in _rows)),
        mo.md(f"The refit only touched {_refit}. Here is the catalog just before that refit "
              "(expand *Events*):"),
        _snapshot,
    ])
    return (staleness_done,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The replacement posterior for `star_05` was computed from `catalog[name].observed_data`. Every
    posterior keeps the observation it was computed from, so PE can always be rerun.

    ## Saving and loading

    `catalog.save(root)` asks each fit to save itself into its own folder and writes the fitter's
    `to_dict()` as JSON. `Catalog.load` reads the kind (`"gaussian"`), looks it up in the registry,
    and calls `GaussianFitter.from_dict` and `GaussianFit.load`. Per-event overrides are restored
    as well, and since every fit is stored with its key, no fit is stale after loading.
    """)
    return


@app.cell
def _(catalog, staleness_done):
    _ = staleness_done
    catalog.save(GAUSS_DIR)
    reloaded = ai.Catalog.load(GAUSS_DIR, types=[Star, StarObservation])
    _e = reloaded.eventnames[2]
    mo.vstack([
        mo.md(f"""
    * representations after loading: `{reloaded.representations}`
    * `{_e}`'s override survived: `{reloaded.fitter("gauss").overrides}`
    * stale after loading: `{reloaded.stale("gauss")}`
    * same fit as before saving: `{bool(np.allclose(reloaded.get_fit(_e, "gauss").cov, catalog.get_fit(_e, "gauss").cov))}`
    """),
        reloaded.get_fit(_e, "gauss"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `@ai.register` works for classes defined in a notebook or a script. If your representation lives
    in a package, declare the kind through an entry point instead, so that `Catalog.load` can find
    it without your package being imported first:

    ```toml
    # pyproject.toml
    [project.entry-points."astroinfer.representations"]
    gaussian = "my_package:GaussianFitter"
    ```

    `astroinfer_tgmm` registers its `"tgmm"` kind this way.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The cost of compression

    With a Gaussian for each event, the evidence of an event becomes an average of its Gaussian
    over draws from the population, $\theta_k\sim p(\theta\mid\Lambda)$:

    $$
    Z_i(\Lambda)\;\propto\;\int \frac{\mathcal N(\theta;\mu_i,\Sigma_i)}{\pi(\theta)}\,p(\theta\mid\Lambda)\,d\theta
    \;\approx\;\frac{1}{M}\sum_{k=1}^{M}\frac{\mathcal N(\theta_k;\mu_i,\Sigma_i)}{\pi(\theta_k)}.
    $$

    The PE prior $\pi$ is flat here, so it only contributes a constant. Below we evaluate the
    hyperparameter likelihood on a grid and compare it with the raw-draws estimator from
    notebook 9. Both include the selection term, with the same injections as notebook 7.
    """)
    return


@app.cell
def _(pop_model, telescope, trigger):
    # Injections for the selection term, as in notebook 7.
    Lg = dict(mu_L=0.5, sigma_L=1.5)
    inj = pop_model.sample(Lg, key=jax.random.key(5), shape=(50000,))
    found = trigger.detect(telescope.sample(Star.from_dict(inj), key=6), np.random.default_rng(7))
    inj_found = {k: jnp.asarray(v)[found] for k, v in inj.items()}
    log_g = pop_model.log_prob(inj_found, Lg)
    return inj_found, log_g


@app.cell
def _(catalog, inj_found, log_g, pop_model, staleness_done):
    _ = staleness_done
    COLUMNS = ["log_L", "distance"]
    raw = catalog.dataproduct(N=1000)
    gauss = catalog.dataproduct(representation="gauss")
    E = raw["prior"].shape[0]
    M = 4000

    def log_alpha(Lambda):
        log_r = pop_model.log_prob(inj_found, Lambda) - log_g
        return jax.scipy.special.logsumexp(log_r) - jnp.log(50000)

    def raw_loglike(Lambda):
        log_w = pop_model.log_prob(raw, Lambda) - jnp.log(raw["prior"])                  # (E, N)
        log_z = jax.scipy.special.logsumexp(log_w, axis=-1) - jnp.log(log_w.shape[-1])
        return jnp.sum(log_z) - E * log_alpha(Lambda)

    def gauss_loglike(Lambda):
        theta = pop_model.sample(Lambda, key=jax.random.key(11), shape=(M,))
        x = jnp.stack([theta[c] for c in COLUMNS], axis=-1)                               # (M, D)
        diff = x[None] - gauss["mean"][:, None]                                            # (E, M, D)
        maha = jnp.einsum("emi,eij,emj->em", diff, jnp.linalg.inv(gauss["cov"]), diff)
        log_k = -0.5 * (maha + jnp.linalg.slogdet(gauss["cov"])[1][:, None]
                        + len(COLUMNS) * jnp.log(2 * jnp.pi))                              # (E, M)
        log_z = jax.scipy.special.logsumexp(log_k, axis=-1) - jnp.log(M)
        return jnp.sum(log_z) - E * log_alpha(Lambda)

    mu_grid = jnp.linspace(-2.0, 1.2, 60)
    sigma_grid = jnp.linspace(0.1, 1.6, 60)
    _MU, _SIG = jnp.meshgrid(mu_grid, sigma_grid)
    _points = {"mu_L": _MU.ravel(), "sigma_L": _SIG.ravel()}
    surfaces = {
        name: np.asarray(jax.lax.map(jax.jit(f), _points, batch_size=100)).reshape(_MU.shape)
        for name, f in [("raw draws", raw_loglike), ("Gaussian", gauss_loglike)]
    }
    return mu_grid, sigma_grid, surfaces


@app.cell(expand_output=True)
def _(Lambda_true, mu_grid, sigma_grid, surfaces):
    def _levels(log_like, masses=(0.95, 0.68)):
        """Likelihood levels enclosing the given probability masses (flat prior on the grid)."""
        p = np.exp(log_like - log_like.max())
        h = np.sort(p.ravel())[::-1]
        c = np.cumsum(h) / h.sum()
        return [h[np.searchsorted(c, m)] for m in masses], p

    _fig, _ax = plt.subplots(figsize=(5.6, 4.4))
    for (_name, _ll), _color in zip(surfaces.items(), [style.POSTERIOR, style.EXTRA]):
        _lv, _p = _levels(_ll)
        _ax.contourf(mu_grid, sigma_grid, _p, levels=[_lv[0], _lv[1], 1.0], colors=[_color, _color],
                     alpha=0.12)
        _ax.contour(mu_grid, sigma_grid, _p, levels=_lv, colors=[_color], linewidths=[1.2, 2.0])
        _ax.plot([], [], color=_color, label=_name)
    _ax.plot(Lambda_true["mu_L"], Lambda_true["sigma_L"], "o", color=style.TRUTH, label="truth")
    _ax.set(xlabel=LABELS["mu_L"], ylabel=LABELS["sigma_L"],
            title="68% / 95% regions: raw draws vs Gaussian fits")
    _ax.legend(loc="upper right")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both likelihoods have the same ridge, and both contain the truth. The Gaussian version is
    looser, though, and extends much further along the ridge towards low $\mu_L$. The Gaussians put
    probability beyond $r = 2\,\mathrm{kpc}$, where the population has none, and they blur the shape
    of each posterior near that edge, so some of the information about luminosity is lost. This is
    why the truncated Gaussian mixtures in `astroinfer_tgmm` model the truncation explicitly.

    ## Where to go next

    The `astroinfer_tgmm` package is a full version of what this notebook builds.
    `TGMMFitter(TGMMConfig(...))` fits truncated Gaussian mixtures, with per-event configurations,
    retries, transformed coordinates and padded data products, and registers them under the kind
    `"tgmm"`.

    The notebooks in `notebooks/gw/01–03` apply the same pieces to gravitational waves. They build a
    mock gravitational-wave catalog, fit its posteriors with TGMMs, and run population inference
    with an event-informed proposal and selection effects.

    ## Summary

    * A representation consists of a fitter (settings, `fit`, `fit_key`, `dataproduct_options`,
      `to_dict` / `from_dict`) and a fit (`fit_key`, `dataproduct`, `save` / `load`, and optionally
      `stack`), subclassed from `BaseFitter` and `BaseFit`.
    * `fit_key` hashes the settings that affect an event's fit together with its `source_key`, so a
      fit goes stale when either of them changes and at no other time.
    * `@ai.register`, or an entry point in a package, lets `Catalog.load` rebuild your kind.
    * `register_representation`, `fit`, `stale`, `dataproduct(representation=...)`, `save` and
      `ai.Catalog.load` work the same way for every kind.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: fitting a subsample

    Large posteriors are slow to fit. Add a `max_draws` setting, so that the fit uses a reproducible
    random subset of at most `max_draws` posterior draws (drawn with `np.random.default_rng(0)`,
    for example).

    1. Write `SubsampledGaussianFitter(GaussianFitter)` with a new `kind` and a `max_draws` setting.
       Does `max_draws` belong in `fit_key`? Should it be overridable per event?
    2. Register it as `"gauss_sub"` on the catalog with `max_draws=200` and fit.
    3. Change `max_draws` and check `catalog.stale("gauss_sub")`. Then change `N`. Did each change
       have the effect you expected?
    """)
    return


@app.cell
def _():
    # your code here
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"Show a solution": mo.md(r"""
    ```python
    @ai.register
    class SubsampledGaussianFitter(GaussianFitter):
        kind = "gaussian_subsampled"     # a new kind, so saved catalogs know which class to use

        def __init__(self, columns, *, max_draws=1000, **kwargs):
            super().__init__(columns, **kwargs)
            self.max_draws = int(max_draws)

        def settings_for(self, eventname):
            # max_draws changes the fit, so it goes into the settings and therefore into
            # fit_key. Per-event overrides of it then work without extra code.
            return {"max_draws": self.max_draws, **super().settings_for(eventname)}

        def fit(self, posterior, *, eventname, source_key):
            s = self.settings_for(eventname)
            n = posterior.batch_shape[0]
            keep = np.sort(np.random.default_rng(0).choice(n, min(n, s["max_draws"]), replace=False))
            return super().fit(posterior[keep], eventname=eventname, source_key=source_key)

        def to_dict(self):
            return {**super().to_dict(), "max_draws": self.max_draws}

        @classmethod
        def from_dict(cls, d):
            return cls(d["columns"], max_draws=d["max_draws"], inflate=d["inflate"],
                       N=d["N"], overrides=d["overrides"])

    catalog.register_representation("gauss_sub", SubsampledGaussianFitter(["log_L", "distance"],
                                                                          max_draws=200))
    catalog.fit("gauss_sub")
    catalog.fitter("gauss_sub").max_draws = 400
    print(len(catalog.stale("gauss_sub")))   # every event: max_draws is in the key
    catalog.fit("gauss_sub")
    catalog.fitter("gauss_sub").N = 50
    print(catalog.stale("gauss_sub"))        # []: N only shapes the data product
    ```

    `posterior[keep]` works because a `PosteriorData` indexes all its per-draw arrays together.
    """)})
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    import subprocess
    import sys
    from dataclasses import dataclass
    from pathlib import Path

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    from numpyro.distributions import constraints

    import astroinfer as ai
    from astroinfer import MeasurementProcess, parameter, parameter_dataclass
    from astroinfer.generative import Distribution, GenerativeModel, TruncatedNormal, product

    import _style as style

    style.apply()

    # The notebook's own folder (mo.notebook_dir() is the cwd when run as a script).
    try:
        TUTORIAL_DIR = Path(__file__).resolve().parent
    except NameError:
        TUTORIAL_DIR = Path(mo.notebook_dir())
    CATALOG_DIR = TUTORIAL_DIR / "data" / "star_catalog"

    N_EVENTS = 40         # detected stars to run PE on
    R_MIN, R_MAX = 0.5, 2.0
    LABELS = dict(log_L=r"$\log_{10} L$", distance=r"$r$ [kpc]",
                  mu_L=r"$\mu_L$", sigma_L=r"$\sigma_L$")



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 9. The star catalog

    In notebooks 7 and 8 the population likelihood went straight from the data $d_i$ to $\Lambda$,
    re-evaluating $\mathcal{L}(d_i\mid\theta)$ at every step. Real surveys rarely work that way. Parameter
    estimation is expensive, so it is run once per event, and the posteriors are stored and shared
    between many population analyses.

    `ai.Catalog` stores them. A catalog holds one `PosteriorData` per *event*, along with metadata.
    It exports the posteriors as an event-major *data product*, `{column: (E, N), "prior": (E, N)}`,
    and it saves itself to a folder and loads itself back. It can also carry compressed
    *representations* of the posteriors, which are the subject of notebook 10.

    At the end of this notebook we rerun the survey's population inference using only the catalog,
    without simulating the telescope again.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src=mo.notebook_dir() / "images" / "09_star_catalog.svg",
             alt="A catalog in which every star holds a cloud of posterior samples",
             width=640)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The survey

    The survey is the one from notebook 7. Stars have a luminosity and a distance, the telescope
    measures a noisy flux and a noisy parallax, and only stars with an observed
    $\log_{10}\hat F > -0.3$ are detected. The population is a truncated
    normal in $\log_{10}L$ and uniform-in-volume distances, $p(r)\propto r^2$.
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

    stars = Star.from_dict(pop_model.sample(Lambda_true, key=jax.random.key(0), shape=(5000,)))
    obs = telescope.sample(stars, key=1)
    det = trigger.detect(obs, np.random.default_rng(2))
    idx = np.flatnonzero(det)[:N_EVENTS]
    data, true_det = obs[idx], stars[idx]

    mo.md(f"{det.sum():,} of {len(det):,} simulated stars are detected. We run PE on the first "
          f"{len(idx)}.")
    return Lambda_true, data, pop_model, telescope, trigger, true_det


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Building the catalog

    `telescope.PE(d, priors, as_posterior=True)` returns a `PosteriorData` (notebook 4). It records
    the PE prior density of every draw as `prior`, plus one `log_prior[...]` factor per parameter.
    The catalog uses both later on.

    A catalog is created with a `RawSamples` setting, which says which columns the data product
    exports and how many draws it keeps per event (`N`). Each posterior is then added with
    `add_event(name, posterior)`. Anything else you want to keep with an event, such as the injected
    truth, goes in `metadata=`.

    PE takes about half a second per star. The catalog is saved to `tutorials/data/star_catalog`. If
    that folder already exists, the cell below loads it instead of rerunning PE, unless you tick the
    box.
    """)
    return


@app.cell
def _():
    pe_priors = dict(log_L=dist.Uniform(-3.0, 5.0), distance=dist.Uniform(R_MIN, R_MAX))
    force_pe = mo.ui.checkbox(label="re-run PE (overwrites tutorials/data/star_catalog)")
    force_pe
    return force_pe, pe_priors


@app.cell
def _(data, force_pe, pe_priors, telescope, true_det):
    if CATALOG_DIR.exists() and not force_pe.value:
        catalog = ai.Catalog.load(CATALOG_DIR, types=[Star, StarObservation])
    else:
        catalog = ai.Catalog(raw=ai.RawSamples(columns=["log_L", "distance"], N=1000))
        for _i in mo.status.progress_bar(range(N_EVENTS), title="Parameter estimation"):
            _posterior = telescope.PE(data[_i], pe_priors, seed=_i, as_posterior=True,
                                      num_warmup=300, num_samples=1500)
            _truth = {k: float(v) for k, v in true_det[_i].to_dict().items()}
            catalog.add_event(f"star_{_i:02d}", _posterior, metadata={"truth": _truth})
        catalog.save(CATALOG_DIR)
    catalog
    return (catalog,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The expanded catalog lists its representations (only `raw` so far), its events, and their
    posteriors. A catalog behaves like a read-only dict of posteriors, so `catalog.eventnames`,
    `len(catalog)`, `"star_03" in catalog` and `catalog["star_03"]` all work.
    """)
    return


@app.cell
def _(catalog):
    event = mo.ui.dropdown(catalog.eventnames, value=catalog.eventnames[3], label="event")
    event
    return (event,)


@app.cell(expand_output=True)
def _(catalog, event):
    _posterior = catalog[event.value]
    _draws = _posterior.variables(["log_L", "distance"])
    _fig, _ = style.corner([_draws], ["log_L", "distance"],
                           labels=[LABELS["log_L"], LABELS["distance"]],
                           truths=_posterior.metadata["truth"],
                           ranges=dict(distance=(R_MIN, R_MAX)))
    mo.hstack([_posterior, _fig], widths=[1, 1], align="start")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The data product

    `catalog.dataproduct()` turns the catalog into arrays that a population likelihood can use.
    Every event is resampled to the same number of draws `N`, and the events are stacked along the
    first axis:
    """)
    return


@app.cell
def _(catalog):
    product_raw = catalog.dataproduct()
    mo.vstack([
        catalog.raw,
        pd.DataFrame({k: [str(tuple(v.shape))] for k, v in product_raw.items()}, index=["shape"]),
    ])
    return (product_raw,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `prior` array holds the PE prior density $\pi(\theta_{ij})$ of each exported draw, counting
    only the exported columns. If you export a subset of the parameters, the catalog multiplies
    together the `log_prior[...]` factors of those parameters and leaves out the others, which are
    marginalised over. This is why PE stores one factor per parameter. Both priors here are
    uniform, so `prior` is the constant $\frac{1}{8}\cdot\frac{1}{1.5}\approx 0.083$.

    The resampling seed for each event is derived from the catalog seed and the event's name. As a
    result, asking for a subset of events, or adding new events, does not change the draws of the
    other events:
    """)
    return


@app.cell
def _(catalog, product_raw):
    _name = catalog.eventnames[5]
    _alone = catalog.dataproduct(eventnames=[_name])
    _same = bool(jnp.all(_alone["log_L"][0] == product_raw["log_L"][5]))
    mo.md(f"""
    * `prior` is constant: min = {float(product_raw["prior"].min()):.4f},
      max = {float(product_raw["prior"].max()):.4f}.
    * `{_name}` exported alone gives the same draws as in the full product: `{_same}`.
    * `catalog.dataproduct(N=200)` changes only the number of draws:
      `{tuple(catalog.dataproduct(N=200)["log_L"].shape)}`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Saving and loading

    A catalog is only written to disk when you call `catalog.save(root)`. Saving to the same folder
    again rewrites only the parts that changed. Here are the files on disk:
    """)
    return


@app.cell
def _(catalog):
    _ = catalog
    _files = sorted(p.relative_to(CATALOG_DIR) for p in CATALOG_DIR.rglob("*") if p.is_file())
    _shown = [str(p) for p in _files[:7]] + [f"... {len(_files) - 7} more files"]
    mo.md("```text\n" + CATALOG_DIR.name + "/\n" + "\n".join("  " + s for s in _shown) + "\n```")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each saved posterior records the import path of its parameter class, for example
    `__main__:Star`. A class defined in a notebook cell cannot be imported from that path in another
    session or script, so you have to pass the classes to `Catalog.load` with `types=`:

    ```python
    catalog = ai.Catalog.load(CATALOG_DIR, types=[Star, StarObservation])
    ```

    `StarObservation` is needed too, because each posterior also stores the observation $d$ it was
    computed from (`posterior.observed_data`). Classes that live in an installed package, such as
    `astroinfer_gw.gwmockcat.GWParams`, are found automatically.
    """)
    return


@app.cell
def _(catalog):
    _ = catalog
    _info = ai.io.read_json(CATALOG_DIR / "representations" / "raw" / "events" / "star_00" / "posterior.json")

    # A fresh Python process that has never seen `Star`, loading without `types=`:
    _code = f"import astroinfer as ai; ai.Catalog.load({str(CATALOG_DIR)!r})"
    _run = subprocess.run([sys.executable, "-c", _code], capture_output=True, text=True)
    _error = _run.stderr.strip().splitlines()[-1] if _run.returncode else "(it loaded)"

    _reloaded = ai.Catalog.load(CATALOG_DIR, types=[Star, StarObservation])
    mo.vstack([
        mo.md(f"Saved parameter type: `{_info['parameter_type']}`; observed data type: "
              f"`{_info['observed_data']['type']}`. A fresh Python process without `types=` gets:"),
        mo.callout(mo.md(f"`{_error}`"), kind="warn"),
        mo.md(f"With `types=`: `{_reloaded!r}`"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Source keys

    `catalog.source_key(name)` is a SHA-256 hash of an event's posterior draws. If you replace a
    posterior with `add_event(name, new_posterior, overwrite=True)`, its key changes. In notebook 10
    the catalog uses this key to tell when a compressed fit of an event is out of date.

    We try this on a copy loaded from disk, so that the catalog above is left as it is:
    """)
    return


@app.cell
def _(catalog, data, pe_priors, telescope):
    _ = catalog
    scratch = ai.Catalog.load(CATALOG_DIR, types=[Star, StarObservation])
    _before = scratch.source_key("star_00")
    _rerun = telescope.PE(data[0], pe_priors, seed=999, as_posterior=True,
                          num_warmup=300, num_samples=1500)
    scratch.add_event("star_00", _rerun, overwrite=True)
    _after = scratch.source_key("star_00")
    mo.md(f"""
    | `star_00` | source key |
    |---|---|
    | saved PE (seed 0) | `{_before[:16]}...` |
    | re-run PE (seed 999) | `{_after[:16]}...` |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Population inference from the catalog

    The posterior draws $\theta_{ij}\sim p(\theta\mid d_i)$ already contain $\mathcal{L}(d_i\mid\theta)$,
    multiplied by the PE prior they were drawn with,
    $p(\theta\mid d_i)\propto \mathcal{L}(d_i\mid\theta)\,\pi(\theta)$. The evidence of each event under the
    population is therefore an average over the stored draws, reweighted from the PE prior to the
    population:

    $$
    Z_i(\Lambda)=\int \mathcal{L}(d_i\mid\theta)\,p(\theta\mid\Lambda)\,d\theta
    \;\propto\;\frac{1}{N}\sum_{j=1}^{N}\frac{p(\theta_{ij}\mid\Lambda)}{\pi(\theta_{ij})}.
    $$

    The data product contains the arrays this needs: the draws, and `prior`, which is $\pi$. With
    the selection term $-E\ln\alpha(\Lambda)$ from notebook 7 added (same injections, same
    estimator), the likelihood class below is about a dozen lines long. `ai.run_sampler` accepts any
    object that has a `.logpdf(Lambda)` method.
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
    mo.md(f"{found.sum():,} of 50,000 injections are found.")
    return inj_found, log_g


@app.class_definition
class RecycledLikelihood:
    """ln L(Lambda) = sum_i ln mean_j p(theta_ij | Lambda) / pi(theta_ij)  -  E ln alpha(Lambda)."""

    def __init__(self, product, pop_model, *, injections=None, log_g=None, n_injections=None):
        self.product, self.pop_model = product, pop_model
        self.injections, self.log_g, self.n_injections = injections, log_g, n_injections
        self.E, self.N = product["prior"].shape

    def event_log_weights(self, Lambda):
        """ln p(theta_ij | Lambda) - ln pi(theta_ij), shape (E, N)."""
        return self.pop_model.log_prob(self.product, Lambda) - jnp.log(self.product["prior"])

    def log_alpha(self, Lambda):
        log_r = self.pop_model.log_prob(self.injections, Lambda) - self.log_g
        return jax.scipy.special.logsumexp(log_r) - jnp.log(self.n_injections)

    def logpdf(self, Lambda):
        log_w = self.event_log_weights(Lambda)
        log_z = jax.scipy.special.logsumexp(log_w, axis=-1) - jnp.log(self.N)
        if self.injections is None:            # no selection term
            return jnp.sum(log_z)
        return jnp.sum(log_z) - self.E * self.log_alpha(Lambda)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Before sampling, we check the Monte Carlo sums at the true $\Lambda$. The effective number of
    draws per event, $N_\mathrm{eff}=(\sum_j w_{ij})^2/\sum_j w_{ij}^2$, should be well above 1 for
    every event. If it is not, that event's $Z_i$ is noisy, and you need more draws or a better
    representation.
    """)
    return


@app.cell
def _(Lambda_true, inj_found, log_g, pop_model, product_raw):
    recycled = RecycledLikelihood(product_raw, pop_model, injections=inj_found, log_g=log_g,
                                  n_injections=50000)
    _log_w = recycled.event_log_weights(Lambda_true)
    _n_eff = np.asarray(jnp.exp(2 * jax.scipy.special.logsumexp(_log_w, axis=-1)
                                - jax.scipy.special.logsumexp(2 * _log_w, axis=-1)))

    _fig, _ax = plt.subplots(figsize=(6.4, 2.6))
    _ax.bar(np.arange(len(_n_eff)), _n_eff, color=style.POSTERIOR, width=0.8)
    _ax.set(xlabel="event", ylabel=r"$N_\mathrm{eff}$", ylim=(0, recycled.N),
            title=f"Effective draws per event at the true Λ (of N = {recycled.N})")
    _fig
    return (recycled,)


@app.cell
def _(pop_model, product_raw, recycled):
    hyper_priors = dict(mu_L=dist.Uniform(-2.0, 2.0), sigma_L=dist.Uniform(0.1, 2.0))
    _kw = dict(seed=1, num_warmup=500, num_samples=2000, summary=False)

    hyper_selected = ai.run_sampler(recycled, hyper_priors, **_kw).samples
    hyper_naive = ai.run_sampler(RecycledLikelihood(product_raw, pop_model), hyper_priors, **_kw).samples
    return hyper_naive, hyper_priors, hyper_selected


@app.cell(expand_output=True)
def _(Lambda_true, hyper_naive, hyper_selected):
    _fig, _ = style.corner(
        [hyper_selected, hyper_naive], ["mu_L", "sigma_L"],
        labels=[LABELS["mu_L"], LABELS["sigma_L"]], colors=[style.POSTERIOR, style.EXTRA],
        truths=Lambda_true, ranges=dict(mu_L=(-2.0, 1.2), sigma_L=(0.1, 1.6)), size=2.6,
        names=["with selection", "without selection"])
    mo.vstack([
        mo.md("### The hyperposterior from stored posteriors"),
        _fig,
        mo.md(f"With selection: $\\mu_L$ = {hyper_selected.mu_L.median():.2f} "
              f"(90%: {hyper_selected.mu_L.quantile(0.05):.2f} to {hyper_selected.mu_L.quantile(0.95):.2f}), "
              f"$\\sigma_L$ = {hyper_selected.sigma_L.median():.2f}. Truth: 0 and 0.7. "
              "Without the selection term, the bias towards bright stars from notebook 7 returns. "
              "With 40 stars the corrected posterior is broad and stretched along a ridge, because "
              "we only see the bright tail, and a low $\\mu_L$ with a wide $\\sigma_L$ explains it "
              "almost as well."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    * `ai.Catalog(raw=ai.RawSamples(columns=[...], N=...))` stores one `PosteriorData` per event.
      `add_event(name, posterior, metadata=...)` adds an event, and `overwrite=True` replaces one.
    * `catalog.dataproduct()` returns `{column: (E, N), "prior": (E, N)}`, where `prior` is the PE
      prior of the exported columns. Each event's resampling depends only on its name.
    * `catalog.save(root)` writes only what changed. `ai.Catalog.load(root, types=[...])` needs the
      classes you defined in the notebook.
    * Population inference from stored posteriors reweights the draws,
      $Z_i\propto\langle p(\theta\mid\Lambda)/\pi(\theta)\rangle_{\theta\sim p(\theta\mid d_i)}$, and
      `ai.run_sampler` runs any object with a `.logpdf(Lambda)` method.

    Raw draws are one way to pass a posterior to a population analysis. In notebook 10 you write a
    compressed representation of your own and add it to the catalog.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: a bigger catalog

    The survey detected far more than 40 stars.

    1. Load the saved catalog into a new variable, run PE on detected stars 40 to 79 and add them
       with `add_event` (`star_40`, ...). Save the result to a new folder, for example
       `tutorials/data/star_catalog_80`.
    2. Re-run the population inference (with selection) on the 80-star data product.
    3. Compare the 40-star and 80-star hyperposteriors in a corner plot. Which part of the
       posterior tightens, and which part hardly changes? Did the draws of the original 40 events in
       the data product change?

    Hint: `data` and `true_det` only hold the first 40 detections, so you need to re-slice
    `obs[det]`.
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
    # Re-create the simulation from above, keeping 80 detections this time.
    _stars = Star.from_dict(pop_model.sample(Lambda_true, key=jax.random.key(0), shape=(5000,)))
    _obs = telescope.sample(_stars, key=1)
    _det = trigger.detect(_obs, np.random.default_rng(2))
    _idx = np.flatnonzero(_det)[:80]
    data80, truth80 = _obs[_idx], _stars[_idx]

    big = ai.Catalog.load(CATALOG_DIR, types=[Star, StarObservation])
    for i in range(40, 80):
        post = telescope.PE(data80[i], pe_priors, seed=i, as_posterior=True,
                            num_warmup=300, num_samples=1500)
        big.add_event(f"star_{i:02d}", post,
                      metadata={"truth": {k: float(v) for k, v in truth80[i].to_dict().items()}})
    big.save(TUTORIAL_DIR / "data" / "star_catalog_80")

    big_product = big.dataproduct()
    lik80 = RecycledLikelihood(big_product, pop_model, injections=inj_found, log_g=log_g,
                               n_injections=50000)
    hyper80 = ai.run_sampler(lik80, hyper_priors, seed=1, num_warmup=500,
                             num_samples=2000, summary=False).samples
    style.corner([hyper_selected, hyper80], ["mu_L", "sigma_L"], truths=Lambda_true,
                 ranges=dict(mu_L=(-2.0, 1.2), sigma_L=(0.1, 1.6)))
    # The bulk near the truth tightens, but the tail along the ridge (low mu_L with wide sigma_L)
    # hardly shrinks, because we only see the bright end of the luminosity function.

    # Seeds depend only on the event names, so the first 40 rows are unchanged.
    print(bool(jnp.all(big_product["log_L"][:40] == product_raw["log_L"])))
    ```
    """)})
    return


if __name__ == "__main__":
    app.run()

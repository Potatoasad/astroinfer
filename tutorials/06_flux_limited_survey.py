import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    from dataclasses import dataclass

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    from numpyro.distributions import constraints
    from scipy.special import expit

    import astroinfer as ai
    from astroinfer import MeasurementProcess, parameter, parameter_dataclass
    from astroinfer.generative import (
        Distribution,
        GenerativeModel,
        TruncatedNormal,
        product,
        truncnorm_pdf,
    )

    import _style as style

    style.apply()

    R_MIN, R_MAX = 0.5, 2.0   # the survey volume: a shell from 0.5 to 2 kpc
    N_SIM = 5_000             # stars simulated in the volume
    N_EVENTS = 150            # detected stars used in the analysis
    N_DRAWS = 2_000           # Monte Carlo draws per star in the population likelihood
    N_INJ = 50_000            # injections for the selection term


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 6 · A flux-limited survey 🔭

    A telescope surveys a shell of stars between 0.5 and 2 kpc. For each star it measures a
    **flux** and a **parallax**, but it only *catalogs* stars whose measured flux beats a limit.
    Faint stars far away never make it in.

    We want the population of stellar luminosities, $p(\log_{10}L\mid\Lambda)$. If we feed the
    cataloged stars into the hierarchical analysis of notebook 5 as if they were a fair sample,
    we get the wrong answer: the catalog is biased towards bright stars. That is **Malmquist bias**,
    and the fix is a *selection term*.

    This notebook adds the last piece of the forward model,

    $$
    \theta\;\xrightarrow{\;K(d\mid\theta)\;}\;d\;\xrightarrow{\;P(\mathrm{det}\mid d)\;}\;\text{in the catalog or not},
    $$

    a **detection trigger**, and shows how to correct the population likelihood for it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The model, recapped

    Everything here is built from pieces you already know.

    * $\theta$ = (log luminosity $\log_{10}L$, distance $r$ in kpc).
    * $d$ = (measured log flux, measured parallax in mas). The flux falls off as $1/r^2$, so
      $\log_{10}F = \log_{10}L - 2\log_{10}r$, and the parallax is $1/r$.
    * The population: $\log_{10}L\sim\mathcal{TN}(\mu_L,\sigma_L)$, and stars uniform in volume,
      $p(r)\propto r^2$, using the `PowerLaw` leaf from notebook 4.
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

    def _display_equations(self):
        return [
            r"\log_{10}\hat F \sim \mathcal{N}\!\left(\log_{10}L - 2\log_{10}r,\ \sigma_F\right)",
            r"\hat\varpi \sim \mathcal{N}\!\left(1/r,\ \sigma_\varpi\right)",
        ]


@app.cell
def _():
    telescope = Telescope()
    pop_model = GenerativeModel(product([
        TruncatedNormal("log_L", "mu_L", "sigma_L", -3.0, 5.0),
        PowerLaw("distance", 2.0, R_MIN, R_MAX)]), order=["log_L", "distance"])
    Lambda_true = dict(mu_L=0.0, sigma_L=0.7)
    telescope
    return Lambda_true, pop_model, telescope


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Detection triggers

    A detection trigger answers one question: given the data $d$, what is the probability that
    this star ends up in the catalog? Note that it reads the **data**, not $\theta$: the telescope
    only knows what it measured. Noise can push a star over the limit, or under it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **🧩 What you implement in a `DetectionTrigger`**

    | method | what it is |
    |---|---|
    | `detection_probability(self, d)` | $P(\mathrm{det}\mid d)\in[0,1]$, **vectorised** over a batch of data |
    | `selection_statistic(self, d)` *(optional)* | the number detection is based on (an SNR, a flux), recorded with injections |
    | `_display_rule()`, `_display_config()` *(optional)* | LaTeX lines and settings for the rich display |

    Unlike a measurement process, a trigger works on a whole batch at once, and plain NumPy is
    fine: detection is never differentiated. `d` arrives as your data class, a dict or a DataFrame,
    and `d["log_flux"]` works for all three. You get `.detect(d, rng)` and
    `.injections(measurement, draws, rng)` for free.

    The built-in `ai.detection.ThresholdTrigger(threshold, statistic=...)` is a hard cut on one
    field of the data.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Our survey uses a hard cut, `ThresholdTrigger(-0.3, statistic="log_flux")`. Real surveys are
    rarely that sharp, so here is a trigger of our own: a **soft** flux limit, where the detection
    probability rises smoothly over a width $w$ around the limit.
    """)
    return


@app.class_definition
class SoftFluxLimit(ai.detection.DetectionTrigger):
    """P(det | d) = sigmoid((log_flux - threshold) / width)."""

    display_name = "Soft flux limit"

    def __init__(self, threshold=-0.3, width=0.1):
        self.threshold, self.width = threshold, width

    def selection_statistic(self, d):
        return np.asarray(d["log_flux"], dtype=np.float64)

    def detection_probability(self, d):
        return expit((self.selection_statistic(d) - self.threshold) / self.width)

    def _display_rule(self):
        return [r"P(\mathrm{det}\mid d) = \mathrm{sigmoid}\!\left("
                r"\frac{\log_{10}\hat F - F_\mathrm{lim}}{w}\right)"]

    def _display_config(self):
        return {"threshold F_lim": self.threshold, "width w": self.width}


@app.cell
def _():
    trigger = ai.detection.ThresholdTrigger(threshold=-0.3, statistic="log_flux")
    soft = SoftFluxLimit(threshold=-0.3, width=0.1)

    _grid = jnp.linspace(-1.0, 0.5, 400)
    _d = StarObservation(log_flux=_grid, parallax=jnp.ones_like(_grid))

    _fig, _ax = plt.subplots(figsize=(6.4, 2.8))
    _ax.plot(_grid, trigger.detection_probability(_d), color=style.INK_2, label="ThresholdTrigger")
    _ax.plot(_grid, soft.detection_probability(_d), color=style.POSTERIOR, label="SoftFluxLimit")
    _ax.set(xlabel=r"measured $\log_{10}\hat F$", ylabel=r"$P(\mathrm{det}\mid d)$",
            title="Two detection triggers")
    _ax.legend(loc="upper left")
    mo.vstack([mo.hstack([trigger, soft], justify="start", gap=2), _fig])
    return (trigger,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulate: truths → data → detection

    There is no simulator object: the whole survey is three explicit lines. Draw true stars from
    the population, observe every one of them, and ask the trigger which are detected.
    `trigger.detect(d, rng)` returns a boolean mask, and a batched `Parameters` object can be indexed
    with it.
    """)
    return


@app.cell
def _(Lambda_true, pop_model, telescope, trigger):
    truths = Star.from_dict(pop_model.sample(Lambda_true, key=jax.random.key(0), shape=(N_SIM,)))
    observed = telescope.sample(truths, key=1)
    detected = trigger.detect(observed, np.random.default_rng(2))

    keep = np.flatnonzero(detected)[:N_EVENTS]
    data, detected_truths = observed[keep], truths[keep]

    mo.md(f"**{detected.sum():,}** of {N_SIM:,} stars are detected ({detected.mean():.0%}). "
          f"The analysis uses the first **{len(keep)}**: `data.batch_shape = {data.batch_shape}`.")
    return data, detected, detected_truths, observed, truths


@app.cell
def _(detected, truths):
    _r = np.asarray(truths.distance)
    _logL = np.asarray(truths.log_L)
    _rr = np.linspace(R_MIN, R_MAX, 200)

    _fig, (_ax, _ax_h) = plt.subplots(1, 2, figsize=(9.6, 3.6), width_ratios=[1.6, 1],
                                      layout="constrained")
    _ax.scatter(_r[~detected], _logL[~detected], s=3, color=style.PRIOR, alpha=0.35,
                linewidths=0, label="missed", rasterized=True)
    _ax.scatter(_r[detected], _logL[detected], s=3, color=style.DATA, alpha=0.6,
                linewidths=0, label="detected", rasterized=True)
    _ax.plot(_rr, -0.3 + 2 * np.log10(_rr), color=style.INK, linewidth=1.4,
             label="flux limit")
    _ax.set(xlabel="distance $r$ [kpc]", ylabel=r"$\log_{10}L$", ylim=(-2.5, 2.5),
            title="Far away, only bright stars make the cut")
    _ax.legend(loc="lower right", markerscale=3)

    _bins = np.linspace(-2.5, 2.5, 50)
    style.density(_ax_h, _logL, bins=_bins, color=style.PRIOR, label="all stars")
    style.density(_ax_h, _logL[detected], bins=_bins, color=style.DATA, label="detected")
    _ax_h.set(xlabel=r"$\log_{10}L$", ylabel="density", title="Luminosities")
    _ax_h.legend(loc="upper left")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A few detected stars sit just below the line: noise pushed their *measured* flux over the
    limit. Overall the detected stars are about half a dex brighter than the population they came
    from. Let's see
    what a naive analysis makes of that.

    ## The naive analysis

    Exactly notebook 5: hyperpriors on $\Lambda=(\mu_L,\sigma_L)$ and
    `telescope.population_inference`. First the habitual check of the Monte Carlo integrals at the
    true $\Lambda$.
    """)
    return


@app.cell
def _(Lambda_true, data, pop_model, telescope):
    hyper_priors = dict(mu_L=dist.Uniform(-2.0, 2.0), sigma_L=dist.Uniform(0.1, 2.0))

    _check = ai.PopulationLikelihood(telescope, data, pop_model, N=N_DRAWS, seed=3)
    _, _neff = jax.jit(_check.event_logpdf_and_neff)(Lambda_true)
    mo.md(f"Per-star $N_\\mathrm{{eff}}$ out of {N_DRAWS:,} draws: median "
          f"**{float(jnp.median(_neff)):.0f}**, minimum **{float(_neff.min()):.0f}**. Healthy.")
    return (hyper_priors,)


@app.cell
def _(data, hyper_priors, pop_model, telescope):
    naive = telescope.population_inference(
        data, pop_model, hyper_priors, N=N_DRAWS, seed=3, num_warmup=300, num_samples=1000,
    ).samples
    naive[["mu_L", "sigma_L"]].describe().loc[["mean", "std"]]
    return (naive,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The truth is $\mu_L=0$, $\sigma_L=0.7$. The naive analysis is confidently wrong: it finds the
    population of *detected* stars, which is brighter and narrower.

    ## The selection term

    The likelihood of notebook 5 multiplies, for each star, the probability of its data. But our
    data are not "a star's data", they are "a star's data, **given that it was detected**". Each
    factor must be normalised by the probability of detection:

    $$
    \mathcal L(\Lambda)=\prod_{i=1}^{E}\frac{Z_i(\Lambda)}{\alpha(\Lambda)},
    \qquad
    \alpha(\Lambda)=\iint P(\mathrm{det}\mid d)\,K(d\mid\theta)\,p(\theta\mid\Lambda)\;\mathrm d d\,\mathrm d\theta .
    $$

    $\alpha(\Lambda)$ is the fraction of the population that the survey can see. A population full of
    faint stars has a small $\alpha$, and that is exactly what the naive analysis was missing: it
    did not pay a price for proposing populations in which detection would have been *likely*.

    ### Estimating $\alpha(\Lambda)$ with injections

    Draw $\theta_j$ from a broad, fixed population $g(\theta)$, simulate each one through the
    telescope once, and trigger. Then by importance sampling

    $$
    \hat\alpha(\Lambda)=\frac{1}{N_\mathrm{inj}}\sum_{j\ \in\ \text{found}}\frac{p(\theta_j\mid\Lambda)}{g(\theta_j)} .
    $$

    `trigger.injections(measurement, draws, rng)` does the simulate-and-trigger step: hand it a
    DataFrame of draws (with $g$'s density in a `prior` column) and it returns the injection table.
    """)
    return


@app.cell
def _(pop_model, telescope, trigger):
    Lambda_g = dict(mu_L=0.5, sigma_L=1.5)          # broader and brighter than any population we expect
    _g_draws = pop_model.sample(Lambda_g, key=jax.random.key(5), shape=(N_INJ,))
    g_table = pd.DataFrame({k: np.asarray(v) for k, v in _g_draws.items()})
    g_table["prior"] = np.exp(np.asarray(pop_model.log_prob(_g_draws, Lambda_g)))

    injections = trigger.injections(telescope, g_table, np.random.default_rng(6))
    injections.head()
    return g_table, injections


@app.cell
def _(injections):
    found = injections[injections.detected]
    injections_found = {k: jnp.asarray(found[k].to_numpy()) for k in ["log_L", "distance"]}
    log_g = jnp.log(jnp.asarray(found["prior"].to_numpy()))
    mo.md(f"**{len(found):,}** of {len(injections):,} injections are found.")
    return injections_found, log_g


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **🧩 Customising the population likelihood: subclass it.**
    `PopulationLikelihood.logpdf(Lambda)` returns $\sum_i\log\hat Z_i(\Lambda)$. Override it, call
    `super().logpdf(Lambda)`, and add whatever extra terms your problem needs. Here: $-E\log\hat\alpha$.
    The subclass's extra `__init__` arguments carry the injections. Everything stays a JAX function
    of $\Lambda$, so NUTS still gets gradients.
    """), kind="info")
    return


@app.class_definition
class SelectedLikelihood(ai.PopulationLikelihood):
    """Population likelihood with the selection term -E log alpha(Lambda) from found injections."""

    def __init__(self, *a, injections, log_g, n_injections, **kw):
        super().__init__(*a, **kw)
        self.injections, self.log_g, self.n_injections = injections, log_g, n_injections

    def log_alpha(self, Lambda):
        log_r = self.pop_model.log_prob(self.injections, Lambda) - self.log_g
        return jax.scipy.special.logsumexp(log_r) - jnp.log(self.n_injections)

    def logpdf(self, Lambda):
        return super().logpdf(Lambda) - self.E * self.log_alpha(Lambda)

    def selection_neff(self, Lambda):
        """Effective number of injections behind alpha-hat (want >> 4 E)."""
        log_r = self.pop_model.log_prob(self.injections, Lambda) - self.log_g
        return jnp.exp(2 * jax.scipy.special.logsumexp(log_r)
                       - jax.scipy.special.logsumexp(2 * log_r))


@app.cell
def _(Lambda_true, data, injections_found, log_g, pop_model, telescope):
    selected_likelihood = SelectedLikelihood(
        telescope, data, pop_model, N=N_DRAWS, seed=3,
        injections=injections_found, log_g=log_g, n_injections=N_INJ,
    )
    _alpha = float(jnp.exp(selected_likelihood.log_alpha(Lambda_true)))
    _neff = float(selected_likelihood.selection_neff(Lambda_true))
    mo.md(f"At the true $\\Lambda$: $\\hat\\alpha$ = **{_alpha:.3f}** (compare the detected fraction "
          f"of the simulation above), and the selection estimate rests on $N_\\mathrm{{eff}}$ = "
          f"**{_neff:,.0f}** injections (want $\\gg 4E$ = {4 * N_EVENTS}).")
    return (selected_likelihood,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `telescope.population_inference` builds a plain `PopulationLikelihood` for you. With a likelihood
    object of your own, call the sampler directly with `ai.run_sampler(likelihood, priors, ...)`.
    """)
    return


@app.cell
def _(hyper_priors, selected_likelihood):
    corrected = ai.run_sampler(selected_likelihood, hyper_priors, seed=1,
                               num_warmup=300, num_samples=1000, summary=False).samples
    corrected[["mu_L", "sigma_L"]].describe().loc[["mean", "std"]]
    return (corrected,)


@app.cell
def _(Lambda_true, corrected, naive):
    _fig, _ = style.corner(
        [naive, corrected], ["mu_L", "sigma_L"],
        labels=[r"$\mu_L$", r"$\sigma_L$"],
        names=["without selection", "with selection"],
        colors=[style.EXTRA, style.POSTERIOR],
        truths=Lambda_true, ranges={"mu_L": (-1.5, 1.2), "sigma_L": (0.2, 1.3)},
        size=2.6,
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With the selection term the truth is back inside the posterior. Notice the long diagonal
    ridge: a lower mean with a wider spread fits just as well. Why? Plot the inferred
    luminosity functions:
    """)
    return


@app.cell
def _(Lambda_true, corrected, detected_truths, naive):
    _x = np.linspace(-2.5, 2.5, 300)
    _fig, _ax = plt.subplots(figsize=(6.4, 3.4))
    for _samples, _color, _name in [(naive, style.EXTRA, "without selection"),
                                    (corrected, style.POSTERIOR, "with selection")]:
        _draws = _samples.sample(60, random_state=0)
        for _j, (_mu, _sig) in enumerate(zip(_draws.mu_L, _draws.sigma_L)):
            _ax.plot(_x, truncnorm_pdf(_x, _mu, _sig, -3.0, 5.0), color=_color, alpha=0.12,
                     linewidth=1, label=_name if _j == 0 else None)
    _ax.plot(_x, truncnorm_pdf(_x, Lambda_true["mu_L"], Lambda_true["sigma_L"], -3.0, 5.0),
             color=style.TRUTH, linestyle="--", linewidth=1.6, label="true population")
    _ax.hist(np.asarray(detected_truths.log_L), bins=20, density=True, histtype="step",
             color=style.DATA, linewidth=1.4, label="detected stars")
    _ax.set(xlabel=r"$\log_{10}L$", ylabel=r"$p(\log_{10}L\mid\Lambda)$", ylim=(0, 1.3),
            title="Inferred luminosity functions")
    _leg = _ax.legend(loc="upper left")
    for _h in _leg.legend_handles:
        _h.set_alpha(1)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    All the corrected curves agree on the **bright side**, where the survey sees stars, and fan out
    on the **faint side**, where it sees almost nothing. Selection effects can remove a bias, but
    they cannot conjure up information about stars you never detected: the faint tail is constrained
    only through the *number* of stars you did not see, via $\alpha(\Lambda)$.

    ## Recap

    * A `DetectionTrigger` implements `detection_probability(d)` (vectorised, NumPy is fine) and
      optionally `selection_statistic(d)`. `ThresholdTrigger` is the built-in hard cut.
    * Simulating a survey is three explicit lines: `pop_model.sample` → `measurement.sample` →
      `trigger.detect`, then index the data with the mask.
    * `trigger.injections(measurement, draws, rng)` scatters draws from a broad $g$ once and
      records detection; found injections give $\hat\alpha(\Lambda)$.
    * Customise the population likelihood by **subclassing** `PopulationLikelihood` and overriding
      `logpdf`; run it with `ai.run_sampler`.
    * Check both Monte Carlo integrals: per-event $N_\mathrm{eff}$ and selection $N_\mathrm{eff}$.

    In **notebook 7** we buy a much better telescope, and the population-draw integrals stop
    working. The fix is a `Proposal`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ✏️ Your turn: a softer survey

    Real flux limits are fuzzy. Redo the analysis with `SoftFluxLimit(threshold=-0.3, width=0.1)`
    in place of the hard cut.

    1. Re-detect the *same* `observed` stars with the soft trigger and keep the first 150.
    2. Rebuild the injection table with the soft trigger.
    3. Run the selection-corrected inference. Which lines of code changed?
    4. Bonus: what goes wrong if you use the soft trigger for the catalog but the *hard* trigger
       for the injections?
    """)
    return


@app.cell
def _():
    # your code here
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({"💡 Show a solution": mo.md(r"""
    ```python
    soft = SoftFluxLimit(threshold=-0.3, width=0.1)

    soft_detected = soft.detect(observed, np.random.default_rng(12))
    soft_data = observed[np.flatnonzero(soft_detected)[:150]]

    soft_inj = soft.injections(telescope, g_table, np.random.default_rng(13))
    soft_found = soft_inj[soft_inj.detected]

    soft_likelihood = SelectedLikelihood(
        telescope, soft_data, pop_model, N=2000, seed=3,
        injections={k: jnp.asarray(soft_found[k].to_numpy()) for k in ["log_L", "distance"]},
        log_g=jnp.log(jnp.asarray(soft_found["prior"].to_numpy())),
        n_injections=len(soft_inj),
    )
    soft_samples = ai.run_sampler(soft_likelihood, hyper_priors, seed=1,
                                  num_warmup=300, num_samples=1000, summary=False).samples
    ```

    Only the trigger changed: the likelihood never looks inside it. With mismatched triggers
    (bonus), $\hat\alpha$ describes a different survey from the one that made the catalog, and
    the bias creeps back in. The injections must go through *exactly* the detection process
    the data went through.
    """)})
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    from dataclasses import dataclass
    from functools import partial

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    from jax.scipy.stats import norm, truncnorm
    from numpyro.distributions import constraints

    import astroinfer as ai
    from astroinfer import MeasurementProcess, parameter, parameter_dataclass
    from astroinfer.generative import Distribution, GenerativeModel, TruncatedNormal, product

    import _style as style

    style.apply()

    R_MIN, R_MAX = 0.5, 2.0   # the survey volume: a shell from 0.5 to 2 kpc
    N_SIM = 5_000             # stars simulated in the volume
    N_EVENTS = 150            # detected stars used in the analysis
    N_INJ = 50_000            # injections for the selection term


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 7 · Sharper telescopes 🔬

    Good news: the observatory has upgraded. The new telescope measures fluxes and parallaxes
    **ten times** more precisely than the one in notebook 6. Better data can only help... right?

    Not if we keep computing the per-star integrals the way we have so far. In this notebook the
    analysis of notebook 6 breaks on the sharper data, we find out why, and we fix it with a
    **`Proposal`**: an importance-sampling distribution that knows where each star's data point.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The survey, unchanged

    Same stars, same population, same flux limit, same selection term as notebook 6. Only the
    telescope's noise levels change.
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
def _():
    # The population, assembled from two named leaves so we can reuse them later.
    luminosity = TruncatedNormal("log_L", "mu_L", "sigma_L", -3.0, 5.0)
    volume = PowerLaw("distance", 2.0, R_MIN, R_MAX)
    pop_model = GenerativeModel(product([luminosity, volume]), order=["log_L", "distance"])
    Lambda_true = dict(mu_L=0.0, sigma_L=0.7)

    trigger = ai.detection.ThresholdTrigger(threshold=-0.3, statistic="log_flux")
    blurry = Telescope()                                            # notebook 6
    sharp = Telescope(sigma_log_flux=0.02, sigma_parallax=0.02)     # the upgrade
    mo.hstack([blurry, sharp], justify="start", gap=2)
    return Lambda_true, blurry, luminosity, pop_model, sharp, trigger, volume


@app.cell
def _(Lambda_true, blurry, pop_model, sharp, trigger):
    truths = Star.from_dict(pop_model.sample(Lambda_true, key=jax.random.key(0), shape=(N_SIM,)))

    def survey(telescope, key):
        """Observe every star, detect, keep the first N_EVENTS."""
        observed = telescope.sample(truths, key=key)
        detected = trigger.detect(observed, np.random.default_rng(key))
        keep = np.flatnonzero(detected)[:N_EVENTS]
        return observed[keep], truths[keep]

    data_blurry, _ = survey(blurry, 1)
    data, detected_truths = survey(sharp, 1)
    data
    return data, data_blurry, detected_truths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The per-star integrals collapse

    Recall what `PopulationLikelihood` computes for star $i$:

    $$
    Z_i(\Lambda)=\int K(d_i\mid\theta)\,p(\theta\mid\Lambda)\,\mathrm d\theta
    \;\approx\;\frac1N\sum_{j=1}^{N}K(d_i\mid\theta_{j}),\qquad \theta_j\sim p(\theta\mid\Lambda).
    $$

    It throws darts from the **whole population** and counts how many land where the likelihood
    is large. With a blurry telescope that region is big and plenty of darts land in it. With the
    sharp telescope it is a tiny box, and almost every dart misses. The effective sample size
    $N_\mathrm{eff}=(\sum_j w_j)^2/\sum_j w_j^2$ tells us how many darts actually count:
    """)
    return


@app.cell
def _(Lambda_true, data, data_blurry, blurry, pop_model, sharp):
    N_POP = 2_000

    def neff_per_star(likelihood):
        _, neff = jax.jit(likelihood.event_logpdf_and_neff)(Lambda_true)
        return np.asarray(neff)

    neff_blurry = neff_per_star(ai.PopulationLikelihood(blurry, data_blurry, pop_model, N=N_POP, seed=3))
    neff_sharp = neff_per_star(ai.PopulationLikelihood(sharp, data, pop_model, N=N_POP, seed=3))

    _bins = np.geomspace(1, N_POP, 40)
    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    _ax.hist(neff_blurry, bins=_bins, color=style.INK_2, alpha=0.6, label="blurry telescope")
    _ax.hist(neff_sharp, bins=_bins, color=style.PRIOR, alpha=0.75, label="sharp telescope")
    _ax.set(xscale="log", xlabel=f"$N_\\mathrm{{eff}}$ per star (out of {N_POP:,} population draws)",
            ylabel="stars", title="Sharper data, starved integrals")
    _ax.legend(loc="upper center")
    mo.vstack([
        _fig,
        mo.md(f"Median $N_\\mathrm{{eff}}$: **{np.median(neff_blurry):.0f}** with the blurry telescope, "
              f"**{np.median(neff_sharp):.1f}** with the sharp one. Many stars rest on a single draw."),
    ])
    return N_POP, neff_sharp, neff_per_star


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A handful of effective draws per star means $\log\hat Z_i(\Lambda)$ is dominated by noise.
    Worse, the noise is not innocent: populations that happen to put their darts near the data
    look *better* than they are. Let's run the notebook 6 analysis anyway and see. First, the
    injections for the selection term, scattered through the sharp telescope:
    """)
    return


@app.cell
def _(pop_model, sharp, trigger):
    Lambda_g = dict(mu_L=0.5, sigma_L=1.5)
    _g_draws = pop_model.sample(Lambda_g, key=jax.random.key(5), shape=(N_INJ,))
    _g_table = pd.DataFrame({k: np.asarray(v) for k, v in _g_draws.items()})
    _g_table["prior"] = np.exp(np.asarray(pop_model.log_prob(_g_draws, Lambda_g)))

    _inj = trigger.injections(sharp, _g_table, np.random.default_rng(6))
    _found = _inj[_inj.detected]
    selection = dict(
        injections={k: jnp.asarray(_found[k].to_numpy()) for k in ["log_L", "distance"]},
        log_g=jnp.log(jnp.asarray(_found["prior"].to_numpy())),
        n_injections=N_INJ,
    )
    hyper_priors = dict(mu_L=dist.Uniform(-2.0, 2.0), sigma_L=dist.Uniform(0.1, 2.0))
    mo.md(f"**{len(_found):,}** of {N_INJ:,} injections found.")
    return hyper_priors, selection


@app.cell
def _(N_POP, data, hyper_priors, pop_model, selection, sharp):
    starved = ai.run_sampler(
        SelectedLikelihood(sharp, data, pop_model, N=N_POP, seed=3, **selection),
        hyper_priors, seed=1, num_warmup=300, num_samples=1000, summary=False,
    ).samples
    starved[["mu_L", "sigma_L"]].describe().loc[["mean", "std"]]
    return (starved,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The truth is $(\mu_L,\sigma_L)=(0,\,0.7)$. That is not a little off: the posterior is narrow,
    confident, and nowhere near it. **Always check $N_\mathrm{eff}$ before trusting a population run.**

    ## Importance sampling with a proposal

    The fix is to throw the darts where the data say the star is. Draw $\theta$ from a proposal
    $q(\theta\mid d_i)$ that hugs each star's likelihood, and reweight:

    $$
    \hat Z_i(\Lambda)=\frac1N\sum_{j=1}^{N}
    \underbrace{\frac{K(d_i\mid\theta_{ij})\,p(\theta_{ij}\mid\Lambda)}{q(\theta_{ij}\mid d_i)}}_{w_{ij}},
    \qquad \theta_{ij}\sim q(\cdot\mid d_i).
    $$

    Any $q$ gives an unbiased estimate as long as it covers the region where $K\cdot p$ is large;
    a $q$ close to $K\cdot p$ gives nearly equal weights and $N_\mathrm{eff}\approx N$.

    A proposal reads data and draws parameters: it is a **measurement process run backwards**,
    and astroinfer treats it exactly like one.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **🧩 What you implement in a `Proposal`** (subclass `ai.Proposal`)

    | attribute / method | what it is |
    |---|---|
    | `parameter_type` | the measurement's **data** class: what $q$ *reads* (here `StarObservation`) |
    | `data_type` | the measurement's **parameter** class: what $q$ *draws* (here `Star`) |
    | `sample_single(self, d, key, Lambda)` | draw **one** $\theta\sim q_\Lambda(\cdot\mid d)$ for **one** star |
    | `logpdf_single(self, theta, d, Lambda)` | the scalar $\log q_\Lambda(\theta\mid d)$ |

    The types are **swapped** relative to the measurement process; astroinfer checks they mirror it.
    `Lambda` is the hyperparameter dict; a proposal that does not depend on it just ignores it.

    One rule: draws must be a **smooth function of `(key, Lambda)`**. The likelihood re-draws with
    the *same* key at every $\Lambda$ (common random numbers), so $\hat Z_i(\Lambda)$ is smooth and
    NUTS can use its gradient. Location-scale transforms and inverse CDFs are fine;
    rejection sampling is not.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Our proposal follows the measurement backwards:

    1. the **true parallax** is near the measured one: $\varpi\sim\mathcal{TN}(\hat\varpi,\ k\sigma_\varpi)$,
       truncated to the survey volume, and $r=1/\varpi$;
    2. given $r$, the **luminosity** follows from the measured flux:
       $\log_{10}L\sim\mathcal N(\log_{10}\hat F+2\log_{10}r,\ k\sigma_F)$.

    $k=1.5$ widens both a little, so $q$ comfortably covers the likelihood. Since we draw $\varpi$
    but score $r$, $\log q$ picks up the Jacobian $|\mathrm d\varpi/\mathrm dr| = 1/r^2$.
    """)
    return


@app.class_definition
class ParallaxProposal(ai.Proposal):
    """q(theta | d): distance from the measured parallax, log L from the measured flux at that distance."""

    display_name = "Parallax proposal"
    parameter_type = StarObservation   # what q reads
    data_type = Star                   # what q draws

    def __init__(self, telescope, widen=1.5, **kw):
        super().__init__(**kw)
        self.telescope, self.widen = telescope, widen

    # --- the distance block -------------------------------------------------
    def parallax_window(self, d):
        """Centre, width and bounds of the truncated normal for the true parallax."""
        s = self.widen * self.telescope.sigma_parallax
        return d.parallax, s, 1.0 / R_MAX, 1.0 / R_MIN

    def draw_distance(self, d, key):
        loc, s, lo, hi = self.parallax_window(d)
        plx = loc + s * jax.random.truncated_normal(key, (lo - loc) / s, (hi - loc) / s)
        return 1.0 / plx

    def log_q_distance(self, r, d):
        loc, s, lo, hi = self.parallax_window(d)
        log_q_plx = truncnorm.logpdf(1.0 / r, (lo - loc) / s, (hi - loc) / s, loc=loc, scale=s)
        return log_q_plx - 2 * jnp.log(r)                # Jacobian |d plx / d r| = 1 / r^2

    # --- the luminosity block -----------------------------------------------
    def luminosity_window(self, d, r):
        return d.log_flux + 2 * jnp.log10(r), self.widen * self.telescope.sigma_log_flux

    # --- the two methods every Proposal implements --------------------------
    def sample_single(self, d, key, Lambda):
        k_r, k_L = jax.random.split(key)
        r = self.draw_distance(d, k_r)
        loc, s = self.luminosity_window(d, r)
        return Star(log_L=loc + s * jax.random.normal(k_L), distance=r)

    def logpdf_single(self, theta, d, Lambda):
        loc, s = self.luminosity_window(d, theta.distance)
        return self.log_q_distance(theta.distance, d) + norm.logpdf(theta.log_L, loc, s)

    def _display_equations(self):
        return [r"\varpi\sim\mathcal{TN}(\hat\varpi,\ k\sigma_\varpi),\quad r = 1/\varpi",
                r"\log_{10}L\sim\mathcal{N}(\log_{10}\hat F + 2\log_{10}r,\ k\sigma_F)"]

    def _display_config(self):
        return {"widen k": self.widen}


@app.cell
def _(sharp):
    proposal = ParallaxProposal(sharp)
    proposal
    return (proposal,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `proposal.sample(d, N=, key=, Lambda=)` and `proposal.logpdf(theta, d, Lambda=)` work just like
    a measurement process's. Here are a thousand darts for one star, from the population and from
    the proposal:
    """)
    return


@app.cell
def _(Lambda_true, data, detected_truths, pop_model, proposal):
    star = 0
    q_draws = proposal.sample(data[star], N=1000, key=0, Lambda=Lambda_true)
    p_draws = pop_model.sample(Lambda_true, key=jax.random.key(1), shape=(1000,))
    _true = detected_truths[star]

    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.6), layout="constrained")
    for _ax, _zoom in zip(_axes, [False, True]):
        _ax.scatter(p_draws["distance"], p_draws["log_L"], s=6, color=style.PRIOR, alpha=0.5,
                    linewidths=0, label="population draws")
        _ax.scatter(q_draws.distance, q_draws.log_L, s=6, color=style.POSTERIOR, alpha=0.5,
                    linewidths=0, label="proposal draws")
        _ax.plot(float(_true.distance), float(_true.log_L), "o", color=style.TRUTH,
                 markersize=6, label="true star")
        _ax.set(xlabel="distance $r$ [kpc]", ylabel=r"$\log_{10}L$")
        if _zoom:
            _cx, _cy = float(_true.distance), float(_true.log_L)
            _hits = int(np.sum((np.abs(np.asarray(p_draws["distance"]) - _cx) < 0.12)
                               & (np.abs(np.asarray(p_draws["log_L"]) - _cy) < 0.25)))
            _ax.set(xlim=(_cx - 0.12, _cx + 0.12), ylim=(_cy - 0.25, _cy + 0.25),
                    title=f"Zoomed in: {_hits} of 1,000 population draws land here")
        else:
            _ax.set(title=f"Star {star}: where the darts land")
            _ax.legend(loc="lower right", markerscale=2)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Pass the proposal to the likelihood with `proposal=`. The draws now come from $q$, so far
    fewer are needed: we use $N=500$ instead of 2,000.
    """)
    return


@app.cell
def _(N_POP, Lambda_true, data, neff_per_star, neff_sharp, pop_model, proposal, sharp):
    N_Q = 500
    neff_proposal = neff_per_star(
        ai.PopulationLikelihood(sharp, data, pop_model, N=N_Q, seed=3, proposal=proposal))

    _bins = np.geomspace(2e-4, 1, 50)
    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    _ax.hist(neff_sharp / N_POP, bins=_bins, color=style.PRIOR, alpha=0.75,
             label=f"population draws (N = {N_POP:,})")
    _ax.hist(neff_proposal / N_Q, bins=_bins, color=style.POSTERIOR, alpha=0.75,
             label=f"proposal draws (N = {N_Q})")
    _ax.set(xscale="log", xlabel=r"$N_\mathrm{eff}/N$ per star", ylabel="stars",
            title="Same data, two ways to throw darts")
    _ax.legend(loc="upper left")
    mo.vstack([
        _fig,
        mo.md(f"With the proposal, median $N_\\mathrm{{eff}}$ = **{np.median(neff_proposal):.0f}** of "
              f"{N_Q} draws (minimum {neff_proposal.min():.0f}), against "
              f"**{np.median(neff_sharp):.1f}** of {N_POP:,} before."),
    ])
    return N_Q, neff_proposal


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Inference with the proposal and the selection term

    `measurement.population_inference` takes the proposal through `proposal=`, and a custom
    likelihood class through `likelihood_cls=`. It builds the likelihood itself, so extra
    constructor arguments, here our injections, are bound with `functools.partial`.
    """)
    return


@app.cell
def _(N_Q, data, hyper_priors, pop_model, proposal, selection, sharp):
    fixed = sharp.population_inference(
        data, pop_model, hyper_priors,
        N=N_Q, seed=3, proposal=proposal,
        likelihood_cls=partial(SelectedLikelihood, **selection),
        num_warmup=300, num_samples=1000,
    ).samples
    fixed[["mu_L", "sigma_L"]].describe().loc[["mean", "std"]]
    return (fixed,)


@app.cell
def _(Lambda_true, fixed, starved):
    _fig, _ = style.corner(
        [starved, fixed], ["mu_L", "sigma_L"],
        labels=[r"$\mu_L$", r"$\sigma_L$"],
        names=["population draws", "proposal"],
        colors=[style.PRIOR, style.POSTERIOR],
        truths=Lambda_true, ranges={"mu_L": (-1.0, 1.3), "sigma_L": (0.3, 2.0)},
        size=2.6,
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Back on target. The proposal costs a few dozen lines once, and every population run after that
    is both cheaper ($N=500$) and correct.

    ## The one override point: `log_weights`

    Everything above used the default weight, implemented in
    `PopulationLikelihood.log_weights(Lambda)`, which returns an `(E, N)` array:

    * no proposal: $\log w = \log K$, with $\theta\sim p(\cdot\mid\Lambda)$;
    * with a proposal: $\log w = \log K + \log p(\theta\mid\Lambda) - \log q(\theta\mid d)$, with $\theta\sim q$.

    Sometimes you know something the default does not. The classic case: draw *some* parameters
    from the population itself. If $q$ draws $\log_{10}L$ from $p(\log_{10}L\mid\Lambda)$, that factor
    appears in both $p$ and $q$ and **cancels exactly**:

    $$
    w = \frac{K\;p(\log_{10}L\mid\Lambda)\,p(r)}{q(r\mid d)\;p(\log_{10}L\mid\Lambda)}
      = \frac{K\;p(r)}{q(r\mid d)} .
    $$

    Override `log_weights` to compute just that. Below, `FluxBlindProposal` reuses the distance block
    and draws the luminosity with `GenerativeModel.sample(Lambda, key=...)`, which is an inverse-CDF
    transform of uniforms and so smooth in $\Lambda$. (In GW population analyses this is how
    poorly-measured spins are handled.)
    """)
    return


@app.class_definition
class FluxBlindProposal(ParallaxProposal):
    """r from the parallax as before; log L from the population p(log L | Lambda)."""

    display_name = "Flux-blind proposal"

    def __init__(self, telescope, luminosity_model, **kw):
        super().__init__(telescope, **kw)
        self.luminosity_model = luminosity_model

    def sample_single(self, d, key, Lambda):
        k_r, k_L = jax.random.split(key)
        log_L = self.luminosity_model.sample(Lambda, key=k_L, shape=())["log_L"]
        return Star(log_L=log_L, distance=self.draw_distance(d, k_r))

    def logpdf_single(self, theta, d, Lambda):
        return (self.log_q_distance(theta.distance, d)
                + self.luminosity_model.log_prob({"log_L": theta.log_L}, Lambda))


@app.class_definition
class CancelledLikelihood(SelectedLikelihood):
    """log w = log K + log p(r) - log q(r | d): the luminosity factor cancels."""

    def __init__(self, *a, distance_model, **kw):
        super().__init__(*a, **kw)
        self.distance_model = distance_model

    def log_weights(self, Lambda):
        theta = self.proposal_draws(Lambda)                 # (E, N) draws from q
        d = self.data.reshape_batch((self.E, 1))            # (E, 1), broadcasts against theta
        return (self.log_kernel_at(theta)
                + self.distance_model.log_prob(theta.to_dict(), Lambda)
                - self.proposal.log_q_distance(theta.distance, d))


@app.cell
def _(Lambda_true, N_Q, data, luminosity, neff_proposal, pop_model, selection, sharp, volume):
    blind = FluxBlindProposal(sharp, luminosity_model=GenerativeModel(luminosity, order=["log_L"]))
    _common = dict(N=N_Q, seed=3, proposal=blind, **selection)

    default_weights = SelectedLikelihood(sharp, data, pop_model, **_common)
    cancelled = CancelledLikelihood(sharp, data, pop_model, **_common,
                                    distance_model=GenerativeModel(volume, order=["distance"]))

    _lz_default, _ = jax.jit(default_weights.event_logpdf_and_neff)(Lambda_true)
    _lz_cancel, _neff_blind = jax.jit(cancelled.event_logpdf_and_neff)(Lambda_true)
    _grad = jax.grad(cancelled.logpdf)({k: jnp.asarray(v) for k, v in Lambda_true.items()})

    mo.md(f"""
    | check at the true $\\Lambda$ | value |
    |---|---|
    | largest difference in $\\log\\hat Z_i$, default vs cancelled weight | `{float(jnp.abs(_lz_default - _lz_cancel).max()):.1e}` |
    | gradient of $\\ln\\mathcal L$ through the $\\Lambda$-dependent draws is finite | `{all(bool(jnp.isfinite(v)) for v in _grad.values())}` |
    | median $N_\\mathrm{{eff}}/N$, flux-blind proposal | **{float(jnp.median(_neff_blind)) / N_Q:.3f}** |
    | median $N_\\mathrm{{eff}}/N$, parallax proposal | **{float(np.median(neff_proposal)) / N_Q:.3f}** |

    The two weights agree to rounding error, and the override never evaluates the cancelling
    factor. Here, though, the flux-blind proposal is much *less* efficient: the sharp telescope pins
    $\\log_{{10}}L$ to 0.02 dex, and the population spreads it over 0.7. Drawing from the population
    pays off only for parameters the data barely constrain. To use it in a run, pass
    `likelihood_cls=partial(CancelledLikelihood, distance_model=..., **selection)` and
    `proposal=blind` to `population_inference`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Recap

    * Population-draw integrals starve when the data are much narrower than the population.
      **Check per-event $N_\mathrm{eff}$**; a few effective draws gives confident, wrong answers.
    * A `Proposal` is a measurement process run backwards: `parameter_type` is the data,
      `data_type` is $\theta$, and you write `sample_single(d, key, Lambda)` and
      `logpdf_single(theta, d, Lambda)`. Draws must be smooth in `(key, Lambda)`.
    * Pass it with `proposal=`; pass a custom likelihood with `likelihood_cls=`
      (bind extra arguments with `functools.partial`).
    * `log_weights(Lambda)` is the single override point for the weight, e.g. when a factor
      shared by $p$ and $q$ cancels.

    So far every analysis started from raw data $d_i$. In **notebook 8** we start from what real
    catalogs publish: each event's *posterior samples*, collected in a `Catalog`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ✏️ Your turn: how wide should a proposal be?

    `ParallaxProposal(sharp, widen=k)` scales both proposal widths by $k$.

    1. For $k$ in `[0.3, 0.6, 1.0, 1.5, 3.0, 10.0]`, compute the median and minimum $N_\mathrm{eff}/N$
       over stars at the true $\Lambda$ (reuse `neff_per_star`).
    2. Plot both against $k$ on a log axis. Where is the sweet spot?
    3. Why does a proposal that is too *narrow* hurt the **minimum** most?
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
    widths = [0.3, 0.6, 1.0, 1.5, 3.0, 10.0]
    medians, minima = [], []
    for k in widths:
        lik = ai.PopulationLikelihood(sharp, data, pop_model, N=500, seed=3,
                                      proposal=ParallaxProposal(sharp, widen=k))
        neff = neff_per_star(lik) / 500
        medians.append(np.median(neff))
        minima.append(neff.min())

    fig, ax = plt.subplots()
    ax.plot(widths, medians, "o-", color=style.POSTERIOR, label="median")
    ax.plot(widths, minima, "o-", color=style.PRIOR, label="minimum")
    ax.set(xscale="log", xlabel="widen $k$", ylabel=r"$N_\mathrm{eff}/N$")
    ax.legend()
    ```

    Around $k\approx 1$ to $1.5$ is best. Too wide wastes draws in empty space, and the
    efficiency falls roughly like $1/k^2$ (two dimensions). Too narrow is worse: $q$ no longer
    covers the tails of $K\cdot p$, the rare draws that land there get enormous weights $p/q$,
    and a single draw can dominate a star's sum. That shows up first in the minimum. A slightly
    wide proposal is always the safer mistake.
    """)})
    return


if __name__ == "__main__":
    app.run()

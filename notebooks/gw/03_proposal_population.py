import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    from pathlib import Path

    import astroinfer as ai
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    import plot_helper as ph
    from astroinfer.generative import (
        Distribution,
        GenerativeModel,
        TruncatedNormal,
        Uniform,
        conditional,
        product,
    )
    from astroinfer_gw.gwmockcat import GWMockCatData, GWMockCatMeasurement, GWParams
    from astroinfer_gw.physics import (
        chirp_mass_from_mass_ratio,
        total_mass_from_chirp_mass,
    )
    from jax.scipy.stats import norm, truncnorm

    ph.restore_plot_defaults()
    ph.set_plot_defaults(latex_font_size=16, usetex=True)

    try:
        NOTEBOOK_DIR = Path(__file__).resolve().parent
    except NameError:
        NOTEBOOK_DIR = Path(mo.notebook_dir())
    RESULTS_DIR = NOTEBOOK_DIR.parent / "data" / "proposal_population"

    N_TRUTHS = 20_000       # population draws to simulate
    SNR_THRESHOLD = 4.0     # detected if the observed SNR exceeds this (as in dev.py)
    N_EVENTS = 100          # detected events in the analysis
    N_DRAWS = 2_000         # importance draws per event
    N_INJECTIONS = 200_000  # injections for the selection term

    M_MIN, M_MAX = 5.0, 100.0
    Q_MIN = 0.05
    Z_MAX = 1.5

    MASS_VARS = ["mass_1_source", "mass_ratio", "redshift"]
    INFORMED_VARS = MASS_VARS + ["projection_factor"]   # drawn following the data
    SPIN_VARS = ["chi_1", "chi_2", "cos_tilt_1", "cos_tilt_2"]   # drawn from the population

    LABELS = dict(alpha=r"$\alpha$", mu_q=r"$\mu_q$", sigma_q=r"$\sigma_q$", lamb=r"$\lambda$",
                  mu_chi_1=r"$\mu_{\chi_1}$", sigma_chi_1=r"$\sigma_{\chi_1}$",
                  mu_chi_2=r"$\mu_{\chi_2}$", sigma_chi_2=r"$\sigma_{\chi_2}$",
                  sigma_0=r"$\sigma_0$", sigma_slope=r"$\sigma_1$")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Population inference with an event-informed proposal

    This notebook simulates a catalog of binary mergers and infers the hyperparameters of their
    population. The evidence of each event, $Z_i(\Lambda)=\int \mathcal{L}(d_i\mid\theta)\,p(\theta\mid\Lambda)\,d\theta$,
    is estimated by importance sampling from a proposal $q_\Lambda(\theta\mid d_i)$:

    $$\hat Z_i(\Lambda)=\frac1N\sum_j \frac{K(d_i\mid\theta_{ij})\,p(\theta_{ij}\mid\Lambda)}{q_\Lambda(\theta_{ij}\mid d_i)},
    \qquad \theta_{ij}\sim q_\Lambda(\cdot\mid d_i).$$

    The proposal treats two groups of parameters differently. The masses, redshift and projection
    factor $(m_1, q, z, \Theta)$ are drawn from the region that the data $d_i$ point to. The spins
    $(\chi_1,\chi_2,\cos\theta_1,\cos\theta_2)$ are drawn from the population itself,
    $p(\cdot\mid q,\Lambda)$. Because the spins come from the population's own conditional
    distribution, that factor cancels in the weight and is never evaluated:

    $$w_{ij} = \frac{K(d_i\mid\theta_{ij})\;p(m_1,q,z,\Theta\mid\Lambda)}{q(m_1,q,z,\Theta\mid d_i)}.$$

    The population has $\Theta\sim U(0,1)$, so $p(\Theta)=1$. $\Theta$ is still drawn from the data,
    because the data constrain it to about $\pm u_\Theta/\hat\rho\approx 0.03$, and draws from the
    population would waste about 95% of the samples.

    The draws are a fixed function of a stored key and $\Lambda$, so $\hat Z_i(\Lambda)$ is smooth in
    $\Lambda$ and NUTS can use its gradient.

    In this population the tilt cosine equals the spin magnitude, $\cos\theta_i\approx\chi_i$, at
    $q=1$, and the relation loosens as $q$ decreases. The notebook checks whether the inference
    recovers this.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Two new population leaves

    `astroinfer.generative` provides truncated normal, uniform and beta leaves. The masses and
    redshift need two more, each written as a `Distribution` with a closed-form CDF and quantile:

    * `PowerLaw`: $p(m_1)\propto m_1^{-\alpha}$ on $[m_\mathrm{min}, m_\mathrm{max}]$;
    * `PowerLawRedshift`: $p(z)\propto \frac{dV_c}{dz}(1+z)^{\lambda-1}$ on $[0, z_\mathrm{max}]$ (Planck15),
      tabulated on a grid.
    """)
    return


@app.cell
def _():
    class PowerLaw(Distribution):
        """p(x) ∝ x^(-alpha) on [lo, hi]; alpha is a hyperparameter name."""

        def __init__(self, var, alpha, lo, hi):
            self.var_names, self.var = [var], var
            self.alpha, self.lo, self.hi = alpha, lo, hi

        def _terms(self, L):
            a = 1.0 - L[self.alpha]
            a = jnp.where(jnp.abs(a) < 1e-6, 1e-6, a)       # the alpha = 1 limit
            return a, self.lo**a, self.hi**a

        def cond_pdf(self, var, val, x, L):
            a, lo_a, hi_a = self._terms(L)
            p = a * val ** (a - 1.0) / (hi_a - lo_a)
            return jnp.where((val >= self.lo) & (val <= self.hi), p, 0.0)

        def cond_cdf(self, var, val, x, L):
            a, lo_a, hi_a = self._terms(L)
            return (jnp.clip(val, self.lo, self.hi) ** a - lo_a) / (hi_a - lo_a)

        def cond_ppf(self, var, u, x, L):
            a, lo_a, hi_a = self._terms(L)
            return (lo_a + u * (hi_a - lo_a)) ** (1.0 / a)

        def bounds(self, var):
            return self.lo, self.hi


    class PowerLawRedshift(Distribution):
        """p(z) ∝ dV_c/dz (1+z)^(lamb-1) on [0, z_max] (Planck15), tabulated on a grid."""

        def __init__(self, var, lamb, z_max, n_grid=2000):
            from astropy.cosmology import Planck15

            self.var_names, self.var = [var], var
            self.lamb, self.z_max = lamb, z_max
            z = np.linspace(0.0, z_max, n_grid + 1)
            self.z = jnp.asarray(z)
            self.dvdz = jnp.asarray(
                4 * np.pi * Planck15.differential_comoving_volume(z).to("Gpc3/sr").value)

        def _cdf_grid(self, L):
            pdf = self.dvdz * (1.0 + self.z) ** (L[self.lamb] - 1.0)
            cdf = jnp.concatenate([jnp.zeros(1),
                                   jnp.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * jnp.diff(self.z))])
            return cdf / cdf[-1], cdf[-1]

        def cond_pdf(self, var, val, x, L):
            _, total = self._cdf_grid(L)
            p = jnp.interp(val, self.z, self.dvdz) * (1.0 + val) ** (L[self.lamb] - 1.0) / total
            return jnp.where((val >= 0.0) & (val <= self.z_max), p, 0.0)

        def cond_cdf(self, var, val, x, L):
            return jnp.interp(val, self.z, self._cdf_grid(L)[0])

        def cond_ppf(self, var, u, x, L):
            return jnp.interp(u, self._cdf_grid(L)[0], self.z)

        def bounds(self, var):
            return 0.0, self.z_max

    return PowerLaw, PowerLawRedshift


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The population

    | sector | model |
    |---|---|
    | masses, redshift | $m_1\sim\mathrm{PL}(-\alpha)_{[5,100]}$, $q\sim\mathcal{TN}(\mu_q,\sigma_q)_{[0.05,1]}$, $z\sim\mathrm{PL}_z(\lambda)_{[0,1.5]}$ |
    | projection factor | $\Theta\sim U(0,1)$ |
    | spin magnitudes | $\chi_i\sim\mathcal{TN}(\mu_{\chi_i},\sigma_{\chi_i})_{[0,1]}$ |
    | spin tilts | $\cos\theta_i \mid \chi_i, q \sim \mathcal{TN}\big(\chi_i,\ \sigma(q)\big)_{[-1,1]}$, $\quad\sigma(q)=\sigma_0+\sigma_1(1-q)$ |

    At $q=1$ the tilt equals the magnitude up to $\sigma_0$; the scatter grows linearly as $q$ drops.

    All four models below are built from the same leaf objects, so they agree with each other by
    construction:

    * `pop_model`: the joint $p(\theta\mid\Lambda)$;
    * `informed_model`: the $(m_1,q,z,\Theta)$ marginal, evaluated in the weight;
    * `spin_model`: the conditional $p(\vec\chi\mid q,\Lambda)$ the proposal draws from
      (`GenerativeModel` checks it only reads its declared external, $q$);
    * `mass_model`: the $(m_1,q,z)$ marginal, for the selection term.
    """)
    return


@app.cell
def _(PowerLaw, PowerLawRedshift):
    def tilt_given_spin(i):
        return TruncatedNormal(
            f"cos_tilt_{i}",
            mu=conditional(lambda x, L: x[f"chi_{i}"], on=f"chi_{i}"),
            sigma=conditional(
                lambda x, L: L["sigma_0"] + L["sigma_slope"] * (1.0 - x["mass_ratio"]),
                on="mass_ratio"),
            a=-1.0, b=1.0,
        )

    mass_sector = product([
        PowerLaw("mass_1_source", "alpha", M_MIN, M_MAX),
        TruncatedNormal("mass_ratio", "mu_q", "sigma_q", Q_MIN, 1.0),
        PowerLawRedshift("redshift", "lamb", Z_MAX),
    ])
    theta_sector = Uniform("projection_factor", 0.0, 1.0)
    spin_sector = product([
        TruncatedNormal("chi_1", "mu_chi_1", "sigma_chi_1", 0.0, 1.0),
        TruncatedNormal("chi_2", "mu_chi_2", "sigma_chi_2", 0.0, 1.0),
        tilt_given_spin(1),
        tilt_given_spin(2),
    ])

    pop_model = GenerativeModel(product([mass_sector, theta_sector, spin_sector]),
                                order=INFORMED_VARS + SPIN_VARS)
    informed_model = GenerativeModel(product([mass_sector, theta_sector]), order=INFORMED_VARS)
    spin_model = GenerativeModel(spin_sector, order=SPIN_VARS, external=["mass_ratio"])
    mass_model = GenerativeModel(mass_sector, order=MASS_VARS)

    Lambda_true = dict(alpha=2.0, mu_q=0.75, sigma_q=0.2, lamb=2.0,
                       mu_chi_1=0.5, sigma_chi_1=0.25, mu_chi_2=0.5, sigma_chi_2=0.25,
                       sigma_0=0.02, sigma_slope=1.0)
    return Lambda_true, informed_model, mass_model, pop_model, spin_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exploring the population

    The sliders set the hyperparameters of the population shown in the plots below. They do not
    change `Lambda_true`, so nothing further down the notebook re-runs. The redshift distribution is
    fixed at $\lambda=2$ and not shown. Each update draws from the same random key, so the
    differences between settings come from $\Lambda$ and not from Monte Carlo noise.
    """)
    return


@app.cell
def _():
    def _slider(lo, hi, value, step, label):
        return mo.ui.slider(lo, hi, value=value, step=step, label=label,
                            show_value=True, debounce=True)

    explore_sliders = mo.ui.dictionary({
        "alpha": _slider(-2.0, 6.0, 2.0, 0.1, r"$\alpha$"),
        "mu_q": _slider(0.3, 1.0, 0.75, 0.01, r"$\mu_q$"),
        "sigma_q": _slider(0.05, 1.0, 0.2, 0.01, r"$\sigma_q$"),
        "mu_chi_1": _slider(0.0, 1.0, 0.5, 0.01, r"$\mu_{\chi_1}$"),
        "sigma_chi_1": _slider(0.05, 1.0, 0.25, 0.01, r"$\sigma_{\chi_1}$"),
        "mu_chi_2": _slider(0.0, 1.0, 0.5, 0.01, r"$\mu_{\chi_2}$"),
        "sigma_chi_2": _slider(0.05, 1.0, 0.25, 0.01, r"$\sigma_{\chi_2}$"),
        "sigma_0": _slider(0.005, 0.5, 0.02, 0.005, r"$\sigma_0$"),
        "sigma_slope": _slider(0.0, 2.0, 1.0, 0.05, r"$\sigma_1$"),
    })

    _s = explore_sliders
    explore_controls = mo.hstack([
        mo.vstack([mo.md("#### Masses"), _s["alpha"], _s["mu_q"], _s["sigma_q"]]),
        mo.vstack([mo.md("#### Spin magnitudes"), _s["mu_chi_1"], _s["sigma_chi_1"],
                   _s["mu_chi_2"], _s["sigma_chi_2"]]),
        mo.vstack([mo.md(r"#### Tilt scatter $\sigma(q)=\sigma_0+\sigma_1(1-q)$"),
                   _s["sigma_0"], _s["sigma_slope"]]),
    ], justify="start", gap=2)
    explore_controls
    return explore_controls, explore_sliders


@app.cell
def _(pop_model):
    N_EXPLORE = 20_000
    # compiled once; the sliders only change the (traced) hyperparameter values
    explore_sample = jax.jit(
        lambda L: pop_model.sample({**L, "lamb": 2.0}, key=jax.random.key(123), shape=(N_EXPLORE,)))
    return (explore_sample,)


@app.cell
def _():
    POP_1D = ["mass_1_source", "mass_ratio", "chi_1", "chi_2",
              "cos_tilt_1", "cos_tilt_2", "chi_eff"]
    POP_BINS = dict(mass_1_source=np.geomspace(M_MIN, M_MAX, 41),
                    mass_ratio=np.linspace(Q_MIN, 1.0, 41),
                    chi_1=np.linspace(0.0, 1.0, 41), chi_2=np.linspace(0.0, 1.0, 41),
                    cos_tilt_1=np.linspace(-1.0, 1.0, 41), cos_tilt_2=np.linspace(-1.0, 1.0, 41),
                    chi_eff=np.linspace(-1.0, 1.0, 41))
    # (x, y, x range, y range) of the 2D panels
    POP_2D = [("chi_1", "cos_tilt_1", (0.0, 1.0), (-1.0, 1.0)),
              ("chi_2", "cos_tilt_2", (0.0, 1.0), (-1.0, 1.0)),
              ("mass_ratio", "tilt_residual_1", (Q_MIN, 1.0), (-2.0, 1.0)),
              ("mass_ratio", "chi_eff", (Q_MIN, 1.0), (-1.0, 1.0))]
    POP_LABELS = dict(mass_1_source=r"$m_1\ [M_\odot]$", mass_ratio=r"$q$",
                      chi_1=r"$\chi_1$", chi_2=r"$\chi_2$",
                      cos_tilt_1=r"$\cos\theta_1$", cos_tilt_2=r"$\cos\theta_2$",
                      chi_eff=r"$\chi_\mathrm{eff}$", tilt_residual_1=r"$\cos\theta_1-\chi_1$")

    def _with_derived(x):
        x = {k: np.asarray(v) for k, v in x.items()}
        q = x["mass_ratio"]
        return {**x,
                "chi_eff": (x["chi_1"] * x["cos_tilt_1"] + q * x["chi_2"] * x["cos_tilt_2"]) / (1.0 + q),
                "tilt_residual_1": x["cos_tilt_1"] - x["chi_1"]}

    def _contours(ax, xv, yv, rx, ry, color):
        """50% and 90% highest-density contours of a 2D histogram."""
        H, xe, ye = np.histogram2d(xv, yv, bins=50, range=[rx, ry])
        h = np.sort(H.ravel())[::-1]
        c = np.cumsum(h) / h.sum()
        levels = [h[np.searchsorted(c, 0.9)], h[np.searchsorted(c, 0.5)]]
        ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T,
                   levels=levels, colors=[color], linewidths=[1.2, 2.0])

    def population_figure():
        """Empty 3x4 grid: 7 one-dimensional marginals, sigma(q), and 4 two-dimensional panels."""
        fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), layout="constrained")
        flat = axes.ravel()
        for ax, v in zip(flat[:7], POP_1D):
            ax.set_xlabel(POP_LABELS[v])
            ax.set_xlim(POP_BINS[v][0], POP_BINS[v][-1])
        flat[0].set(xscale="log", yscale="log")
        for ax in axes[:2, 0]:
            ax.set_ylabel("density")
        flat[7].set(xlabel=r"$q$", ylabel=r"$\sigma(q)=\sigma_0+\sigma_1(1-q)$", xlim=(Q_MIN, 1.0))
        for ax, (vx, vy, rx, ry) in zip(axes[2], POP_2D):
            ax.set(xlabel=POP_LABELS[vx], ylabel=POP_LABELS[vy], xlim=rx, ylim=ry)
        for ax in axes[2, :2]:
            ax.plot([0, 1], [0, 1], color="0.5", linewidth=1, linestyle="--", zorder=3)
        return fig, axes

    def plot_population(axes, x, L, *, color, label=None, points=None, contours=False, n_points=4000):
        """Draw one population on a `population_figure` grid.

        x : samples {var: (M,)} for one Lambda, or {var: (K, M)} for K hyperposterior draws,
            in which case the 1D panels and sigma(q) show the median and 90% band over draws.
        L : the hyperparameters behind x (floats, or (K,) arrays); only sigma_0, sigma_slope are used.
        points : None, "q" (colour by mass ratio) or "plain" scatter in the 2D panels.
        contours : 50%/90% contours of the pooled samples in the 2D panels.
        Returns the q-coloured scatter (for a colorbar) when points="q".
        """
        x = _with_derived(x)
        flat = axes.ravel()
        for i, (ax, v) in enumerate(zip(flat[:7], POP_1D)):
            rows = np.atleast_2d(x[v])
            dens = np.stack([np.histogram(r, bins=POP_BINS[v], density=True)[0] for r in rows])
            if len(dens) > 1:
                lo, med, hi = np.quantile(dens, [0.05, 0.5, 0.95], axis=0)
                ax.stairs(hi, POP_BINS[v], baseline=lo, fill=True, color=color, alpha=0.3, linewidth=0)
                ax.stairs(med, POP_BINS[v], color=color, linewidth=1.5, label=label if i == 0 else None)
            else:
                ax.stairs(dens[0], POP_BINS[v], color=color, linewidth=1.5, label=label if i == 0 else None)

        qq = np.linspace(Q_MIN, 1.0, 200)
        sig = (np.atleast_1d(L["sigma_0"])[:, None]
               + np.atleast_1d(L["sigma_slope"])[:, None] * (1.0 - qq))
        if len(sig) > 1:
            lo, med, hi = np.quantile(sig, [0.05, 0.5, 0.95], axis=0)
            flat[7].fill_between(qq, lo, hi, color=color, alpha=0.3, linewidth=0)
            flat[7].plot(qq, med, color=color, linewidth=2)
        else:
            flat[7].plot(qq, sig[0], color=color, linewidth=2)
        flat[7].set_ylim(0.0, max(flat[7].get_ylim()[1], 1.05 * sig.max()))

        mappable = None
        for ax, (vx, vy, rx, ry) in zip(axes[2], POP_2D):
            xv, yv = x[vx].ravel(), x[vy].ravel()
            if points == "q" and vx != "mass_ratio":
                mappable = ax.scatter(xv[:n_points], yv[:n_points], c=x["mass_ratio"].ravel()[:n_points],
                                      s=2, alpha=0.35, linewidths=0, cmap="viridis",
                                      vmin=Q_MIN, vmax=1.0, rasterized=True)
            elif points is not None:
                ax.scatter(xv[:n_points], yv[:n_points], s=2, alpha=0.35, linewidths=0,
                           color=color, rasterized=True)
            if contours:
                _contours(ax, xv, yv, rx, ry, color)
        return mappable

    return plot_population, population_figure


@app.cell
def _(explore_sample, explore_sliders, plot_population, population_figure):
    _L = {k: float(v) for k, v in explore_sliders.value.items()}
    # plain mathtext keeps redraws fast (the notebook default renders through LaTeX)
    with plt.rc_context({"text.usetex": False}):
        _fig, _axes = population_figure()
        _sc = plot_population(_axes, explore_sample(_L), _L, color="0.2", points="q")
        _fig.colorbar(_sc, ax=_axes[2, 1]).ax.set_title(r"$q$")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulating the catalog
    """)
    return


@app.cell
def _(Lambda_true, pop_model):
    measurement = GWMockCatMeasurement()
    trigger = ai.detection.ThresholdTrigger(threshold=SNR_THRESHOLD, statistic="snr")

    truths = GWParams.from_dict(pop_model.sample(Lambda_true, key=jax.random.key(0), shape=(N_TRUTHS,)))
    observed = measurement.sample(truths, key=jax.random.key(1))
    detected = trigger.detect(observed, np.random.default_rng(2))

    _keep = np.flatnonzero(detected)[:N_EVENTS]
    data, detected_truths = observed[_keep], truths[_keep]

    mo.md(f"{detected.sum():,} of {N_TRUTHS:,} sources are detected, and the analysis uses the first "
          f"{len(_keep)}. Their observed SNRs span {float(data.snr.min()):.1f} to "
          f"{float(data.snr.max()):.1f} (median {float(jnp.median(data.snr)):.1f}).")
    return data, detected_truths, measurement, trigger, truths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The plots show the tilt cosine against the spin magnitude in the true population, for
    near-equal and for unequal mass ratios.
    """)
    return


@app.cell
def _(truths):
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    _q = np.asarray(truths.mass_ratio)
    for _ax, (_sel, _title) in zip(_axes, [(_q > 0.9, r"$q > 0.9$"), (_q < 0.6, r"$q < 0.6$")]):
        _idx = np.flatnonzero(_sel)[:3000]
        _ax.scatter(np.asarray(truths.chi_1)[_idx], np.asarray(truths.cos_tilt_1)[_idx],
                    s=3, alpha=0.3, color="0.25", linewidths=0)
        _ax.set_title(_title)
        _ax.set_xlabel(r"$\chi_1$")
        _ax.set_xlim(0, 1)
        _ax.set_ylim(-1, 1)
    _axes[0].set_ylabel(r"$\cos\theta_1$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The proposal

    For one event, the data restrict $(m_1, q, z, \Theta)$ to a small region. With $w=1/\hat\rho$, as in
    the measurement model, the proposal draws:

    * $\Theta\sim\mathcal{TN}(\hat\Theta,\ k_\Theta u_\Theta w)_{[0,1]}$, because the projection factor
      is observed directly;
    * $\ln\mathcal{M}_\mathrm{det}\sim\mathcal N(\ln\hat{\mathcal M}_\mathrm{det},\ k_\mathcal{M}\,u_\mathcal{M}w)$,
      because the chirp mass is also observed directly;
    * $q\sim U\big(q(\hat\eta - k_\eta u_\eta w),\ q(\hat\eta + k_\eta u_\eta w)\big)$, by inverting the
      symmetric mass ratio over a window of $\pm k_\eta$ standard deviations, which is wide near
      $q=1$, where $\eta$ is flat;
    * $\ln z\mid\Theta\sim\mathcal N(\ln\hat z(\Theta),\ k_z/\hat\rho)$, where $\hat z(\Theta)$ solves
      $\rho_\mathrm{opt}(\mathcal M_\mathrm{det}, q, d_L(z))\,\Theta=\hat\rho$. This is centred on the
      drawn value of $\Theta$, so the proposal follows the ridge between the SNR, $\Theta$ and the
      distance.

    From these, $m_1=\mathcal M_\mathrm{det}/\big[(1+z)\,c(q)\big]$, and $\log q(m_1,q,z,\Theta\mid d)$
    includes the Jacobian of that transformation. Finally the spins are drawn from `spin_model`,
    $\vec\chi\sim p(\cdot\mid q,\Lambda)$, with a separate key, so that they are also common random
    numbers across $\Lambda$.

    The proposal is a `Proposal`, a measurement process run backwards: its `parameter_type` is the
    data class and its `data_type` is the parameter class.
    """)
    return


@app.cell
def _():
    def q_of_eta(eta):
        eta = jnp.clip(eta, 1e-6, 0.25)
        return (1 - 2 * eta - jnp.sqrt(1 - 4 * eta)) / (2 * eta)


    def truncnorm_draw(key, loc, scale, lo, hi):
        return loc + scale * jax.random.truncated_normal(key, (lo - loc) / scale, (hi - loc) / scale)


    def truncnorm_logpdf(x, loc, scale, lo, hi):
        return truncnorm.logpdf(x, (lo - loc) / scale, (hi - loc) / scale, loc=loc, scale=scale)


    class DataInformedProposal(ai.Proposal):
        """q_Lambda(theta | d) = q(m1, q, z, Theta | d) p(spins | q, Lambda)."""

        parameter_type = GWMockCatData
        data_type = GWParams

        def __init__(self, measurement, spin_model, k_mc=1.5, k_eta=4.0, k_theta=1.3, k_z=1.3, **kw):
            super().__init__(**kw)
            self.measurement, self.spin_model = measurement, spin_model
            self.k_mc, self.k_eta, self.k_theta, self.k_z = k_mc, k_eta, k_theta, k_z
            self._z_grid = jnp.asarray(measurement.cosmology.z)
            self._dl_grid = jnp.asarray(measurement.cosmology.d_L)

        def region(self, d):
            """Proposal parameters for (Theta, ln Mc_det, q) of event d."""
            m = self.measurement
            w = 1.0 / jnp.maximum(d.snr, m.snr_floor)
            de = self.k_eta * m.u_eta * w
            q_lo = jnp.maximum(q_of_eta(d.symmetric_mass_ratio - de), Q_MIN)
            q_hi = jnp.where(d.symmetric_mass_ratio + de >= 0.25, 1.0,
                             q_of_eta(d.symmetric_mass_ratio + de))
            return dict(theta_loc=d.projection_factor_obs, theta_sd=self.k_theta * m.u_projection * w,
                        ln_mc=d.log_chirp_mass_det, ln_mc_sd=self.k_mc * m.u_chirp_mass * w,
                        q_lo=q_lo, q_hi=jnp.maximum(q_hi, q_lo + 1e-3))

        def redshift_region(self, d, r, theta):
            """ln z mean and sd given Theta: solve snr = rho_opt(Mc_det, q, d_L(z)) * Theta for d_L."""
            snr_model = self.measurement.snr_model
            mc = jnp.exp(r["ln_mc"])
            m_tot = total_mass_from_chirp_mass(mc, 0.5 * (r["q_lo"] + r["q_hi"]))
            snr_hat = jnp.maximum(d.snr, 1.0)
            d_l = (snr_model.reference_snr * (mc / snr_model.reference_chirp_mass) ** (5 / 6)
                   * snr_model.bns_horizon / (1 + (m_tot / snr_model.m_cut) ** 2) * theta / snr_hat)
            z_hat = jnp.clip(jnp.interp(d_l, self._dl_grid, self._z_grid), 1e-3, Z_MAX)
            return jnp.log(z_hat), self.k_z / snr_hat

        def sample_single(self, d, key, Lambda):
            k_theta, k_mc, k_q, k_z, k_spin = jax.random.split(key, 5)
            r = self.region(d)
            theta = truncnorm_draw(k_theta, r["theta_loc"], r["theta_sd"], 0.0, 1.0)
            mc_det = jnp.exp(r["ln_mc"] + r["ln_mc_sd"] * jax.random.normal(k_mc))
            q = r["q_lo"] + (r["q_hi"] - r["q_lo"]) * jax.random.uniform(k_q)
            ln_z, ln_z_sd = self.redshift_region(d, r, theta)
            z = jnp.exp(ln_z + ln_z_sd * jax.random.normal(k_z))
            m1 = mc_det / ((1 + z) * chirp_mass_from_mass_ratio(1.0, q))
            spins = self.spin_model.sample(Lambda, key=k_spin, shape=(),
                                           conditioned_on=dict(mass_ratio=q))
            return GWParams(mass_1_source=m1, mass_ratio=q, redshift=z, projection_factor=theta,
                            **{v: spins[v] for v in SPIN_VARS})

        def log_q_informed(self, theta, d):
            """log q(m1, q, z, Theta | d), with the Jacobian of (ln Mc_det, ln z) -> (m1, z)."""
            r = self.region(d)
            m1, q, z = theta.mass_1_source, theta.mass_ratio, theta.redshift
            ln_z, ln_z_sd = self.redshift_region(d, r, theta.projection_factor)
            mc_det = m1 * chirp_mass_from_mass_ratio(1.0, q) * (1 + z)
            return (truncnorm_logpdf(theta.projection_factor, r["theta_loc"], r["theta_sd"], 0.0, 1.0)
                    + norm.logpdf(jnp.log(mc_det), r["ln_mc"], r["ln_mc_sd"]) - jnp.log(m1)
                    + jnp.where((q >= r["q_lo"]) & (q <= r["q_hi"]),
                                -jnp.log(r["q_hi"] - r["q_lo"]), -jnp.inf)
                    + norm.logpdf(jnp.log(z), ln_z, ln_z_sd) - jnp.log(z))

        def logpdf_single(self, theta, d, Lambda):
            return self.log_q_informed(theta, d) + self.spin_model.log_prob(theta.to_dict(), Lambda)

    return (DataInformedProposal,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The likelihood with selection

    The likelihood subclasses `PopulationLikelihood` and overrides `log_weights` with the weight
    above, in which the spin factor cancels. Detection depends only on $(m_1,q,z,\Theta)$, and
    $p(\Theta)$ has no hyperparameters, so the selection term needs only the mass and redshift model:

    $$\ln\mathcal L(\Lambda)=\sum_i\ln\hat Z_i(\Lambda) - N_\mathrm{events}\ln\hat\alpha(\Lambda),\qquad
    \hat\alpha(\Lambda)=\frac{1}{N_\mathrm{inj}}\sum_{\mathrm{found}}\frac{p(m_1,q,z\mid\Lambda)}{g(m_1,q,z)},$$

    The injections are drawn from the same mass model with broader hyperparameters, and $g$ is their
    density.
    """)
    return


@app.class_definition
class CancelledSpinLikelihood(ai.PopulationLikelihood):
    """log w = log L + log p(m1, q, z, Theta | Lambda) - log q(m1, q, z, Theta | d); the spins
    cancel. logpdf adds the selection term -E log alpha(Lambda)."""

    def __init__(self, *args, informed_model, mass_model, injections, injection_log_prob,
                 n_injections, **kw):
        super().__init__(*args, **kw)
        self.informed_model, self.mass_model = informed_model, mass_model
        self.injections = injections                    # found injections {var: array}
        self.injection_log_prob = injection_log_prob    # log g at the found injections
        self.n_injections = n_injections                # all injections drawn

    def log_weights(self, Lambda):
        theta = self.proposal_draws(Lambda)
        d = self.data.reshape_batch((self.E, 1))
        return (self.log_likelihood_at(theta)
                + self.informed_model.log_prob(theta.to_dict(), Lambda)
                - self.proposal.log_q_informed(theta, d))

    def log_selection(self, Lambda):
        log_r = self.mass_model.log_prob(self.injections, Lambda) - self.injection_log_prob
        lse = jax.scipy.special.logsumexp(log_r)
        log_alpha = lse - jnp.log(self.n_injections)
        n_eff = jnp.exp(2 * lse - jax.scipy.special.logsumexp(2 * log_r))
        return log_alpha, n_eff

    def logpdf(self, Lambda):
        log_l = super().logpdf(Lambda)
        log_alpha, n_eff = self.log_selection(Lambda)
        self.diagnostics = {**self.diagnostics, "N_eff_selection": n_eff}
        return log_l - self.E * log_alpha


@app.cell
def _(Lambda_true, mass_model, measurement, pop_model, trigger):
    Lambda_injection = {**Lambda_true, "alpha": 0.5, "mu_q": 0.6, "sigma_q": 0.5, "lamb": 1.0}
    _draws = pop_model.sample(Lambda_injection, key=jax.random.key(5), shape=(N_INJECTIONS,))
    _found = trigger.detect(measurement.sample(GWParams.from_dict(_draws), key=jax.random.key(6)),
                            np.random.default_rng(7))
    found_injections = {k: jnp.asarray(_draws[k])[_found] for k in MASS_VARS}
    injection_log_prob = mass_model.log_prob(found_injections, Lambda_injection)

    mo.md(f"{_found.sum():,} of {N_INJECTIONS:,} injections are found.")
    return found_injections, injection_log_prob


@app.cell
def _(
    DataInformedProposal,
    data,
    found_injections,
    informed_model,
    injection_log_prob,
    mass_model,
    measurement,
    pop_model,
    spin_model,
):
    proposal = DataInformedProposal(measurement, spin_model)

    def make_likelihood(seed):
        return CancelledSpinLikelihood(
            measurement, data, pop_model, N=N_DRAWS, seed=seed, proposal=proposal,
            informed_model=informed_model, mass_model=mass_model, injections=found_injections,
            injection_log_prob=injection_log_prob, n_injections=N_INJECTIONS)

    likelihood = make_likelihood(seed=3)
    proposal
    return likelihood, make_likelihood


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checks at the true $\Lambda$

    Before sampling, we check the effective sample size of each event at the true $\Lambda$, and the
    Monte Carlo scatter of $\ln\mathcal L$ between two independent proposal keys.
    """)
    return


@app.cell
def _(Lambda_true, likelihood, make_likelihood):
    _lp, n_eff_true = jax.jit(likelihood.event_logpdf_and_neff)(Lambda_true)
    _lp_other, _ = jax.jit(make_likelihood(seed=11).event_logpdf_and_neff)(Lambda_true)
    _log_alpha, _n_eff_sel = likelihood.log_selection(Lambda_true)

    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.hist(np.asarray(n_eff_true), bins=np.geomspace(1, N_DRAWS, 30), color="0.35")
    _ax.set_xscale("log")
    _ax.set_xlabel(r"effective draws per event, $N_\mathrm{eff}$")
    _ax.set_ylabel("events")
    _fig.tight_layout()

    mo.vstack([
        mo.md(
            f"Out of {N_DRAWS:,} draws per event, the median $N_\\mathrm{{eff}}$ is "
            f"{float(jnp.median(n_eff_true)):.0f} and the minimum is {float(n_eff_true.min()):.1f}. "
            f"Two proposal keys give $\\sum_i\\ln\\hat Z_i$ = {float(_lp.sum()):.2f} and "
            f"{float(_lp_other.sum()):.2f}. The selection term has $N_\\mathrm{{eff}}$ = "
            f"{float(_n_eff_sel):,.0f}, which should be much larger than "
            f"$4N_\\mathrm{{events}}$ = {4 * N_EVENTS}."),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hyperparameter inference

    Samples saved in `notebooks/data/proposal_population/` are loaded if they exist. Tick the box to
    re-run the sampler, which takes about 15 to 20 minutes on a laptop CPU.
    """)
    return


@app.cell
def _():
    hyper_priors = dict(
        alpha=dist.Uniform(-2.0, 6.0),
        mu_q=dist.Uniform(0.3, 1.0), sigma_q=dist.Uniform(0.05, 1.0),
        lamb=dist.Uniform(-2.0, 6.0),
        mu_chi_1=dist.Uniform(0.0, 1.0), sigma_chi_1=dist.Uniform(0.05, 1.0),
        mu_chi_2=dist.Uniform(0.0, 1.0), sigma_chi_2=dist.Uniform(0.05, 1.0),
        sigma_0=dist.LogUniform(0.005, 0.5), sigma_slope=dist.Uniform(0.0, 2.0),
    )
    force_run = mo.ui.checkbox(label="re-run the sampler (overwrites data/proposal_population)")
    force_run
    return force_run, hyper_priors


@app.cell
def _(force_run, hyper_priors, likelihood):
    _path = RESULTS_DIR / "hyperposterior.csv"
    if _path.exists() and not force_run.value:
        hyperposterior = pd.read_csv(_path)
    else:
        _sampler = ai.run_sampler(likelihood, hyper_priors, seed=1, num_warmup=500,
                                  num_samples=1000, summary=False,
                                  sampler_kwargs={"dense_mass": True})
        hyperposterior = _sampler.samples
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        hyperposterior.to_csv(_path, index=False)
    hyperposterior.describe().T[["mean", "std", "min", "max"]]
    return (hyperposterior,)


@app.cell
def _(hyperposterior):
    mo.md(f"""
    Along the chain, the smallest per-event $N_\\mathrm{{eff}}$ has a median of
    {hyperposterior['N_eff_min'].median():.1f} and a minimum of {hyperposterior['N_eff_min'].min():.1f}.
    The smallest selection $N_\\mathrm{{eff}}$ is {hyperposterior['N_eff_selection'].min():,.0f}.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Mass and redshift sector
    """)
    return


@app.cell(expand_output=True)
def _(Lambda_true, hyperposterior):
    _vars = ["alpha", "mu_q", "sigma_q", "lamb"]
    ph.make_corner_plot([hyperposterior[_vars]], model_labels=["hyperposterior"], variables=_vars,
                        variable_labels=[LABELS[v] for v in _vars],
                        truth={v: Lambda_true[v] for v in _vars}, scatter_kwargs=dict(s=3, alpha=0.2))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Spin sector
    """)
    return


@app.cell(expand_output=True)
def _(Lambda_true, hyperposterior):
    _vars = ["mu_chi_1", "sigma_chi_1", "mu_chi_2", "sigma_chi_2", "sigma_0", "sigma_slope"]
    ph.make_corner_plot([hyperposterior[_vars]], model_labels=["hyperposterior"], variables=_vars,
                        variable_labels=[LABELS[v] for v in _vars],
                        truth={v: Lambda_true[v] for v in _vars}, scatter_kwargs=dict(s=3, alpha=0.2))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The tilt scatter $\sigma(q)$

    The plot shows the inferred $\sigma(q)=\sigma_0+\sigma_1(1-q)$, its median and 90% band, against
    the true curve. The points are the true values of $|\cos\theta_1-\chi_1|$ and
    $|\cos\theta_2-\chi_2|$ for the detected events.
    """)
    return


@app.cell
def _(Lambda_true, detected_truths, hyperposterior):
    _q = np.linspace(Q_MIN, 1.0, 200)
    _sig = (hyperposterior["sigma_0"].to_numpy()[:, None]
            + hyperposterior["sigma_slope"].to_numpy()[:, None] * (1 - _q[None, :]))
    _lo, _med, _hi = np.quantile(_sig, [0.05, 0.5, 0.95], axis=0)

    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _qd = np.asarray(detected_truths.mass_ratio)
    for _i in (1, 2):
        _res = np.abs(np.asarray(getattr(detected_truths, f"cos_tilt_{_i}"))
                      - np.asarray(getattr(detected_truths, f"chi_{_i}")))
        _ax.scatter(_qd, _res, s=10, color="0.6", linewidths=0,
                    label="detected events (truth)" if _i == 1 else None)
    _ax.fill_between(_q, _lo, _hi, color=ph.default_pallete[1], alpha=0.3, linewidth=0,
                     label="inferred, 90\\%")
    _ax.plot(_q, _med, color=ph.default_pallete[1], lw=2, label="inferred, median")
    _ax.plot(_q, Lambda_true["sigma_0"] + Lambda_true["sigma_slope"] * (1 - _q),
             color="black", lw=1.5, ls="--", label="truth")
    _ax.set_xlabel(r"$q$")
    _ax.set_ylabel(r"$\sigma(q)$")
    _ax.set_xlim(Q_MIN, 1.0)
    _ax.set_ylim(0, None)
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Population predictive distribution

    These panels show the population averaged over the hyperposterior, on the same axes as the
    explorer above, using 200 hyperposterior draws with 5,000 sources each. The one-dimensional
    panels and $\sigma(q)$ show the median and 90% band across draws. The band includes some Monte
    Carlo noise from using 5,000 sources per draw. The two-dimensional panels show the 50% and 90%
    contours of all draws pooled together.

    The grey reference is the population set by the sliders, which are shown again here. They start
    at the true values, so by default the grey reference is the true population.
    """)
    return


@app.cell
def _(Lambda_true, hyperposterior, pop_model):
    N_PPD_HYPER, N_PPD_DRAWS = 200, 5_000
    _rows = hyperposterior.iloc[np.linspace(0, len(hyperposterior) - 1, N_PPD_HYPER).astype(int)]
    ppd_Lambda = {k: _rows[k].to_numpy() for k in Lambda_true}
    _draw = jax.jit(jax.vmap(lambda L, k: pop_model.sample(L, key=k, shape=(N_PPD_DRAWS,))))
    ppd = _draw({k: jnp.asarray(v) for k, v in ppd_Lambda.items()},
                jax.random.split(jax.random.key(321), N_PPD_HYPER))
    ppd = {k: np.asarray(v) for k, v in ppd.items()}   # {var: (N_PPD_HYPER, N_PPD_DRAWS)}
    return ppd, ppd_Lambda


@app.cell
def _(
    explore_controls,
    explore_sample,
    explore_sliders,
    plot_population,
    population_figure,
    ppd,
    ppd_Lambda,
):
    _L = {k: float(v) for k, v in explore_sliders.value.items()}
    with plt.rc_context({"text.usetex": False}):
        _fig, _axes = population_figure()
        plot_population(_axes, explore_sample(_L), _L, color="0.45", label="explorer (sliders)",
                        points="plain")
        plot_population(_axes, ppd, ppd_Lambda, color=ph.default_pallete[1],
                        label="PPD (median, 90%)", contours=True)
        _fig.legend(loc="outside upper center", ncols=2, frameon=False)
    mo.vstack([explore_controls, _fig])
    return


if __name__ == "__main__":
    app.run()

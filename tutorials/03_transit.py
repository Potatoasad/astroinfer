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
    from numpyro.distributions import constraints

    from astroinfer import MeasurementProcess, parameter, parameter_dataclass

    import _style as style

    style.apply()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. A transiting planet

    When a planet passes in front of its star, it blocks a small part of the star's light. If the
    orbit is seen edge on, this happens once per orbit, and the brightness of the star dips by the
    fraction of its disk that the planet covers. A telescope measures the brightness of a star
    every 30 minutes for 20 days. From that *light curve*, we want the orbital period of the
    planet, the time of its first transit and its radius.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src=mo.notebook_dir() / "images" / "03_transit.svg",
             alt="A planet crossing its star, and the dips it makes in the light curve",
             width=640)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The model

    The parameters $\theta$ are the orbital period $P$, the time of the first transit $T_0$, and
    the radius $R_p$ of the planet in units of the radius of the star. A planet of radius $R_p$
    covers a fraction $R_p^2$ of the disk, so that is the depth of each dip. We keep the model
    simple. The star has the same brightness across its disk, every transit lasts $\tau = 0.2$
    days, and the noise is white: independent and Gaussian, with a standard deviation $\sigma$
    that is the same for every exposure.

    The data are three arrays along the same axis: the times $t_i$ of the exposures, the measured
    fluxes $\hat F_i$ and their errors $\sigma_i$. Carrying the times in the data means the
    likelihood, and anything else that looks at a light curve, does not need to know how the
    telescope was scheduled.

    A `@parameter_dataclass` is an ordinary dataclass, so it can have methods. `LightCurve` gets
    a `plot` method, which draws error bars, or with `noise=False` plain lines. A batch of light
    curves has `time` and `flux` of shape `(N, n)`, and `plot` draws their transposes, one line
    per light curve.
    """)
    return


@app.class_definition
@parameter_dataclass(name="Planet", latex=r"\theta")
class Planet:
    period: float = parameter(name="Orbital period [days]", latex="P",
                              support=constraints.positive)
    t0: float = parameter(name="Time of first transit [days]", latex="T_0")
    radius: float = parameter(name="Planet radius [stellar radii]", latex="R_p",
                              support=constraints.unit_interval)


@app.class_definition
@parameter_dataclass(name="Light curve", latex="d")
class LightCurve:
    time: jax.Array = parameter(name="Time [days]", latex="t",
                                index=("time",), index_latex=("i",))
    flux: jax.Array = parameter(name="Relative flux", latex=r"\hat F",
                                index=("time",), index_latex=("i",))
    flux_err: jax.Array = parameter(name="Flux error", latex=r"\sigma",
                                    index=("time",), index_latex=("i",))

    def plot(self, ax, noise=True, **kwargs):
        """Draw the light curve on ax, with error bars if noise."""
        if noise:
            ax.errorbar(self.time, self.flux, yerr=self.flux_err, fmt=".",
                        ms=2, lw=0.5, **kwargs)
        else:  # a batch of light curves is drawn as one line each
            ax.plot(self.time.T, self.flux.T, lw=1, **kwargs)
        ax.set(xlabel="time [days]", ylabel="relative flux")
        return ax.figure, ax


@app.class_definition
@dataclass
class Photometry(MeasurementProcess):
    days: float = 20.0         # length of the observation
    cadence: float = 1 / 48    # one exposure every 30 minutes
    duration: float = 0.2      # transit duration [days]
    sigma: float = 0.002       # white noise per exposure

    display_name = "Transit photometry"
    parameter_type = Planet      # theta is a Planet
    data_type = LightCurve       # d is a LightCurve

    def light_curve(self, theta: Planet, t: jax.Array) -> jax.Array:
        """Flux of the star at times t: a box with smoothed edges."""
        P = theta.period
        phase = (t - theta.t0 + P / 2) % P - P / 2
        edge = jnp.abs(phase) - self.duration / 2
        return 1 - theta.radius**2 * jax.nn.sigmoid(-edge / 0.01)

    def sample_single(self, theta: Planet, key: jax.Array) -> LightCurve:
        t = jnp.arange(0, self.days, self.cadence)
        noise = self.sigma * jax.random.normal(key, t.shape)
        return LightCurve(time=t, flux=self.light_curve(theta, t) + noise,
                          flux_err=jnp.full_like(t, self.sigma))

    def logpdf_single(self, d: LightCurve, theta: Planet) -> jax.Array:
        model = self.light_curve(theta, d.time)
        return dist.Normal(model, d.flux_err).log_prob(d.flux).sum()

    def _display_equations(self):
        return [r"\hat F_i \sim \mathcal{N}\!\left(1 - R_p^2\, b(t_i),\ \sigma_i\right)",
                r"b(t) = 1 \text{ within } \tau/2 \text{ of } T_0 + kP, \text{ else } 0"]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `light_curve` is a helper method of the measurement process, used by both `sample_single` and
    `logpdf_single`. It computes the phase of each time, the distance from the middle of the
    nearest transit, and turns it into a box. A sharp box has zero gradient almost everywhere and
    jumps at its edges, which stops NUTS from moving $P$ and $T_0$, so the edges are smoothed with
    a sigmoid over about 0.01 days, shorter than one exposure.

    We pass every setting of the telescope explicitly, even where it equals the default:
    """)
    return


@app.cell
def _():
    telescope = Photometry(days=20.0, cadence=1 / 48, duration=0.2, sigma=0.002)
    telescope
    return (telescope,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulating a light curve

    The true planet has a period of 3 days, a first transit at 1.2 days and a radius of 0.1
    stellar radii, about that of Jupiter around the Sun. One light curve from the telescope:
    """)
    return


@app.cell
def _(telescope):
    truth = Planet(period=3.0, t0=1.2, radius=0.1)
    lightcurve = telescope.sample(truth, key=0)
    lightcurve
    return lightcurve, truth


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It has 960 exposures. On the left is the whole light curve, drawn with its `plot` method. On
    the right, every time is replaced by its phase, the time since the nearest transit for the
    true period, which stacks the seven transits on top of each other:
    """)
    return


@app.cell
def _(lightcurve, truth):
    _phase = (lightcurve.time - truth.t0 + truth.period / 2) % truth.period - truth.period / 2

    _fig, (_ax, _ax_fold) = plt.subplots(1, 2, figsize=(10, 3.2), width_ratios=[2, 1],
                                         layout="constrained")
    lightcurve.plot(_ax, noise=True, color=style.DATA)
    _ax.set_title("Simulated light curve")
    _ax_fold.errorbar(_phase, lightcurve.flux, yerr=lightcurve.flux_err, fmt=".", ms=2,
                      lw=0.5, color=style.DATA)
    _ax_fold.set(xlim=(-0.5, 0.5), xlabel="time since transit [days]",
                 title="Folded on the true period")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each dip is $R_p^2 = 0.01$ deep, five times the noise $\sigma = 0.002$, and lasts about ten
    exposures.

    ## Parameter estimation

    The priors are uniform: periods between 2 and 4 days, a first transit in the first 2 days,
    and radii up to 0.3. The prior on $T_0$ is shorter than the shortest period, so that $T_0$
    and $T_0 + P$ cannot both describe the same transit. With `as_posterior=True`, `PE` returns a
    `PosteriorData`, which keeps the samples as a batch of `Planet` objects, with one axis for the
    chain and one for the draw. Notebook 4 looks at it in detail.
    """)
    return


@app.cell
def _(lightcurve, telescope):
    pe_priors = {"period": dist.Uniform(2, 4), "t0": dist.Uniform(0, 2),
                 "radius": dist.Uniform(0, 0.3)}
    posterior = telescope.PE(lightcurve, pe_priors, seed=0, as_posterior=True)
    posterior
    return pe_priors, posterior


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `stack_samples()` merges the chain and draw axes into one, which is the shape a scatter plot
    wants. The radius against the period, and the time of first transit:
    """)
    return


@app.cell
def _(posterior, truth):
    samples = posterior.stack_samples().theta

    _fig, (_ax, _ax_t0) = plt.subplots(1, 2, figsize=(9.6, 3.4), layout="constrained")
    _ax.scatter(samples.period, samples.radius, s=2, alpha=0.2, color=style.POSTERIOR,
                rasterized=True)
    _ax.plot(truth.period, truth.radius, "x", ms=10, mew=2, color=style.TRUTH, label="truth")
    _ax.set(xlabel="orbital period $P$ [days]", ylabel="planet radius $R_p$ [stellar radii]",
            title="Posterior samples of $P$ and $R_p$")
    _ax.legend(loc="upper right")
    style.density(_ax_t0, samples.t0, color=style.POSTERIOR, label="posterior")
    style.truth(_ax_t0, truth.t0)
    _ax_t0.set(xlabel="time of first transit $T_0$ [days]", ylabel="density",
               title="Posterior of $T_0$")
    _ax_t0.legend(loc="upper right")
    _fig
    return (samples,)


@app.cell(hide_code=True)
def _(samples):
    _P, _T0, _R = (np.asarray(x) for x in (samples.period, samples.t0, samples.radius))
    mo.md(rf"""
    The posterior gives $P = {_P.mean():.4f} \pm {_P.std():.4f}$ days,
    $T_0 = {_T0.mean():.3f} \pm {_T0.std():.3f}$ days and
    $R_p = {_R.mean():.4f} \pm {_R.std():.4f}$. The period is known to about a minute and a half,
    because an error in $P$ grows with every orbit, and seven transits pin it down. $P$ and $R_p$
    are uncorrelated: the depth of the dips does not depend on when they happen.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Posterior predictive check

    A *posterior predictive* light curve is one simulated from a draw of the posterior. If the
    model is right, the data should look like one of them. `posterior_predictive(posterior,
    draws=100, key=1)` picks 100 posterior draws at random and calls `sample` on them. We call it
    on a copy of the telescope with no noise, which gives the light curve that each draw
    predicts, and draw them with `plot(noise=False)` over the data:
    """)
    return


@app.cell
def _():
    noiseless = Photometry(days=20.0, cadence=1 / 48, duration=0.2, sigma=0.0)
    return (noiseless,)


@app.cell
def _(lightcurve, noiseless, posterior, truth):
    predictive = noiseless.posterior_predictive(posterior, draws=100, key=1)

    _fig, (_ax, _ax_zoom) = plt.subplots(1, 2, figsize=(10, 3.2), width_ratios=[2, 1],
                                         layout="constrained")
    for _a in (_ax, _ax_zoom):
        lightcurve.plot(_a, noise=True, color=style.DATA, alpha=0.4)
        predictive.plot(_a, noise=False, color=style.POSTERIOR, alpha=0.05)
    _ax.set_title("Posterior predictive light curves and the data")
    _ax_zoom.set(xlim=(truth.t0 - 0.4, truth.t0 + 0.4), title="The first transit")
    _fig
    return (predictive,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The 100 predicted light curves lie on top of each other and go through the middle of the
    data, in and out of transit.

    ## Summary

    * A `@parameter_dataclass` can have methods. They act on a single object or a batch, in the
      same way as its fields.
    * A measurement process can have helper methods too. Here `light_curve` is shared by
      `sample_single` and `logpdf_single`.
    * `PE(..., as_posterior=True)` returns a `PosteriorData`. `stack_samples()` merges its chain
      and draw axes, and `.theta` is a batch of the parameter type.
    * `posterior_predictive(posterior, draws=, key=)` simulates data from posterior draws. A copy
      of the measurement process without noise gives the predicted signal.

    Notebook 4 looks at `PosteriorData` in more detail, and at how to give your own objects a
    display like the ones above.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: errors that are too small

    Suppose the pipeline reports errors of 0.002 for every exposure, but the real noise is 0.004.

    1. Simulate a light curve whose noise is 0.004, then set its `flux_err` to 0.002. Since
       `LightCurve` is frozen, build a new one from the fields of the old one.
    2. Run PE with `Photometry`. Compare the width of the posterior of $R_p$ with the one above.
       With this much noise, the likelihood of $P$ has local maxima where only some of the
       transits line up, and NUTS started anywhere in the wide priors above can get stuck in one.
       Use priors close to the period found above instead: $P$ between 2.9 and 3.1 days and $T_0$
       between 1.1 and 1.3 days.
    3. Add a field `jitter` to the parameters, and use the standard deviation
       $\sqrt{\sigma_i^2 + s^2}$ in the likelihood, where $s$ is the jitter. Run PE with a prior
       on $s$ between 0 and 0.01. What value of $s$ do you find, and how wide is the posterior of
       $R_p$ now?

    There is an empty cell below for your code.
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
    noisy = Photometry(days=20.0, cadence=1 / 48, duration=0.2, sigma=0.004)
    real = noisy.sample(Planet(period=3.0, t0=1.2, radius=0.1), key=2)
    reported = LightCurve(time=real.time, flux=real.flux,
                          flux_err=jnp.full_like(real.flux, 0.002))

    narrow = {"period": dist.Uniform(2.9, 3.1), "t0": dist.Uniform(1.1, 1.3),
              "radius": dist.Uniform(0, 0.3)}
    trusting = Photometry(days=20.0, cadence=1 / 48, duration=0.2, sigma=0.002)
    naive = trusting.PE(reported, narrow, seed=0)


    @parameter_dataclass(name="Planet with jitter", latex=r"\theta")
    class JitteryPlanet:
        period: float = parameter(name="Orbital period [days]", latex="P")
        t0: float = parameter(name="Time of first transit [days]", latex="T_0")
        radius: float = parameter(name="Planet radius [stellar radii]", latex="R_p")
        jitter: float = parameter(name="Extra white noise", latex="s")


    @dataclass
    class JitteryPhotometry(Photometry):
        parameter_type = JitteryPlanet

        def logpdf_single(self, d, theta):
            model = self.light_curve(theta, d.time)
            scale = jnp.sqrt(d.flux_err**2 + theta.jitter**2)
            return dist.Normal(model, scale).log_prob(d.flux).sum()


    careful = JitteryPhotometry(days=20.0, cadence=1 / 48, duration=0.2, sigma=0.002)
    fitted = careful.PE(reported, {**narrow, "jitter": dist.Uniform(0, 0.01)}, seed=0)
    # naive["radius"].std() is about half of fitted["radius"].std(): the naive posterior is too
    # narrow. fitted["jitter"] is close to sqrt(0.004**2 - 0.002**2), about 0.0035.
    ```
    """)})
    return


if __name__ == "__main__":
    app.run()

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

    import astroinfer as ai
    from astroinfer import MeasurementProcess, batched_method, parameter, parameter_dataclass

    import _style as style

    style.apply()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. The lighthouse

    A lighthouse stands somewhere off a straight coast, at position $\alpha$ along the shore and a
    distance $\beta$ out to sea. It rotates and emits short flashes at uniformly random angles. You
    can't see the lighthouse, but a row of detectors along the shore records where each flash hits
    the coast. Where is the lighthouse?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src=mo.notebook_dir() / "images" / "02_lighthouse.svg",
             alt="A lighthouse off a straight coast, flashing towards detectors on the shore",
             width=640)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A flash at angle $\phi\sim U(-\tfrac\pi2,\tfrac\pi2)$ lands at $\hat x = \alpha + \beta\tan\phi$,
    so every recorded position is a draw from a *Cauchy distribution*,

    $$
    \mathcal{L}(\hat x_k\mid\alpha,\beta) = \frac{\beta}{\pi\left[\beta^2 + (\hat x_k-\alpha)^2\right]} .
    $$

    This is Gull's (1988) lighthouse problem. We use it to look at the part of astroinfer that
    notebook 1 skipped over: what a `@parameter_dataclass` object is, and what you can do with one.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The model

    The class definitions below introduce a few things that were not in notebook 1.
    `support=constraints.positive` on $\beta$ declares that the lighthouse is out at sea; we use it
    in the section on constrained parameters. `Flashes.x` has a core axis `"flash"`, so one
    observation is a whole night of flashes. Finally, `flash_density` is an ordinary method on the
    parameter class, decorated with `@batched_method`, which is explained further down.
    """)
    return


@app.class_definition
@parameter_dataclass(name="Lighthouse", latex=r"\theta")
class Lighthouse:
    alpha: float = parameter(name="Position along the shore [km]", latex=r"\alpha")
    beta: float = parameter(name="Distance out to sea [km]", latex=r"\beta",
                            support=constraints.positive)

    @batched_method
    def flash_density(self, x):
        """Density of flash positions along the shore, evaluated on a grid x."""
        return self.beta / (jnp.pi * (self.beta**2 + (x - self.alpha) ** 2))


@app.class_definition
@parameter_dataclass(name="Flashes", latex=r"\hat{x}")
class Flashes:
    x: float = parameter(name="Shore position of each flash [km]", latex=r"\hat{x}",
                         index=("flash",), index_latex=("k",))


@app.class_definition
@dataclass
class Shoreline(MeasurementProcess):
    n_flashes: int = 30

    display_name = "Shoreline detectors"
    parameter_type = Lighthouse
    data_type = Flashes

    def sample_single(self, theta, key):
        angle = jax.random.uniform(key, (self.n_flashes,), minval=-jnp.pi / 2, maxval=jnp.pi / 2)
        return Flashes(x=theta.alpha + theta.beta * jnp.tan(angle))

    def logpdf_single(self, d, theta):
        log_k = jnp.log(theta.beta / jnp.pi) - jnp.log(theta.beta**2 + (d.x - theta.alpha) ** 2)
        return jnp.sum(log_k, axis=-1)


@app.cell
def _():
    shoreline = Shoreline(n_flashes=30)
    lighthouse = Lighthouse(alpha=1.0, beta=1.5)
    tonight = shoreline.sample(lighthouse, key=3)
    mo.hstack([lighthouse, shoreline, tonight], justify="start", gap=2, align="start")
    return lighthouse, shoreline, tonight


@app.cell
def _(lighthouse, tonight):
    def draw_night(ax, theta, flashes, xlim=(-8, 10)):
        """The lighthouse, its rays, and where they hit the shore."""
        a, b = float(theta.alpha), float(theta.beta)
        xs = np.asarray(flashes.x)
        for _x in xs:
            ax.plot([a, _x], [b, 0], color=style.YELLOW, alpha=0.35, linewidth=1)
        ax.axhline(0, color=style.INK_2, linewidth=1.5)
        ax.plot(xs, np.zeros_like(xs), "|", color=style.DATA, markersize=14, mew=2,
                label="recorded flashes")
        ax.plot(a, b, marker="^", color=style.INK, markersize=12, linestyle="none",
                label="lighthouse (hidden)")
        ax.set(xlim=xlim, ylim=(-0.4, max(3.0, b + 1)), xlabel=r"position along the shore $x$ [km]",
               ylabel="out to sea [km]")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    _fig, _ax = plt.subplots(figsize=(7.5, 3.0))
    draw_night(_ax, lighthouse, tonight)
    _off = int(np.sum((np.asarray(tonight.x) < -8) | (np.asarray(tonight.x) > 10)))
    _ax.set_title(f"{tonight.x.shape[0]} flashes ({_off} land off the plot)")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Averaging the flashes

    An obvious first estimate of $\alpha$ is the average flash position. For a Cauchy distribution
    this does not work, because the mean of $n$ Cauchy draws is itself Cauchy with the same width,
    however large $n$ is. A single flash at a grazing angle lands kilometres away and drags the mean
    with it. The median does converge, and the likelihood, which we use below, does better still.
    """)
    return


@app.cell
def _(lighthouse):
    _long_night = Shoreline(n_flashes=5000).sample(lighthouse, key=11)
    _x = np.asarray(_long_night.x)
    _n = np.arange(1, len(_x) + 1)
    _running_median = np.array([np.median(_x[:k]) for k in _n])

    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    _ax.plot(_n, np.cumsum(_x) / _n, color=style.PRIOR, label="running mean")
    _ax.plot(_n, _running_median, color=style.POSTERIOR, label="running median")
    style.truth(_ax, lighthouse.alpha, orientation="horizontal", label=r"true $\alpha$")
    _ax.set(xscale="log", ylim=(-4, 6), xlabel="number of flashes", ylabel=r"estimate of $\alpha$",
            title="Running mean and median of the flash positions")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## `Parameters` objects

    An object made by `@parameter_dataclass` is a frozen dataclass, with the usual fields, `==` and
    constructor. It is also a JAX pytree, so `jax.jit`, `jax.grad` and `jax.vmap` accept it and
    return the same type. And its fields have named axes: trailing *core* axes declared with
    `index=`, and any number of leading *batch* axes, which all fields share and which broadcast
    as in NumPy.

    ### Broadcasting

    If one field is an array and the others are scalars, the object holds a batch of lighthouses.
    The fields broadcast against each other, and `.broadcast()` expands every field to the full
    batch shape:
    """)
    return


@app.cell
def _():
    three = Lighthouse(alpha=jnp.array([-2.0, 0.0, 2.0]), beta=1.0)
    mo.md(f"""
    | expression | result |
    |---|---|
    | `three.batch_shape` | `{three.batch_shape}` |
    | `three.beta` | `{three.beta}` (still a scalar) |
    | `three.broadcast().beta` | `{np.asarray(three.broadcast().beta)}` |
    | `three.reshape_batch((3, 1)).batch_shape` | `{three.reshape_batch((3, 1)).batch_shape}` |
    | `Lighthouse.parameters` | `{Lighthouse.parameters}` |
    """)
    return (three,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Indexing

    Sampling 4 nights for each of the 3 lighthouses gives `Flashes` with batch shape `(3, 4)`. New
    batch axes are added in front, and the core axis (30 flashes) stays at the end. You can index
    the batch axes with integers, slices or boolean masks, and pick out a field by name:
    """)
    return


@app.cell
def _(shoreline, three):
    nights = shoreline.sample(three, N=4, key=0)
    _rows = {
        "nights": nights,
        "nights[0]": nights[0],
        "nights[0, 1:]": nights[0, 1:],
        "nights[np.array([True, False, True])]": nights[np.array([True, False, True])],
    }
    mo.md(
        "| expression | batch_shape | `x.shape` |\n|---|---|---|\n"
        + "\n".join(f"| `{k}` | `{v.batch_shape}` | `{v.x.shape}` |" for k, v in _rows.items())
        + f"\n| `nights['x']` | | `{nights['x'].shape}` |"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Batched methods

    `flash_density(x)` is written for a single lighthouse and a grid `x`. On a batch of three
    lighthouses, plain NumPy broadcasting would try to combine shapes `(3,)` and `(400,)` and fail.
    `@batched_method` instead applies `vmap` to the method over every batch axis, so the result has
    shape `(3, 400)`, with one curve per lighthouse:
    """)
    return


@app.cell
def _(three):
    x_grid = jnp.linspace(-8, 8, 400)
    curves = three.flash_density(x_grid)

    _fig, _ax = plt.subplots(figsize=(6.4, 2.8))
    for _curve, _a, _color in zip(curves, three.alpha, style.SERIES):
        _ax.plot(x_grid, _curve, color=_color, label=rf"$\alpha = {float(_a):g}$")
    _ax.set(xlabel=r"$x$ [km]", ylabel="flash density", title=f"three.flash_density(x): shape {curves.shape}")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _fig
    return (x_grid,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Converting to and from dicts

    `from_dict` accepts either a dict or a DataFrame and ignores any extra columns. This is how
    posterior samples are turned into typed objects. `to_dict()` converts back to a dict. The
    objects are immutable, but their fields may hold NumPy arrays, so call `.copy()` before editing
    an array in place.
    """)
    return


@app.cell
def _():
    _table = pd.DataFrame({"alpha": [0.5, 1.0, 1.5], "beta": [1.0, 1.5, 2.0],
                           "keeper": ["Ada", "Bo", "Cy"]})
    from_table = Lighthouse.from_dict(_table)
    mo.vstack([
        mo.md("`Lighthouse.from_dict(table)`, with an extra `keeper` column that is ignored:"),
        _table,
        from_table,
        mo.md(f"`from_table.to_dict()` = `{ {k: np.asarray(v).tolist() for k, v in from_table.to_dict().items()} }`"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Arithmetic and gradients

    Objects of the same type can be added and subtracted field by field, and multiplied by numbers.
    Because they are pytrees, `jax.grad` of a function of a `Lighthouse` returns a `Lighthouse`
    holding $\partial/\partial\alpha$ and $\partial/\partial\beta$. A gradient step is then written
    as it is in the maths, `theta + lr * grad`:
    """)
    return


@app.cell
def _(shoreline, tonight):
    guess = Lighthouse(alpha=0.0, beta=1.0)
    grad = jax.grad(lambda theta: shoreline.logpdf(tonight, theta))(guess)
    stepped = guess + 0.02 * grad
    mo.md(f"""
    | | $\\alpha$ | $\\beta$ |
    |---|---|---|
    | `guess` | {float(guess.alpha):.3f} | {float(guess.beta):.3f} |
    | `grad = jax.grad(log_like)(guess)` (a `{type(grad).__name__}`) | {float(grad.alpha):.3f} | {float(grad.beta):.3f} |
    | `guess + 0.02 * grad` | {float(stepped.alpha):.3f} | {float(stepped.beta):.3f} |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## The likelihood surface

    A 2-D grid of lighthouses is a `Lighthouse` with batch shape `(n_alpha, n_beta)`. To make one,
    give `alpha` shape `(n_alpha, 1)` and `beta` shape `(1, n_beta)` and let them broadcast.

    To compare three different nights, we sample them as a batch of 3 and pass `outer=True`.
    `.logpdf` then evaluates every lighthouse against every night, giving shape
    `(n_alpha, n_beta, 3)`. Without `outer=True` the two batch shapes are broadcast together, which
    is what you want when they line up one-to-one.
    """)
    return


@app.cell
def _(lighthouse, shoreline):
    alpha_grid = jnp.linspace(-3, 5, 161)
    beta_grid = jnp.linspace(0.05, 5, 120)
    grid = Lighthouse(alpha=alpha_grid[:, None], beta=beta_grid[None, :])

    three_nights = shoreline.sample(lighthouse, N=3, key=21)
    log_like_maps = shoreline.logpdf(three_nights, grid, outer=True)

    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True, layout="constrained")
    for _i, _ax in enumerate(_axes):
        _L = np.asarray(log_like_maps[..., _i])
        _ax.pcolormesh(alpha_grid, beta_grid, np.exp(_L - _L.max()).T, cmap="Blues", shading="auto")
        _ax.plot(lighthouse.alpha, lighthouse.beta, "^", mfc="none", mec=style.INK, mew=1.8, markersize=10)
        _ax.set(title=f"night {_i + 1}", xlabel=r"$\alpha$ [km]")
        _ax.grid(False)
    _axes[0].set_ylabel(r"$\beta$ [km]")
    mo.vstack([
        grid,
        mo.md(f"The *Batch* line of `grid` shows its two batch axes, {grid.batch_shape[0]} values "
              f"of $\\alpha$ by {grid.batch_shape[1]} of $\\beta$. `log_like_maps` is a plain "
              f"array of shape `{log_like_maps.shape}`, one likelihood map per night:"),
        _fig,
    ])
    return alpha_grid, beta_grid, grid


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Constrained parameters and the latent space

    Next we find the maximum-likelihood lighthouse for tonight by gradient ascent. A step can push
    $\beta$ below zero, where the likelihood is meaningless, so we optimise in *unconstrained
    coordinates* instead: map each field to the whole real line, take the step there, and map back.

    The `support=` declarations make this possible. `theta.to_unconstrained()` returns a
    `Lighthouse` holding the *latent* point $z$. Here $z_\beta = \log\beta$, and $z_\alpha=\alpha$
    because $\alpha$ has no declared support. `Lighthouse.from_unconstrained(z)` maps any real $z$
    back inside the supports. The latent point has the same type, so arithmetic and `jax.grad`
    work on it as before.
    """)
    return


@app.cell
def _(lighthouse):
    _z = lighthouse.to_unconstrained()
    mo.vstack([
        ai.LatentSpace(Lighthouse),
        mo.md(f"`lighthouse.to_unconstrained()` gives $z_\\alpha$ = {float(_z.alpha):.3f} and "
              f"$z_\\beta$ = {float(_z.beta):.3f} = $\\log 1.5$. Mapping back gives "
              f"$\\beta$ = {float(Lighthouse.from_unconstrained(_z).beta):.3f}."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **What you implement for constrained parameters**

    Only `support=` on the field, set to a
    [numpyro constraint](https://num.pyro.ai/en/stable/distributions.html#constraints) such as
    `constraints.positive`, `constraints.interval(0, 1)` or `constraints.unit_interval`.

    If one task needs different supports, build a space by hand, for example
    `ai.LatentSpace(Lighthouse, {"alpha": dist.Uniform(-10, 10)})`, and pass it to
    `theta.to_unconstrained(space)`. The dict values can be constraints or distributions; for a
    distribution its `.support` is used.

    Optimisation does not need a Jacobian. If you move a density into latent space, for example to
    sample there, add `space.log_abs_det_jacobian(z)`.
    """), kind="info")
    return


@app.cell
def _(shoreline, tonight):
    def log_like_latent(z):
        return shoreline.logpdf(tonight, Lighthouse.from_unconstrained(z))

    ascent_step = jax.jit(jax.grad(log_like_latent))

    z = Lighthouse(alpha=-2.0, beta=4.0).to_unconstrained()
    path = [Lighthouse.from_unconstrained(z)]
    for _ in range(400):
        z = z + 0.01 * ascent_step(z)
        path.append(Lighthouse.from_unconstrained(z))
    mle = path[-1]
    path_alpha = np.array([float(p.alpha) for p in path])
    path_beta = np.array([float(p.beta) for p in path])
    mo.md(f"After 400 steps the maximum-likelihood estimate is $\\hat\\alpha$ = "
          f"{float(mle.alpha):.3f}, $\\hat\\beta$ = {float(mle.beta):.3f}.")
    return mle, path_alpha, path_beta


@app.cell
def _(
    alpha_grid,
    beta_grid,
    grid,
    lighthouse,
    path_alpha,
    path_beta,
    shoreline,
    tonight,
):
    _L = np.asarray(shoreline.logpdf(tonight, grid))
    _fig, _ax = plt.subplots(figsize=(6.4, 3.8))
    _ax.pcolormesh(alpha_grid, beta_grid, np.exp(_L - _L.max()).T, cmap="Blues", shading="auto")
    _ax.plot(path_alpha, path_beta, color=style.PRIOR, linewidth=2, label="gradient ascent in $z$")
    _ax.plot(path_alpha[0], path_beta[0], "o", color=style.PRIOR, markersize=7)
    _ax.plot(path_alpha[-1], path_beta[-1], "*", color=style.PRIOR, markersize=14, label="MLE")
    _ax.plot(lighthouse.alpha, lighthouse.beta, "^", mfc="none", mec=style.INK, mew=1.8, markersize=11,
             label="truth", zorder=5)
    _ax.grid(False)
    _ax.set(xlabel=r"$\alpha$ [km]", ylabel=r"$\beta$ [km]", title="Tonight's likelihood")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## The posterior

    We run `.PE` as in notebook 1, with uniform priors over a 20 km stretch of coast and up to 6 km
    out to sea. NUTS keeps $\beta$ positive through the support of its prior. `from_dict` then
    turns the draws into a batch of lighthouses, which we use for the plots below.
    """)
    return


@app.cell
def _(shoreline, tonight):
    pe_priors = dict(alpha=dist.Uniform(-10.0, 10.0), beta=dist.Uniform(0.0, 6.0))
    posterior = shoreline.PE(tonight, pe_priors, seed=0, num_samples=4000)
    posterior_lighthouses = Lighthouse.from_dict(posterior)
    posterior_lighthouses
    return posterior, posterior_lighthouses


@app.cell
def _(lighthouse, mle, posterior):
    _fig, _axes = style.corner(
        [posterior], ["alpha", "beta"], labels=[r"$\alpha$ [km]", r"$\beta$ [km]"],
        names=["posterior"], truths={"alpha": float(lighthouse.alpha), "beta": float(lighthouse.beta)},
    )
    _axes[1, 0].plot(float(mle.alpha), float(mle.beta), "*", color=style.PRIOR, markersize=14)
    _axes[0, 0].plot([], [], "*", color=style.PRIOR, markersize=12, linestyle="none", label="MLE")
    style.corner_legend(_fig, _axes)
    _fig
    return


@app.cell
def _(posterior_lighthouses, tonight, x_grid):
    # Posterior predictive: one flash-density curve per posterior lighthouse (a batch of 4000).
    _curves = np.asarray(posterior_lighthouses[:400].flash_density(x_grid))
    _lo, _mid, _hi = np.percentile(_curves, [5, 50, 95], axis=0)

    _fig, _ax = plt.subplots(figsize=(6.4, 2.8))
    _ax.fill_between(x_grid, _lo, _hi, color=style.POSTERIOR, alpha=0.25, linewidth=0,
                     label="posterior 90%")
    _ax.plot(x_grid, _mid, color=style.POSTERIOR, label="posterior median")
    _ax.plot(np.asarray(tonight.x), np.zeros(tonight.x.shape), "|", color=style.DATA, markersize=14,
             mew=2, label="tonight's flashes")
    _ax.set(xlim=(-8, 8), xlabel=r"$x$ [km]", ylabel="flash density",
            title="Posterior predictive flash density")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    * Batch axes come first and broadcast across fields; core axes (`index=`) come last and belong
      to a single object. `batch_shape`, `broadcast`, `reshape_batch` and integer, slice or mask
      indexing act on the batch axes only.
    * Objects are pytrees, so they support `+`, `-` and multiplication by a number, and `jax.grad`
      returns the same type.
    * `@batched_method` lets you write a method for one object and call it on a batch.
    * `.logpdf(d, theta, outer=True)` evaluates every $\theta$ against every $d$.
    * `support=` together with `to_unconstrained` and `from_unconstrained` keeps an optimiser inside
      the allowed region.

    Notebook 3 fits the light curve of a star with a transiting planet. It gives the data class a
    `plot` method and checks the fit against posterior predictive light curves.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: a short coastline

    Suppose your detectors only cover the stretch $-L\le x\le L$ with $L = 3$ km, and flashes
    outside it are never recorded. Each recorded flash is then a draw from a truncated Cauchy
    distribution:

    $$
    \mathcal{L}(\hat x_k\mid\alpha,\beta) = \frac{\text{Cauchy}(\hat x_k;\alpha,\beta)}
    {F(L) - F(-L)}, \qquad F(x) = \tfrac12 + \tfrac1\pi\arctan\frac{x-\alpha}{\beta}.
    $$

    1. Write `ShortCoast(MeasurementProcess)` with fields `n_flashes` and `L`. Sample by drawing the
       angle uniformly between $\arctan\frac{-L-\alpha}{\beta}$ and $\arctan\frac{L-\alpha}{\beta}$.
    2. Put the lighthouse at $\alpha = 2.5$, $\beta = 1$ and run PE with both `ShortCoast` and
       `Shoreline` on the same data. Which one finds the lighthouse?
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
    @dataclass
    class ShortCoast(MeasurementProcess):
        n_flashes: int = 30
        L: float = 3.0

        parameter_type = Lighthouse
        data_type = Flashes

        def angle_range(self, theta):
            lo = jnp.arctan((-self.L - theta.alpha) / theta.beta)
            hi = jnp.arctan((self.L - theta.alpha) / theta.beta)
            return lo, hi

        def sample_single(self, theta, key):
            lo, hi = self.angle_range(theta)
            angle = jax.random.uniform(key, (self.n_flashes,), minval=lo, maxval=hi)
            return Flashes(x=theta.alpha + theta.beta * jnp.tan(angle))

        def logpdf_single(self, d, theta):
            lo, hi = self.angle_range(theta)
            log_cauchy = jnp.log(theta.beta / jnp.pi) - jnp.log(theta.beta**2 + (d.x - theta.alpha)**2)
            log_norm = jnp.log((hi - lo) / jnp.pi)          # F(L) - F(-L)
            inside = jnp.abs(d.x) <= self.L
            return jnp.sum(jnp.where(inside, log_cauchy - log_norm, -jnp.inf), axis=-1)

    coast = ShortCoast()
    d = coast.sample(Lighthouse(alpha=2.5, beta=1.0), key=0)
    priors = dict(alpha=dist.Uniform(-10, 10), beta=dist.Uniform(0, 6))
    right = coast.PE(d, priors, seed=0)
    wrong = Shoreline().PE(d, priors, seed=0)
    # `wrong` places the lighthouse closer to the middle of the detectors and nearer the shore.
    ```
    """)})
    return


if __name__ == "__main__":
    app.run()

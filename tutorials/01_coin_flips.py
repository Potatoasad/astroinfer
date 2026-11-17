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
    from scipy import stats

    import astroinfer as ai
    from astroinfer import MeasurementProcess, parameter, parameter_dataclass

    import _style as style

    style.apply()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Is this coin rigged?

    This is the first of ten notebooks on astroinfer. By the end of the series you should be able
    to write a forward model for your own experiment and run both single-event and population
    inference on it. Each notebook works through one problem:

    | # | problem | topics |
    |---|---|---|
    | 1 | is this coin rigged? | parameters, data, and a first `MeasurementProcess` |
    | 2 | the lighthouse | batches, core axes, JAX, constrained parameters |
    | 3 | a transiting planet | methods on a data class, posterior predictive checks |
    | 4 | the doctor's office | `PosteriorData`, derived parameters, custom displays |
    | 5 | the coin mint | population models with `astroinfer.generative` |
    | 6 | auditing the mint | hierarchical inference with `PopulationLikelihood` |
    | 7 | a flux-limited survey | detection triggers and selection effects |
    | 8 | sharper telescopes | importance sampling with a `Proposal` |
    | 9 | the star catalog | `Catalog`, data products, save / load |
    | 10 | your own representation | the `Fit` / `Fitter` protocol |

    ## Parameters, data and the measurement process

    An analysis in astroinfer is built from three objects, all of which you write yourself. The
    *parameters* $\theta$ are what nature knows, the *data* $d$ are what you get to see, and the
    *measurement process* $\mathcal{L}(d\mid\theta)$ connects the two:

    $$
    \theta \;\xrightarrow{\quad \mathcal{L}(d\mid\theta)\quad}\; d
    $$

    The parameters and the data are each a `@parameter_dataclass`. The measurement process is a
    subclass of `MeasurementProcess` and has two jobs: simulate $d$ given $\theta$, and evaluate the
    probability of $d$ given $\theta$. Once you have these, sampling the posterior
    $p(\theta\mid d)\propto \mathcal{L}(d\mid\theta)\,\pi(\theta)$ takes one method call.

    We'll start with the simplest experiment there is, flipping a coin.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src=mo.notebook_dir() / "images" / "01_coin_flips.svg",
             alt="A coin being flipped, and the list of heads and tails it produces",
             width=640)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The parameters

    A coin has one parameter, the probability $p$ that it lands heads. We describe it with a
    `@parameter_dataclass`, which is a frozen dataclass and also a JAX pytree (notebook 2 explains
    why that is useful). Each field is declared with `parameter(...)`. Both of its arguments are
    optional: `name` is a readable description, and `latex` is the symbol used when the object is
    displayed.
    """)
    return


@app.class_definition
@parameter_dataclass(name="Coin", latex=r"\mathrm{coin}")
class Coin:
    p: float = parameter(name="Probability of heads", latex="p")


@app.cell
def _():
    coin = Coin(p=0.7)
    coin
    return (coin,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Clicking the object's name expands or collapses it. Every astroinfer object has a display like
    this one, which is a quick way to check what you are working with.

    ## The data

    One flip gives one boolean, whether the coin landed heads:
    """)
    return


@app.class_definition
@parameter_dataclass(name="Flip", latex=r"\hat{h}")
class Flip:
    heads: bool = parameter(name="Landed heads", latex=r"\hat{h}")


@app.cell
def _():
    Flip(heads=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The measurement process

    For a single flip the likelihood is $\mathcal{L}(\hat h\mid p) = p^{\hat h}(1-p)^{1-\hat h}$. To
    turn it into a measurement process, subclass `MeasurementProcess`, name the parameter and data
    types, and write two methods, one that draws a flip and one that evaluates its log likelihood:
    """)
    return


@app.class_definition
class FlipOnce(MeasurementProcess):
    display_name = "Flip a coin once"
    parameter_type = Coin
    data_type = Flip

    def sample_single(self, theta, key):
        return Flip(heads=jax.random.bernoulli(key, theta.p))

    def logpdf_single(self, d, theta):
        return jnp.where(d.heads, jnp.log(theta.p), jnp.log1p(-theta.p))


@app.cell
def _():
    flipper = FlipOnce()
    flipper
    return (flipper,)


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **What you implement in a `MeasurementProcess`**

    | attribute / method | description |
    |---|---|
    | `parameter_type` | your $\theta$ class |
    | `data_type` | your $d$ class |
    | `sample_single(self, theta, key)` | draw one $d\sim \mathcal{L}(\cdot\mid\theta)$ for a single $\theta$, using the JAX `key` |
    | `logpdf_single(self, d, theta)` | return the scalar $\log \mathcal{L}(d\mid\theta)$ for a single $d$ and $\theta$ |

    Both methods should handle a single $\theta$ and a single observation. The base class
    vectorises them with `jax.vmap` over whatever batch you pass in, handles the random keys, and
    provides `.sample`, `.logpdf` and `.PE`. Write the methods with `jax.numpy` so that they stay
    differentiable.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulating data

    `flipper.sample(theta)` draws one observation for each $\theta$. To get many independent
    observations, pass `N=`. They come back as a single `Flip` object whose fields have an extra
    leading *batch axis*:
    """)
    return


@app.cell
def _(coin, flipper):
    one_flip = flipper.sample(coin)
    thousand_flips = flipper.sample(coin, N=1000)
    mo.vstack([one_flip, thousand_flips])
    return one_flip, thousand_flips


@app.cell(hide_code=True)
def _(coin, thousand_flips):
    mo.md(rf"""
    `thousand_flips` holds all 1000 flips in one `Flip` object. Its display has the title
    $\{{\hat h_n\}}$ and a *Batch* line $n \in 1\ldots 1000$, so `thousand_flips.batch_shape` is
    `(1000,)`. Indexing works as for an array: `thousand_flips[0]` is the first flip and
    `thousand_flips[:10]` the first ten. The fraction of heads is
    {float(thousand_flips.heads.mean()):.3f}, and the true $p$ is {coin.p}.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `.sample` also broadcasts over a batch of $\theta$. With three coins and `N=500` we get a batch
    of shape `(3, 500)`:
    """)
    return


@app.cell
def _(flipper):
    three_coins = Coin(p=jnp.array([0.1, 0.5, 0.9]))
    many = flipper.sample(three_coins, N=500)

    _fig, _ax = plt.subplots(figsize=(6.4, 2.6))
    _running = np.cumsum(np.asarray(many.heads), axis=1) / np.arange(1, 501)
    for _row, _p, _color in zip(_running, three_coins.p, style.SERIES):
        _ax.plot(_row, color=_color, label=f"$p = {float(_p):.1f}$")
        _ax.axhline(float(_p), color=_color, linestyle=":", linewidth=1)
    _ax.set(xlabel="number of flips", ylabel="running fraction of heads", ylim=(0, 1),
            title=f"batch_shape = {many.batch_shape}")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A measurement process keeps its own JAX key in `flipper.key`. Each call to `.sample` splits it,
    so repeated calls give different flips. For a reproducible draw, pass `key=` (a JAX key or an
    integer); this leaves the internal key alone. `flipper.seed(123)` resets the internal key.
    """)
    return


@app.cell
def _(coin, flipper):
    _a = flipper.sample(coin, N=8, key=42)
    _b = flipper.sample(coin, N=8, key=42)
    _c = flipper.sample(coin, N=8)
    mo.md(f"""
    | call | flips |
    |---|---|
    | `sample(coin, N=8, key=42)` | `{np.asarray(_a.heads).astype(int)}` |
    | `sample(coin, N=8, key=42)` | `{np.asarray(_b.heads).astype(int)}` (same as above) |
    | `sample(coin, N=8)` | `{np.asarray(_c.heads).astype(int)}` (different) |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The likelihood

    `flipper.logpdf(d, theta)` evaluates $\log \mathcal{L}(d\mid\theta)$, broadcasting the batch shapes of $d$
    and $\theta$ against each other. Passing one flip and a grid of coins gives the likelihood as a
    function of $p$:
    """)
    return


@app.cell
def _(flipper, one_flip):
    p_grid = jnp.linspace(0.001, 0.999, 300)
    log_like = flipper.logpdf(one_flip, Coin(p=p_grid))      # shape (300,)

    _fig, _ax = plt.subplots(figsize=(6.4, 2.8))
    _ax.plot(p_grid, jnp.exp(log_like), color=style.POSTERIOR)
    _ax.set(xlabel="$p$", ylabel=r"$\mathcal{L}(\hat h \mid p)$",
            title=f"Likelihood after one flip ({'heads' if one_flip.heads else 'tails'})")
    _fig
    return (p_grid,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Parameter estimation

    `.PE` needs a prior for every field of $\theta$, given as a dict of
    [numpyro distributions](https://num.pyro.ai/en/stable/distributions.html). It runs NUTS on
    $\mathcal{L}(d\mid\theta)\,\pi(\theta)$ and returns the posterior draws in a DataFrame, along with the log
    likelihood and the prior density of each draw. Those last two columns come up again in
    notebook 9. With `just_prior=True` it samples the prior instead, which we plot below for
    comparison.
    """)
    return


@app.cell
def _(flipper, one_flip):
    pe_priors = dict(p=dist.Uniform(0.0, 1.0))

    posterior_one = flipper.PE(one_flip, pe_priors, seed=0, num_samples=4000)
    prior_draws = flipper.PE(one_flip, pe_priors, seed=0, num_samples=4000, just_prior=True)
    posterior_one.head()
    return pe_priors, posterior_one, prior_draws


@app.cell
def _(coin, one_flip, p_grid, posterior_one, prior_draws):
    _h = int(one_flip.heads)
    _fig, _ax = plt.subplots(figsize=(6.4, 3.2))
    style.density(_ax, prior_draws.p, color=style.PRIOR, label="prior", range=(0, 1), bins=40)
    style.density(_ax, posterior_one.p, color=style.POSTERIOR, label="posterior (NUTS)",
                  range=(0, 1), bins=40)
    _ax.plot(p_grid, stats.beta(1 + _h, 2 - _h).pdf(p_grid), color=style.INK_2, linewidth=1.2,
             label="exact: Beta")
    style.truth(_ax, coin.p)
    _ax.set(xlabel="$p$", ylabel="density", xlim=(0, 1),
            title=f"Prior and posterior after one flip ({'heads' if _h else 'tails'})")
    _ax.legend(loc="upper left")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Many flips

    A single flip hardly moves the posterior away from the prior, so we now flip the coin $n$ times
    and treat the whole sequence as one observation. The field `heads` becomes a vector. We tell
    astroinfer this with `index=`, which gives the field a *core axis* named `"flip"`
    (`index_latex=` sets the symbol used in displays).

    A core axis is part of a single observation: twenty flips make up one `Flips` object. Batch
    axes, like the `N=1000` above, count how many observations there are. astroinfer keeps the two
    separate so that it knows which axes to vectorise over.
    """)
    return


@app.class_definition
@parameter_dataclass(name="Flips", latex=r"\hat{h}")
class Flips:
    heads: bool = parameter(name="Landed heads", latex=r"\hat{h}",
                            index=("flip",), index_latex=("k",))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The measurement process now has a setting, the number of flips. Measurement processes are
    dataclasses, so a setting is an ordinary field. The flips are independent, so `logpdf_single`
    sums the log likelihood over the core axis,
    $\log \mathcal{L}(\hat h_{1..n}\mid p) = \sum_k \log \mathcal{L}(\hat h_k\mid p)$:
    """)
    return


@app.class_definition
@dataclass
class FlipMany(MeasurementProcess):
    n_flips: int = 20

    display_name = "Flip a coin n times"
    parameter_type = Coin
    data_type = Flips

    def sample_single(self, theta, key):
        return Flips(heads=jax.random.bernoulli(key, theta.p, shape=(self.n_flips,)))

    def logpdf_single(self, d, theta):
        per_flip = jnp.where(d.heads, jnp.log(theta.p), jnp.log1p(-theta.p))
        return jnp.sum(per_flip, axis=-1)


@app.cell
def _():
    _demo = FlipMany(n_flips=20)
    _d = _demo.sample(Coin(p=0.7), N=5)
    mo.vstack([_demo, _d,
               mo.md(f"Five observations of 20 flips: `heads.shape = {_d.heads.shape}`, "
                     f"`batch_shape = {_d.batch_shape}`. The expanded display shows the batch axis "
                     "$n$ and the core axis $k$.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Detecting a biased coin

    Use the sliders to choose a true $p$ and a number of flips. The cell below simulates the
    experiment, runs PE, and plots the result next to the exact posterior, which for a uniform prior
    is $p\mid\hat h \sim \mathrm{Beta}(1+n_\mathrm{heads},\;1+n_\mathrm{tails})$.
    """)
    return


@app.cell
def _():
    true_p = mo.ui.slider(0.05, 0.95, value=0.6, step=0.05, label="true $p$", show_value=True)
    n_flips = mo.ui.slider(1, 400, value=40, step=1, label="number of flips", show_value=True,
                           debounce=True)
    mo.hstack([true_p, n_flips], justify="start", gap=2)
    return n_flips, true_p


@app.cell
def _(n_flips, p_grid, pe_priors, true_p):
    experiment = FlipMany(n_flips=n_flips.value)
    observed = experiment.sample(Coin(p=true_p.value), key=7)
    posterior_many = experiment.PE(observed, pe_priors, seed=1, num_samples=4000)

    _heads = int(observed.heads.sum())
    _exact = stats.beta(1 + _heads, 1 + n_flips.value - _heads)
    _p_heads_biased = float((posterior_many.p > 0.5).mean())

    _fig, _ax = plt.subplots(figsize=(6.4, 3.2))
    style.density(_ax, posterior_many.p, color=style.POSTERIOR, label="posterior (NUTS)",
                  range=(0, 1), bins=60)
    _ax.plot(p_grid, _exact.pdf(p_grid), color=style.INK_2, linewidth=1.2, label="exact: Beta")
    style.truth(_ax, true_p.value)
    _ax.axvline(0.5, color=style.MUTED, linewidth=1, label="fair coin")
    _ax.set(xlabel="$p$", ylabel="density", xlim=(0, 1),
            title=f"{_heads} heads in {n_flips.value} flips")
    _ax.legend(loc="upper left")

    mo.vstack([
        _fig,
        mo.md(f"$P(p > 0.5\\mid \\hat h)$ = {_p_heads_biased:.3f}. "
              "Try a true $p$ of 0.55: how many flips do you need before you can tell?"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    * `@parameter_dataclass` and `parameter(name=, latex=, index=)` describe both $\theta$ and $d$.
      Axes declared with `index=` are core axes; any extra leading axes are batch axes.
    * A `MeasurementProcess` needs `parameter_type`, `data_type`, `sample_single` and
      `logpdf_single`, written for one $\theta$ and one $d$.
    * The base class then provides `.sample(theta, N=, key=)`, `.logpdf(d, theta)` and
      `.PE(d, priors)`, batched over whatever shapes you pass.

    Notebook 2 looks at `@parameter_dataclass` objects in more detail: batches, JAX gradients, and
    parameters that have to stay within a range.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: the blurry camera

    Suppose you film each flip, and the camera misreads the face with probability
    $\varepsilon = 0.1$. A recorded flip is then heads with probability
    $p(1-\varepsilon) + (1-p)\varepsilon$.

    1. Write `BlurryFlips(MeasurementProcess)` with fields `n_flips` and `epsilon`, reusing `Coin`
       and `Flips`.
    2. Simulate 200 flips of a $p = 0.8$ coin and run PE.
    3. Run `FlipMany` on the same data and compare the two posteriors. Which one is biased, and in
       which direction?

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
    @dataclass
    class BlurryFlips(MeasurementProcess):
        n_flips: int = 200
        epsilon: float = 0.1

        parameter_type = Coin
        data_type = Flips

        def recorded_p(self, p):
            return p * (1 - self.epsilon) + (1 - p) * self.epsilon

        def sample_single(self, theta, key):
            q = self.recorded_p(theta.p)
            return Flips(heads=jax.random.bernoulli(key, q, shape=(self.n_flips,)))

        def logpdf_single(self, d, theta):
            q = self.recorded_p(theta.p)
            return jnp.sum(jnp.where(d.heads, jnp.log(q), jnp.log1p(-q)), axis=-1)

    camera = BlurryFlips()
    d = camera.sample(Coin(p=0.8), key=0)
    blurry = camera.PE(d, dict(p=dist.Uniform(0, 1)), seed=0)
    naive = FlipMany(n_flips=200).PE(d, dict(p=dist.Uniform(0, 1)), seed=0)
    # The naive posterior is pulled towards 0.5, because misreads make the coin look fairer
    # than it is.
    ```
    """)})
    return


if __name__ == "__main__":
    app.run()

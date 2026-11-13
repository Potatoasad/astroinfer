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
    # 1 · Is this coin rigged? 🪙

    Welcome to **astroinfer**. In this series you will build up, one problem at a time, everything
    you need to write your own forward model and run single-event and population inference on it.

    | # | problem | you learn |
    |---|---|---|
    | **1** | **is this coin rigged?** | **parameters, data, and your first `MeasurementProcess`** |
    | 2 | the lighthouse | batches, core axes, JAX, constrained parameters |
    | 3 | the doctor's office | `PosteriorData`, derived parameters, custom displays |
    | 4 | the coin mint | population models with `astroinfer.generative` |
    | 5 | auditing the mint | hierarchical inference with `PopulationLikelihood` |
    | 6 | a flux-limited survey | detection triggers and selection effects |
    | 7 | sharper telescopes | importance sampling with a `Proposal` |
    | 8 | the star catalog | `Catalog`, data products, save / load |
    | 9 | your own representation | the `Fit` / `Fitter` protocol |

    ## The whole idea in one picture

    Every analysis in astroinfer is built from three objects you write yourself:

    $$
    \underbrace{\theta}_{\text{what nature knows}}
    \;\xrightarrow{\quad K(d\mid\theta)\quad}\;
    \underbrace{d}_{\text{what you get to see}}
    $$

    * $\theta$, the **parameters**: a `@parameter_dataclass`;
    * $d$, the **data**: another `@parameter_dataclass`;
    * $K(d\mid\theta)$, the **measurement process**: a subclass of `MeasurementProcess` that knows how
      to *simulate* $d$ from $\theta$ and how to *score* $d$ given $\theta$.

    Once those exist, parameter estimation $p(\theta\mid d)\propto K(d\mid\theta)\,\pi(\theta)$ is one
    method call. Let's build all three for the simplest experiment there is.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1 · Describe the parameters $\theta$

    A coin has one parameter: the probability $p$ that it lands heads. A `@parameter_dataclass` is a
    frozen dataclass (and a JAX pytree, more on that in notebook 2). Each field is declared with
    `parameter(...)`, which takes an optional human-readable `name` and a `latex` symbol used when the
    object is displayed.
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
    Click the object above to expand it. Every astroinfer object has a rich display like this, so
    you can always just *look* at what you have.

    ## Step 2 · Describe the data $d$

    One flip gives one boolean: did it land heads?
    """)
    return


@app.class_definition
@parameter_dataclass(name="Flip", latex=r"\hat{h}")
class Flip:
    heads: bool = parameter(name="Landed heads", latex=r"\hat{h}")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3 · Write the measurement process $K(d\mid\theta)$

    This is the heart of every model. For a single flip,
    $K(\hat h\mid p) = p^{\hat h}(1-p)^{1-\hat h}$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **🧩 What you implement in a `MeasurementProcess`**

    | attribute / method | what it is |
    |---|---|
    | `parameter_type` | your $\theta$ class |
    | `data_type` | your $d$ class |
    | `sample_single(self, theta, key)` | draw **one** $d\sim K(\cdot\mid\theta)$ for **one** $\theta$, using the JAX `key` |
    | `logpdf_single(self, d, theta)` | return the scalar $\log K(d\mid\theta)$ for **one** $d$ and **one** $\theta$ |

    Write them for a *single* $\theta$ and a *single* observation. The base class vectorises them
    (with `jax.vmap`) over any batch you hand it, manages random keys, and adds `.sample`, `.logpdf`
    and `.PE`. Use `jax.numpy` inside them so everything stays differentiable.
    """), kind="info")
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
    mo.md(r"""
    ## Simulate: `.sample`

    `flipper.sample(theta)` draws one observation per $\theta$. Pass `N=` for many independent
    observations; they come back as **one** `Flip` object whose fields carry a leading *batch axis*.
    """)
    return


@app.cell
def _(coin, flipper):
    one_flip = flipper.sample(coin)
    thousand_flips = flipper.sample(coin, N=1000)

    mo.vstack([
        one_flip,
        thousand_flips,
        mo.md(f"`thousand_flips.batch_shape = {thousand_flips.batch_shape}`, and the fraction of "
              f"heads is **{float(thousand_flips.heads.mean()):.3f}** (true $p$ = {coin.p})."),
    ])
    return (one_flip,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `.sample` broadcasts over batches of $\theta$ too. Three coins, 500 flips each, gives a batch of
    shape `(3, 500)`, with no loops in sight:
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
    mo.callout(mo.md(r"""
    **🎲 Randomness, without the bookkeeping.** Every measurement process carries a JAX key
    (`flipper.key`). Calling `.sample(theta)` splits it, so repeated calls give fresh noise. Pass
    `key=` (a JAX key or an `int`) when you want a *reproducible* draw; that leaves the internal key
    untouched. `flipper.seed(123)` resets it.
    """), kind="neutral")
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
    | `sample(coin, N=8, key=42)` | `{np.asarray(_b.heads).astype(int)}`  (identical) |
    | `sample(coin, N=8)` | `{np.asarray(_c.heads).astype(int)}`  (fresh) |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Score: `.logpdf`

    `flipper.logpdf(d, theta)` evaluates $\log K(d\mid\theta)$, broadcasting the batch shapes of $d$
    and $\theta$ against each other. Hand it a *grid* of coins and you get the whole likelihood
    curve in one call:
    """)
    return


@app.cell
def _(flipper, one_flip):
    p_grid = jnp.linspace(0.001, 0.999, 300)
    log_like = flipper.logpdf(one_flip, Coin(p=p_grid))      # shape (300,)

    _fig, _ax = plt.subplots(figsize=(6.4, 2.8))
    _ax.plot(p_grid, jnp.exp(log_like), color=style.POSTERIOR)
    _ax.set(xlabel="$p$", ylabel=r"$K(\hat h \mid p)$",
            title=f"Likelihood after one flip ({'heads' if one_flip.heads else 'tails'})")
    _fig
    return (p_grid,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Infer: `.PE`

    Parameter estimation needs a prior for every field of $\theta$, given as a dict of
    [numpyro distributions](https://num.pyro.ai/en/stable/distributions.html). `.PE` runs NUTS on
    $K(d\mid\theta)\,\pi(\theta)$ and returns the draws as a DataFrame, together with the log
    likelihood and the prior density of every draw (you will see why those matter in notebook 8).

    Passing `just_prior=True` samples the prior alone, which is handy for comparison.
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
            title=f"One flip ({'heads' if _h else 'tails'}) barely moves the prior")
    _ax.legend(loc="upper left")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Many flips: *core axes*

    One flip does not tell us much. Let's flip the coin $n$ times and treat the whole sequence as
    **one** observation. The field `heads` is now a vector, and we say so with `index=`: it has a
    *core axis* called `"flip"`. (Use `index_latex=` for the symbol shown in displays.)

    Core axes are part of what a *single* observation is. Batch axes, like the `N=1000` above, count
    *how many* observations there are. Keeping the two apart is what lets astroinfer batch anything.
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
    dataclasses, so settings are just fields. Inside `logpdf_single` the flips are independent, so
    the log likelihood **sums over the core axis**:
    $\log K(\hat h_{1..n}\mid p) = \sum_k \log K(\hat h_k\mid p)$.
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
                     f"`batch_shape = {_d.batch_shape}`. Expand the object to see the batch axis "
                     "$n$ and the core axis $k$.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is it rigged? Play with it 🎛️

    Choose a true $p$ and a number of flips. The cell below simulates the experiment, runs PE, and
    compares it with the exact answer: with a uniform prior,
    $p\mid\hat h \sim \mathrm{Beta}(1+n_\mathrm{heads},\;1+n_\mathrm{tails})$.
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
        mo.md(f"$P(p > 0.5\\mid \\hat h)$ = **{_p_heads_biased:.3f}**. "
              "Try a true $p$ of 0.55: how many flips do you need before you can tell?"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Recap

    * `@parameter_dataclass` + `parameter(name=, latex=, index=)` describe $\theta$ and $d$.
      `index=` declares **core axes** (part of one object); leading axes are **batch axes**.
    * A `MeasurementProcess` needs `parameter_type`, `data_type`, `sample_single`, `logpdf_single`,
      all written for **one** $\theta$ and **one** $d$.
    * You get `.sample(theta, N=, key=)`, `.logpdf(d, theta)` and `.PE(d, priors)` for free, batched
      over whatever shapes you pass.

    Next, in **notebook 2**, we look under the hood of `@parameter_dataclass`: batches, JAX
    gradients, and parameters that must stay inside a range.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ✏️ Your turn: the blurry camera

    You film each flip, but the camera misreads the face with probability $\varepsilon = 0.1$. Each
    *recorded* flip is heads with probability $p(1-\varepsilon) + (1-p)\varepsilon$.

    1. Write `BlurryFlips(MeasurementProcess)` with fields `n_flips` and `epsilon`. Reuse `Coin` and
       `Flips`.
    2. Simulate 200 flips of a $p = 0.8$ coin and run PE.
    3. Compare with `FlipMany`'s posterior on the *same* data. Which one is biased, and which way?

    Write it in the empty cell below.
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
    # naive is pulled towards 0.5: misreads make the coin look fairer than it is.
    ```
    """)})
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    import time

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from astroinfer import parameter, parameter_dataclass
    from astroinfer.generative import (
        Beta,
        Distribution,
        GenerativeModel,
        TruncatedNormal,
        Uniform,
        conditional,
        mixture,
        product,
    )

    import _style as style

    style.apply()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. The coin mint

    So far every coin had one true bias $p$, and we inferred it. Now consider a *mint* that stamps
    out thousands of coins. Imperfections in the press give each coin a slightly different bias, and
    the biases follow a *population distribution*

    $$
    p \sim \pi(p\mid\Lambda),
    $$

    where the *hyperparameters* $\Lambda$ describe the mint, for example its typical bias, how
    sloppy it is, and how many of its machines are worn out.

    This notebook shows how to write down $\pi(\theta\mid\Lambda)$ with `astroinfer.generative`.
    Notebook 5 then infers $\Lambda$ from flipped coins.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The parameters are the `Coin` class from notebook 1, with one field `p`. We only need the class
    here; the flips come back in notebook 5.
    """)
    return


@app.class_definition
@parameter_dataclass(name="Coin", latex=r"\mathrm{coin}")
class Coin:
    p: float = parameter(name="Probability of heads", latex="p")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Population models

    You could write a population with numpyro or scipy, but hierarchical inference (notebook 5)
    needs two things from a population model. It has to draw samples
    $\theta\sim\pi(\theta\mid\Lambda)$ that are a smooth, differentiable function of $\Lambda$, so
    that the sampler can follow gradients, and it has to evaluate $\log\pi(\theta\mid\Lambda)$ at
    given values. `astroinfer.generative` does both. Each variable is drawn by pushing a fixed
    uniform number $u\in(0,1)$ through the inverse CDF, $\theta = F^{-1}(u;\Lambda)$, so if $u$ is
    held fixed and $\Lambda$ changes a little, $\theta$ changes a little too.

    ## Leaves and models

    A *leaf* is a one-dimensional distribution for one named variable. Its parameters can be numbers
    or the names of hyperparameters, given as strings and looked up in $\Lambda$ at call time:
    """)
    return


@app.cell
def _():
    mint = GenerativeModel(
        TruncatedNormal("p", mu="mu", sigma="sigma", a=0.0, b=1.0),   # p ~ TN(mu, sigma) on [0, 1]
        order=["p"],                                                   # the order variables are drawn in
    )
    Lambda = dict(mu=0.6, sigma=0.12)

    draws = mint.sample(Lambda, key=jax.random.key(0), shape=(20_000,))
    coins = Coin.from_dict(draws)          # a dict of arrays -> a batch of 20,000 coins
    coins
    return Lambda, coins, mint


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `.sample(Lambda, key=, shape=)` returns a plain `{variable: array}` dict, which `Coin.from_dict`
    accepts. `.log_prob(values, Lambda)` evaluates the log density, and on a grid of $p$ it gives
    the density curve:
    """)
    return


@app.cell
def _(Lambda, coins, mint):
    p_grid = jnp.linspace(0.0, 1.0, 400)
    _density = jnp.exp(mint.log_prob({"p": p_grid}, Lambda))

    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    style.density(_ax, coins.p, color=style.PRIOR, label="20,000 stamped coins", range=(0, 1), bins=60)
    _ax.plot(p_grid, _density, color=style.INK_2, linewidth=1.2, label=r"$\pi(p\mid\Lambda)$ from .log_prob")
    _ax.set(xlabel="$p$", ylabel="density", xlim=(0, 1),
            title=rf"The mint at $\mu$ = {Lambda['mu']}, $\sigma$ = {Lambda['sigma']}")
    _ax.legend(loc="upper left")
    _fig
    return (p_grid,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The built-in leaves are `Uniform(var, a, b)`, `TruncatedNormal(var, mu, sigma, a, b)` and
    `Beta(var, alpha, beta)`. Each distribution parameter can be given in one of three ways:

    | parameter | example | meaning |
    |---|---|---|
    | a number | `sigma=0.02` | fixed |
    | a string | `mu="mu"` | the hyperparameter `Lambda["mu"]` |
    | `conditional(fn, on=)` | see below | computed from variables drawn earlier |

    `Uniform` and `TruncatedNormal` have a closed-form quantile, so a draw is one formula. `Beta`
    does not, so astroinfer inverts its CDF numerically by bisection, which is still differentiable
    but noticeably slower:
    """)
    return


@app.cell
def _():
    def _timed(model, params):
        f = jax.jit(lambda L: model.sample(L, key=jax.random.key(0), shape=(100_000,)))
        jax.block_until_ready(f(params))                    # compile once
        t0 = time.perf_counter()
        for _ in range(10):
            jax.block_until_ready(f(params))
        return (time.perf_counter() - t0) / 10 * 1e3

    _tn = _timed(GenerativeModel(TruncatedNormal("p", "mu", "sigma", 0.0, 1.0), order=["p"]),
                 dict(mu=0.6, sigma=0.12))
    _beta = _timed(GenerativeModel(Beta("p", "alpha", "beta"), order=["p"]),
                   dict(alpha=9.0, beta=6.0))
    mo.md(f"""
    | leaf | 100,000 draws |
    |---|---|
    | `TruncatedNormal` (closed-form quantile) | {_tn:.2f} ms |
    | `Beta` (bisection) | {_beta:.2f} ms ({_beta / _tn:.0f}x slower) |

    Prefer leaves with a closed-form quantile in code that runs many times, such as a population
    likelihood.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mixtures

    Suppose the mint has two presses. The fair press stamps near-perfect coins with $p\approx 0.5$,
    and the worn press stamps biased ones. A fraction $f_\mathrm{fair}$ of all coins come from the
    fair press:

    $$
    \pi(p\mid\Lambda) = f_\mathrm{fair}\,\mathcal{TN}(p;\,0.5,\,0.02)
                      + (1-f_\mathrm{fair})\,\mathcal{TN}(p;\,\mu_\mathrm{worn},\,\sigma_\mathrm{worn}).
    $$

    `mixture(models, weights)` takes the components, which must all describe the same variables,
    and the names of $K-1$ weight hyperparameters. The last weight is one minus the others.
    """)
    return


@app.cell
def _():
    fair_press = TruncatedNormal("p", mu=0.5, sigma=0.02, a=0.0, b=1.0)
    worn_press = TruncatedNormal("p", mu="mu_worn", sigma="sigma_worn", a=0.0, b=1.0)

    two_press_mint = GenerativeModel(mixture([fair_press, worn_press], weights=["f_fair"]),
                                     order=["p"])
    Lambda_two = dict(f_fair=0.6, mu_worn=0.75, sigma_worn=0.08)
    return Lambda_two, two_press_mint


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Common random numbers

    The histogram below is redrawn from the same random key each time you move a slider, so the
    coins don't jump around between settings and each coin moves smoothly as $\Lambda$ changes. The
    sampler is compiled with `jax.jit` once, and the sliders only change its inputs.
    """)
    return


@app.cell
def _():
    knobs = mo.ui.dictionary({
        "f_fair": mo.ui.slider(0.0, 1.0, value=0.6, step=0.02, label=r"$f_\mathrm{fair}$", show_value=True),
        "mu_worn": mo.ui.slider(0.05, 0.95, value=0.75, step=0.01, label=r"$\mu_\mathrm{worn}$", show_value=True),
        "sigma_worn": mo.ui.slider(0.02, 0.4, value=0.08, step=0.01, label=r"$\sigma_\mathrm{worn}$", show_value=True),
    })
    mo.hstack([knobs["f_fair"], knobs["mu_worn"], knobs["sigma_worn"]], justify="start", gap=2)
    return (knobs,)


@app.cell
def _(two_press_mint):
    stamp = jax.jit(lambda L: two_press_mint.sample(L, key=jax.random.key(1), shape=(20_000,)))
    return (stamp,)


@app.cell
def _(knobs, p_grid, stamp, two_press_mint):
    _L = {k: float(v) for k, v in knobs.value.items()}
    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    style.density(_ax, stamp(_L)["p"], color=style.PRIOR, label="stamped coins", range=(0, 1), bins=80)
    _ax.plot(p_grid, jnp.exp(two_press_mint.log_prob({"p": p_grid}, _L)), color=style.INK_2,
             linewidth=1.2, label=r"$\pi(p\mid\Lambda)$")
    _ax.set(xlabel="$p$", ylabel="density", xlim=(0, 1), ylim=(0, None), title="The two-press mint")
    _ax.legend(loc="upper left")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each draw is $\theta = F^{-1}(u;\Lambda)$ with $u$ fixed. Below, 20 coins (20 fixed values of
    $u$) are stamped again as $\mu_\mathrm{worn}$ goes from 0.3 to 0.9. The left panel uses the same
    `key` every time (common random numbers), and every coin traces a smooth curve. The right panel
    uses a new key for each setting, and the values scatter randomly. Only the left one has a useful
    derivative with respect to $\mu_\mathrm{worn}$.
    """)
    return


@app.cell
def _(Lambda_two, two_press_mint):
    _mus = jnp.linspace(0.3, 0.9, 80)
    _same = jax.vmap(lambda m: two_press_mint.sample({**Lambda_two, "mu_worn": m},
                                                     key=jax.random.key(3), shape=(20,))["p"])(_mus)
    _fresh = jax.vmap(lambda m, k: two_press_mint.sample({**Lambda_two, "mu_worn": m},
                                                         key=k, shape=(20,))["p"])(
        _mus, jax.random.split(jax.random.key(4), len(_mus)))

    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True, layout="constrained")
    for _ax, _vals, _title in [(_axes[0], _same, "same key at every setting"),
                               (_axes[1], _fresh, "new key at every setting")]:
        _ax.plot(_mus, _vals, color=style.PRIOR, linewidth=0.9, alpha=0.6)
        _ax.set(xlabel=r"$\mu_\mathrm{worn}$", title=_title)
    _axes[0].set_ylabel("$p$ of each coin")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can also pass the uniforms yourself with `seeds={variable: u_array}`, which is what the
    likelihood in notebook 5 does. Because the draws are differentiable, you can take `jax.grad`
    with respect to $\Lambda$ of any quantity computed from samples. The table shows the gradient of
    the mint's average bias with respect to each hyperparameter, next to a finite-difference
    estimate:
    """)
    return


@app.cell
def _(Lambda_two, two_press_mint):
    seeds = {"p": jax.random.uniform(jax.random.key(5), (50_000,))}

    def mean_bias(L):
        return two_press_mint.sample(L, seeds=seeds)["p"].mean()

    _L = {k: jnp.asarray(v) for k, v in Lambda_two.items()}
    _grad = jax.grad(mean_bias)(_L)
    _eps = 1e-5
    _fd = {k: (mean_bias({**_L, k: _L[k] + _eps}) - mean_bias({**_L, k: _L[k] - _eps})) / (2 * _eps)
           for k in _L}
    mo.md("| hyperparameter | `jax.grad` | finite difference |\n|---|---|---|\n" + "\n".join(
        f"| `{k}` | {float(_grad[k]):+.5f} | {float(_fd[k]):+.5f} |" for k in _L))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The mixture has no closed-form quantile, so its draws come from bisection. The gradient is still
    exact, because astroinfer does not differentiate through the bisection loop. It differentiates
    the root of $F(\theta;\Lambda)=u$ with the implicit function theorem,
    $\partial\theta/\partial\Lambda = -(\partial F/\partial\Lambda)/\pi(\theta)$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Several variables

    Coins have more than one property. Suppose the mint also varies the thickness $t$ of each coin,
    $t\sim U(1.5, 2.5)$ mm, and thicker coins are more biased:

    $$
    p\mid t \sim \mathcal{TN}\big(0.5 + k\,(t - 2),\ \sigma\big)_{[0,1]}.
    $$

    Three pieces of `astroinfer.generative` are needed for this. `product([...])` combines leaves
    for different variables. `conditional(fn, on=)` makes a distribution parameter a function
    `fn(x, L)` of earlier draws `x` and hyperparameters `L`, where `on=` names the variables `fn`
    reads. `order=` sets the order in which variables are drawn, and a conditional may only read
    variables that come before it.
    """)
    return


@app.cell
def _():
    p_given_thickness = TruncatedNormal(
        "p",
        mu=conditional(lambda x, L: 0.5 + L["k"] * (x["thickness"] - 2.0), on="thickness"),
        sigma="sigma",
        a=0.0, b=1.0,
    )
    thick_mint = GenerativeModel(
        product([Uniform("thickness", 1.5, 2.5), p_given_thickness]),
        order=["thickness", "p"],
    )
    Lambda_thick = dict(k=0.3, sigma=0.04)
    thick_coins = thick_mint.sample(Lambda_thick, key=jax.random.key(6), shape=(3000,))

    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    _ax.scatter(thick_coins["thickness"], thick_coins["p"], s=4, alpha=0.4, color=style.PRIOR,
                linewidths=0, rasterized=True)
    _ax.set(xlabel="thickness $t$ [mm]", ylabel="$p$", title="Bias against thickness")
    _fig
    return Lambda_thick, p_given_thickness, thick_mint


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `GenerativeModel` checks the order when you build it, so a wrong order raises an error straight
    away instead of somewhere inside a sampler. Here is what happens if we draw `p` first:
    """)
    return


@app.cell
def _(thick_mint):
    try:
        GenerativeModel(thick_mint.root, order=["p", "thickness"])
        _msg = "no error?!"
    except ValueError as _err:
        _msg = str(_err)
    mo.callout(mo.md(f"`ValueError`: {_msg}"), kind="danger")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### External variables

    Sometimes a variable comes from somewhere else, because it was measured or another model drew
    it. Declare it in `external=` and pass its values at call time with `conditioned_on=`. Here we
    stamp one coin at each of three thicknesses that we choose:
    """)
    return


@app.cell
def _(Lambda_thick, p_given_thickness):
    p_only = GenerativeModel(p_given_thickness, order=["p"], external=["thickness"])
    _t = jnp.array([1.5, 2.0, 2.5])
    _out = p_only.sample(Lambda_thick, key=jax.random.key(7), shape=(3,),
                         conditioned_on={"thickness": _t})
    mo.md("| thickness | stamped $p$ |\n|---|---|\n" + "\n".join(
        f"| {float(a):.1f} mm | {float(b):.3f} |" for a, b in zip(_t, _out["p"])))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Writing your own leaf

    If none of the built-in leaves fit, subclass `Distribution`. As an example we write a power
    law, $\pi(x)\propto x^{\gamma}$ on $[x_\mathrm{lo}, x_\mathrm{hi}]$. Notebook 6 uses this class
    for stars spread uniformly in volume around us, $\pi(r)\propto r^2$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **What you implement in a `Distribution`**

    Every method receives `var` (which variable), `x` (the dict of variables drawn so far) and `L`
    (the hyperparameters), so any leaf can be conditional.

    | method | returns |
    |---|---|
    | `var_names` (attribute) | the variables this leaf describes, e.g. `["distance"]` |
    | `cond_pdf(var, val, x, L)` | the density at `val` (0 outside the support) |
    | `cond_cdf(var, val, x, L)` | the CDF at `val` |
    | `bounds(var)` | the support `(a, b)` (used by bisection and `log_prob`) |
    | `cond_ppf(var, u, x, L)` | optional: the closed-form quantile. Return `None` (the default) to fall back to bisection |
    | `deps(var)` | optional: the names of earlier variables you read, used to check the order |

    Write the methods with `jax.numpy` so that they are differentiable.
    """), kind="info")
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To see what `cond_ppf` gains you, make a copy without it. Both versions give the same draws from
    the same seeds, because bisection finds the same quantile, but the closed form saves 40
    bisection steps per draw:
    """)
    return


@app.cell
def _():
    class PowerLawNoQuantile(PowerLaw):
        def cond_ppf(self, var, u, x, L):
            return None                         # -> astroinfer inverts cond_cdf by bisection

    volume = GenerativeModel(PowerLaw("distance", 2.0, 0.5, 2.0), order=["distance"])
    volume_slow = GenerativeModel(PowerLawNoQuantile("distance", 2.0, 0.5, 2.0), order=["distance"])

    _fast = volume.sample({}, key=jax.random.key(8), shape=(50_000,))["distance"]
    _slow = volume_slow.sample({}, key=jax.random.key(8), shape=(50_000,))["distance"]

    _r = jnp.linspace(0.4, 2.1, 400)
    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    style.density(_ax, _fast, color=style.PRIOR, label="draws", range=(0.5, 2.0), bins=45)
    _ax.plot(_r, jnp.exp(volume.log_prob({"distance": _r}, {})), color=style.INK_2, linewidth=1.2,
             label=r"$\pi(r)\propto r^2$")
    _ax.set(xlabel="distance $r$ [kpc]", ylabel="density", title="Stars uniform in volume")
    _ax.legend(loc="upper left")
    mo.vstack([_fig, mo.md(
        f"Largest difference between closed-form and bisection draws: "
        f"{float(jnp.abs(_fast - _slow).max()):.1e} kpc.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    * A population model $\pi(\theta\mid\Lambda)$ is a `GenerativeModel(root, order, external=)`.
      The root is built from leaves (`Uniform`, `TruncatedNormal`, `Beta`, or your own
      `Distribution`) combined with `product` for different variables and `mixture` for the same
      variables.
    * A distribution parameter is a number, a hyperparameter name, or `conditional(fn, on=)` of
      earlier draws. The order is checked when the model is built.
    * `.sample(Lambda, key=, shape=)`, `.sample(Lambda, seeds=)` and `.log_prob(values, Lambda)`.
      With a fixed key or fixed seeds, the draws are differentiable in $\Lambda$.

    In notebook 5 we flip coins from the mint and infer $\Lambda$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: the third press

    The mint buys a third press that is badly broken and stamps coins with $p\sim U(0.9, 1.0)$.

    1. Build a three-component mixture (fair, worn, broken). How many weight names do you need?
    2. Sample 20,000 coins at $f_\mathrm{fair}=0.5$, $f_\mathrm{worn}=0.3$ and plot them.
    3. Use `jax.grad` to find how the fraction of coins with $p>0.85$ changes with $f_\mathrm{fair}$.
       (A hard cut has zero gradient, so use a smooth step such as
       `jax.nn.sigmoid((p - 0.85) / 0.01)`.)
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
    broken_press = Uniform("p", 0.9, 1.0)
    three_press_mint = GenerativeModel(
        mixture([fair_press, worn_press, broken_press], weights=["f_fair", "f_worn"]),  # K - 1 = 2
        order=["p"],
    )
    L3 = dict(f_fair=0.5, f_worn=0.3, mu_worn=0.75, sigma_worn=0.08)
    p = three_press_mint.sample(L3, key=jax.random.key(0), shape=(20_000,))["p"]

    fig, ax = plt.subplots()
    style.density(ax, p, color=style.PRIOR, range=(0, 1), bins=80)

    u = {"p": jax.random.uniform(jax.random.key(1), (20_000,))}
    def frac_very_biased(L):
        p = three_press_mint.sample(L, seeds=u)["p"]
        return jax.nn.sigmoid((p - 0.85) / 0.01).mean()

    jax.grad(frac_very_biased)({k: jnp.asarray(v) for k, v in L3.items()})["f_fair"]
    # About -0.2: more fair coins means fewer very biased ones.
    ```
    """)})
    return


if __name__ == "__main__":
    app.run()

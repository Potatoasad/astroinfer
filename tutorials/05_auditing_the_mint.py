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
    from jax.scipy.stats import truncnorm

    import astroinfer as ai
    from astroinfer import MeasurementProcess, parameter, parameter_dataclass
    from astroinfer.generative import GenerativeModel, TruncatedNormal

    import _style as style

    style.apply()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 5 · Auditing the mint 🔍

    A regulator buys $E = 40$ coins from the mint of notebook 4 and flips each one 30 times. From
    those flips alone, what can they say about the **mint**, i.e. its hyperparameters $\Lambda$?

    This is **hierarchical inference**: every coin has its own unknown $p_i$, drawn from
    $\pi(p\mid\Lambda)$, and we want $\Lambda$ without ever pinning down any single $p_i$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Recap from notebooks 1 and 4.** Each coin is a `Coin`, each experiment a `Flips` vector, and
    `FlipMany` is the measurement process. The mint stamps
    $p\sim\mathcal{TN}(\mu,\sigma)_{[0,1]}$.
    """)
    return


@app.class_definition
@parameter_dataclass(name="Coin", latex=r"\mathrm{coin}")
class Coin:
    p: float = parameter(name="Probability of heads", latex="p")


@app.class_definition
@parameter_dataclass(name="Flips", latex=r"\hat{h}")
class Flips:
    heads: bool = parameter(name="Landed heads", latex=r"\hat{h}",
                            index=("flip",), index_latex=("k",))


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
    mint = GenerativeModel(TruncatedNormal("p", mu="mu", sigma="sigma", a=0.0, b=1.0), order=["p"])
    Lambda_true = dict(mu=0.6, sigma=0.12)
    E, N_FLIPS = 40, 30
    return E, Lambda_true, N_FLIPS, mint


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulate the audit

    Two lines, both from earlier notebooks: stamp $E$ coins from the mint, then flip each one. The
    data are a single `Flips` object with batch shape `(E,)`, one row per coin.
    """)
    return


@app.cell
def _(E, Lambda_true, N_FLIPS, mint):
    true_coins = Coin.from_dict(mint.sample(Lambda_true, key=jax.random.key(0), shape=(E,)))
    flipper = FlipMany(n_flips=N_FLIPS)
    data = flipper.sample(true_coins, key=1)
    data
    return data, flipper, true_coins


@app.cell
def _(N_FLIPS, data, true_coins):
    _frac = np.asarray(data.heads.mean(axis=-1))
    _order = np.argsort(np.asarray(true_coins.p))
    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    _ax.scatter(np.arange(len(_order)), _frac[_order], s=30, color=style.DATA,
                label=f"observed fraction of heads ({N_FLIPS} flips)", zorder=3)
    _ax.scatter(np.arange(len(_order)), np.asarray(true_coins.p)[_order], s=30, marker="_",
                color=style.TRUTH, linewidths=2, label="true $p_i$", zorder=4)
    _ax.set(xlabel="coin (sorted by true $p$)", ylabel="$p$", ylim=(0, 1),
            title="Each coin on its own is only loosely measured")
    _ax.legend(loc="upper left")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The population likelihood

    The coins are independent, so the likelihood of $\Lambda$ is a product over coins of each coin's
    **evidence** under the population, with its own $p_i$ integrated out:

    $$
    \mathcal{L}(\Lambda) = \prod_{i=1}^{E} Z_i(\Lambda),\qquad
    Z_i(\Lambda) = \int K(d_i\mid\theta)\,\pi(\theta\mid\Lambda)\,d\theta .
    $$

    astroinfer estimates each integral by Monte Carlo, drawing $N$ coins from the population:

    $$
    \hat Z_i(\Lambda) = \frac{1}{N}\sum_{j=1}^{N} K(d_i\mid\theta_{ij}),\qquad
    \theta_{ij} = F^{-1}(u_{ij};\Lambda).
    $$

    The uniforms $u_{ij}$ are drawn **once** and kept fixed, which is the common-random-numbers trick
    from notebook 4. So $\hat Z_i(\Lambda)$ is a smooth function of $\Lambda$ and NUTS can follow its
    gradient.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **🧩 What you implement: nothing new!** `ai.PopulationLikelihood` is built entirely from pieces
    you already wrote:

    | argument | what it is |
    |---|---|
    | `measurement` | your `MeasurementProcess`; its `logpdf` gives $K(d_i\mid\theta_{ij})$ |
    | `data` | all observations, one `data_type` object with batch shape `(E,)` |
    | `pop_model` | a `GenerativeModel` that draws **every** field of `parameter_type` |
    | `N`, `seed` | Monte Carlo draws per event, and the seed of the fixed uniforms |

    The variable names in the population model must match the field names of your parameters.
    """), kind="info")
    return


@app.cell
def _(Lambda_true, data, flipper, mint):
    likelihood = ai.PopulationLikelihood(flipper, data, mint, N=1000, seed=0)
    L_true = {k: jnp.asarray(v) for k, v in Lambda_true.items()}

    log_K = likelihood.log_kernel(L_true)                  # log K(d_i | theta_ij): shape (E, N)
    log_Z, n_eff = likelihood.event_logpdf_and_neff(L_true)  # per coin: shape (E,), (E,)
    mo.md(f"`log_kernel` has shape `{log_K.shape}`: one row per coin, one column per draw. "
          f"`event_logpdf_and_neff` gives $\\log\\hat Z_i$ and an effective sample size per coin.")
    return L_true, likelihood, n_eff


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Is $N$ big enough? Check the effective sample size

    Each draw is weighted by $w_{ij} = K(d_i\mid\theta_{ij})$. If a few draws carry all the weight,
    the estimate is noisy. The effective sample size
    $N_\mathrm{eff} = (\sum_j w_{ij})^2 / \sum_j w_{ij}^2$ counts how many draws really contribute.
    Coins whose data sit far from the bulk of the population get the fewest useful draws:
    """)
    return


@app.cell
def _(Lambda_true, data, likelihood, n_eff):
    _frac = np.asarray(data.heads.mean(axis=-1))
    _fig, _ax = plt.subplots(figsize=(6.4, 3.0))
    _ax.scatter(_frac, np.asarray(n_eff), s=30, color=style.POSTERIOR, zorder=3)
    _ax.axvline(Lambda_true["mu"], color=style.MUTED, linewidth=1, label=r"population mean $\mu$")
    _ax.axhline(likelihood.N, color=style.INK_2, linestyle=":", linewidth=1, label=f"N = {likelihood.N}")
    _ax.set(xlabel="observed fraction of heads", ylabel=r"$N_\mathrm{eff}$", yscale="log",
            title=r"$N_\mathrm{eff}$ per coin at the true $\Lambda$")
    _ax.legend(loc="lower center")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Gradients and diagnostics

    `likelihood.logpdf(Lambda)` is the scalar $\log\hat{\mathcal L}$. It is differentiable, so
    `jax.value_and_grad` works on it. A quick finite-difference check is a good habit before
    sampling. Each call also leaves a `diagnostics` dict behind, which the sampler records for every
    draw.
    """)
    return


@app.cell
def _(L_true, likelihood):
    _value, _grad = jax.value_and_grad(likelihood.logpdf)(L_true)
    _eps = 1e-6
    _fd = {k: (likelihood.logpdf({**L_true, k: L_true[k] + _eps})
               - likelihood.logpdf({**L_true, k: L_true[k] - _eps})) / (2 * _eps) for k in L_true}
    likelihood.logpdf(L_true)                     # a plain call refreshes .diagnostics
    _diag = {k: float(v) for k, v in likelihood.diagnostics.items()}
    _symbol = {"mu": r"\mu", "sigma": r"\sigma"}
    mo.md(
        f"$\\log\\hat{{\\mathcal L}}(\\Lambda_\\mathrm{{true}})$ = **{float(_value):.3f}**\n\n"
        "| | `jax.grad` | finite difference |\n|---|---|---|\n"
        + "\n".join(f"| $\\partial/\\partial{_symbol[k]}$ | "
                    f"{float(_grad[k]):+.4f} | {float(_fd[k]):+.4f} |" for k in L_true)
        + "\n\n`likelihood.diagnostics` = `"
        + ", ".join(f"{k}: {v:.3g}" for k, v in _diag.items()) + "`"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## An exact answer to check against

    This toy problem is small enough to do *exactly*: each $Z_i$ is a 1-D integral over $p$, so we
    can compute it by brute-force quadrature on a fine grid, for every $(\mu,\sigma)$ on a grid. With
    flat hyperpriors this gives the exact hyperposterior, which is our ground truth.
    """)
    return


@app.cell
def _(N_FLIPS, data):
    def exact_log_likelihood(mus, sigmas, n_grid=2000):
        """log L(mu, sigma) on a grid by quadrature over p; shape (len(sigmas), len(mus))."""
        heads = data.heads.sum(axis=-1)[:, None]                          # (E, 1)
        p = jnp.linspace(1e-4, 1 - 1e-4, n_grid)
        log_K = heads * jnp.log(p) + (N_FLIPS - heads) * jnp.log1p(-p)   # (E, P)
        shift = log_K.max(axis=1, keepdims=True)
        MU, SG = jnp.meshgrid(mus, sigmas)
        a, b = (0.0 - MU.ravel()) / SG.ravel(), (1.0 - MU.ravel()) / SG.ravel()
        prior = truncnorm.pdf(p[:, None], a, b, loc=MU.ravel(), scale=SG.ravel())  # (P, G)
        Z = jnp.exp(log_K - shift) @ prior * (p[1] - p[0])                # (E, G)
        return (jnp.log(Z) + shift).sum(axis=0).reshape(MU.shape)

    mu_grid = jnp.linspace(0.4, 0.8, 160)
    sigma_grid = jnp.linspace(0.01, 0.35, 160)
    exact_grid = exact_log_likelihood(mu_grid, sigma_grid)
    return exact_grid, exact_log_likelihood, mu_grid, sigma_grid


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How big should $N$ be? 🤔

    Below are slices of $\log\hat{\mathcal L}$ for three values of $N$, against the exact curve.
    Every curve is **smooth**, thanks to the fixed uniforms. But smooth is not the same as *right*.
    With $N=10$ the curve is both shifted down and pulled sideways: $\log\hat Z$ is biased low when
    $N_\mathrm{eff}$ is small, and the bias depends on $\Lambda$. The sampler would happily converge
    to the wrong place.
    """)
    return


@app.cell
def _(Lambda_true, data, exact_log_likelihood, flipper, mint, mu_grid, sigma_grid):
    _sig0, _mu0 = jnp.asarray([Lambda_true["sigma"]]), jnp.asarray([Lambda_true["mu"]])
    _exact_mu = exact_log_likelihood(mu_grid, _sig0)[0]
    _exact_sig = exact_log_likelihood(_mu0, sigma_grid)[:, 0]

    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 3.3), sharey=True, layout="constrained")
    for _N, _color in zip([10, 100, 1000], [style.EXTRA, style.PRIOR, style.POSTERIOR]):
        _lik = ai.PopulationLikelihood(flipper, data, mint, N=_N, seed=0)
        _f = jax.jit(jax.vmap(lambda m, s: _lik.logpdf(dict(mu=m, sigma=s))))
        _axes[0].plot(mu_grid, _f(mu_grid, jnp.full_like(mu_grid, Lambda_true["sigma"])),
                      color=_color, label=f"N = {_N}")
        _axes[1].plot(sigma_grid, _f(jnp.full_like(sigma_grid, Lambda_true["mu"]), sigma_grid),
                      color=_color)
    _axes[0].plot(mu_grid, _exact_mu, color=style.INK, linestyle="--", linewidth=1.4, label="exact")
    _axes[1].plot(sigma_grid, _exact_sig, color=style.INK, linestyle="--", linewidth=1.4)
    _top = float(_exact_mu.max())
    _axes[0].set(xlabel=r"$\mu$  (at true $\sigma$)", ylabel=r"$\log\hat{\mathcal{L}}$", ylim=(_top - 25, _top + 2))
    _axes[1].set(xlabel=r"$\sigma$  (at true $\mu$)")
    _axes[0].legend(loc="lower center")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sample the hyperposterior

    Give the hyperparameters priors (numpyro distributions again) and call
    `measurement.population_inference`. It builds the `PopulationLikelihood` for you and runs NUTS.
    It returns the gravpop sampler; `.samples` is a DataFrame of $\Lambda$ draws, plus the
    diagnostics recorded at every draw.
    """)
    return


@app.cell
def _(data, flipper, mint):
    hyper_priors = dict(mu=dist.Uniform(0.0, 1.0), sigma=dist.Uniform(0.01, 0.5))

    sampler = flipper.population_inference(data, mint, hyper_priors, N=1000, seed=0,
                                           num_warmup=300, num_samples=1500)
    hyperposterior = sampler.samples
    hyperposterior.describe().T[["mean", "std", "min", "max"]].round(3)
    return hyper_priors, hyperposterior


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `population_inference` is a convenience wrapper. The explicit version is
    `ai.run_sampler(ai.PopulationLikelihood(...), hyper_priors, ...)`, which is the one to use once
    you start subclassing the likelihood. Here it is, used to see what an $N=10$ analysis would have
    told us:
    """)
    return


@app.cell
def _(data, flipper, hyper_priors, mint):
    _too_few = ai.PopulationLikelihood(flipper, data, mint, N=10, seed=0)
    hyperposterior_n10 = ai.run_sampler(_too_few, hyper_priors, seed=0, num_warmup=300,
                                        num_samples=1500, summary=False).samples
    return (hyperposterior_n10,)


@app.cell
def _(Lambda_true, exact_grid, hyperposterior, hyperposterior_n10, mu_grid, sigma_grid):
    def _hpd_levels(log_l, fractions=(0.9, 0.5)):
        w = np.exp(np.asarray(log_l) - np.max(log_l)).ravel()
        s = np.sort(w)[::-1]
        c = np.cumsum(s) / s.sum()
        return [s[np.searchsorted(c, f)] for f in fractions], w.reshape(np.shape(log_l))

    _fig, _axes = style.corner(
        [hyperposterior, hyperposterior_n10], ["mu", "sigma"],
        labels=[r"$\mu$", r"$\sigma$"], names=["N = 1000", "N = 10"],
        colors=[style.POSTERIOR, style.EXTRA], truths=Lambda_true,
        ranges={"mu": (0.45, 0.75), "sigma": (0.0, 0.25)}, size=2.8)
    _levels, _w = _hpd_levels(exact_grid)
    _axes[1, 0].contour(np.asarray(mu_grid), np.asarray(sigma_grid), _w, levels=_levels,
                        colors=[style.INK_2], linewidths=[1.0, 1.6])
    _axes[0, 0].plot([], [], color=style.INK_2, label="exact (50%, 90%)")
    style.corner_legend(_fig, _axes)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The $N=1000$ hyperposterior (blue) lands on the exact contours. The $N=10$ one is confidently
    wrong: too narrow a population, because too few draws ever land near the outlying coins.

    **Always look at the recorded diagnostics.** If the smallest per-coin $N_\mathrm{eff}$ along
    the chain gets close to 1, increase $N$ or, better, sample smarter (notebook 7):
    """)
    return


@app.cell
def _(hyperposterior, hyperposterior_n10):
    mo.md("| run | median `N_eff_min` | lowest `N_eff_min` |\n|---|---|---|\n"
          + "\n".join(f"| {name} | {h['N_eff_min'].median():.1f} | {h['N_eff_min'].min():.1f} |"
                      for name, h in [("N = 1000", hyperposterior), ("N = 10", hyperposterior_n10)]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Recap

    * The population likelihood is $\prod_i Z_i(\Lambda)$ with
      $Z_i=\int K(d_i\mid\theta)\,\pi(\theta\mid\Lambda)\,d\theta$, estimated by Monte Carlo with
      **fixed** uniforms, so it is smooth and differentiable in $\Lambda$.
    * `ai.PopulationLikelihood(measurement, data, pop_model, N=, seed=)` needs only your measurement
      process, a batch of data with shape `(E,)`, and a `GenerativeModel`.
    * Check it before sampling: `event_logpdf_and_neff` (is $N$ enough for every event?),
      `jax.value_and_grad(likelihood.logpdf)`, and `diagnostics`.
    * Sample with `measurement.population_inference(...)` or
      `ai.run_sampler(likelihood, priors, ...)`.

    Two more knobs, for later: `conditioned_on={...}` passes fixed values to a population with
    `external` variables, and `likelihood_cls=` swaps in your own subclass of `PopulationLikelihood`.
    In **notebook 6** we need exactly that: a survey that only *sees* bright stars.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ✏️ Your turn: which press made these?

    Use the two-press mint from notebook 4: a fair press ($p\sim\mathcal{TN}(0.5, 0.02)$) and a
    worn press ($\mathcal{TN}(\mu_\mathrm{worn}, 0.08)$, with $\sigma_\mathrm{worn}$ held fixed this
    time), with a fraction $f_\mathrm{fair}$ from the fair press.

    1. Simulate 60 coins at $f_\mathrm{fair}=0.5$, $\mu_\mathrm{worn}=0.75$, and flip each 50 times.
    2. Infer $(f_\mathrm{fair}, \mu_\mathrm{worn})$ and make a corner plot with the truth.
    3. Check `N_eff_min` along the chain. Is $N=500$ enough?
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
    from astroinfer.generative import mixture

    two_press = GenerativeModel(
        mixture([TruncatedNormal("p", 0.5, 0.02, 0.0, 1.0),
                 TruncatedNormal("p", "mu_worn", 0.08, 0.0, 1.0)], weights=["f_fair"]),
        order=["p"],
    )
    L_two = dict(f_fair=0.5, mu_worn=0.75)
    coins = Coin.from_dict(two_press.sample(L_two, key=jax.random.key(0), shape=(60,)))
    flipper50 = FlipMany(n_flips=50)
    d = flipper50.sample(coins, key=1)

    post = flipper50.population_inference(
        d, two_press,
        dict(f_fair=dist.Uniform(0, 1), mu_worn=dist.Uniform(0.5, 1.0)),
        N=500, seed=0, num_warmup=300, num_samples=1000,
    ).samples
    fig, _ = style.corner([post], ["f_fair", "mu_worn"], truths=L_two)
    post["N_eff_min"].describe()
    ```
    """)})
    return


if __name__ == "__main__":
    app.run()

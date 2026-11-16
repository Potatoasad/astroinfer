---
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

We want to know whether a coin is biased. The parameter $\theta$ is the probability $p$ that
the coin lands heads, the data $d$ are the outcomes of 50 flips, and the measurement process
$\mathcal{L}(d\mid\theta)$ says how likely those outcomes are for a given $p$. Each of the three is a
class you write. To run the code yourself, install astroinfer first, as described in
[Installation](installation.md).

To begin, import the libraries used on this page:

```{code-cell} python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from scipy import stats

from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)
```

The parameters and the data are each a *parameter type*, a dataclass made with
`@parameter_dataclass` whose fields are declared with `parameter`. The coin has one field. The
50 flips of one experiment are stored in a single field with a *core axis* named `"flip"`,
declared with `index=`, and written $i$ in displays:

```{code-cell} python
@parameter_dataclass(name="Coin", latex=r"\theta")
class Coin:
    p: float = parameter(name="Probability of heads", latex="p")

@parameter_dataclass(name="Flips", latex="d")
class Flips:
    heads: bool = parameter(name="Landed heads", index=("flip",), index_latex=("i",))

coin = Coin(p=0.7)
coin
```

The measurement process is a subclass of `MeasurementProcess`. It names the two parameter types
and defines two methods, both written for a single $\theta$ and a single $d$: `sample_single`
draws one experiment with a JAX random key, and `logpdf_single` returns
$\log \mathcal{L}(d\mid\theta) = \sum_k \log p^{h_k}(1-p)^{1-h_k}$. The number of flips is an ordinary
dataclass field.

```{code-cell} python
@dataclass
class FlipCoin(MeasurementProcess):
    n_flips: int = 50

    parameter_type = Coin
    data_type = Flips

    def sample_single(self, theta, key):
        heads = jax.random.bernoulli(key, theta.p, (self.n_flips,))
        return Flips(heads=heads)

    def logpdf_single(self, d, theta):
        p = theta.p
        return jnp.where(d.heads, jnp.log(p), jnp.log1p(-p)).sum()

experiment = FlipCoin(n_flips=50)
experiment
```

`sample` runs the experiment. With `N=1000` it runs it 1000 times. The integer `key` makes the
draw reproducible.

```{code-cell} python
d = experiment.sample(coin, N=1000, key=0)
d
```

The result is a single `Flips` object that holds all 1000 experiments, a *batch* of data. Its
display shows this in three places. The title $\{d_n\}$ marks a collection indexed by $n$, the
*Batch* line gives $n \in 1\ldots 1000$, and the field `heads` carries the indices $n$ and $i$,
with shape `(1000, 50)`. The *Dimensions* line lists the core axis `flip` of
length 50, which belongs to each experiment and is not part of the batch: `d.batch_shape` is
`(1000,)`.

A batch works like an array of experiments. Indexing selects from it and returns another
`Flips` object, so `d[:10]` is the first ten experiments and `d[0]` is a single one, with no
*Batch* line:

```{code-cell} python
d[0]
```

`sample` and `logpdf` accept a batch anywhere they accept a single object, and apply the methods
you wrote to every element. The fraction of heads in each experiment scatters around the true
value of 0.7:

```{code-cell} python
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.hist(d.heads.mean(axis=-1), bins=jnp.linspace(0.01, 0.99, 50), color="#418FC5")
ax.axvline(coin.p, color="#17243D", linestyle="--")
ax.set(xlabel="fraction of heads in 50 flips", ylabel="experiments");
```

`logpdf` evaluates $\log \mathcal{L}(d\mid\theta)$, broadcasting the batch shapes of $d$ and $\theta$
against each other. Passing the first experiment and a grid of 200 coins gives the likelihood
as a function of $p$:

```{code-cell} python
p_grid = jnp.linspace(0.01, 0.99, 200)
log_like = experiment.logpdf(d[0], Coin(p=p_grid))
log_like.shape
```

`PE` samples the posterior $p(\theta\mid d)\propto \mathcal{L}(d\mid\theta)\,\pi(\theta)$ with NUTS. It
takes a prior for every field as a numpyro distribution, and other keyword arguments go to the
sampler:

```{code-cell} python
priors = {"p": dist.Uniform(0, 1)}
posterior = experiment.PE(d[0], priors, seed=0, num_samples=4000, summary=False)
posterior.head()
```

The result is a pandas DataFrame with one row per draw, holding the parameters, the log
likelihood and the prior density. For a uniform prior the exact posterior is
$\mathrm{Beta}(1 + n_\mathrm{heads},\, 1 + n_\mathrm{tails})$, which the NUTS draws match:

```{code-cell} python
n_heads = int(d[0].heads.sum())
exact = stats.beta(1 + n_heads, 1 + 50 - n_heads)

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.hist(posterior["p"], bins=50, density=True, color="#418FC5", alpha=0.6, label="NUTS draws")
ax.plot(p_grid, exact.pdf(p_grid), color="#17243D", label="exact Beta posterior")
ax.axvline(coin.p, color="#17243D", linestyle="--", label="true $p$")
ax.set(xlabel="$p$", ylabel="density", title=f"{n_heads} heads in 50 flips")
ax.legend();
```

With `as_posterior=True`, `PE` returns a `PosteriorData` instead, which keeps the draws as a
batch of `Coin` objects together with the observed data. It is described in
[Measurement processes](user-guide/measurement-processes.md#parameter-estimation).

## Next steps

The [quickstart](quickstart.md) uses a model with two parameters to go on from here to
populations of objects and catalogs of posteriors, and [Installation](installation.md) shows how
to install astroinfer and its extensions. The [examples](examples.md) work
through nine problems in more depth, starting with this coin, and
[Why astroinfer?](why-astroinfer.md) describes what the library takes care of.

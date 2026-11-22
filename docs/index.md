---
title: astroinfer
kernelspec:
  name: python3
  display_name: Python 3
site:
  hide_outline: true
  hide_title_block: true
---

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)

from pathlib import Path


def run_snippet(name):
    """Run snippets/<name>.py here; the diagrams in each tab show its pieces."""
    path = Path("snippets") / f"{name}.py"
    exec(compile(path.read_text(), str(path), "exec"), globals())
```

```{image} assets/astroinfer-logo.svg
:alt: astroinfer
:width: 300px
:align: left
:class: logo-light
```

```{image} assets/astroinfer-logo-dark.svg
:alt: astroinfer
:width: 300px
:align: left
:class: logo-dark
```

**astroinfer** is a JAX-based framework for astrophysical simulation and Bayesian inference.
Define a physical system and its measurement process once, then use the same model to simulate
observations, estimate individual parameters, and model and infer population distributions.

{button}`Get started <getting-started.md>` {button}`Examples <examples.md>` {button}`Installation <installation.md>`

:::::::{tab-set}

::::::{tab-item} Coin flips

```{code-cell} python
:tags: [remove-cell]

run_snippet("coin_flip")
```

Imagine an experiment in which you flip a coin 50 times and write down the whole list of heads
and tails as your data. Here is how you would implement it:

:::::{div}
:class: rail

::::{div}
:class: rail-gut

:::{div}
:class: dag-node dag-theta

$\theta$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Parameters $\theta$
:::

A coin has a probability $p$ of landing heads.

```{literalinclude} snippets/coin_flip.py
:start-after: --- theta
:end-before: --- data
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-process

$\mathcal{L}(d\mid\theta)$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Measurement process $\mathcal{L}(d\mid\theta)$
:::

Flipping the coin 50 times gives a list of heads and tails. `sample_single` simulates the 50
flips, and `logpdf_single` returns the log probability of a given list for a coin with
probability $p$.

```{literalinclude} snippets/coin_flip.py
:start-after: --- measurement
:end-before: --- end
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-node

$d$
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Data $d$
:::

The data are the 50 flips, `True` for heads. `index` makes `heads` an array along an axis named
`flip`.

```{literalinclude} snippets/coin_flip.py
:start-after: --- data
:end-before: --- measurement
:language: python
```

::::

:::::

Every astroinfer object has a display in a notebook. The experiment shows its methods and its
settings:

```{code-cell} python
experiment = FlipCoin(n_flips=50)
experiment
```

A coin with $p = 0.7$ shows its one parameter, with the name given in `Coin`:

```{code-cell} python
coin = Coin(p=0.7)
coin
```

We ask the measurement process to run the 50-flip experiment 1000 times for this coin, and get
back a batch of data:

```{code-cell} python
d = experiment.sample(coin, N=1000, key=0)
d
```

`d` is one `Flips` object that holds all 1000 experiments. Its `heads` field is a 1000 by 50
array of `True` and `False`, with one row per experiment and one column per flip.

The likelihood works on the whole batch, and returns one value of $\log\mathcal{L}(d\mid\theta)$
for each of the 1000 experiments:

```{code-cell} python
log_like = experiment.logpdf(d, coin)
log_like.shape
```

`sample_single` and `logpdf_single` were written for one experiment. astroinfer applies them to
batches of any shape with `jax.vmap` and manages the random keys.

Parameter estimation samples the posterior of $p$ with NUTS. Here we give it the first
experiment, `d[0]`, and a uniform prior on $p$:

```{code-cell} python
priors = {"p": dist.Uniform(0, 1)}
posterior = experiment.PE(d[0], priors, seed=0)
posterior["p"].mean()
```

The posterior is much narrower than the prior after 50 flips:

```{code-cell} python
p_grid = jnp.linspace(0.001, 0.999, 200)
prior_density = jnp.exp(priors["p"].log_prob(p_grid))

fig, ax = plt.subplots(figsize=(6, 2.8))
ax.hist(posterior["p"], bins=40, density=True, color="#418FC5",
        label="posterior")
ax.plot(p_grid, prior_density, color="#E0A348", lw=2, label="prior")
ax.axvline(coin.p, color="#17243D", ls="--", label="true $p$")
ax.set(xlabel="probability of heads $p$", ylabel="density",
       title="Prior and posterior of $p$ from the first experiment")
ax.legend();
```

:::{dropdown} Imports for this example
```{literalinclude} snippets/coin_flip.py
:start-after: --- imports
:end-before: --- theta
:language: python
```
:::

::::::

::::::{tab-item} Transit

```{code-cell} python
:tags: [remove-cell]

run_snippet("transit")
```

A planet that passes in front of its star blocks some of the star's light, once per orbit. A
telescope records the brightness of the star for 20 days, and from that light curve we want the
orbital period of the planet, the time of its first transit and its radius. We assume a star of
uniform brightness, a fixed transit duration and white noise:

:::::{div}
:class: rail

::::{div}
:class: rail-gut

:::{div}
:class: dag-node dag-theta

$\theta$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Parameters $\theta$
:::

A planet has an orbital period $P$, a time of first transit $T_0$ and a radius $R_p$, in units
of the radius of the star.

```{literalinclude} snippets/transit.py
:start-after: --- theta
:end-before: --- data
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-process

$\mathcal{L}(d\mid\theta)$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Measurement process $\mathcal{L}(d\mid\theta)$
:::

During a transit the planet blocks a fraction $R_p^2$ of the light. `light_curve` writes this as
a box with smoothed edges, so that its gradient is defined everywhere. The telescope adds white
Gaussian noise with standard deviation `sigma` to every exposure.

```{literalinclude} snippets/transit.py
:start-after: --- measurement
:end-before: --- end
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-node

$d$
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Data $d$
:::

The times, the measured fluxes and their errors, all along an axis named `time`, written $i$.

```{literalinclude} snippets/transit.py
:start-after: --- data
:end-before: --- measurement
:language: python
```

::::

:::::

We simulate the light curve of a planet with a period of 3 days, a first transit at 1.2 days
and a radius of 0.1 stellar radii. The `plot` method of `LightCurve` draws it with error
bars:

```{code-cell} python
telescope = Photometry(days=20.0, cadence=1 / 48, duration=0.2,
                       sigma=0.002)
truth = Planet(period=3.0, t0=1.2, radius=0.1)
lightcurve = telescope.sample(truth, key=0)

fig, ax = plt.subplots(figsize=(7, 3))
lightcurve.plot(ax, noise=True, color="#17243D")
ax.set_title("Simulated light curve");
```

The seven transits, each $R_p^2 = 0.01$ deep, stand out above the noise. We run parameter estimation with
uniform priors that span a range of periods, transit times and radii:

```{code-cell} python
priors = {"period": dist.Uniform(2, 4), "t0": dist.Uniform(0, 2),
          "radius": dist.Uniform(0, 0.3)}
posterior = telescope.PE(lightcurve, priors, seed=0, as_posterior=True)
```

`posterior.theta` holds the samples as a batch of `Planet` objects, indexed by chain and draw;
`stack_samples()` merges those two axes into one. The radius and the period against each other,
and the time of first transit, with the true values:

```{code-cell} python
samples = posterior.stack_samples().theta
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
ax1.scatter(samples.period, samples.radius, s=2, alpha=0.2, color="#418FC5")
ax1.plot(truth.period, truth.radius, "x", ms=10, mew=2, color="#E0A348")
ax1.set(xlabel="orbital period $P$ [days]",
        ylabel="planet radius $R_p$ [stellar radii]",
        title="Posterior samples of $P$ and $R_p$")
ax2.hist(samples.t0, bins=50, color="#418FC5")
ax2.axvline(truth.t0, color="#E0A348", lw=2)
ax2.set(xlabel="time of first transit $T_0$ [days]", ylabel="samples",
        title="Posterior of $T_0$");
```

For the posterior predictive, `posterior_predictive` draws noiseless light curves from 100
posterior samples, picked at random, with a copy of the telescope that has `sigma=0`. With
`noise=False`, `plot` draws the batch of 100 light curves as lines over the data:

```{code-cell} python
noiseless = Photometry(days=20.0, cadence=1 / 48, duration=0.2,
                       sigma=0.0)
predictive = noiseless.posterior_predictive(posterior, draws=100, key=1)

fig, ax = plt.subplots(figsize=(7, 3))
lightcurve.plot(ax, noise=True, color="#17243D", alpha=0.3)
predictive.plot(ax, noise=False, color="#418FC5", alpha=0.05)
ax.set_title("Posterior predictive light curves and the data");
```

:::{dropdown} Imports for this example
```{literalinclude} snippets/transit.py
:start-after: --- imports
:end-before: --- theta
:language: python
```
:::

::::::

::::::{tab-item} Star population

```{code-cell} python
:tags: [remove-cell]

run_snippet("star_population")
```

A telescope measures the flux and the parallax of 200 stars. Each star has a luminosity $L$ and
a distance $r$, and the luminosities follow a population distribution with mean $\mu_L$ and
width $\sigma_L$ in $\log_{10}L$. We want those two hyperparameters, $\Lambda=(\mu_L,\sigma_L)$,
without knowing the luminosity or distance of any single star. The model has a population level
above $\theta$:

:::::{div}
:class: rail

::::{div}
:class: rail-gut

:::{div}
:class: dag-node dag-theta

$\Lambda$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Hyperparameters $\Lambda$
:::

The mean $\mu_L$ and width $\sigma_L$ of the distribution of $\log_{10}L$.

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-process

$\pi(\theta\mid\Lambda)$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Population model $\pi(\theta\mid\Lambda)$
:::

$\log_{10}L$ follows a normal distribution truncated to $[-3, 5]$, and the distance is uniform
between 0.5 and 2 kpc. The variable names match the fields of `Star`, and the hyperparameter
names are the keys of $\Lambda$.

```{literalinclude} snippets/star_population.py
:start-after: --- population
:end-before: --- theta
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-node

$\theta$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Parameters $\theta$
:::

Each star has a log luminosity and a distance.

```{literalinclude} snippets/star_population.py
:start-after: --- theta
:end-before: --- data
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-process

$\mathcal{L}(d\mid\theta)$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Measurement process $\mathcal{L}(d\mid\theta)$
:::

The flux falls off as $1/r^2$, so $\log_{10}F = \log_{10}L - 2\log_{10}r$, and the parallax in
mas is $1/r$ for $r$ in kpc. The telescope measures both with Gaussian noise.

```{literalinclude} snippets/star_population.py
:start-after: --- measurement
:end-before: --- end
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-node

$d$
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Data $d$
:::

The measured log flux and parallax of one star.

```{literalinclude} snippets/star_population.py
:start-after: --- data
:end-before: --- measurement
:language: python
```

::::

:::::

We draw 200 stars from the population with $\mu_L = 0$ and $\sigma_L = 0.7$ and observe each
one. `population.sample` returns a dictionary of arrays, and `Star.from_dict` turns it into a
batch of 200 `Star` objects:

```{code-cell} python
Lambda_true = {"mu_L": 0.0, "sigma_L": 0.7}
draws = population.sample(Lambda_true, key=jax.random.key(0), shape=(200,))
stars = Star.from_dict(draws)

telescope = Telescope(sigma_log_flux=0.2, sigma_parallax=0.3)
data = telescope.sample(stars, key=1)
data
```

Nearby stars have large parallaxes, and the measured flux falls with distance, with a spread
that comes from the spread of luminosities:

```{code-cell} python
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.scatter(data.parallax, data.log_flux, s=8, color="#418FC5")
ax.set(xlabel=r"measured parallax $\hat\varpi$ [mas]",
       ylabel=r"measured $\log_{10}\hat F$",
       title="Measured parallax and log flux of 200 stars");
```

`population_inference` samples $\Lambda$ with NUTS. For each star, the likelihood of $\Lambda$
integrates over the unknown $\theta$ with 2000 Monte Carlo draws from the population:

```{code-cell} python
hyper_priors = {"mu_L": dist.Uniform(-2, 2), "sigma_L": dist.Uniform(0.1, 2)}
posterior = telescope.population_inference(data, population, hyper_priors,
                                           N=2000, seed=0)
posterior.summary()
```

The posterior gives $\mu_L = 0.05 \pm 0.06$ and $\sigma_L = 0.74 \pm 0.04$, within one standard
deviation of the true values of 0 and 0.7.

On the left are the posterior samples of $\Lambda$. On the right, each line is the population
distribution $\pi(\log_{10}L\mid\Lambda)$ for one of 100 posterior samples, drawn over a
histogram of the true luminosities of the 200 stars, which the analysis never saw:

```{code-cell} python
samples = posterior.stack_samples().to_dataframe()
hyper = samples.sample(100, random_state=0)
mu, sigma = hyper["mu_L"].to_numpy(), hyper["sigma_L"].to_numpy()
grid = jnp.linspace(-3, 3, 200)
pdfs = jnp.exp(dist.TruncatedNormal(mu[:, None], sigma[:, None], low=-3.0,
                                    high=5.0).log_prob(grid))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
ax1.scatter(samples["mu_L"], samples["sigma_L"], s=2,
            alpha=0.2, color="#418FC5")
ax1.plot(Lambda_true["mu_L"], Lambda_true["sigma_L"], "x", ms=10, mew=2,
         color="#E0A348")
ax1.set(xlabel=r"$\mu_L$", ylabel=r"$\sigma_L$",
        title=r"Posterior samples of $\Lambda$")
ax2.hist(stars.log_L, bins=25, density=True, color="#E8EDF3",
         edgecolor="#8592a6", label=r"true $\log_{10}L$")
ax2.plot(grid, pdfs.T, color="#418FC5", alpha=0.1)
ax2.set(xlabel=r"$\log_{10}L$", ylabel="density",
        title="Inferred population and true luminosities")
ax2.legend();
```

:::{dropdown} Imports for this example
```{literalinclude} snippets/star_population.py
:start-after: --- imports
:end-before: --- population
:language: python
```
:::

::::::

::::::{tab-item} Selection effects

```{code-cell} python
:tags: [remove-cell]

run_snippet("star_population")
run_snippet("star_selection")
```

The stars of the Star population tab are now seen by a survey that catalogs a star only when its
measured flux is above a limit. Faint stars far away never reach the catalog, so the catalog is
brighter than the population it came from. To infer $\Lambda=(\mu_L,\sigma_L)$ from it, the
model needs one more step after the data, the probability that a measurement is detected:

:::::{div}
:class: rail

::::{div}
:class: rail-gut

:::{div}
:class: dag-node dag-theta

$\Lambda$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Hyperparameters $\Lambda$
:::

The mean $\mu_L$ and width $\sigma_L$ of the distribution of $\log_{10}L$.

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-process

$\pi(\theta\mid\Lambda)$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Population model $\pi(\theta\mid\Lambda)$
:::

$\log_{10}L$ follows a normal distribution truncated to $[-3, 5]$, and the distance is uniform
between 0.5 and 2 kpc. This is the `population` of the Star population tab.

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-node

$\theta$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Parameters $\theta$
:::

Each star has a log luminosity and a distance, held in a `Star`.

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-process

$\mathcal{L}(d\mid\theta)$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Measurement process $\mathcal{L}(d\mid\theta)$
:::

The `Telescope` measures the log flux $\log_{10}L - 2\log_{10}r$ and the parallax $1/r$ with
Gaussian noise.

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-node

$d$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Data $d$
:::

The measured log flux and parallax of one star, held in a `StarObservation`.

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-process

$P(\mathrm{det}\mid d)$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Detection trigger $P(\mathrm{det}\mid d)$
:::

The survey catalogs a star when its measured log flux is above $-0.3$. The trigger reads the
data, not $\theta$, so noise can push a star over the limit or under it. `ThresholdTrigger` is
the built-in hard cut on one field of the data.

```{literalinclude} snippets/star_selection.py
:start-after: --- trigger
:end-before: --- likelihood
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-node

$\{d_i\}$
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Catalog $\{d_i\}$
:::

The measurements of the $E$ detected stars. The others are never recorded.

::::

:::::

We draw 1000 stars from the population with $\mu_L = 0$ and $\sigma_L = 0.7$, observe each one
and ask the trigger which are detected. `trigger.detect` returns a boolean mask, which indexes
the batch of observations:

```{code-cell} python
Lambda_true = {"mu_L": 0.0, "sigma_L": 0.7}
draws = population.sample(Lambda_true, key=jax.random.key(0), shape=(1000,))
stars = Star.from_dict(draws)

telescope = Telescope(sigma_log_flux=0.2, sigma_parallax=0.3)
observed = telescope.sample(stars, key=1)
detected = trigger.detect(observed, np.random.default_rng(2))
data = observed[detected]
data
```

The catalog holds 588 of the 1000 stars. Below the line, where the true flux is under the limit,
the survey detects only the few stars that noise pushed over it:

```{code-cell} python
r_grid = jnp.linspace(0.5, 2.0, 100)

fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
ax.scatter(stars.distance[~detected], stars.log_L[~detected], s=6,
           color="#C9D2DE", label="missed")
ax.scatter(stars.distance[detected], stars.log_L[detected], s=6,
           color="#418FC5", label="detected")
ax.plot(r_grid, -0.3 + 2 * jnp.log10(r_grid), color="#17243D",
        label="flux limit")
ax.set(xlabel="distance $r$ [kpc]", ylabel=r"$\log_{10}L$",
       title="True luminosity and distance of 1000 stars")
ax.legend(loc="lower right");
```

Analysing the catalog as if it held every star, as in the Star population tab, gives the
population of the detected stars:

```{code-cell} python
hyper_priors = {"mu_L": dist.Uniform(-2, 2), "sigma_L": dist.Uniform(0.1, 2)}
naive = telescope.population_inference(data, population, hyper_priors, N=2000,
                                       seed=0, num_warmup=500,
                                       num_samples=1000, summary=False)
naive.summary()
```

It gives $\mu_L = 0.49 \pm 0.02$ and $\sigma_L = 0.46 \pm 0.02$, far from the true values of 0
and 0.7. Each factor of the likelihood is the evidence $Z_i(\Lambda)$ of one star, but the
catalog only holds detected stars, so each factor has to be divided by the probability
$\alpha(\Lambda)$ that a star drawn from the population is detected:

$$
\mathcal L(\Lambda)=\prod_{i=1}^{E}\frac{Z_i(\Lambda)}{\alpha(\Lambda)},
\qquad
\alpha(\Lambda)=\iint P(\mathrm{det}\mid d)\,\mathcal{L}(d\mid\theta)\,\pi(\theta\mid\Lambda)
\,\mathrm{d}d\,\mathrm{d}\theta .
$$

We estimate $\alpha(\Lambda)$ with *injections*: draws $\theta_j$ from a broad, fixed
distribution $g(\theta)$, each simulated once through the telescope and the trigger. Reweighting
the detected ones gives

$$
\hat\alpha(\Lambda)=\frac{1}{N_\mathrm{inj}}\sum_{j\in\mathrm{found}}
\frac{\pi(\theta_j\mid\Lambda)}{g(\theta_j)} .
$$

Here $g$ is the population with a brighter mean and a larger width. `trigger.injections` takes a
DataFrame of draws, with $g(\theta_j)$ in a `prior` column, and adds the simulated data and
whether each draw was detected:

```{code-cell} python
Lambda_g = {"mu_L": 0.5, "sigma_L": 1.5}
g_draws = population.sample(Lambda_g, key=jax.random.key(3), shape=(50_000,))
draw_table = pd.DataFrame({k: np.asarray(v) for k, v in g_draws.items()})
draw_table["prior"] = np.exp(np.asarray(population.log_prob(g_draws, Lambda_g)))

injections = trigger.injections(telescope, draw_table, np.random.default_rng(4))
int(injections["detected"].sum())
```

The selection term goes in a subclass of `PopulationLikelihood`. It keeps the found injections
and overrides `logpdf` to subtract $E\log\hat\alpha(\Lambda)$ from the sum of the
$\log\hat Z_i(\Lambda)$:

:::{div}
:class: landing-code

```{literalinclude} snippets/star_selection.py
:start-after: --- likelihood
:end-before: --- end
:language: python
```
:::

`population_inference` builds the likelihood from the class passed as `likelihood_cls`, and
`functools.partial` binds the injections to it:

```{code-cell} python
corrected = telescope.population_inference(
    data, population, hyper_priors, N=2000, seed=0, num_warmup=500,
    num_samples=1000, summary=False,
    likelihood_cls=partial(SelectedLikelihood, injections=injections))
corrected.summary()
```

The corrected posterior gives $\mu_L = -0.09 \pm 0.10$ and $\sigma_L = 0.76 \pm 0.05$, within
about one standard deviation of the true values. On the left are the posterior samples of
$\Lambda$ from both analyses. On the right, each line is the population distribution
$\pi(\log_{10}L\mid\Lambda)$ for one of 50 posterior samples, over the true population:

```{code-cell} python
grid = jnp.linspace(-3, 3, 200)

def population_pdfs(posterior, n=50):
    hyper = posterior.stack_samples().to_dataframe().sample(n, random_state=0)
    mu = hyper["mu_L"].to_numpy()[:, None]
    sigma = hyper["sigma_L"].to_numpy()[:, None]
    return jnp.exp(dist.TruncatedNormal(mu, sigma, low=-3.0, high=5.0)
                   .log_prob(grid))

true_pdf = jnp.exp(dist.TruncatedNormal(0.0, 0.7, low=-3.0, high=5.0)
                   .log_prob(grid))
analyses = [("without selection term", naive, "#8592a6"),
            ("with selection term", corrected, "#418FC5")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
for label, posterior, color in analyses:
    samples = posterior.stack_samples().to_dataframe()
    ax1.scatter(samples["mu_L"], samples["sigma_L"], s=2, alpha=0.2,
                color=color)
    ax2.plot(grid, population_pdfs(posterior).T, color=color, alpha=0.15)
    ax2.plot([], [], color=color, label=label)
ax1.plot(Lambda_true["mu_L"], Lambda_true["sigma_L"], "x", ms=10, mew=2,
         color="#E0A348")
ax1.set(xlabel=r"$\mu_L$", ylabel=r"$\sigma_L$",
        title=r"Posterior samples of $\Lambda$")
ax2.plot(grid, true_pdf, color="#E0A348", lw=2, ls="--",
         label="true population")
ax2.set(xlabel=r"$\log_{10}L$", ylabel="density",
        title="Inferred and true populations")
ax2.legend(fontsize=8);
```

The corrected curves agree on the bright side, where the survey sees most stars, and spread
out on the faint side, which the catalog constrains only through the stars it did not detect.
[Example 7](examples/07_flux_limited_survey.ipynb) works through this survey in more detail.

:::{dropdown} Definitions from the Star population tab
```{literalinclude} snippets/star_population.py
:start-after: --- population
:end-before: --- end
:language: python
```
:::

:::{dropdown} Imports for this example
```{literalinclude} snippets/star_population.py
:start-after: --- imports
:end-before: --- population
:language: python
```

```{literalinclude} snippets/star_selection.py
:start-after: --- imports
:end-before: --- trigger
:language: python
```
:::

::::::

::::::{tab-item} Template

```{code-cell} python
:tags: [remove-cell]

run_snippet("template")
```

A template to copy for your own problem. Every field takes a `name` and a LaTeX symbol for the
displays, an `index` if it is an array, and a `support` if its values are restricted. The two
methods of the measurement process work on one $\theta$ and one $d$, with no batch axes, and the
type hints say what goes in and out.

:::::{div}
:class: rail

::::{div}
:class: rail-gut

:::{div}
:class: dag-node dag-theta

$\theta$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Parameters $\theta$
:::

The parameters of one object. `x` is a positive scalar, and `ys` is a vector along an axis
named `component` with values between 0 and 1.

```{literalinclude} snippets/template.py
:start-after: --- theta
:end-before: --- data
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-process

$\mathcal{L}(d\mid\theta)$
:::

:::{div}
:class: rail-stem rail-arrow
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Measurement process $\mathcal{L}(d\mid\theta)$
:::

Settings of the instrument are dataclass fields. `sample_single` simulates one data set, and
`logpdf_single` returns $\log\mathcal{L}(d\mid\theta)$ for one data set. Fill in the two bodies.

```{literalinclude} snippets/template.py
:start-after: --- measurement
:end-before: --- end
:language: python
```

::::

::::{div}
:class: rail-gut

:::{div}
:class: rail-stem
:::

:::{div}
:class: dag-node

$d$
:::

::::

::::{div}
:class: rail-body

:::{div}
:class: rail-label

Data $d$
:::

The data of one object, one value per reading.

```{literalinclude} snippets/template.py
:start-after: --- data
:end-before: --- measurement
:language: python
```

::::

:::::

The names and symbols appear in the displays:

```{code-cell} python
MyParameters(x=1.0, ys=jnp.array([0.2, 0.5, 0.9]))
```

```{code-cell} python
MyMeasurement()
```

Once the two methods are filled in, the rest works as in the other examples:

```python
measurement = MyMeasurement(n_readings=100, noise=0.1)
theta = MyParameters(x=1.0, ys=jnp.array([0.2, 0.5, 0.9]))

d = measurement.sample(theta, N=1000, key=0)   # a batch of 1000 data sets
measurement.logpdf(d, theta)                   # 1000 values of log L(d | theta)
```

:::{dropdown} Imports for this example
```{literalinclude} snippets/template.py
:start-after: --- imports
:end-before: --- theta
:language: python
```
:::

::::::

:::::::

## Beyond one object

The same measurement process is the starting point for population inference. A population
model $\pi(\theta\mid\Lambda)$ describes how $\theta$ is spread over many objects, and
astroinfer samples the hyperparameters $\Lambda$ from a batch of measurements, with a selection
term when only some objects are detected. The posteriors of single objects can be kept in a
`Catalog`, which saves them to disk and exports them as arrays for a population analysis.

## Where to go next

::::{grid} 1 1 2 2

:::{card} Getting started
:link: getting-started.md
The coin flip example step by step, with outputs.
:::

:::{card} Examples
:link: examples.md
Ten worked problems, from a coin flip to selection effects and custom catalog
representations.
:::

:::{card} User guide
:link: user-guide/terminology.md
Parameter types, measurement processes, population models, population inference and
catalogs, one page each.
:::

:::{card} API reference
:link: api.md
Every public class and function.
:::

::::

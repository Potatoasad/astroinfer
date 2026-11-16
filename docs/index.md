---
title: astroinfer
site:
  hide_outline: true
  hide_title_block: true
---

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
:end-before: --- inference
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
`flip`, written $i$.

```{literalinclude} snippets/coin_flip.py
:start-after: --- data
:end-before: --- measurement
:language: python
```

::::

:::::

The measurement process now runs the experiment 1000 times, evaluates the likelihood of every
outcome, and samples the posterior of $p$ given the first one:

::::{div}
:class: landing-code

```{literalinclude} snippets/coin_flip.py
:start-after: --- inference
:end-before: --- end
:language: python
```

::::

`sample_single` and `logpdf_single` were written for one coin and one experiment. astroinfer
applies them to batches of any shape with `jax.vmap`, manages the random keys, and passes the
likelihood to NUTS.

:::{dropdown} Imports for this example
```{literalinclude} snippets/coin_flip.py
:start-after: --- imports
:end-before: --- theta
:language: python
```
:::

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
Nine worked problems, from a coin flip to selection effects and custom catalog
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

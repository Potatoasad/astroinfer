---
kernelspec:
  name: python3
  display_name: Python 3
---

# Distributions over parameter types

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

A *conditional distribution* is a probability distribution $p(x\mid c)$ whose output $x$ and
conditions $c$ are parameter objects. Every such distribution in astroinfer is a subclass of
`ConditionalDistribution`. Measurement processes $\mathcal{L}(d\mid\theta)$, proposals
$q_\Lambda(\theta\mid d)$ and population models $\pi(\theta\mid\Lambda)$ are subclasses, and so
are the structured distributions, products and mixtures described on this page. All of them draw
batches of objects with `sample` and evaluate log densities with `logpdf`.

The examples describe binaries with two masses and two spins, and a set of hyperparameters for
their population:

```{code-cell} python
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints

import astroinfer as ai
from astroinfer import parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)

@parameter_dataclass(name="Masses", latex="m")
class Masses:
    m1: float = parameter(name="Primary mass", latex="m_1", support=constraints.positive)
    q: float = parameter(name="Mass ratio", latex="q", support=constraints.unit_interval)

@parameter_dataclass(name="Spins", latex=r"\chi")
class Spins:
    chi1: float = parameter(name="Primary spin", latex=r"\chi_1", support=constraints.unit_interval)
    chi2: float = parameter(name="Secondary spin", latex=r"\chi_2", support=constraints.unit_interval)

@parameter_dataclass(name="Binary", latex=r"\theta")
class Binary:
    masses: Masses
    spins: Spins

@parameter_dataclass(name="Hyperparameters", latex=r"\Lambda")
class Hyper:
    mu_m: float = parameter(latex=r"\mu_m")
    sigma_m: float = parameter(latex=r"\sigma_m")
    mu_chi: float = parameter(latex=r"\mu_\chi")
    sigma_chi: float = parameter(latex=r"\sigma_\chi")
```

`Binary` has two nested fields; see [Nested parameter types](parameters.md#nested-parameter-types).

## Writing a conditional distribution

Subclass `ConditionalDistribution` and fill in these members:

| member | what it is |
|---|---|
| `data_type` | the type of the output $x$ |
| `parameter_type` | the type of the conditions $c$; `None` (the default) for a distribution without conditions |
| `sample_single(self, *conditions, key)` | one draw of $x$ for a single set of conditions, using the JAX random key `key` |
| `logpdf_single(self, *data, *conditions)` | the scalar $\log p(x\mid c)$ for a single $x$ and a single set of conditions |

`data_type` and `parameter_type` each take one of four forms, called *schemas*, which decide how
the values are passed:

| schema | example | arguments |
|---|---|---|
| none | `parameter_type = None` | no arguments |
| single | `parameter_type = Hyper` | one positional argument |
| tuple | `parameter_type = (Masses, Hyper)` | several positional arguments, in that order |
| mapping | `parameter_type = {"masses": Masses, "hyper": Hyper}` | keyword arguments with those names |

A parameter type is always one argument; it is never split into its fields. A type that is not a
parameter type, such as `float` or `typing.Any`, marks an *untyped* argument, which is passed on
unchanged. The single methods receive their arguments in the same form: `sample_single` gets the
conditions and then `key=`, and `logpdf_single` gets the data and then the conditions, with
mapping schemas as keywords.

Here the spins of a binary follow a normal distribution truncated to $[0, 1]$, whose mean is
$\mu_\chi$ times the mass ratio. The conditions are the masses and the hyperparameters, passed by
name:

```{code-cell} python
class SpinModel(ai.ConditionalDistribution):
    data_type = Spins
    parameter_type = {"masses": Masses, "hyper": Hyper}

    def law(self, masses, hyper):
        return dist.TruncatedNormal(hyper.mu_chi * masses.q, hyper.sigma_chi, low=0.0, high=1.0)

    def sample_single(self, masses, hyper, key):
        key1, key2 = jax.random.split(key)
        law = self.law(masses, hyper)
        return Spins(chi1=law.sample(key1), chi2=law.sample(key2))

    def logpdf_single(self, spins, masses, hyper):
        law = self.law(masses, hyper)
        return law.log_prob(spins.chi1) + law.log_prob(spins.chi2)

spin_model = SpinModel()
spin_model
```

The display shows the density with the symbols of its types, and one row per method. Conditions
in a mapping schema are passed as keywords. `N=` sets the number of draws for each set of
conditions:

```{code-cell} python
masses = Masses(m1=jnp.array([10.0, 30.0, 60.0]), q=jnp.array([0.2, 0.5, 0.9]))
hyper = Hyper(mu_m=30.0, sigma_m=5.0, mu_chi=0.6, sigma_chi=0.1)

spins = spin_model.sample(key=0, N=4, masses=masses, hyper=hyper)
spins.batch_shape
```

The three binaries give the first batch axis and the four draws the second. `logpdf` takes the
data first and then the conditions:

```{code-cell} python
spin_model.logpdf(spins, masses=masses, hyper=hyper).shape
```

The conditions of a mapping schema may also be given as one dict with those names. A positional
argument after the conditions is the key:

```{code-cell} python
spin_model.sample({"masses": masses, "hyper": hyper}, 0).batch_shape
```

`log_prob` is another name for `logpdf`, as in NumPyro. Keyword arguments that are not condition
names are passed unchanged to `sample_single` and `logpdf_single`, so the single methods can take
settings as extra keywords. The `key=` argument behaves as it does for
[measurement processes](measurement-processes.md#simulating-data): without it, the distribution
uses its own key, `spin_model.key`, and advances it on every call.

## Batching

`sample` and `logpdf` apply the single methods to every element of a batch, following these
rules:

1. Only arguments that are parameter objects are batched. A dict or DataFrame given for an
   argument declared with a parameter type is converted to that type first, and is then batched
   too. Every other argument, such as a number, a plain array or a dict for an untyped argument,
   reaches every single call whole.
2. The batch shapes of the conditions broadcast against each other, as in NumPy.
3. `N` is `None`, an integer or a tuple of integers. `sample` appends these sample axes after the
   batch axes of the conditions.
4. When the data have more batch axes than the conditions, the condition axes line up with the
   leading data axes.

Rules 3 and 4 make `logpdf(sample(c, N=n), c)` line up for every distribution, as in the example
above. Here two values of the hyperparameters meet a single set of masses, and `N` is a tuple:

```{code-cell} python
two_hypers = Hyper(mu_m=30.0, sigma_m=5.0, mu_chi=jnp.array([0.2, 0.8]), sigma_chi=0.1)
spin_model.sample(key=1, N=(5, 3), masses=masses[0], hyper=two_hypers).batch_shape
```

The batch shape `(2, 5, 3)` holds the axis of the two hyperparameter values, followed by the
sample axes.

With `outer=True`, `logpdf` pairs every data element with every condition element, with the
condition axes first. Five spin values against the three binaries give a $3\times 5$ array:

```{code-cell} python
grid = Spins(chi1=jnp.linspace(0.1, 0.9, 5), chi2=0.3)
spin_model.logpdf(grid, masses=masses, hyper=hyper, outer=True).shape
```

Without `outer=True`, the same call fails, because the data batch shape `(5,)` does not start
with the condition batch shape `(3,)`:

```{code-cell} python
:tags: [raises-exception]

spin_model.logpdf(grid, masses=masses, hyper=hyper)
```

[Sharp bits](../sharp-bits.md) has examples of rules 1 and 4, which are easy to get wrong.

## Binding conditions

`condition(...)` takes conditions in the same form as `sample` and returns a *bound
distribution*, a `BoundConditionalDistribution` with those conditions fixed. A mapping schema
may be bound in part. The result is then a distribution conditional on the remaining names:

```{code-cell} python
given_masses = spin_model.condition(masses=masses)
given_masses
```

The bound values are shown under *Configuration*. The bound distribution takes the remaining
conditions, and draws the same values as the parent for the same key:

```{code-cell} python
a = given_masses.sample(key=2, hyper=hyper)
b = spin_model.sample(key=2, masses=masses, hyper=hyper)
bool(jnp.all(a.chi1 == b.chi1))
```

Binding the rest leaves a distribution without conditions:

```{code-cell} python
fixed = given_masses.condition(hyper=hyper)
fixed.sample(key=3, N=2).batch_shape, fixed.logpdf(spins).shape
```

A bound distribution has its own key, split from the parent's when it is made.

## Structured distributions

Many distributions over a parameter type can be written one field at a time. Every parameter type
has a class method `distribution(**fields)` that returns a *structured distribution*, a
`StructuredDistribution` over that type. Each keyword names a field and gives one of:

- a NumPyro distribution, for a field that holds arrays;
- a fixed value (see [Fixed fields](#fixed-fields));
- a function of the conditions that returns a NumPyro distribution (see
  [Conditions](#conditions));
- `ai.conditional(fn, on=...)`, for a field whose distribution depends on fields drawn before it
  (see [Conditional fields](#conditional-fields));
- for a nested field, a structured distribution over the nested type, a fixed object of that
  type, or any conditional distribution whose output is that type.

A uniform prior on the primary mass and the mass ratio:

```{code-cell} python
mass_prior = Masses.distribution(m1=dist.Uniform(5.0, 100.0), q=dist.Uniform(0.1, 1.0))
mass_prior
```

`sample` returns `Masses` objects:

```{code-cell} python
m = mass_prior.sample(key=0, N=1000)
m
```

The log density is the sum of the fields' log densities:

```{code-cell} python
m0 = Masses(m1=30.0, q=0.5)
mass_prior.logpdf(m0), dist.Uniform(5.0, 100.0).log_prob(30.0) + dist.Uniform(0.1, 1.0).log_prob(0.5)
```

Both are $-4.449$, the logarithm of $1/(95 \times 0.9)$.

Every field must be given, either here or as a default in the dataclass. A misspelt or missing
field raises an error that names it, here `Masses.q`:

```{code-cell} python
:tags: [raises-exception]

Masses.distribution(m1=dist.Uniform(5.0, 100.0))
```

A NumPyro distribution must describe one value of its field. For a field without core axes, its
`batch_shape` and `event_shape` must both be empty. A field with core axes needs one event
dimension per core axis, which `.to_event` provides (see [Sharp bits](../sharp-bits.md)).

Each field is drawn with its own key, `jax.random.fold_in(key, i)`, where `i` is the field's
position in the dataclass. Turning one field into a fixed value therefore leaves the draws of the
other fields unchanged.

### Fixed fields

A field given a value instead of a distribution is a *fixed field*. It is not random: every draw
holds the given value, and it adds nothing to the log density. A field that is left out but has a
default in the dataclass is fixed at its default. Here the secondary spin is fixed at zero:

```{code-cell} python
spin_prior = Spins.distribution(chi1=dist.Uniform(0.0, 1.0), chi2=0.0)
spin_prior.sample(key=1, N=3)["chi2"]
```

`logpdf` of an object whose fixed fields differ from the declared values is $-\infty$, since the
distribution gives such an object zero probability:

```{code-cell} python
spin_prior.logpdf(Spins(chi1=jnp.array([0.4, 0.4]), chi2=jnp.array([0.0, 0.1])))
```

The first object matches the fixed value and gets the log density of $\chi_1$ alone, 0; the
second, with $\chi_2 = 0.1$, gets $-\infty$.

The comparison uses a relative tolerance of $10^{-6}$, and is written with `jnp.where`, so it
works under `jax.jit` and keeps the gradient finite. To skip it, build the distribution with
`ai.StructuredDistribution(Spins, fields, check_fixed=False)`, where `fields` is the dict of
keywords you would pass to `distribution`.

### Nested fields

A nested field takes a structured distribution over its own type. Paths into nested fields are
written with dots:

```{code-cell} python
binary_prior = Binary.distribution(
    masses=mass_prior,
    spins=Spins.distribution(chi1=dist.Beta(2.0, 5.0), chi2=dist.Beta(2.0, 5.0)),
)
binary_prior
```

The display shows each nested field as its own section. `random_paths()` and `fixed_paths()` list
the random and fixed fields by path:

```{code-cell} python
binary_prior.sample(key=2, N=500).batch_shape, binary_prior.random_paths()
```

A nested field can also be fixed to one object:

```{code-cell} python
non_spinning = Binary.distribution(masses=mass_prior, spins=Spins(chi1=0.0, chi2=0.0))
non_spinning.fixed_paths()
```

## Conditional fields

`ai.conditional(fn, on=paths)` makes a field whose distribution depends on fields drawn before
it. `fn(x)` receives `x`, the dict of values drawn so far keyed by path, and returns a NumPyro
distribution. `on` names the paths that `fn` reads, as a string or a list of strings. Here the
mass ratio is bounded below so that the secondary mass, $q\,m_1$, is at least 5:

```{code-cell} python
masses_above_5 = Masses.distribution(
    m1=dist.Uniform(5.0, 100.0),
    q=ai.conditional(lambda x: dist.Uniform(5.0 / x["m1"], 1.0), on="m1"),
)
m = masses_above_5.sample(key=3, N=10_000)
float(jnp.min(m.q * m.m1))
```

The smallest secondary mass in 10,000 draws is 5.003. The log density is the chain rule along
the draw order, $\log p(m_1) + \log p(q\mid m_1)$.

Paths are relative to the structured distribution that declares the field. A path that is not
found there is looked up in the enclosing distributions, out to the outermost one. `masses_above_5`
therefore keeps working when it is nested inside `Binary`, and a field of the spins can read
`"masses.m1"`, which exists only once the spins sit beside the masses. Here heavier binaries have
smaller primary spins:

```{code-cell} python
mass_dependent_spins = Binary.distribution(
    masses=masses_above_5,
    spins=Spins.distribution(
        chi1=ai.conditional(lambda x: dist.Beta(2.0, 2.0 + x["masses.m1"] / 10.0), on="masses.m1"),
        chi2=dist.Beta(2.0, 5.0),
    ),
)
mass_dependent_spins
```

The display writes a conditional field as $p(\chi_1\mid m_1)$, with the symbols of the fields it
reads. A path that names a field of the enclosing types but does not exist raises an error when the
distribution is built:

```{code-cell} python
:tags: [raises-exception]

Binary.distribution(
    masses=mass_prior,
    spins=Spins.distribution(
        chi1=ai.conditional(lambda x: dist.Beta(2.0, x["masses.m2"]), on="masses.m2"),
        chi2=0.0,
    ),
)
```

A path that cannot be resolved yet, such as `"masses.m1"` in a distribution over `Spins` alone,
raises when the distribution is first used.

`ai.conditional` has the same meaning as `astroinfer.generative.conditional` (see
[Population models](population-models.md#conditional-parameters)), but describes a field of a
parameter type rather than a parameter of a generative leaf. `ai.product` and `ai.mixture` below
likewise differ from their namesakes in `astroinfer.generative`.

### Draw order

`order` lists the random fields in the order they are drawn. By default it follows the fields
depth first, with every field moved after the fields it reads:

```{code-cell} python
mass_dependent_spins.order
```

`with_order(paths)` returns a copy that draws in the given order, which must list every random
field once. Each field must come after the fields named in its `on`:

```{code-cell} python
reordered = mass_dependent_spins.with_order(["masses.m1", "spins.chi1", "spins.chi2", "masses.q"])
reordered.order
```

The key of each field depends on its position in the dataclass, not in the order, so a valid
reordering gives the same draws for the same key:

```{code-cell} python
first, second = mass_dependent_spins.sample(key=4, N=3), reordered.sample(key=4, N=3)
bool(jnp.all(first.spins.chi1 == second.spins.chi1))
```

An order in which a field comes before a field it reads raises an error. So do fields that read
each other in a cycle:

```{code-cell} python
:tags: [raises-exception]

mass_dependent_spins.with_order(["spins.chi1", "masses.m1", "masses.q", "spins.chi2"])
```

## Conditions

A field may also be a function that returns a NumPyro distribution from the conditions. The
conditions are declared with `.given(type)`, which returns a copy with that condition schema; the
type takes any of the four forms above. Here a population of binaries has a truncated normal
primary mass whose mean and width are hyperparameters:

```{code-cell} python
mass_model = Masses.distribution(
    m1=lambda L: dist.TruncatedNormal(L.mu_m, L.sigma_m, low=5.0, high=100.0),
    q=ai.conditional(lambda x, L: dist.Uniform(5.0 / x["m1"], 1.0), on="m1"),
)
population = Binary.distribution(
    masses=mass_model,
    spins=Spins.distribution(
        chi1=lambda L: dist.TruncatedNormal(L.mu_chi, L.sigma_chi, low=0.0, high=1.0),
        chi2=lambda L: dist.TruncatedNormal(L.mu_chi, L.sigma_chi, low=0.0, high=1.0),
    ),
).given(Hyper)
population
```

`.given` passes the condition type down to nested structured distributions that have none of
their own, here `mass_model` and the spins. Once it is set, every function field receives the
conditions, and so does every conditional field, after `x`; the mass ratio above is written
`lambda x, L: ...` for that reason.
The conditions are not given as a keyword of `distribution`, because any keyword there could be
a field name.

The conditions follow the batching rules. Two values of $\mu_m$ give two batches of draws:

```{code-cell} python
two_mass_scales = Hyper(mu_m=jnp.array([20.0, 60.0]), sigma_m=5.0, mu_chi=0.3, sigma_chi=0.1)
theta = population.sample(two_mass_scales, key=5, N=2000)
theta.batch_shape, theta.masses.m1.mean(axis=1)
```

The mean primary masses of the two batches are 20.1 and 60.0. A dict is converted to `Hyper`
and is one set of conditions:

```{code-cell} python
Lambda = {"mu_m": 30.0, "sigma_m": 5.0, "mu_chi": 0.3, "sigma_chi": 0.1}
population.logpdf(population.sample(Lambda, key=6, N=3), Lambda)
```

With a mapping schema the functions receive the conditions as keywords, so their argument names
must match:

```{code-cell} python
mass_population = Masses.distribution(
    m1=lambda hyper: dist.TruncatedNormal(hyper.mu_m, hyper.sigma_m, low=5.0, high=100.0),
    q=dist.Uniform(0.1, 1.0),
).given({"hyper": Hyper})
mass_population.sample(key=7, N=2, hyper=Lambda).batch_shape
```

A structured distribution without `.given` takes the condition type of its first nested field or
block that has one.

### Conditional distributions as fields

A nested field can hold any conditional distribution whose output is the field's type, such as an
existing class or one whose density does not split field by field. It is drawn as one unit. Its
conditions are taken from the conditions of the structured distribution: all of them for an equal
single or tuple schema, or the names it needs from a mapping schema. It cannot read fields drawn
before it; use a conditional field for that. Here a spin model written as a class, conditioned on
the hyperparameters, takes the place of `Spins.distribution(...)`:

```{code-cell} python
class SpinPopulation(ai.ConditionalDistribution):
    data_type = Spins
    parameter_type = Hyper

    def law(self, hyper):
        return dist.TruncatedNormal(hyper.mu_chi, hyper.sigma_chi, low=0.0, high=1.0)

    def sample_single(self, hyper, key):
        key1, key2 = jax.random.split(key)
        return Spins(chi1=self.law(hyper).sample(key1), chi2=self.law(hyper).sample(key2))

    def logpdf_single(self, spins, hyper):
        return self.law(hyper).log_prob(spins.chi1) + self.law(hyper).log_prob(spins.chi2)

with_block = Binary.distribution(masses=mass_model, spins=SpinPopulation()).given(Hyper)
with_block.order
```

The block appears in `order` under the path of its field. Its density is the same as that of the
field-by-field spins above, so the two populations agree:

```{code-cell} python
theta = population.sample(Lambda, key=8, N=4)
with_block.logpdf(theta, Lambda) - population.logpdf(theta, Lambda)
```

The differences are zero up to rounding, of order $10^{-16}$.

## Products

`ai.product(a, b, ...)` is the independent product $p(x_a, x_b\mid c) = p_a(x_a\mid c)\,p_b(x_b\mid
c)$. With positional components the output is a tuple; with keyword components it is a dict.
Its conditions are the union of the components' conditions: mapping schemas merge by name and
must agree on the type of a shared name, single and tuple schemas must be equal, and a component
without conditions accepts any. Here the masses and spins of the population above form a product
conditioned on `Hyper`. `mass_model` has no condition type of its own, so it is given one here:

```{code-cell} python
masses_and_spins = ai.product(masses=mass_model.given(Hyper), spins=SpinPopulation())
masses_and_spins
```

`sample` returns a dict, and `logpdf` takes one:

```{code-cell} python
draws = masses_and_spins.sample(Lambda, key=9, N=3)
masses_and_spins.logpdf(draws, Lambda)
```

The output of a product is a tuple or a dict, not a `Binary`. When every component is over a
field of one parameter type, as here, `Binary.distribution(masses=..., spins=...)` gives the same
density with a `Binary` as its output.

## Mixtures

`ai.mixture(components, weights=w)` is the mixture $p(x\mid c) = \sum_k w_k\,p_k(x\mid c)$ of
distributions over one output type. Here 70% of binaries have small spins and 30% have large
ones:

```{code-cell} python
small = Spins.distribution(chi1=dist.Beta(2.0, 8.0), chi2=dist.Beta(2.0, 8.0))
large = Spins.distribution(chi1=dist.Beta(8.0, 2.0), chi2=dist.Beta(8.0, 2.0))
spin_mixture = ai.mixture([small, large], weights=jnp.array([0.7, 0.3]))
spin_mixture
```

The weights are checked to be finite, nonnegative and to sum to one within $10^{-6}$. The log
density marginalizes over the component with `logsumexp`, so it is differentiable with respect to
the weights. `logits=` instead of `weights=` gives unnormalized log weights, which are passed
through a softmax:

```{code-cell} python
s0 = Spins(chi1=0.2, chi2=0.9)
same = ai.mixture([small, large], logits=jnp.log(jnp.array([0.7, 0.3])) + 1.0)
spin_mixture.logpdf(s0), same.logpdf(s0)
```

Both give $-7.162$, since a constant added to every logit does not change the weights.

A draw picks one component for the whole object, so both spins of a binary come from the same
component:

```{code-cell} python
s = spin_mixture.sample(key=10, N=20_000)
float(jnp.mean(s.chi1 > 0.5)), float(jnp.corrcoef(s.chi1, s.chi2)[0, 1])
```

About 30% of primary spins are above 0.5, and the two spins have a correlation coefficient of
0.84, although each component draws them independently. Every component is sampled for every
object, so a draw from a mixture of $K$ components costs $K$ draws.

The components must have the same output type and draw objects with the same leaf shapes.
Structured components must also have the same support at every path, here the unit interval of
the beta distributions; pass `support=` to `ai.mixture` to set one when they differ.

### Weights that depend on the conditions

`weights` (or `logits`) may be a function of the conditions. Its argument names become conditions
of the mixture, in addition to those of the components. Here the fraction of binaries with large
spins rises with the primary mass, with a steepness `alpha`:

```{code-cell} python
def large_spin_fraction(masses, alpha):
    w = jax.nn.sigmoid(alpha * (masses.m1 - 40.0))
    return jnp.stack([1.0 - w, w])

spins_by_mass = ai.mixture([small, large], weights=large_spin_fraction)
spins_by_mass
```

The components have no conditions, so the mixture is conditioned on `masses` and `alpha`, the
arguments of the weight function. These names are untyped, so by the batching rules a `Masses`
object is batched and the number `alpha` reaches every object whole. Spins drawn for 5000 binaries
from the mass prior follow their masses:

```{code-cell} python
m = mass_prior.sample(key=11, N=5000)
s = spins_by_mass.sample(key=12, masses=m, alpha=0.5)
float(s.chi1[m.m1 < 30].mean()), float(s.chi1[m.m1 > 50].mean())
```

The mean primary spin is 0.20 below 30 solar masses and 0.80 above 50. The joint density of the
masses and spins is the sum of the two log densities, and its gradient with respect to `alpha` is
finite:

```{code-cell} python
def log_joint(alpha):
    return jnp.sum(mass_prior.logpdf(m) + spins_by_mass.logpdf(s, masses=m, alpha=alpha))

jax.grad(log_joint)(0.5)
```

Pass `parameter_type=` to give the conditions types, for example
`parameter_type={"masses": Masses, "alpha": float}`, so that a dict of masses is converted to
`Masses`. It must name every condition of the components.

## Graphs

`graph()` draws a distribution as a directed acyclic graph. It reads the definition of the
distribution and runs nothing. For a structured distribution, each field is a node, grouped by
nested type; a conditional field has an edge from every field in its `on` list, a field written
as a function of the conditions has an edge from the condition node, and a fixed field is drawn
dashed with its value:

```{code-cell} python
mass_dependent_spins.graph()
```

```{code-cell} python
population.graph()
```

A mixture shows its weights, its components and the conditions the weights read:

```{code-cell} python
spins_by_mass.graph()
```

Other distributions show their conditions pointing to their output, $\theta\to d$ for a
measurement process. A graph has `nodes`, `edges` and `groups` attributes, `to_svg()` returns
the drawing as a string, and `to_dot()` returns graphviz DOT source; `to_graphviz()` needs the
graphviz package. Labels are drawn as plain text, so LaTeX in a label is shown with Unicode
symbols and subscripts.

## Products of densities over the same object

`ai.product` multiplies distributions over different objects. A product of two densities over the
same object, $p(x) \propto p_a(x)\,p_b(x)$, is not a class in astroinfer. Write it as the sum of
the log densities instead. Here a uniform mass prior is combined with a second, independent
constraint on the same masses:

```{code-cell} python
constraint = Masses.distribution(m1=dist.Normal(35.0, 5.0), q=dist.Beta(5.0, 2.0))

def log_combined(m):
    return mass_prior.logpdf(m) + constraint.logpdf(m)

log_combined(Masses(m1=jnp.array([20.0, 35.0, 120.0]), q=0.7))
```

The third value is $-\infty$ because 120 is outside the uniform prior. The sum leaves out the
normalizing constant $\int p_a(x)\,p_b(x)\,dx$. That is harmless only when the constant does not
depend on anything being inferred; when $p_a$ or $p_b$ depends on hyperparameters, so does the
constant, and the sum is then not the log density of the product.

In a NumPyro model whose masses are drawn from `mass_prior`, add the second log density with
`ai.factor`, which passes it to `numpyro.factor`:

```py
# inside a NumPyro model, with `masses` a Masses object drawn from mass_prior
ai.factor("mass_constraint", constraint.logpdf(masses))
```

[NumPyro models](numpyro-models.md) shows how to use these distributions inside NumPyro models,
with `ai.sample`, `ai.factor` and `ai.infer`.

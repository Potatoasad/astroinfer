---
kernelspec:
  name: python3
  display_name: Python 3
---

# Parameter types

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

A *parameter type* is a frozen dataclass whose fields are arrays with named axes. astroinfer
uses parameter types for both the parameters $\theta$ and the data $d$ of a model. This page
uses the imports from the [quickstart](../quickstart.md):

```{code-cell} python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from numpyro.distributions import constraints

import astroinfer as ai
from astroinfer import batched_method, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)
```

## Creating a parameter type

Decorate a class with `@parameter_dataclass` and declare each field with `parameter(...)`:

```{code-cell} python
@parameter_dataclass(name="Source", latex=r"\theta")
class Source:
    x: float = parameter(name="Position [arcsec]", latex="x")
    flux: float = parameter(name="Flux", latex="f")
```

Objects are created with keyword arguments, as for any dataclass:

```{code-cell} python
src = Source(x=0.2, flux=3.0)
src
```

Every argument of the decorator and of `parameter` is optional:

| argument | of | meaning |
|---|---|---|
| `name` | decorator | title of the object's display (default: the class name) |
| `latex` | decorator | symbol for the object in displays (default: the class name) |
| `name` | `parameter` | readable description of the field |
| `latex` | `parameter` | symbol for the field in displays |
| `index`, `index_latex` | `parameter` | names and symbols of the field's core axes (see [Core axes](#core-axes)) |
| `support` | `parameter` | a numpyro constraint on the field's values (see [Supports and latent spaces](#supports-and-latent-spaces)) |
| `default`, `default_factory` | `parameter` | as in `dataclasses.field` |

Objects are frozen. Assigning to a field raises an error:

```{code-cell} python
:tags: [raises-exception]

src.x = 1.0
```

## Class-level information

`parameters` lists the field names, in definition order. It works on the class and on
instances:

```{code-cell} python
Source.parameters, src.parameters
```

`field_specs()` returns the `ParameterSpec` of each field, which holds the arguments given to
`parameter`:

```{code-cell} python
Source.field_specs()["flux"]
```

`name` and `latex` return the values given to the decorator:

```{code-cell} python
src.name, src.latex
```

## Batch axes

Any field can hold an array. Leading axes of the fields are *batch axes*, and the object then
holds a batch of objects. `batch_shape` is the shape of that batch:

```{code-cell} python
sources = Source(x=jnp.array([-0.5, 0.0, 0.5]), flux=2.0)
sources.batch_shape, src.batch_shape
```

`is_batched()` is `True` when the batch shape is not empty:

```{code-cell} python
sources.is_batched(), src.is_batched()
```

The batch shape is the broadcast of the fields' batch shapes, following NumPy's rules. Each field
is stored as it was given, so here `flux` is still a single number:

```{code-cell} python
sources.flux
```

`broadcast()` returns a copy in which every field has the full batch shape. Given a shape, it
broadcasts to that shape instead:

```{code-cell} python
sources.broadcast().flux, src.broadcast((2, 2)).batch_shape
```

`reshape_batch(shape)` changes the batch shape. The number of elements has to stay the same:

```{code-cell} python
sources.reshape_batch((3, 1)).batch_shape
```

## Core axes

Some fields are arrays even for a single object, for example a spectrum with one value per
channel. `index=` names the trailing axes of such a field, which are its *core axes*:

```{code-cell} python
@parameter_dataclass(name="Spectrum", latex="s")
class Spectrum:
    flux: float = parameter(name="Flux per channel", latex="f",
                            index=("channel",), index_latex=("c",))
```

Core axes are always the trailing axes of a field, and they are not part of `batch_shape`:

```{code-cell} python
one = Spectrum(flux=jnp.ones(4))
three = Spectrum(flux=jnp.ones((3, 4)))
one.batch_shape, three.batch_shape
```

`broadcast()`, `reshape_batch()` and indexing act on the batch axes only:

```{code-cell} python
three.reshape_batch((3, 1)).flux.shape, three[0].flux.shape
```

`index_latex` gives the symbol of each core axis in displays. When it is given, it must have one
entry for each name in `index`.

## Indexing

Indexing selects from the batch axes or picks out a field:

| index | example | result |
|---|---|---|
| integer | `sources[1]` | one object, with that batch axis removed |
| slice | `sources[1:]` | an object with a shorter batch |
| boolean mask | `sources[sources.x >= 0]` | the elements where the mask is `True` |
| integer array | `sources[jnp.array([0, 2])]` | the elements at those positions |
| field name | `sources["x"]` | the field's array, as stored |

```{code-cell} python
sources[1]
```

The other kinds of index keep a batch axis:

```{code-cell} python
sources[1:].batch_shape, sources[sources.x >= 0].batch_shape, sources[jnp.array([0, 2])].batch_shape
```

```{code-cell} python
sources["x"]
```

Array indices must be one-dimensional, and a tuple of indices can have at most one entry per
batch axis. An object without batch axes cannot be indexed by position:

```{code-cell} python
:tags: [raises-exception]

src[0]
```

## Converting to and from dicts and tables

`to_dict()` returns the fields as a dict:

```{code-cell} python
sources.to_dict()
```

`from_dict` builds an object from a dict or a pandas DataFrame. Columns that are not fields are
ignored, and DataFrame columns become NumPy arrays:

```{code-cell} python
table = pd.DataFrame({"x": [0.1, 0.2], "flux": [1.0, 2.0], "label": ["a", "b"]})
from_table = Source.from_dict(table)
from_table
```

A missing field raises an error:

```{code-cell} python
:tags: [raises-exception]

Source.from_dict({"x": 0.1})
```

`ai.coerce(obj, cls)` returns `obj` unchanged when it is already a `cls`, and calls
`cls.from_dict` when it is a dict or DataFrame. Functions that accept either form use it:

```{code-cell} python
type(ai.coerce({"x": 0.0, "flux": 1.0}, Source)).__name__, ai.coerce(src, Source) is src
```

Objects are frozen, but a field may hold a NumPy array, which can still be changed in place.
`copy()` returns an object whose arrays are new copies, so changing one does not change the
other:

```{code-cell} python
copied = from_table.copy()
copied.x[0] = 99.0
from_table.x, copied.x
```

## Arithmetic

Two objects of the same type can be added and subtracted, field by field. An object can be
negated, multiplied by a number and divided by a number:

```{code-cell} python
src + src
```

```{code-cell} python
-src / 2
```

Batch axes broadcast as usual:

```{code-cell} python
sources + src
```

Adding objects of different types raises a `TypeError`.

## Pytrees and JAX transformations

Every parameter type is registered as a JAX *pytree*: JAX treats the fields as the leaves of a
tree, so its transformations accept parameter objects and return objects of the same type.

```{code-cell} python
jax.tree_util.tree_leaves(src)
```

`jax.grad` of a function of a `Source` returns a `Source` holding the derivatives with respect
to each field:

```{code-cell} python
def brightness(s):
    return s.flux * jnp.exp(-s.x**2)

gradient = jax.grad(brightness)(src)
gradient
```

`jax.jit` and `jax.vmap` work the same way. `vmap` maps over the leading axis of every field:

```{code-cell} python
jax.jit(brightness)(src), jax.vmap(brightness)(sources.broadcast())
```

Because the gradient has the same type, a gradient step is written as arithmetic on objects:

```{code-cell} python
src + 0.1 * gradient
```

## Batched methods

A parameter type can have methods. A method written for a single object often fails on a batch,
because NumPy broadcasting lines up the batch axes with the method's own arrays:

```{code-cell} python
@parameter_dataclass(name="Peak")
class Peak:
    centre: float = parameter(name="Centre", latex=r"\mu")
    width: float = parameter(name="Width", latex=r"\sigma")

    def profile(self, grid):
        return jnp.exp(-0.5 * ((grid - self.centre) / self.width) ** 2)

grid = jnp.linspace(-1.0, 1.0, 5)
Peak(centre=0.0, width=0.3).profile(grid).shape
```

```{code-cell} python
:tags: [raises-exception]

Peak(centre=jnp.array([-0.5, 0.0, 0.5]), width=0.3).profile(grid)
```

Decorate the method with `@batched_method` instead. It applies `jax.vmap` once per batch axis,
so the method body always sees a single object, and the batch axes come first in the result:

```{code-cell} python
@parameter_dataclass(name="Peak")
class BatchedPeak:
    centre: float = parameter(name="Centre", latex=r"\mu")
    width: float = parameter(name="Width", latex=r"\sigma")

    @batched_method
    def profile(self, grid):
        return jnp.exp(-0.5 * ((grid - self.centre) / self.width) ** 2)

BatchedPeak(centre=jnp.array([-0.5, 0.0, 0.5]), width=0.3).profile(grid).shape
```

Arguments other than `self` are passed to every element unchanged. The method body must be
written with `jax.numpy`, because it runs under `jax.vmap`.

## Supports and latent spaces

`support=` declares the values a field may take, as a
[numpyro constraint](https://num.pyro.ai/en/stable/distributions.html#constraints). A field
without a support may take any real value.

```{code-cell} python
@parameter_dataclass(name="Disk")
class Disk:
    x: float = parameter(name="Position", latex="x")
    radius: float = parameter(name="Radius", latex="r", support=constraints.positive)

disk = Disk(x=0.2, radius=3.0)
```

`to_unconstrained()` maps each field onto the whole real line using its support, and returns an
object of the same type. `from_unconstrained(z)` maps back:

```{code-cell} python
z = disk.to_unconstrained()
z
```

```{code-cell} python
Disk.from_unconstrained(z)
```

A positive field is mapped with a logarithm, and a field without a support is unchanged. Values
on the boundary of a support map to $\pm\infty$.

The mapping is a `LatentSpace`. `ai.LatentSpace.default(Disk)` is the space built from the
declared supports. `ai.LatentSpace(Disk, supports)` replaces the supports of the fields named in
`supports`, given either as constraints or as numpyro distributions, whose `.support` is used:

```{code-cell} python
box = ai.LatentSpace(Disk, {"x": dist.Uniform(-1.0, 1.0)})
box.supports
```

Pass the space to `to_unconstrained` and `from_unconstrained`, or call its own methods,
`from_parameter` and `to_parameter`:

```{code-cell} python
z_box = disk.to_unconstrained(box)
z_box
```

```{code-cell} python
box.to_parameter(z_box)
```

An optimiser can work on the latent point directly. A probability density moved into the latent
space needs the log Jacobian of the mapping, which `log_abs_det_jacobian(z)` returns, summed over
the fields and core axes:

```{code-cell} python
box.log_abs_det_jacobian(z_box)
```

Naming a field the type does not have raises an error:

```{code-cell} python
:tags: [raises-exception]

ai.LatentSpace(Disk, {"y": constraints.positive})
```

For an optimisation in the latent space, see [the lighthouse tutorial](https://github.com/Potatoasad/astroinfer/blob/main/tutorials/02_lighthouse.py).

## Nested parameter types

A field annotated with another parameter type is a *nested field*, and a type with nested fields
is a *nested parameter type*. A nested field holds one object of its own type:

```{code-cell} python
@parameter_dataclass(name="Position", latex=r"\mathbf{r}")
class Position:
    x: float = parameter(name="x [arcsec]", latex="x")
    y: float = parameter(name="y [arcsec]", latex="y")

@parameter_dataclass(name="Star", latex=r"\theta")
class Star:
    position: Position = parameter(name="Sky position")
    flux: float = parameter(name="Flux", latex="f", support=constraints.positive)

star = Star(position=Position(x=0.1, y=-0.2), flux=3.0)
star
```

The display shows the nested field as a section of its own, titled with the `name` given to
`parameter`. A nested field may set `name` and `latex`, but not `index` or `support`, which belong
to the fields of the nested type:

```{code-cell} python
:tags: [raises-exception]

@parameter_dataclass
class BadStar:
    position: Position = parameter(support=constraints.positive)
```

The annotation may also be a string, such as `position: "Position"`, for a type defined later in
the module.

`parameters` lists the fields of the outer type only. `nested_types()` maps each nested field to
its type, and `flat_field_specs()` gives the `ParameterSpec` of every array field, with nested
fields named by their *path*, the field names joined by dots:

```{code-cell} python
Star.parameters, Star.nested_types(), list(Star.flat_field_specs())
```

### Batching and indexing

Batch axes are shared by every array field in the tree, and broadcast as for a flat type. Here the
position is a single point and the flux has three values:

```{code-cell} python
stars = Star(position=Position(x=0.0, y=0.0), flux=jnp.array([1.0, 2.0, 3.0]))
stars.batch_shape, stars.position.batch_shape, stars.broadcast().position.x.shape
```

The outer object has batch shape `(3,)`, while the nested position, stored as given, has batch
shape `()` until `broadcast()` gives it the full shape. Named core axes also share one namespace
across the tree, so two fields that use the same axis name must agree on its length.

Integer, slice and array indices select from the batch of every field in the tree. A string may
be a path, which reaches into nested fields; the name of a nested field returns the nested object:

```{code-cell} python
stars[1:].batch_shape, stars["position.x"], type(stars["position"]).__name__
```

Arithmetic, `copy()` and the JAX transformations act on every array field in the tree. The
gradient of a function of a `Star` is a `Star` with a `Position` inside:

```{code-cell} python
jax.grad(lambda s: s.flux * jnp.exp(-s.position.x**2 - s.position.y**2))(star)
```

### Dicts and tables

`to_dict()` returns nested dicts, and `to_flat_dict()` returns one entry per path:

```{code-cell} python
star.to_dict()
```

```{code-cell} python
star.to_flat_dict()
```

`from_dict` accepts either form. A nested field may be given as a nested dict, as an object of its
type, or as keys with dotted paths, so a DataFrame with a `position.x` column converts directly:

```{code-cell} python
star_table = pd.DataFrame({"position.x": [0.1, 0.4], "position.y": [0.0, 0.2], "flux": [1.0, 2.0]})
Star.from_dict(star_table)
```

The same paths name the columns of tables of draws. `PosteriorData.to_dataframe()`, `summary()`,
`to_arviz()` and catalog files use `position.x` and `position.y`, and
`PosteriorData.from_dataframe` reads them back. The columns of a posterior over `Star`:

```{code-cell} python
ai.PosteriorData(theta=Star(position=Position(x=jnp.zeros(4), y=jnp.ones(4)),
                            flux=jnp.linspace(1.0, 2.0, 4))).to_dataframe().columns.tolist()
```

### Latent spaces

A `LatentSpace` of a nested type maps every array field by its own support. Overrides are keyed by
path:

```{code-cell} python
star_space = ai.LatentSpace(Star, {"position.x": dist.Uniform(-1.0, 1.0)})
star_space.supports
```

`children` holds the latent space of each nested field, and `log_abs_det_jacobian` sums over the
whole tree. To build distributions over nested types field by field, see
[Distributions over parameter types](distributions.md#nested-fields).

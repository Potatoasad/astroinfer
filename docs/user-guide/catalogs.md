---
kernelspec:
  name: python3
  display_name: Python 3
---

# Catalogs

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

A *catalog* holds one posterior per *event*, together with any compressed representations fitted
to those posteriors. It exports the posteriors as event-major arrays for population analyses, and
it saves itself to a folder and loads itself back.

The examples on this page use the source and camera from the [quickstart](../quickstart.md):

```{code-cell} python
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import astroinfer as ai
from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)

@parameter_dataclass(name="Source", latex=r"\theta")
class Source:
    x: float = parameter(name="Position [arcsec]", latex="x")
    flux: float = parameter(name="Flux", latex="f")

@parameter_dataclass(name="Measurement", latex="d")
class Measurement:
    x_hat: float = parameter(name="Measured position [arcsec]", latex=r"\hat x")
    flux_hat: float = parameter(name="Measured flux", latex=r"\hat f")

@dataclass
class Camera(MeasurementProcess):
    sigma_x: float = 0.1
    sigma_flux: float = 0.3

    parameter_type = Source
    data_type = Measurement

    def sample_single(self, theta, key):
        key_x, key_flux = jax.random.split(key)
        return Measurement(x_hat=theta.x + self.sigma_x * jax.random.normal(key_x),
                           flux_hat=theta.flux + self.sigma_flux * jax.random.normal(key_flux))

    def logpdf_single(self, d, theta):
        norm = jax.scipy.stats.norm
        return (norm.logpdf(d.x_hat, theta.x, self.sigma_x)
                + norm.logpdf(d.flux_hat, theta.flux, self.sigma_flux))

camera = Camera()
priors = {"x": dist.Uniform(-1.0, 1.0), "flux": dist.Uniform(0.0, 10.0)}
sources = Source(x=jnp.array([-0.5, 0.0, 0.5]), flux=jnp.array([2.0, 3.0, 4.0]))
measured = camera.sample(sources, key=0)
posteriors = [camera.PE(measured[i], priors, seed=i, as_posterior=True,
                        num_warmup=300, num_samples=1000, summary=False)
              for i in range(3)]
```

## Creating a catalog

A catalog is created with a `RawSamples` setting, which lists the columns the catalog exports and
the default number of draws per event, `N`. Events are added with `add_event(name, posterior)`:

```{code-cell} python
catalog = ai.Catalog(raw=ai.RawSamples(columns=["x", "flux"], N=500))
for i, post in enumerate(posteriors):
    catalog.add_event(f"src_{i}", post, metadata={"true_flux": float(sources.flux[i])})
catalog
```

`metadata=` is merged into the posterior's own metadata:

```{code-cell} python
catalog["src_1"].metadata
```

The posteriors can also be passed to the constructor, as a dict from event names to posteriors.
`raw` may be a plain list of columns, in which case `N` defaults to 2000, and `seed` sets the
seed used for resampling (42 by default):

```{code-cell} python
ai.Catalog({"a": posteriors[0], "b": posteriors[1]}, raw=["x"], seed=7)
```

An exported column must be a parameter or an extra of every posterior:

```{code-cell} python
:tags: [raises-exception]

ai.Catalog({"a": posteriors[0]}, raw=["magnitude"])
```

A posterior must have exactly one draw axis. Flatten chains and draws into one axis before adding
it:

```{code-cell} python
:tags: [raises-exception]

two_chains = ai.PosteriorData(theta=Source(x=jnp.zeros((2, 500)), flux=jnp.ones((2, 500))))
catalog.add_event("chains", two_chains)
```

Adding an event under an existing name raises an error unless you pass `overwrite=True`, and
`remove_event(name)` removes an event together with all of its fits:

```{code-cell} python
:tags: [raises-exception]

catalog.add_event("src_0", posteriors[0])
```

`set_raw(raw)` changes the exported columns or `N` of an existing catalog.

## Accessing events

A catalog behaves like a read-only dict from event names to posteriors:

| expression | result |
|---|---|
| `len(catalog)` | number of events |
| `catalog.eventnames` | event names, in the order they were added |
| `name in catalog` | whether an event exists |
| `catalog[name]`, `catalog.posterior(name)` | the event's `PosteriorData` |
| `for name in catalog` | iterates over event names |
| `catalog.posteriors` | all posteriors, in event order |
| `catalog.source_key(name)` | the content hash of the event's posterior |

```{code-cell} python
len(catalog), catalog.eventnames, "src_1" in catalog, catalog["src_1"].batch_shape
```

## The raw data product

`dataproduct()` turns the catalog into arrays that a population likelihood can use. Every event is
resampled to `N` draws, and the events are stacked along the first axis:

```{code-cell} python
product = catalog.dataproduct()
{name: array.shape for name, array in product.items()}
```

`prior` is the prior density that the exported draws were sampled under, for the exported
columns only. It is computed with `PosteriorData.marginal_prior` from the `log_prior[...]` columns
that `PE` stores. Exporting only the flux gives the density of the $U(0, 10)$ flux prior:

```{code-cell} python
flux_only = ai.Catalog(dict(zip(catalog.eventnames, posteriors)), raw=["flux"])
flux_only.dataproduct(N=4)["prior"]
```

Keyword arguments override the catalog's settings for one call:

| argument | effect |
|---|---|
| `N=` | number of draws per event, instead of `raw.N` |
| `eventnames=` | which events to export, and in which order |
| `seed=` | the base seed for resampling, instead of `catalog.seed` |
| `representation=` | which representation to export; see [Representations](#representations) |

When `N` equals the number of stored draws, the draws are returned in their stored order.
Otherwise `N` of them are drawn at random, with replacement only if `N` is larger.

Each event's draws depend only on the base seed and the event's name. Exporting a subset of the
events, or exporting them in a different order, does not change any event's draws:

```{code-cell} python
alone = catalog.dataproduct(eventnames=["src_2"])
reordered = catalog.dataproduct(eventnames=["src_2", "src_0", "src_1"])
bool(jnp.all(alone["flux"][0] == product["flux"][2])), bool(jnp.all(reordered["flux"][0] == product["flux"][2]))
```

A different base seed gives different draws:

```{code-cell} python
bool(jnp.all(catalog.dataproduct(seed=1)["flux"] == product["flux"]))
```

## Source keys

`source_key(name)` is the `content_hash` of an event's posterior. Fits use it to notice that a
posterior has changed:

```{code-cell} python
key_before = catalog.source_key("src_0")
catalog.add_event("src_0", posteriors[0][:500], overwrite=True)
key_before == posteriors[0].content_hash(), catalog.source_key("src_0") == key_before
```

## Saving and loading

`save(root)` writes the catalog to a folder and returns its resolved path. A catalog is only
written when you call `save`. Here we save to a temporary folder and list the files:

```{code-cell} python
import tempfile
from pathlib import Path

tmp = tempfile.TemporaryDirectory()
root = Path(tmp.name) / "catalog"
catalog.save(root)
sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
```

`catalog.json` records the event order, the representations and the seed. Each representation has
a folder with its settings in `fitter.json` and one folder per event. For the raw representation,
each event folder holds the posterior's arrays in `posterior.h5` and its metadata, observed data
and type paths in `posterior.json`.

`ai.Catalog.load(root)` reads a catalog back. Each saved posterior records the import path of its
parameter type and of its observed data, for example `__main__:Source`. In the session that
defined the classes that path can be imported, but in any other session it cannot. Loading the
catalog in a new Python process fails:

```{code-cell} python
import subprocess
import sys

script = f"import astroinfer as ai; ai.Catalog.load({str(root)!r})"
result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
print(result.stderr.strip().splitlines()[-1])
```

Pass the classes with `types=`:

```{code-cell} python
loaded = ai.Catalog.load(root, types=[Source, Measurement])
loaded
```

Classes from an installed package are imported automatically and do not need `types=`.

The loaded catalog has the same events, posteriors, metadata, observed data and seed, so it
produces the same data product:

```{code-cell} python
(loaded.eventnames == catalog.eventnames,
 all(loaded.source_key(e) == catalog.source_key(e) for e in catalog),
 bool(jnp.all(loaded.dataproduct()["flux"] == catalog.dataproduct()["flux"])),
 loaded["src_1"].metadata["true_flux"],
 type(loaded["src_1"].observed_data).__name__)
```

Metadata is saved as JSON. A metadata value that cannot be written as JSON is dropped with a
warning when the catalog is saved:

```{code-cell} python
import warnings

odd = ai.Catalog({"a": posteriors[0]}, raw=["x"])
odd.add_event("a", posteriors[0], metadata={"note": object()}, overwrite=True)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    odd.save(Path(tmp.name) / "odd")
[str(w.message) for w in caught]
```

Saving again to the folder a catalog was saved to, or loaded from, only rewrites what changed:
new or replaced events, refitted representations, every `fitter.json` and `catalog.json`.
Folders of events or representations that were removed are deleted. Here only `src_0` is
rewritten:

```{code-cell} python
def stamps(root):
    return {e: (root / "representations/raw/events" / e / "posterior.h5").stat().st_mtime_ns
            for e in ["src_0", "src_1", "src_2"]}

before = stamps(root)
catalog.add_event("src_0", posteriors[0], overwrite=True)
catalog.save(root)
after = stamps(root)
{e: before[e] != after[e] for e in before}
```

`save` refuses to write into a folder that is not empty and is not a catalog:

```{code-cell} python
:tags: [raises-exception]

other = Path(tmp.name) / "other"
other.mkdir()
(other / "notes.txt").write_text("unrelated")
catalog.save(other)
```

## Representations

Besides the raw draws, a catalog can hold other *representations* of each posterior, such as a
fitted density. A representation is a pair of classes, a *fit* and a *fitter*, and the catalog
uses only the methods listed below. That is why any kind of representation can be fitted, checked,
exported, saved and loaded in the same way.

A fit holds the result for one event. Subclass `ai.representations.BaseFit`, which provides a
display and the default `stack`:

| member | what it does |
|---|---|
| `fit_key` (property) | the key the fit was made with |
| `dataproduct(*, N, seed, **options)` | this event's arrays |
| `save(directory)` | writes the fit into an empty folder |
| `load(directory, ctx)` (classmethod) | rebuilds the fit; `ctx.resolve(path)` resolves saved type paths using `types=` |
| `stack(products)` (classmethod) | combines the events' arrays; the default stacks arrays of equal shape along a new first axis |

A fitter holds the settings and makes one fit per event. Subclass
`ai.representations.BaseFitter`, which provides a display and an empty `dataproduct_options`:

| member | what it does |
|---|---|
| `kind` (class attribute) | a unique name for this kind of representation |
| `fit_type` (class attribute) | the fit class |
| `fit(posterior, *, eventname, source_key)` | fits one event |
| `fit_key(eventname, source_key)` | a hash of everything that determines the fit, including `source_key` |
| `dataproduct_options(eventname)` | settings that only affect the data product, such as `N` |
| `to_dict()`, `from_dict(d)` (classmethod) | the settings as JSON, so the fitter can be saved |

This representation stores the mean and standard deviation of each column. `config_hash` hashes a
JSON-like dict, and `ai.io.write_json` and `ai.io.read_json` write and read JSON files that may
contain NumPy arrays:

```{code-cell} python
from typing import ClassVar

from astroinfer.representations import BaseFit, BaseFitter, config_hash

class SummaryFit(BaseFit):
    def __init__(self, columns, mean, sd, *, fit_key):
        self.columns, self.mean, self.sd = list(columns), np.asarray(mean), np.asarray(sd)
        self._fit_key = fit_key

    @property
    def fit_key(self):
        return self._fit_key

    def dataproduct(self, *, N, seed):
        return {"mean": jnp.asarray(self.mean), "sd": jnp.asarray(self.sd)}

    def save(self, directory):
        ai.io.write_json(directory / "summary.json", {"columns": self.columns, "mean": self.mean,
                                                      "sd": self.sd, "fit_key": self.fit_key})

    @classmethod
    def load(cls, directory, ctx):
        s = ai.io.read_json(directory / "summary.json")
        return cls(s["columns"], s["mean"], s["sd"], fit_key=s["fit_key"])

@ai.register
class SummaryFitter(BaseFitter):
    kind: ClassVar[str] = "summary"
    fit_type: ClassVar[type] = SummaryFit

    def __init__(self, columns):
        self.columns = list(columns)

    def fit_key(self, eventname, source_key):
        return config_hash({"kind": self.kind, "columns": self.columns, "source": source_key})

    def fit(self, posterior, *, eventname, source_key):
        values = posterior.variables(self.columns)
        x = np.stack([np.asarray(values[c]) for c in self.columns], axis=-1)
        return SummaryFit(self.columns, x.mean(axis=0), x.std(axis=0),
                          fit_key=self.fit_key(eventname, source_key))

    def to_dict(self):
        return {"columns": self.columns}

    @classmethod
    def from_dict(cls, d):
        return cls(d["columns"])
```

`@ai.register` records the fitter under its `kind`, so that `Catalog.load` can rebuild it. A
package registers its kinds through the `astroinfer.representations` entry point instead, so that
`load` finds them without the package being imported first. `astroinfer_tgmm` does this:

```toml
[project.entry-points."astroinfer.representations"]
tgmm = "astroinfer_tgmm:TGMMFitter"
```

### Fitting

`register_representation(name, fitter)` attaches a fitter under a name of your choice, so one
catalog can hold several representations of the same kind. `missing(name)` lists the events
without a fit, and `fit(name)` fits them and returns their names:

```{code-cell} python
catalog.register_representation("summary", SummaryFitter(["x", "flux"]))
catalog.missing("summary"), catalog.fit("summary"), catalog.fit("summary")
```

The second call fits nothing, because every fit exists and is up to date. `get_fit(event, name)`
returns one fit, and `fits(name)` returns a dict of all of them:

```{code-cell} python
catalog.get_fit("src_1", "summary").mean, list(catalog.fits("summary"))
```

| call | what it fits |
|---|---|
| `fit(name)` | every event whose fit is missing or stale |
| `fit(name, eventnames=[...])` | the same, among the listed events only |
| `fit(name, refit=True)` | every event, whatever its state |
| `fit(name, errors="warn")` | as above, but an event whose fit raises is skipped with a warning and keeps its previous fit |

The raw representation is not fitted, and `add_representation(event, name, fit)` stores a fit
that was made outside the catalog.

### Stale fits

A fit is *stale* when its `fit_key` differs from the key its fitter would compute now. Replacing a
posterior changes its source key, so its fits become stale:

```{code-cell} python
catalog.add_event("src_2", posteriors[2][:500], overwrite=True)
catalog.is_stale("src_2", "summary"), catalog.stale("summary")
```

Settings that only affect the data product belong in `dataproduct_options` rather than in
`fit_key`, so changing them leaves the fits as they are. Settings that enter `fit_key` do make the
fits stale. Here the fitter is replaced with one that summarises `x` only:

```{code-cell} python
catalog.register_representation("summary", SummaryFitter(["x"]), overwrite=True)
catalog.stale("summary")
```

`dataproduct` refuses to export stale fits unless you pass `allow_stale=True`. It always refuses
when an event has no fit at all:

```{code-cell} python
:tags: [raises-exception]

catalog.dataproduct(representation="summary")
```

```{code-cell} python
{name: array.shape for name, array in
 catalog.dataproduct(representation="summary", allow_stale=True).items()}
```

After a refit the data product has one column, matching the new settings:

```{code-cell} python
catalog.fit("summary")
{name: array.shape for name, array in catalog.dataproduct(representation="summary").items()}
```

`remove_representation(name)` removes a representation and its fits. The raw representation
cannot be removed.

Tutorial `tutorials/09_your_own_representation.py` builds a Gaussian representation with
per-event settings, and the `astroinfer_tgmm` package provides truncated Gaussian mixtures.

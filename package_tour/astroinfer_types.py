import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    import tempfile
    from dataclasses import dataclass
    from pathlib import Path
    from typing import ClassVar

    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd
    from numpyro.distributions import constraints

    import astroinfer as ai
    from astroinfer import MeasurementProcess, batched_method, parameter, parameter_dataclass
    from astroinfer.detection import DetectionTrigger, ThresholdTrigger
    from astroinfer.display import html_table, kv_table
    from astroinfer.generative import (
        GenerativeModel,
        TruncatedNormal,
        Uniform,
        conditional,
        mixture,
        product,
    )
    from astroinfer.representations import BaseFit, BaseFitter, config_hash


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # astroinfer's types

    This notebook goes through the types in astroinfer and the methods you use on them. The
    running example is a source with a position and a flux, measured once by a camera with
    Gaussian noise.

    Every astroinfer object displays itself when it is the last expression in a cell. Click on a
    display to expand it.

    ## Parameter types

    A parameter type is a class decorated with `@parameter_dataclass`, with each field declared
    by `parameter(...)`. The `name` and `latex` arguments set how the object is displayed.
    """)
    return


@app.class_definition
@parameter_dataclass(name="Source", latex=r"\theta")
class Source:
    x: float = parameter(name="Position [arcsec]", latex="x")
    flux: float = parameter(name="Flux", latex="f")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Objects are created like any dataclass:
    """)
    return


@app.cell
def _():
    src = Source(x=0.2, flux=3.0)
    src
    return (src,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `Source.parameters` lists the field names, and `to_dict()` returns the values:
    """)
    return


@app.cell
def _(src):
    Source.parameters, src.to_dict()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The objects are frozen, so assigning to a field raises an error:
    """)
    return


@app.cell
def _(src):
    try:
        src.x = 1.0
    except Exception as _err:
        _message = f"{type(_err).__name__}: {_err}"
    _message
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Batches

    A field can hold an array instead of a number. The object then holds a *batch* of sources,
    and `batch_shape` is the shape of that batch. Fields broadcast against each other as NumPy
    arrays do, so a single flux is shared by all three sources here:
    """)
    return


@app.cell
def _():
    sources = Source(x=jnp.array([-0.5, 0.0, 0.5]), flux=2.0)
    sources
    return (sources,)


@app.cell
def _(sources, src):
    sources.batch_shape, src.batch_shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The flux is still stored as a single number. `broadcast()` returns a copy in which every
    field has the full batch shape, and `reshape_batch(shape)` changes the batch shape:
    """)
    return


@app.cell
def _(sources):
    sources.flux, sources.broadcast().flux, sources.reshape_batch((3, 1)).batch_shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Indexing with an integer, a slice or a boolean mask selects from the batch and returns another
    `Source`. Indexing with a field name returns that field:
    """)
    return


@app.cell
def _(sources):
    sources[1].to_dict(), sources[sources.x >= 0].batch_shape, sources["x"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Building objects from tables

    `from_dict` builds an object from a dict of arrays or from a pandas DataFrame, with one row
    per batch element. Columns that are not fields are ignored:
    """)
    return


@app.cell
def _():
    table = pd.DataFrame({"x": [0.1, 0.2], "flux": [1.0, 2.0], "label": ["a", "b"]})
    Source.from_dict(table).to_dict()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Arithmetic and JAX

    Two objects of the same type can be added or subtracted field by field, and an object can be
    multiplied by a number:
    """)
    return


@app.cell
def _(src):
    (src + src).to_dict(), (src * 0.5).to_dict()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Every parameter type is registered as a JAX *pytree*, so JAX transformations accept a
    `Source` and return one. The gradient of a function of a `Source` is a `Source` holding the
    derivative with respect to each field:
    """)
    return


@app.cell
def _(src):
    def brightness(s):
        return s.flux * jnp.exp(-s.x**2)

    _gradient = jax.grad(brightness)(src)
    type(_gradient).__name__, _gradient.to_dict(), jax.jit(brightness)(src)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Methods

    A parameter type can have ordinary methods. This one evaluates a Gaussian profile on a grid
    of positions, so for a single peak it returns one value per grid point:
    """)
    return


@app.class_definition
@parameter_dataclass(name="Peak")
class Peak:
    centre: float = parameter(name="Centre", latex=r"\mu")
    width: float = parameter(name="Width", latex=r"\sigma")

    def profile(self, grid):
        z = (grid - self.centre) / self.width
        return jnp.exp(-0.5 * z**2)


@app.cell
def _():
    grid = jnp.linspace(-1.0, 1.0, 5)
    Peak(centre=0.0, width=0.3).profile(grid)
    return (grid,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For a batch of three peaks, `self.centre` has shape `(3,)` while `grid` has shape `(5,)`, and
    the two do not broadcast:
    """)
    return


@app.cell
def _(grid):
    three_centres = jnp.array([-0.5, 0.0, 0.5])
    try:
        Peak(centre=three_centres, width=0.3).profile(grid)
    except TypeError as _err:
        _message = f"TypeError: {_err}"
    _message
    return (three_centres,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The decorator `@batched_method` fixes this. It applies `jax.vmap` once for each batch axis,
    so the method body sees one peak at a time, and the batch axes are added in front of the
    result. Here it is applied in a subclass:
    """)
    return


@app.class_definition
@parameter_dataclass(name="Peak")
class BatchedPeak(Peak):
    @batched_method
    def profile(self, grid):
        z = (grid - self.centre) / self.width
        return jnp.exp(-0.5 * z**2)


@app.cell
def _(grid, three_centres):
    BatchedPeak(centre=three_centres, width=0.3).profile(grid).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Core axes

    Some fields are arrays even for a single object, such as a spectrum with one value per
    channel. Passing `index=` to `parameter` names the trailing axes of such a field. These are
    *core axes*: they belong to each object and are not part of `batch_shape`. `index_latex`
    gives the symbol used for the axis in the display.
    """)
    return


@app.class_definition
@parameter_dataclass(name="Spectrum", latex="s")
class Spectrum:
    flux: float = parameter(name="Flux per channel", latex="f", index=("channel",),
                            index_latex=("c",))


@app.cell
def _():
    one_spectrum = Spectrum(flux=jnp.ones(4))
    three_spectra = Spectrum(flux=jnp.ones((3, 4)))
    one_spectrum.batch_shape, three_spectra.batch_shape, three_spectra[0].flux.shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Supports and the latent space

    `parameter` also takes `support=`, a numpyro constraint describing the values a field may take.
    The radius of a disk has to be positive:
    """)
    return


@app.class_definition
@parameter_dataclass(name="Disk")
class Disk:
    x: float = parameter(name="Position", latex="x")
    radius: float = parameter(name="Radius", latex="r", support=constraints.positive)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Optimisers and samplers often work best on unconstrained real numbers. `to_unconstrained()`
    maps each field onto the whole real line and returns an object of the same type, and
    `from_unconstrained` maps it back. A positive field is mapped with a logarithm, and a field
    without a support is left as it is:
    """)
    return


@app.cell
def _():
    disk = Disk(x=0.2, radius=3.0)
    z = disk.to_unconstrained()
    z.to_dict(), Disk.from_unconstrained(z).to_dict()
    return (disk,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The mapping is a `LatentSpace`. You can build one yourself to replace some of the declared
    supports, for example to keep `x` between $-1$ and $1$:
    """)
    return


@app.cell
def _():
    box = ai.LatentSpace(Disk, {"x": constraints.interval(-1.0, 1.0)})
    box
    return (box,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Pass the space to `to_unconstrained` to use it. If you move a probability density into the
    latent space, add the log Jacobian of the mapping, `log_abs_det_jacobian(z)`:
    """)
    return


@app.cell
def _(box, disk):
    z_box = disk.to_unconstrained(box)
    z_box.to_dict(), box.log_abs_det_jacobian(z_box)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Displays

    The displays above come from `ai.RichDisplay`, and any class can inherit from it. It uses up
    to three methods: `_display_title()` returns the title, `_display_math()` returns LaTeX shown
    next to the title (or `None`), and `_display_body()` returns the HTML shown when the display is
    expanded. `astroinfer.display` has helpers for the body: `kv_table(dict)` makes a two-column
    table and `html_table(header, rows)` makes a table with a header row.
    """)
    return


@app.class_definition
class ObservingRun(ai.RichDisplay):
    def __init__(self, telescope, nights, settings):
        self.telescope, self.nights, self.settings = telescope, nights, dict(settings)

    def _display_title(self):
        return f"Observing run on {self.telescope}"

    def _display_math(self):
        return rf"n_\mathrm{{nights}} = {self.nights}"

    def _display_body(self):
        return kv_table(self.settings)


@app.cell
def _():
    ObservingRun("the tour telescope", 3, {"filter": "r", "exposure [s]": 30})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Measurement processes

    The data are a second parameter type. Our camera measures the position and the flux of a
    source once:
    """)
    return


@app.class_definition
@parameter_dataclass(name="Measurement", latex="d")
class Measurement:
    x_hat: float = parameter(name="Measured position [arcsec]", latex=r"\hat x")
    flux_hat: float = parameter(name="Measured flux", latex=r"\hat f")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A measurement process describes how data are produced from parameters, $\mathcal{L}(d\mid\theta)$. You
    write one by subclassing `MeasurementProcess` and filling in:

    * `parameter_type` and `data_type`, the two parameter types;
    * `sample_single(self, theta, key)`, which draws one `Measurement` for one `Source`. JAX
      random functions take an explicit *key*, and `jax.random.split` turns one key into two
      independent keys;
    * `logpdf_single(self, d, theta)`, which returns $\log \mathcal{L}(d\mid\theta)$ for one measurement
      and one source;
    * optionally `derived_parameters(self, theta)`, which receives a dict of arrays and returns
      extra quantities computed from the parameters. They are added to the output of parameter
      estimation.

    Measurement processes are dataclasses, so settings such as the noise levels are ordinary
    fields. `display_name` sets the title of the display.
    """)
    return


@app.class_definition
@dataclass
class Camera(MeasurementProcess):
    sigma_x: float = 0.1
    sigma_flux: float = 0.3

    display_name = "Camera"
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

    def derived_parameters(self, theta):
        return {"magnitude": -2.5 * np.log10(theta["flux"])}


@app.cell
def _():
    camera = Camera()
    camera
    return (camera,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Simulating data

    `sample(theta, key=...)` calls `sample_single` for every element of the batch. An integer key
    makes the draw reproducible:
    """)
    return


@app.cell
def _(camera, src):
    d = camera.sample(src, key=0)
    d.to_dict()
    return (d,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A batch of sources gives a batch of measurements. `N=` adds a trailing batch axis with that
    many independent measurements of each source:
    """)
    return


@app.cell
def _(camera, sources, src):
    (camera.sample(sources, key=1).batch_shape,
     camera.sample(src, key=2, N=4).batch_shape,
     camera.sample(sources, key=3, N=4).batch_shape)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Without `key=`, the camera uses a key of its own and advances it, so repeated calls give
    different noise. `seed(n)` resets that key:
    """)
    return


@app.cell
def _(camera, src):
    camera.seed(5)
    _first, _second = camera.sample(src), camera.sample(src)
    camera.seed(5)
    _again = camera.sample(src)
    float(_first.x_hat), float(_second.x_hat), float(_again.x_hat)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Evaluating the likelihood

    `logpdf(d, theta)` evaluates $\log \mathcal{L}(d\mid\theta)$. The batch shapes of `d` and `theta`
    broadcast against each other, so one measurement against three sources gives three values:
    """)
    return


@app.cell
def _(camera, d, sources, src):
    camera.logpdf(d, src), camera.logpdf(d, sources)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With `outer=True`, every source is evaluated against every measurement. The source axes come
    first in the result:
    """)
    return


@app.cell
def _(camera, sources):
    _two_measurements = camera.sample(sources[:2], key=4)
    camera.logpdf(_two_measurements, sources, outer=True).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Parameter estimation

    `PE(d, priors)` samples the posterior $p(\theta\mid d)$ with NUTS. `priors` holds a numpyro
    distribution for every field. Other keyword arguments go to the sampler. Here they ask for
    fewer warm-up steps and draws than the default, and turn off the summary the sampler prints.
    """)
    return


@app.cell
def _(camera, d):
    pe_priors = {"x": dist.Uniform(-1.0, 1.0), "flux": dist.Uniform(0.0, 10.0)}
    pe_table = camera.PE(d, pe_priors, seed=0, num_warmup=300, num_samples=1000, summary=False)
    pe_table.head()
    return (pe_priors,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each row is one draw. Next to `x` and `flux` are `magnitude`, from `derived_parameters`;
    `log_likelihood`, the value of $\log \mathcal{L}(d\mid\theta)$; `prior`, the prior density of the draw;
    and `log_prior[x]` and `log_prior[flux]`, the log prior density of each field separately.

    ## Posterior data

    With `as_posterior=True`, `PE` returns a `PosteriorData` instead of a DataFrame:
    """)
    return


@app.cell
def _(camera, d, pe_priors):
    posterior = camera.PE(d, pe_priors, seed=0, as_posterior=True,
                          num_warmup=300, num_samples=1000, summary=False)
    posterior
    return (posterior,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `theta` holds the draws as a batch of `Source` objects. The other columns are split into
    `sample_stats` (the log likelihood and the priors) and `extras` (everything else, here the
    magnitude). `metadata` describes the run, and `observed_data` is the measurement the posterior
    is conditioned on:
    """)
    return


@app.cell
def _(posterior):
    (posterior.theta.batch_shape, list(posterior.sample_stats), list(posterior.extras),
     posterior.metadata, posterior.observed_data.to_dict())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Indexing with a column name returns that column, whichever group it is in. Indexing with an
    integer, a slice or a boolean mask selects draws and returns a smaller `PosteriorData`:
    """)
    return


@app.cell
def _(posterior):
    (posterior["flux"].shape, float(posterior["magnitude"].mean()),
     posterior[posterior["flux"] > 3.0].batch_shape)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `variables(names)` collects several columns into a dict. `marginal_prior(columns)` returns
    the prior density of each draw for some of the fields only, built from the `log_prior[...]`
    columns. For `x` it is the density of $U(-1, 1)$:
    """)
    return


@app.cell
def _(posterior):
    list(posterior.variables(["x", "magnitude"])), posterior.marginal_prior(["x"])[:3]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `content_hash()` is a fingerprint of all the per-draw arrays, so it changes when the draws
    change:
    """)
    return


@app.cell
def _(posterior):
    posterior.content_hash()[:16], posterior[:500].content_hash()[:16]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `to_dataframe()` gives back the table that `PE` returns by default, and
    `PosteriorData.from_dataframe(table, Source)` turns such a table, for example one from another
    sampler, into a `PosteriorData`. The round trip keeps the draws:
    """)
    return


@app.cell
def _(posterior):
    _round_trip = ai.PosteriorData.from_dataframe(posterior.to_dataframe(), Source)
    _round_trip.content_hash() == posterior.content_hash()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A `PosteriorData` can also be created directly. Only `theta` is required:
    """)
    return


@app.cell
def _():
    ai.PosteriorData(
        theta=Source(x=jnp.zeros(5), flux=jnp.linspace(1.0, 2.0, 5)),
        sample_stats={"prior": jnp.full(5, 0.05)},
        metadata={"origin": "written by hand"},
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Population models

    A population model describes how parameters are distributed over many sources,
    $\pi(\theta\mid\Lambda)$, where $\Lambda$ are *hyperparameters*. Population models are built
    from the pieces in `astroinfer.generative`.

    The smallest model has a single *leaf*, a one-dimensional distribution for one named variable.
    The parameters of a leaf are numbers or the names of hyperparameters. Here the flux follows a
    normal distribution with mean `"mu"` and width `"sigma"`, truncated to the range 0 to 20.
    `GenerativeModel(root, order)` takes the distribution and the order in which its variables
    are drawn, and `sample(Lambda, key=, shape=)` returns a dict of arrays:
    """)
    return


@app.cell
def _():
    flux_model = GenerativeModel(TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0), order=["flux"])
    Lambda = {"mu": 3.0, "sigma": 1.0}
    flux_draws = flux_model.sample(Lambda, key=jax.random.key(0), shape=(5,))
    flux_draws
    return Lambda, flux_draws, flux_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `log_prob(values, Lambda)` evaluates the log density of such a dict:
    """)
    return


@app.cell
def _(Lambda, flux_draws, flux_model):
    flux_model.log_prob(flux_draws, Lambda)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `product` combines leaves for different variables. Adding a uniform distribution for the
    position gives a model for a whole `Source`, so its draws can be passed to
    `Source.from_dict`:
    """)
    return


@app.cell
def _(Lambda):
    population = GenerativeModel(
        product([Uniform("x", -1.0, 1.0), TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0)]),
        order=["x", "flux"],
    )
    pop_sources = Source.from_dict(population.sample(Lambda, key=jax.random.key(1), shape=(1000,)))
    pop_sources.batch_shape
    return pop_sources, population


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each variable is drawn by passing a uniform random number through the inverse of its
    cumulative distribution function. `sample` also accepts these uniform numbers directly, as
    `seeds={variable: array}`. With the seeds held fixed, the draws are a smooth function of
    $\Lambda$, so quantities computed from them can be differentiated:
    """)
    return


@app.cell
def _(Lambda, population):
    seeds = {"x": jax.random.uniform(jax.random.key(2), (1000,)),
             "flux": jax.random.uniform(jax.random.key(3), (1000,))}

    def mean_flux(L):
        return population.sample(L, seeds=seeds)["flux"].mean()

    jax.grad(mean_flux)(Lambda)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A distribution parameter can depend on a variable drawn earlier, through
    `conditional(fn, on=)`. The function `fn(x, L)` receives the variables drawn so far and the
    hyperparameters. A variable that the model does not draw itself is declared with `external=`
    and passed in at sampling time with `conditioned_on=`. Here sources further from the centre
    are fainter:
    """)
    return


@app.cell
def _(Lambda):
    flux_given_x = GenerativeModel(
        TruncatedNormal("flux", conditional(lambda x, L: L["mu"] - x["x"] ** 2, on="x"),
                        "sigma", 0.0, 20.0),
        order=["flux"], external=["x"],
    )
    flux_given_x.sample(Lambda, key=jax.random.key(4), shape=(3,),
                        conditioned_on={"x": jnp.array([-1.0, 0.0, 1.0])})
    return (flux_given_x,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `GenerativeModel` checks the order when it is built. Drawing `flux` before the `x` it depends
    on raises an error:
    """)
    return


@app.cell
def _(flux_given_x):
    try:
        GenerativeModel(product([Uniform("x", -1.0, 1.0), flux_given_x.root]), order=["flux", "x"])
    except ValueError as _err:
        _message = f"ValueError: {_err}"
    _message
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `mixture(models, weights)` combines models of the same variables. `weights` names one
    hyperparameter for every component except the last, whose weight is one minus the others:
    """)
    return


@app.cell
def _(Lambda):
    two_groups = GenerativeModel(
        mixture([TruncatedNormal("flux", 1.0, 0.2, 0.0, 20.0),
                 TruncatedNormal("flux", "mu", "sigma", 0.0, 20.0)], weights=["f_faint"]),
        order=["flux"],
    )
    two_groups.sample({**Lambda, "f_faint": 0.3}, key=jax.random.key(5), shape=(6,))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Detection

    To simulate a survey, draw sources from the population and measure each one:
    """)
    return


@app.cell
def _(camera, pop_sources):
    pop_data = camera.sample(pop_sources, key=6)
    pop_data.batch_shape
    return (pop_data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A detection trigger decides which measurements are detected. `ThresholdTrigger(threshold,
    statistic=)` detects a measurement when one of its fields is above the threshold.
    `detect(d, rng)` takes a NumPy random generator and returns a boolean mask over the batch,
    which can be used to index the data:
    """)
    return


@app.cell
def _(pop_data):
    hard = ThresholdTrigger(2.5, statistic="flux_hat")
    _mask = hard.detect(pop_data, np.random.default_rng(0))
    int(_mask.sum()), pop_data[_mask].batch_shape
    return (hard,)


@app.cell
def _(hard):
    hard
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For any other rule, subclass `DetectionTrigger` and implement `detection_probability(d)`,
    which returns a detection probability for each measurement in the batch. A trigger works on the
    whole batch at once with NumPy, because detection is never differentiated. You can also
    implement `selection_statistic(d)`, the number that detection is based on. Here the detection
    probability rises smoothly around the threshold:
    """)
    return


@app.class_definition
class SoftFluxLimit(DetectionTrigger):
    display_name = "Soft flux limit"

    def __init__(self, threshold, width):
        self.threshold, self.width = float(threshold), float(width)

    def selection_statistic(self, d):
        return np.asarray(d["flux_hat"], dtype=np.float64)

    def detection_probability(self, d):
        return 1.0 / (1.0 + np.exp(-(self.selection_statistic(d) - self.threshold) / self.width))


@app.cell
def _(pop_data):
    soft = SoftFluxLimit(threshold=2.5, width=0.2)
    int(soft.detect(pop_data, np.random.default_rng(0)).sum())
    return (soft,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `injections(measurement, draws, rng)` takes a DataFrame of parameter draws, with a `prior`
    column holding the density each draw was taken from. It measures every draw once and applies
    the trigger. The returned table adds the measured data as `observed_<field>`, the selection
    statistic, the detection probability and whether the draw was detected:
    """)
    return


@app.cell
def _(Lambda, camera, population, soft):
    _draws = {k: np.asarray(v) for k, v in
              population.sample(Lambda, key=jax.random.key(7), shape=(2000,)).items()}
    draw_table = pd.DataFrame(_draws)
    draw_table["prior"] = np.exp(np.asarray(population.log_prob(_draws, Lambda)))
    soft.injections(camera, draw_table, np.random.default_rng(1)).head()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Likelihoods and the sampler

    The sampler behind `PE` comes from the gravpop package. It accepts any object with a
    `logpdf(params)` method, and `ai.run_sampler(likelihood, priors, ...)` runs it and returns the
    sampler. `PE` uses `ai.FixedDataLikelihood(measurement, d)`, whose `logpdf(theta)` is
    `measurement.logpdf(d, theta)`:
    """)
    return


@app.cell
def _(camera, d, src):
    ai.FixedDataLikelihood(camera, d).logpdf(src), camera.logpdf(d, src)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `ai.PopulationLikelihood(measurement, data, pop_model, N=, seed=)` is the likelihood of the
    hyperparameters $\Lambda$, given measurements of a batch of independent sources. For each
    source it averages $\mathcal{L}(d_i\mid\theta)$ over `N` draws from the population. The uniform seeds of
    those draws are fixed by `seed`, so the result is differentiable in $\Lambda$. Here we use the
    first 50 simulated measurements, before any detection cut:
    """)
    return


@app.cell
def _(Lambda, camera, pop_data, population):
    likelihood = ai.PopulationLikelihood(camera, pop_data[:50], population, N=500, seed=0)
    likelihood.logpdf(Lambda)
    return (likelihood,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `event_logpdf_and_neff(Lambda)` returns the log of each source's average and its effective
    number of draws. A small effective number of draws means the average is noisy and `N` should
    be larger:
    """)
    return


@app.cell
def _(Lambda, likelihood):
    _log_z, _neff = likelihood.event_logpdf_and_neff(Lambda)
    _log_z.shape, float(jnp.median(_neff)), float(_neff.min())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `run_sampler` samples $\Lambda$, given priors for the hyperparameters. The sampler's `samples`
    attribute is a DataFrame of draws:
    """)
    return


@app.cell
def _(likelihood):
    hyper_priors = {"mu": dist.Uniform(0.0, 6.0), "sigma": dist.Uniform(0.2, 3.0)}
    hyper = ai.run_sampler(likelihood, hyper_priors, seed=0,
                           num_warmup=300, num_samples=1000, summary=False)
    hyper.samples[["mu", "sigma"]].describe().loc[["mean", "std"]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `camera.population_inference(data, pop_model, priors, N=...)` builds the
    `PopulationLikelihood` and calls `run_sampler` in one step. To change the likelihood, for
    example to correct for detection, subclass `PopulationLikelihood` and override
    `logpdf(Lambda)`; tutorial 7 does this. Tutorial 8 introduces `Proposal`, which draws the
    parameters of each source from a distribution close to its likelihood instead of from the
    population.

    ## Catalogs

    A `Catalog` stores one `PosteriorData` per event. It is created with a `RawSamples` setting,
    which gives the columns the catalog exports and how many draws it exports per event.
    `add_event(name, posterior, metadata=)` adds an event. Here we run `PE` on measurements of
    three sources:
    """)
    return


@app.cell
def _(camera, pe_priors, sources):
    _measured = camera.sample(sources, key=8)
    catalog = ai.Catalog(raw=ai.RawSamples(columns=["x", "flux"], N=500))
    for _i in range(3):
        _post = camera.PE(_measured[_i], pe_priors, seed=_i, as_posterior=True,
                          num_warmup=300, num_samples=1000, summary=False)
        catalog.add_event(f"src_{_i}", _post, metadata={"true_x": float(sources.x[_i])})
    catalog
    return (catalog,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A catalog behaves like a read-only dict from event names to posteriors:
    """)
    return


@app.cell
def _(catalog):
    len(catalog), catalog.eventnames, "src_1" in catalog, catalog["src_1"].batch_shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `dataproduct()` resamples every event to the same number of draws and stacks them, with the
    events along the first axis. `prior` is the prior density of the exported columns, computed
    with `marginal_prior`:
    """)
    return


@app.cell
def _(catalog):
    {name: array.shape for name, array in catalog.dataproduct().items()}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `source_key(name)` is the `content_hash` of an event's posterior:
    """)
    return


@app.cell
def _(catalog):
    catalog.source_key("src_1") == catalog["src_1"].content_hash()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Saving and loading

    `save(root)` writes the catalog to a folder, and `ai.Catalog.load(root, types=[...])` reads
    it back. Each saved posterior records where its classes can be imported from. Classes defined
    in a notebook cannot be imported from there, so `load` is given them with `types=`. We save
    to a temporary folder:
    """)
    return


@app.cell
def _(catalog):
    with tempfile.TemporaryDirectory() as _tmp:
        _root = Path(_tmp) / "catalog"
        catalog.save(_root)
        _files = sorted(str(p.relative_to(_root)) for p in _root.rglob("*") if p.is_file())
        _loaded = ai.Catalog.load(_root, types=[Source, Measurement])
    _files[:5], _loaded.eventnames
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Representations

    Besides the raw draws, a catalog can hold other *representations* of each posterior, such as a
    fitted density. A representation consists of two classes.

    The *fit* stores the result for one event. It subclasses `BaseFit`, which is a `RichDisplay`,
    and needs a `fit_key` property, a `dataproduct(N, seed)` method that returns this event's
    arrays, and `save(directory)` and `load(directory, ctx)` to write itself to a folder and read
    itself back. `ai.io.write_json` and `ai.io.read_json` write and read JSON files and accept
    NumPy arrays. This fit stores the mean and standard deviation of each column:
    """)
    return


@app.class_definition
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

    def _display_title(self):
        return "Summary fit"

    def _display_body(self):
        rows = [[c, f"{m:.3f}", f"{s:.3f}"] for c, m, s in zip(self.columns, self.mean, self.sd)]
        return html_table(["column", "mean", "sd"], rows)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The *fitter* holds the settings and makes one fit per event. It subclasses `BaseFitter` and
    has:

    * `kind`, a unique name for this kind of representation, and `fit_type`, the fit class;
    * `fit(posterior, eventname=, source_key=)`, which makes the fit for one event;
    * `fit_key(eventname, source_key)`, a hash of everything that determines the fit, including
      the posterior's `source_key`. `config_hash` hashes a dict of settings;
    * `to_dict()` and `from_dict(d)`, which store the settings so the fitter can be saved.

    The decorator `@ai.register` records the `kind`, so that `Catalog.load` knows which class to
    rebuild.
    """)
    return


@app.class_definition
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `register_representation(name, fitter)` attaches a fitter to the catalog. `missing(name)`
    lists the events without a fit, and `fit(name)` fits them and returns their names:
    """)
    return


@app.cell
def _(catalog):
    catalog.register_representation("summary", SummaryFitter(["x", "flux"]))
    _missing = catalog.missing("summary")
    _fitted = catalog.fit("summary")
    _missing, _fitted
    return


@app.cell
def _(catalog):
    catalog.get_fit("src_1", "summary")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A fit is *stale* when its `fit_key` no longer matches the one the fitter would compute now.
    Replacing a posterior with `add_event(..., overwrite=True)` changes its `source_key`, so its
    fit becomes stale, and the next `fit` call refits only that event:
    """)
    return


@app.cell
def _(catalog):
    catalog.add_event("src_0", catalog["src_0"][:500], overwrite=True)
    _stale = catalog.stale("summary")
    _refitted = catalog.fit("summary")
    _stale, _refitted
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `dataproduct(representation=name)` stacks every fit's data product along a leading event
    axis:
    """)
    return


@app.cell
def _(catalog):
    {name: array.shape for name, array in catalog.dataproduct(representation="summary").items()}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The numbered tutorials in `tutorials/` apply these types to concrete problems and explain the
    statistics.
    """)
    return


if __name__ == "__main__":
    app.run()

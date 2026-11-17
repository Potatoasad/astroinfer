import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import jax

    jax.config.update("jax_enable_x64", True)

    from dataclasses import dataclass, replace

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro.distributions as dist
    import pandas as pd

    import astroinfer as ai
    from astroinfer import MeasurementProcess, PosteriorData, parameter, parameter_dataclass
    from astroinfer.display import html_table, kv_table

    import _style as style

    style.apply()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. The doctor's office

    A patient walks into a doctor's office. The doctor is in a hurry and writes down only one
    number, the patient's BMI, $w/h^2$, measured with a little noise. What can we say about the
    patient's weight $w$ and height $h$?

    As we'll see, one number does not say much. The office has doorways of different heights,
    though, and the receptionist notes whether the patient ducks under each one. Those notes turn
    out to carry a lot of information.

    Along the way the notebook covers derived parameters (quantities like BMI that are functions
    of $\theta$), the `PosteriorData` object that `.PE(..., as_posterior=True)` returns, and custom
    displays for measurement processes and for any object of your own.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src=mo.notebook_dir() / "images" / "04_doctors_office.svg",
             alt="A patient on a scale next to a height rod, and their chart",
             width=640)
    return


@app.class_definition
@parameter_dataclass(name="Person", latex=r"\theta")
class Person:
    weight: float = parameter(name="Weight [kg]", latex="w")
    height: float = parameter(name="Height [m]", latex="h")

    def bmi(self):
        return self.weight / self.height**2


@app.class_definition
@parameter_dataclass(name="Chart", latex="d")
class Chart:
    BMI: float = parameter(name="Measured BMI", latex=r"\widehat{\mathrm{BMI}}")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The measurement

    The measurement model is $\widehat{\mathrm{BMI}}\sim\mathcal N(w/h^2,\ \sigma^2)$. Besides
    `sample_single` and `logpdf_single`, the class below overrides three optional methods:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | method | description |
    |---|---|
    | `derived_parameters(self, theta)` | takes `{name: array}` for $\theta$ and returns `{name: array}` of extra quantities, which are added to the PE output and to `with_derived(table)` |
    | `_display_equations(self)` | a list of LaTeX lines shown under *Model* in the display |
    | `_display_config(self)` | `{label: value}` shown under *Configuration* (by default, the dataclass fields) |

    `derived_parameters` is applied to tables of draws, so it receives plain arrays rather than a
    `Person`. Write it with array operations.
    """)
    return


@app.class_definition
@dataclass
class MeasureBMI(MeasurementProcess):
    sigma_bmi: float = 0.5

    display_name = "BMI measurement"
    parameter_type = Person
    data_type = Chart

    def sample_single(self, theta, key):
        return Chart(BMI=theta.bmi() + self.sigma_bmi * jax.random.normal(key))

    def logpdf_single(self, d, theta):
        return jax.scipy.stats.norm.logpdf(d.BMI, theta.bmi(), self.sigma_bmi)

    def derived_parameters(self, theta):
        return {"BMI": theta["weight"] / theta["height"] ** 2,
                "weight_lb": 2.20462 * theta["weight"]}

    def _display_equations(self):
        return [r"\widehat{\mathrm{BMI}} \sim \mathcal{N}\!\left(w/h^2,\ \sigma_\mathrm{BMI}^2\right)"]

    def _display_config(self):
        return {"BMI noise σ": self.sigma_bmi}


@app.cell
def _():
    office = MeasureBMI(sigma_bmi=0.5)
    office
    return (office,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the expanded display, the *Model* line and the *Configuration* table come from the two
    display methods. `with_derived` adds the derived columns to any table that has the parameter
    columns:
    """)
    return


@app.cell
def _(office):
    waiting_room = pd.DataFrame({"name": ["Ada", "Bo", "Cy"],
                                 "weight": [62.0, 95.0, 70.0], "height": [1.75, 1.80, 1.62]})
    office.with_derived(waiting_room)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Parameter estimation

    Our first patient weighs 62 kg and is 1.75 m tall. The office measures their BMI, and the chart
    is all we get to see:
    """)
    return


@app.cell
def _(office):
    patient = Person(weight=62.0, height=1.75)
    chart = office.sample(patient, key=1)
    mo.hstack([patient, chart], justify="start", gap=2, align="start")
    return chart, patient


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The PE prior describes the people who come into the office, $w\sim\mathcal N(75, 15)$ kg and
    $h\sim\mathcal N(1.70, 0.10)$ m, truncated to sensible values. With `as_posterior=True`, `.PE`
    returns a `PosteriorData` instead of a DataFrame:
    """)
    return


@app.cell
def _(chart, office):
    pe_priors = dict(
        weight=dist.TruncatedNormal(75.0, 15.0, low=30.0, high=200.0),
        height=dist.TruncatedNormal(1.70, 0.10, low=1.0, high=2.5),
    )
    posterior = office.PE(chart, pe_priors, seed=0, num_samples=4000, as_posterior=True)
    posterior
    return pe_priors, posterior


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A `PosteriorData` has six parts, which you can see by expanding the object above:

    | attribute | contents | here |
    |---|---|---|
    | `theta` | the draws, as your parameter type with one batch axis per sample axis | a `Person` |
    | `sample_dims` | the names of the sample axes | `("chain", "draw")` |
    | `sample_stats` | per-draw inference quantities | `log_likelihood`, `prior`, `log_prior[weight]`, `log_prior[height]`, and the NUTS statistics such as `accept_prob` and `diverging` |
    | `extras` | other per-draw columns, such as derived parameters | `BMI`, `weight_lb` |
    | `metadata` | information about the run | sampler, seed, number of draws |
    | `observed_data` | the $d$ the posterior is conditioned on | a `Chart` |
    """)
    return


@app.cell
def _(chart, patient, posterior):
    def plot_people(ax, people, *, color, label, n=2500):
        # ravel merges the chain and draw axes, so [:n] takes the first n draws.
        w, h = np.ravel(people.weight)[:n], np.ravel(people.height)[:n]
        ax.scatter(w, h, s=4, alpha=0.35, color=color, linewidths=0, rasterized=True)
        ax.plot([], [], "o", color=color, label=label)

    def bmi_contours(ax, observed=None, levels=(15, 20, 25, 30, 35)):
        W, H = np.meshgrid(np.linspace(30, 130, 200), np.linspace(1.3, 2.1, 200))
        cs = ax.contour(W, H, W / H**2, levels=levels, colors=style.MUTED, linewidths=0.8)
        ax.clabel(cs, fmt="BMI %d", fontsize=8)
        if observed is not None:
            ax.contour(W, H, W / H**2, levels=[observed], colors=style.DATA, linewidths=2)
            ax.plot([], [], color=style.DATA, label="measured BMI")
        ax.set(xlim=(30, 130), ylim=(1.3, 2.1), xlabel="weight $w$ [kg]", ylabel="height $h$ [m]")

    def mark_truth(ax, person):
        ax.plot(float(person.weight), float(person.height), "X", color=style.TRUTH, markersize=10,
                label="truth", zorder=5)

    _fig, _ax = plt.subplots(figsize=(6.4, 4.0))
    bmi_contours(_ax, observed=float(chart.BMI))
    plot_people(_ax, posterior.theta, color=style.POSTERIOR, label="posterior")
    mark_truth(_ax, patient)
    _ax.set_title("Posterior from the BMI alone")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _fig
    return bmi_contours, mark_truth, plot_people


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The posterior is a narrow band along the curve $w/h^2 = \widehat{\mathrm{BMI}}$. The BMI
    fixes which curve the patient is on, but the position along the curve comes only from the
    prior. `posterior.theta` is a `Person`, so `plot_people` can read `.weight` and `.height`
    directly, and `posterior.theta.bmi()` works as well.

    ## Working with a `PosteriorData`

    Indexing by name returns a column, whether it is a parameter, a sample statistic or an extra.

    NUTS draws have two sample axes, a chain and a draw, named in `sample_dims`; here there is one
    chain of 4000 draws. Indexing with integers, slices or boolean masks selects along those axes
    in order and returns a smaller `PosteriorData`, with every per-draw array sliced the same way.
    `isel` does the same by axis name. A boolean mask selects along one axis, so a selection by
    value is made after `stack_samples()`, which merges the sample axes into one:
    """)
    return


@app.cell
def _(posterior):
    flat = posterior.stack_samples()
    overweight = flat[flat["BMI"] > 20.5]
    first_100 = posterior[:, :100]
    mo.md(f"""
    | expression | result |
    |---|---|
    | `posterior.sizes` | `{posterior.sizes}` |
    | `posterior["height"]` | array of shape `{posterior["height"].shape}` |
    | `posterior["BMI"]` (an extra) | mean {float(posterior["BMI"].mean()):.2f} |
    | `posterior["log_likelihood"]` (a sample stat) | shape `{posterior["log_likelihood"].shape}` |
    | `posterior[:, :100]` | `{first_100!r}` |
    | `posterior.isel(draw=slice(100)).sizes` | `{posterior.isel(draw=slice(100)).sizes}` |
    | `flat = posterior.stack_samples()` | `{flat.sizes}` |
    | `flat[flat["BMI"] > 20.5]` | {overweight.num_samples} draws |
    | `posterior.variable_names` | `{posterior.variable_names}` |
    | `posterior.metadata` | `{posterior.metadata}` |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A `PosteriorData` can be converted to and from a DataFrame. `to_dataframe` stacks the sample
    axes first and keeps the chain of every draw as a column. `from_dataframe` sorts the columns
    back into `theta`, `sample_stats` (recognised names such as `prior`, `log_prior[...]` and
    `log_likelihood`) and `extras` (everything else). You can use it to bring in posteriors from
    other samplers:
    """)
    return


@app.cell
def _(posterior):
    frame = posterior.to_dataframe()
    rebuilt = PosteriorData.from_dataframe(frame, parameter_type=Person,
                                           metadata={"source": "a DataFrame"})
    mo.vstack([frame.head(), rebuilt])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The prior of a subset of columns

    PE stores the prior density of every draw both as a whole (`prior`) and factor by factor
    (`log_prior[weight]`, `log_prior[height]`). A catalog may later export only some columns of a
    posterior, and then it needs the prior of those columns alone. `marginal_prior(columns)` builds
    it from the factors. It raises an error when it cannot, for example for a derived column like
    BMI, which has no stored prior:
    """)
    return


@app.cell
def _(pe_priors, posterior):
    _height_prior = posterior.marginal_prior(["height"])
    _check = np.exp(pe_priors["height"].log_prob(posterior["height"]))
    try:
        posterior.marginal_prior(["BMI"])
        _bmi_msg = "no error?"
    except ValueError as _err:
        _bmi_msg = str(_err)
    mo.md(f"""
    * `marginal_prior(["height"])` matches the height prior's density: max difference
      `{float(np.max(np.abs(_height_prior - _check))):.1e}`.
    * `marginal_prior(["weight", "height"])` is the full `prior` column.
    * `marginal_prior(["BMI"])` raises: *{_bmi_msg}*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Content hash

    `content_hash()` is a hash of every per-draw array. Catalogs (notebook 9) use it to detect when
    a posterior has changed, so that anything fitted to it is known to be out of date. Slicing the
    posterior changes the hash:
    """)
    return


@app.cell
def _(posterior):
    mo.md(f"""
    | | `content_hash()[:16]` |
    |---|---|
    | `posterior` | `{posterior.content_hash()[:16]}` |
    | `posterior` again | `{posterior.content_hash()[:16]}` |
    | `posterior[:, :100]` | `{posterior[:, :100].content_hash()[:16]}` |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## The doorways

    The office has doorways of heights $1.55, 1.65, 1.75, 1.85$ m, and the patient walks through
    each of them three times. Someone of height $h$ ducks under a door of height $H_i$ with
    probability $\sigma\!\left(\frac{h - H_i}{s}\right)$, a logistic curve of width $s = 3$ cm.

    The data class gets a second field with a core axis `"door"`, holding the twelve yes/no
    observations.
    """)
    return


@app.class_definition
@parameter_dataclass(name="Chart with doors", latex="d")
class ChartWithDoors:
    BMI: float = parameter(name="Measured BMI", latex=r"\widehat{\mathrm{BMI}}")
    ducked: bool = parameter(name="Ducked under the door", latex=r"\hat{u}",
                             index=("door",), index_latex=("i",))


@app.class_definition
@dataclass
class MeasureBMIAndDoors(MeasurementProcess):
    sigma_bmi: float = 0.5
    door_heights: tuple = (1.55, 1.65, 1.75, 1.85) * 3
    fall_off: float = 0.03

    display_name = "BMI and doorways"
    parameter_type = Person
    data_type = ChartWithDoors

    def duck_probability(self, height):
        return jax.nn.sigmoid((height - jnp.asarray(self.door_heights)) / self.fall_off)

    def sample_single(self, theta, key):
        k_bmi, k_duck = jax.random.split(key)
        return ChartWithDoors(
            BMI=theta.bmi() + self.sigma_bmi * jax.random.normal(k_bmi),
            ducked=jax.random.bernoulli(k_duck, self.duck_probability(theta.height)),
        )

    def logpdf_single(self, d, theta):
        p = self.duck_probability(theta.height)
        log_doors = jnp.sum(jnp.where(d.ducked, jnp.log(p), jnp.log1p(-p)), axis=-1)
        return jax.scipy.stats.norm.logpdf(d.BMI, theta.bmi(), self.sigma_bmi) + log_doors

    def derived_parameters(self, theta):
        return {"BMI": theta["weight"] / theta["height"] ** 2}

    def _display_equations(self):
        return [r"\widehat{\mathrm{BMI}} \sim \mathcal{N}\!\left(w/h^2,\ \sigma_\mathrm{BMI}^2\right)",
                r"\hat{u}_i \sim \mathrm{Bernoulli}\!\left(\mathrm{sigmoid}\big((h - H_i)/s\big)\right)"]

    def _display_config(self):
        return {"BMI noise σ": self.sigma_bmi, "door heights H [m]": sorted(set(self.door_heights)),
                "visits": len(self.door_heights) // len(set(self.door_heights)),
                "fall-off s [m]": self.fall_off}


@app.cell
def _():
    office_with_doors = MeasureBMIAndDoors()

    _h = jnp.linspace(1.4, 2.0, 300)
    _p = office_with_doors.duck_probability(_h[:, None])      # (300, 12)
    _fig, _ax = plt.subplots(figsize=(6.4, 2.6))
    for _i, (_H, _color) in enumerate(zip(sorted(set(office_with_doors.door_heights)), style.SERIES)):
        _ax.plot(_h, _p[:, _i], color=_color, label=f"door $H = {_H}$ m")
    _ax.set(xlabel="patient height $h$ [m]", ylabel="P(duck)", title="Probability of ducking under each door")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    mo.vstack([office_with_doors, _fig])
    return (office_with_doors,)


@app.cell
def _(office_with_doors):
    second_patient = Person(weight=58.0, height=1.62)
    door_chart = office_with_doors.sample(second_patient, key=4)
    mo.vstack([
        door_chart,
        mo.md(f"Ducked under: `{np.asarray(door_chart.ducked).astype(int)}` "
              f"(doors `{office_with_doors.door_heights}`)"),
    ])
    return door_chart, second_patient


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We run PE twice on the same visit, once with all the data and once with only the BMI (a `Chart`
    built from the same number). `PosteriorData` is a dataclass, so we can add the patient's name
    to the metadata with `dataclasses.replace`:
    """)
    return


@app.cell
def _(door_chart, office, office_with_doors, pe_priors):
    with_doors = office_with_doors.PE(door_chart, pe_priors, seed=0, num_samples=4000,
                                      as_posterior=True)
    with_doors = replace(with_doors, metadata={**with_doors.metadata, "patient": "Bo"})
    bmi_only = office.PE(Chart(BMI=door_chart.BMI), pe_priors, seed=0, num_samples=4000,
                         as_posterior=True)
    return bmi_only, with_doors


@app.cell
def _(
    bmi_contours,
    bmi_only,
    door_chart,
    mark_truth,
    plot_people,
    second_patient,
    with_doors,
):
    _fig, _ax = plt.subplots(figsize=(6.4, 4.0))
    bmi_contours(_ax, observed=float(door_chart.BMI))
    plot_people(_ax, bmi_only.theta, color=style.PRIOR, label="BMI only")
    plot_people(_ax, with_doors.theta, color=style.POSTERIOR, label="BMI + doors")
    mark_truth(_ax, second_patient)
    _ax.set_title("Posteriors with and without the doorways")
    _ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The twelve yes/no answers from the doorways pin down the height, which the BMI alone could not
    do.

    ---
    ## A display for your own objects

    Any class can have the collapsible display, with a LaTeX header, that astroinfer objects use.
    Subclass `ai.RichDisplay` and fill in three methods.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(r"""
    **What you implement in a `RichDisplay`**

    | method | returns |
    |---|---|
    | `_display_title(self)` | the title |
    | `_display_math(self)` | LaTeX shown next to the title (or `None`) |
    | `_display_body(self)` | HTML shown when the display is expanded |

    `astroinfer.display` has small HTML helpers you can use in the body, such as
    `kv_table({label: value})` and `html_table(header, rows)`.
    """), kind="info")
    return


@app.class_definition
class PatientChart(ai.RichDisplay):
    """A one-glance summary of a patient's posterior."""

    def __init__(self, name, posterior):
        self.name, self.posterior = name, posterior

    def _display_title(self):
        return f"Patient chart · {self.name}"

    def _display_math(self):
        return r"p(w,\,h \mid d)"

    def _display_body(self):
        rows = []
        for column, unit in [("weight", "kg"), ("height", "m"), ("BMI", "")]:
            lo, mid, hi = np.percentile(np.asarray(self.posterior[column]), [5, 50, 95])
            rows.append([column, f"{mid:.2f} {unit}", f"[{lo:.2f}, {hi:.2f}]"])
        p_tall = float(np.mean(np.asarray(self.posterior["height"]) > 1.70))
        return (html_table(["", "median", "90% interval"], rows)
                + kv_table({"P(taller than 1.70 m)": f"{p_tall:.1%}",
                            "draws": self.posterior.num_samples}))


@app.cell
def _(bmi_only, with_doors):
    mo.vstack([PatientChart("Bo, BMI only", bmi_only), PatientChart("Bo, BMI + doors", with_doors)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    * `derived_parameters(theta)` adds functions of $\theta$ to the PE output (as `extras`) and to
      `with_derived(table)`.
    * `PE(..., as_posterior=True)` returns a `PosteriorData`, which can be indexed by name or by
      draw, converted to and from a DataFrame, and asked for the prior of a subset of columns
      (`marginal_prior`).
    * `_display_equations` and `_display_config` customise a measurement process's display; any
      subclass of `ai.RichDisplay` gets a display from `_display_title`, `_display_math` and
      `_display_body`.

    Every analysis so far has been about a single object: one coin, one lighthouse, one patient.
    Notebook 5 turns to populations, the distributions that such objects are drawn from.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: the bathroom scale

    The office buys a scale that weighs patients with a noise of 2 kg.

    1. Write a data class `ChartWithScale` with fields `BMI` and `weight` (the scale reading).
    2. Write `MeasureBMIAndScale(MeasurementProcess)` with `sigma_bmi` and `sigma_scale`, a
       `derived_parameters` that returns BMI, and a `_display_equations` with both lines.
    3. Run PE on our first patient (62 kg, 1.75 m) and draw the posterior on top of the BMI-only one.
       Why does knowing the weight also pin the height?
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
    @parameter_dataclass(name="Chart with scale", latex="d")
    class ChartWithScale:
        BMI: float = parameter(name="Measured BMI", latex=r"\widehat{\mathrm{BMI}}")
        weight: float = parameter(name="Scale reading [kg]", latex=r"\hat{w}")

    @dataclass
    class MeasureBMIAndScale(MeasurementProcess):
        sigma_bmi: float = 0.5
        sigma_scale: float = 2.0

        parameter_type = Person
        data_type = ChartWithScale

        def sample_single(self, theta, key):
            k1, k2 = jax.random.split(key)
            return ChartWithScale(
                BMI=theta.bmi() + self.sigma_bmi * jax.random.normal(k1),
                weight=theta.weight + self.sigma_scale * jax.random.normal(k2),
            )

        def logpdf_single(self, d, theta):
            norm = jax.scipy.stats.norm
            return (norm.logpdf(d.BMI, theta.bmi(), self.sigma_bmi)
                    + norm.logpdf(d.weight, theta.weight, self.sigma_scale))

        def derived_parameters(self, theta):
            return {"BMI": theta["weight"] / theta["height"] ** 2}

        def _display_equations(self):
            return [r"\widehat{\mathrm{BMI}} \sim \mathcal{N}(w/h^2, \sigma_\mathrm{BMI}^2)",
                    r"\hat{w} \sim \mathcal{N}(w, \sigma_\mathrm{scale}^2)"]

    scale_office = MeasureBMIAndScale()
    d = scale_office.sample(Person(weight=62.0, height=1.75), key=0)
    post = scale_office.PE(d, pe_priors, seed=0, as_posterior=True)

    fig, ax = plt.subplots()
    bmi_contours(ax, observed=float(d.BMI))
    plot_people(ax, posterior.theta, color=style.PRIOR, label="BMI only")
    plot_people(ax, post.theta, color=style.POSTERIOR, label="BMI + scale")
    ax.legend()
    # h = sqrt(w / BMI), so knowing w picks out one point on the BMI curve.
    ```
    """)})
    return


if __name__ == "__main__":
    app.run()

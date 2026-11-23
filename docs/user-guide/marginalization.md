---
kernelspec:
  name: python3
  display_name: Python 3
---

# Integrating values out of a model

```{code-cell} python
:tags: [remove-cell]

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
```

Many hierarchical models have latent values that can be integrated out exactly. When the prior
of a value and the likelihood of the data are *conjugate*, the integral of their product over the
value has a closed form, and so does the distribution of the value given the data. Integrating
such values out of a model, or *marginalizing* them, leaves the sampler fewer dimensions to
explore, often removes the funnels that hierarchical models are prone to, and loses nothing: the
values can be drawn afterwards from their exact distribution given the data.

In an astroinfer model, `ai.sample(name, prior, ..., marginalize=True)` marks a value to be
integrated out. The observed site that uses the value computes the integral, using a description
of its likelihood that you write as one short method. `ai.infer` then samples the other
parameters and draws the marginalized values afterwards, so the posterior holds them as in an
ordinary run. The sections below introduce the parts one at a time, each with an example;
[NumPyro models](numpyro-models.md) introduces `ai.sample` and `ai.infer`.

```{code-cell} python
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro import handlers
from numpyro.distributions import constraints

import astroinfer as ai
from astroinfer import parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)

SETTINGS = dict(seed=0, num_warmup=1000, num_samples=4000, summary=False,
                mcmc_kwargs={"progress_bar": False})

def compare(posteriors, names):
    """Posterior mean, standard deviation and effective sample size of each run."""
    rows = {}
    for label, posterior in posteriors.items():
        table = posterior.summary(names)
        for name in names:
            rows[(name, label)] = {"mean": table.loc[name, "mean"], "sd": table.loc[name, "sd"],
                                   "ess": table.loc[name, "ess"]}
        rows[("divergences", label)] = {"mean": int(posterior.sample_stats["diverging"].sum())}
    return pd.DataFrame(rows).T.round(4)
```

## A first example

A set of points is measured with Gaussian errors of known size. The true values $x_i$ of the
points come from a population $\mathcal N(\mu, \sigma)$, and each measurement is
$\hat x_i \sim \mathcal N(x_i, \epsilon_i)$. We want $\mu$ and $\sigma$. The true value and the
measurement are two parameter types; the error bar is part of the measurement:

```{code-cell} python
@parameter_dataclass(name="True value", latex="x")
class TrueValue:
    x: float = parameter(name="True value", latex="x")

@parameter_dataclass(name="Measurement", latex=r"\hat x")
class Measurement:
    y: float = parameter(name="Measured value", latex=r"\hat x")
    error: float = parameter(name="Quoted error", latex=r"\epsilon", support=constraints.positive)
```

The likelihood is a conditional distribution of the measurement given the true value. Besides
`logpdf_single`, it declares that it is conjugate to a Normal prior on `x`, and describes itself
as a function of `x`:

```{code-cell} python
class MeasuredWithError(ai.ConditionalDistribution):
    data_type = Measurement
    parameter_type = TrueValue
    conjugate_likelihood = {"x": ai.marginal.NormalNormal}

    def logpdf_single(self, d, value):
        return dist.Normal(value.x, d.error).log_prob(d.y)

    def conjugate_likelihood_single(self, path, d, value):
        # As a function of x: the measurement is Normal around x with this error.
        return d.y, d.error

measured = MeasuredWithError()
```

`conjugate_likelihood` maps the place of the value among the conditions, here the field `x` of
the condition, to a *conjugate family*, a class that knows how to integrate a prior and a
likelihood of matching forms. `NormalNormal` is the family of a Normal prior on a scalar and
Normal measurements of it. `conjugate_likelihood_single` returns the likelihood in the form the
family documents, here `(observed, error)`. The class has no `sample_single`: an observed site
takes the shape of its data from the data.

We simulate 30 points with $\mu = 3$, $\sigma = 1.5$ and errors between 0.5 and 2:

```{code-cell} python
N = 30
k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
x_true = 3.0 + 1.5 * jax.random.normal(k1, (N,))
errors = jax.random.uniform(k2, (N,), minval=0.5, maxval=2.0)
data = Measurement(y=x_true + errors * jax.random.normal(k3, (N,)), error=errors)
```

The model is written in terms of the true values. Whether they are sampled or integrated out is
the one keyword:

```{code-cell} python
def points_model(marginalize):
    def model(data):
        mu = ai.sample("mu", dist.Normal(0.0, 10.0))
        sigma = ai.sample("sigma", dist.HalfNormal(5.0))
        with numpyro.plate("points", data.batch_shape[0]):
            x = ai.sample("x", dist.Normal(mu, sigma), marginalize=marginalize)
            ai.sample("y", measured, TrueValue(x=x), obs=data)
    return model

full = ai.infer(points_model(False), data, **SETTINGS)
marginal = ai.infer(points_model(True), data, **SETTINGS)
compare({"sampled": full, "marginalized": marginal}, ["mu", "sigma"])
```

The two runs agree. The posterior mean of $\mu$ is 3.68 sampled and 3.69 marginalized, with a
standard deviation of 0.40 in both, and that of $\sigma$ is 1.61 with a standard deviation of 0.38
in both; the true values are 3 and 1.5. The marginalized run has an effective sample size of 4168
for $\sigma$ in 4000 draws, against 1086 for the sampled run, which has to move $\sigma$ and the
30 true values together.

## What `marginalize=True` does

With `marginalize=True`, `ai.sample` creates no sample site. NumPyro's trace of the model has
two latent sites, the hyperparameters:

```{code-cell} python
trace = handlers.trace(handlers.seed(points_model(True), rng_seed=0)).get_trace(data)
[name for name, site in trace.items() if site["type"] == "sample" and not site["is_observed"]]
```

`ai.sample` returns a *stand-in* for the value instead: an array of the value's shape, filled
with NaN. The observed site that receives it among its conditions does not evaluate its
`logpdf`. It looks up the family, asks the prior and the likelihood for their descriptions, and
adds the integral of their product over the value to the model as a factor. For this family the
integral is the familiar result that each measurement is $\mathcal N(\mu, \sqrt{\sigma^2 +
\epsilon_i^2})$, which we can check against the model's log density at one value of the
hyperparameters:

```{code-cell} python
from numpyro.infer.util import log_density
from scipy import stats

params = {"mu": 3.2, "sigma": 1.4}
with handlers.block():
    model_value = log_density(points_model(True), (data,), {}, params)[0]
by_hand = (dist.Normal(0.0, 10.0).log_prob(3.2) + dist.HalfNormal(5.0).log_prob(1.4)
           + stats.norm(3.2, np.sqrt(1.4**2 + np.asarray(data.error) ** 2)).logpdf(np.asarray(data.y)).sum())
float(model_value), float(by_hand)
```

The two agree. The value is inside the plate, so it is integrated out once per point.

After sampling, `ai.infer` runs the model once more for every posterior draw and draws each
point's true value from its distribution given the data and the hyperparameters of that draw. The
posterior has the field `x` as the sampled run does:

```{code-cell} python
marginal.parameter_names, marginal.theta.x.shape, marginal.metadata["marginalized"]
```

```{code-cell} python
pd.DataFrame({
    "measured": np.asarray(data.y[:3]), "error": np.asarray(data.error[:3]),
    "sampled": np.asarray(full.theta.x[..., :3].mean((0, 1))),
    "marginalized, then drawn": np.asarray(marginal.theta.x[..., :3].mean((0, 1))),
    "true": np.asarray(x_true[:3]),
}).round(3)
```

The second point was measured at 4.49 with an error of 1.89. Both runs pull it towards the
population, to 4.02 and 4.03; its true value is 2.28.

## Describing the likelihood

A likelihood that can integrate a value out has two members:

| member | what it is |
|---|---|
| `conjugate_likelihood` | a dict from the *path* of the value among the conditions to a conjugate family |
| `conjugate_likelihood_single(self, path, d, *conditions)` | the likelihood as a function of the value at `path`, for one element of the batch, in the form the family documents |

The path locates the value. For a single condition, as above, it is a field of the condition
(`"x"`), or `""` for the whole condition. For a mapping schema its first part is a condition name,
as in `"velocity"` or `"velocity.v"`; for a tuple schema it is an index. The method receives the
data and the conditions as `logpdf_single` does, with the stand-in in place of the value, so it
must not read the value: it would read NaN.

The built-in families and the forms they take:

| family | prior | likelihood, as a function of the value $a$ |
|---|---|---|
| `NormalNormal` | `(loc, scale)` or `dist.Normal` | `(observed, error)`, observed $\sim\mathcal N(a, \text{error})$, or `dist.Normal(observed, error)` |
| `GammaPoisson` | `(concentration, rate)` or `dist.Gamma` | `(counts, exposure)`, counts $\sim$ Poisson($a\,\times$ exposure) |
| `LinearGaussian` | `(mean, scale_tril)`, `dist.MultivariateNormal`, `dist.Normal` | a dict with `observed`, `design`, `noise_cov` or `noise_scale_tril`, and optionally `offset` and `mask`; observed $\sim\mathcal N(\text{design}\cdot a + \text{offset}, \text{noise})$ |

In every family the likelihood's arrays may hold several measurements of the same value; they are
combined.

## Your own prior: the distance to a cluster

The prior can be a conditional distribution you wrote, as well as a NumPyro distribution. Gaia
measures the parallaxes of 60 stars of an open cluster at 500 pc. Each star's true parallax is
drawn from a Normal around the cluster's parallax $\varpi_c$, with a spread $\sigma_\varpi$ set
by the depth of the cluster, 0.03 mas here, smaller than most of the error bars, which range
from 0.02 to 0.25 mas. Gaia's parallaxes are
offset from the true ones by a global zero point.

```{code-cell} python
@parameter_dataclass(name="Cluster", latex=r"\Lambda")
class Cluster:
    parallax: float = parameter(name="Cluster parallax [mas]", latex=r"\varpi_c")
    depth: float = parameter(name="Spread of true parallaxes [mas]", latex=r"\sigma_\varpi",
                             support=constraints.positive)

@parameter_dataclass(name="Star", latex=r"\theta")
class Star:
    parallax: float = parameter(name="True parallax [mas]", latex=r"\varpi")

@parameter_dataclass(name="Gaia parallax", latex="d")
class GaiaParallax:
    parallax: float = parameter(name="Measured parallax [mas]", latex=r"\hat\varpi")
    error: float = parameter(name="Quoted error [mas]", latex=r"\epsilon", support=constraints.positive)
```

A prior class declares its family with `conjugate_prior` and describes itself with
`conjugate_prior_single`, which receives the conditions as `sample_single` does:

```{code-cell} python
class ClusterDepth(ai.ConditionalDistribution):
    data_type = Star
    parameter_type = Cluster
    conjugate_prior = ai.marginal.NormalNormal

    def sample_single(self, cluster, key):
        return Star(parallax=cluster.parallax + cluster.depth * jax.random.normal(key))

    def logpdf_single(self, star, cluster):
        return dist.Normal(cluster.parallax, cluster.depth).log_prob(star.parallax)

    def conjugate_prior_single(self, cluster):
        return dist.Normal(cluster.parallax, cluster.depth)

class GaiaMeasurement(ai.ConditionalDistribution):
    data_type = GaiaParallax
    parameter_type = Star
    conjugate_likelihood = {"parallax": ai.marginal.NormalNormal}
    zero_point = -0.017   # mas

    def logpdf_single(self, d, star):
        return dist.Normal(star.parallax + self.zero_point, d.error).log_prob(d.parallax)

    def conjugate_likelihood_single(self, path, d, star):
        return dist.Normal(d.parallax - self.zero_point, d.error)

depth_model, gaia = ClusterDepth(), GaiaMeasurement()
```

The likelihood's description moves the known zero point to the data, so that the family sees a
measurement centred on the true parallax. Both methods return a NumPyro `Normal`, which
`NormalNormal` accepts in place of a tuple. The data:

```{code-cell} python
truth = Cluster(parallax=2.0, depth=0.03)
k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
stars = depth_model.sample(truth, key=k1, N=60)
parallax_errors = jax.random.uniform(k2, (60,), minval=0.02, maxval=0.25)
gaia_data = GaiaParallax(
    parallax=stars.parallax + gaia.zero_point + parallax_errors * jax.random.normal(k3, (60,)),
    error=parallax_errors,
)

def cluster_model(prior, marginalize):
    def model(data):
        cluster = Cluster(parallax=ai.sample("cluster_parallax", dist.Uniform(0.5, 5.0)),
                          depth=ai.sample("depth", dist.HalfNormal(0.2)))
        with numpyro.plate("stars", data.batch_shape[0]):
            star = ai.sample("star", prior, cluster, marginalize=marginalize)
            ai.sample("d", gaia, star, obs=data)
    return model

strict = dict(SETTINGS, target_accept_prob=0.95, max_tree_depth=10)
cluster_full = ai.infer(cluster_model(depth_model, False), gaia_data, **strict)
cluster_marginal = ai.infer(cluster_model(depth_model, True), gaia_data, **strict)
compare({"sampled": cluster_full, "marginalized": cluster_marginal}, ["cluster_parallax", "depth"])
```

The sampled run has 44 divergent transitions and an effective sample size of 52 for the depth in
4000 draws, although it runs with a target acceptance of 0.95 and trees up to depth 10. This is
the funnel of hierarchical models: when the depth is small, all 60 true parallaxes are held close
to the cluster's, and no single step size suits both that region and the rest. The marginalized
run samples two parameters, has no divergences, and has an effective sample size of 1518 for the
depth. Both find a cluster parallax of 2.012 to 2.013 mas with a standard deviation of 0.013 mas,
and a depth of 0.026 to 0.027 mas with a standard deviation of 0.017 to 0.018 mas; the true values
are 2.0 and 0.03 mas.

The stars come back as `Star` objects, one per star and posterior draw:

```{code-cell} python
cluster_marginal.theta.star
```

## Pair rules

A conjugate family needs a description from each class. A *pair rule* is the alternative: one
class written for a specific prior class and likelihood class, which reads whatever it needs from
the two objects and their arguments, so the classes need no conjugacy members at all. It has two
methods,

```
log_marginal(self, prior, likelihood, data, prior_conditions, likelihood_conditions)
draw(self, key, prior, likelihood, data, prior_conditions, likelihood_conditions)
```

where `prior` and `likelihood` are the distribution objects and the other arguments are for one
element of the batch. Conditions are passed as the schema gives them: the object for a single
condition, a tuple for a tuple schema, a dict for a mapping schema. `draw` returns one value, in
its type. Here is the cluster again, with a prior class that declares no family:

```{code-cell} python
class PlainClusterDepth(ClusterDepth):
    conjugate_prior = None

@ai.marginal.pair(PlainClusterDepth, GaiaMeasurement)
class ClusterDepthGaia:
    def _sums(self, cluster, likelihood, d):
        # The exponent of prior times likelihood is -(P x^2 - 2 h x + Q) / 2.
        observed = d.parallax - likelihood.zero_point
        P = 1 / cluster.depth**2 + 1 / d.error**2
        h = cluster.parallax / cluster.depth**2 + observed / d.error**2
        Q = cluster.parallax**2 / cluster.depth**2 + observed**2 / d.error**2
        return P, h, Q

    def log_marginal(self, prior, likelihood, data, prior_conditions, likelihood_conditions):
        P, h, Q = self._sums(prior_conditions, likelihood, data)
        return (-0.5 * (Q - h**2 / P) - 0.5 * jnp.log(P) - jnp.log(prior_conditions.depth)
                - jnp.log(data.error) - 0.5 * jnp.log(2 * jnp.pi))

    def draw(self, key, prior, likelihood, data, prior_conditions, likelihood_conditions):
        P, h, _ = self._sums(prior_conditions, likelihood, data)
        return Star(parallax=h / P + jax.random.normal(key) / jnp.sqrt(P))

cluster_pair = ai.infer(cluster_model(PlainClusterDepth(), True), gaia_data, **strict)
compare({"family": cluster_marginal, "pair rule": cluster_pair}, ["cluster_parallax", "depth"])
```

The rule reads the zero point from the likelihood object and the error from the data. It gives
the same posterior as the family, draw for draw, since both compute the same integral. A pair rule
is registered for its two classes and their subclasses, and takes precedence over any family the
classes declare. The trade-off is reuse: a family serves every class that describes itself in its
form, while a pair rule serves one combination.

## Counts: Gamma and Poisson

Forty faint X-ray sources are observed in the same three pointings, of 20, 35 and 50 ks. Each
source's counts in a pointing are Poisson with mean its count rate times the exposure, and the
rates follow a Gamma population whose shape and scale we infer. A Gamma prior on a rate is
conjugate to Poisson counts:

```{code-cell} python
@parameter_dataclass(name="Source", latex=r"\lambda")
class Source:
    rate: float = parameter(name="Count rate [counts/ks]", latex=r"\lambda",
                            support=constraints.positive)

@parameter_dataclass(name="Counts", latex="k")
class Counts:
    k: float = parameter(name="Counts in each pointing", latex="k", index=("pointing",))

class Pointings(ai.ConditionalDistribution):
    data_type = Counts
    parameter_type = Source
    conjugate_likelihood = {"rate": ai.marginal.GammaPoisson}
    exposure = jnp.array([20.0, 35.0, 50.0])   # ks

    def sample_single(self, source, key):
        return Counts(k=dist.Poisson(source.rate * self.exposure).sample(key).astype(float))

    def logpdf_single(self, d, source):
        return jnp.sum(dist.Poisson(source.rate * self.exposure).log_prob(d.k))

    def conjugate_likelihood_single(self, path, d, source):
        return d.k, self.exposure

pointings = Pointings()
true_rates = dist.Gamma(3.0, 1.0 / 0.05).sample(jax.random.key(2), (40,))
counts = pointings.sample(Source(rate=true_rates), key=3)

def xray_model(counts):
    shape = ai.sample("shape", dist.LogNormal(0.0, 1.0))
    scale = ai.sample("scale", dist.LogNormal(-3.0, 1.0))
    with numpyro.plate("sources", counts.batch_shape[0]):
        rate = ai.sample("rate", dist.Gamma(shape, 1.0 / scale), marginalize=True)
        ai.sample("counts", pointings, Source(rate=rate), obs=counts)

xray = ai.infer(xray_model, counts, **SETTINGS)
xray.summary(["shape", "scale"])
```

The true shape and scale of the population are 3 and 0.05 counts/ks. The posterior means are
2.85 and 0.061, with standard deviations of 0.76 and 0.018. The sampler explores two dimensions
instead of 42, and `xray.theta.rate` holds the 40 rates, drawn afterwards.

## Vectors: the velocities of cluster stars

`LinearGaussian` integrates out a vector whose prior is Gaussian and whose data are Gaussian with
a mean linear in it. Gaia measures the two proper motions of 40 cluster stars, and a radial
velocity for about 60% of them. Each star's measurements are a projection of its 3D velocity
$\vec v_i$, set by the star's position and parallax, with a correlated noise covariance. The
velocities follow a Gaussian population with mean $\vec\mu$ and a full covariance, which we infer:

```{code-cell} python
K_PM = 4.740470446   # km/s per (mas/yr) at a parallax of 1 mas

@parameter_dataclass(name="Velocity", latex=r"\vec v")
class Velocity:
    v: float = parameter(name="Velocity, ICRS cartesian [km/s]", latex=r"\vec v", index=("xyz",))

@parameter_dataclass(name="Star", latex="s")
class StarInfo:
    ra: float = parameter(name="Right ascension [rad]", latex=r"\alpha")
    dec: float = parameter(name="Declination [rad]", latex=r"\delta")
    parallax: float = parameter(name="Parallax [mas]", latex=r"\varpi")
    cov: float = parameter(name="Noise covariance of (pmra, pmdec, rv)", latex="C",
                           index=("obs", "obs"))
    has_rv: float = parameter(name="1 if the star has a radial velocity", latex="m")

@parameter_dataclass(name="Kinematic data", latex="d")
class Motion:
    y: float = parameter(name="(pmra [mas/yr], pmdec [mas/yr], rv [km/s])", latex="d",
                         index=("obs",))

class GaiaKinematics(ai.ConditionalDistribution):
    data_type = Motion
    parameter_type = {"velocity": Velocity, "star": StarInfo}
    conjugate_likelihood = {"velocity": ai.marginal.LinearGaussian}

    def design_matrix_single(self, star):
        """(obs, xyz): the velocity projected onto east, north and the line of sight."""
        a, d = star.ra, star.dec
        east = jnp.array([-jnp.sin(a), jnp.cos(a), 0.0])
        north = jnp.array([-jnp.sin(d) * jnp.cos(a), -jnp.sin(d) * jnp.sin(a), jnp.cos(d)])
        radial = jnp.array([jnp.cos(d) * jnp.cos(a), jnp.cos(d) * jnp.sin(a), jnp.sin(d)])
        to_units = jnp.array([star.parallax / K_PM, star.parallax / K_PM, 1.0])
        return jnp.stack([east, north, radial]) * to_units[:, None]

    def sample_single(self, velocity, star, key):
        mean = self.design_matrix_single(star) @ velocity.v
        return Motion(y=jax.random.multivariate_normal(key, mean, star.cov))

    def logpdf_single(self, d, velocity, star):
        m = jnp.array([1.0, 1.0, star.has_rv])
        cov = star.cov * jnp.outer(m, m) + jnp.diag(1.0 - m)
        r = (d.y - self.design_matrix_single(star) @ velocity.v) * m
        return (-0.5 * r @ jnp.linalg.solve(cov, r) - 0.5 * jnp.linalg.slogdet(cov)[1]
                - 0.5 * jnp.sum(m) * jnp.log(2 * jnp.pi))

    def conjugate_likelihood_single(self, path, d, velocity, star):
        return {
            "observed": d.y,
            "design": Velocity(v=self.design_matrix_single(star)),   # one Velocity per row
            "noise_cov": star.cov,
            "mask": jnp.array([1.0, 1.0, star.has_rv]),              # a missing rv drops out
        }

kinematics = GaiaKinematics()
```

The design is given as `Velocity` objects, one per measured component, whose field `v` holds that
component's projection. The family packs them into a matrix itself, so the order of the columns
never has to be written out; this matters for a value with several fields. The mask leaves out
the radial velocity of the stars that have none, whatever is stored there. The data:

```{code-cell} python
keys = jax.random.split(jax.random.key(4), 7)
mu_true, sigma_true = jnp.array([-5.0, 12.0, 3.0]), jnp.array([1.0, 2.0, 1.5])
v_true = mu_true + sigma_true * jax.random.normal(keys[0], (40, 3))
B = 0.05 * jax.random.normal(keys[1], (40, 3, 3))
star_info = StarInfo(
    ra=1.2 + 0.3 * jax.random.uniform(keys[2], (40,)),
    dec=0.4 + 0.3 * jax.random.uniform(keys[3], (40,)),
    parallax=2.0 + 0.05 * jax.random.normal(keys[4], (40,)),
    cov=jnp.einsum("nij,nkj->nik", B, B) + jnp.diag(jnp.array([0.01, 0.01, 1.0])),
    has_rv=(jax.random.uniform(keys[5], (40,)) < 0.6).astype(float),
)
motions = kinematics.sample(velocity=Velocity(v=v_true), star=star_info, key=keys[6])

def kinematics_model(stars, data):
    mu = ai.sample("mu", dist.Normal(0.0, 50.0).expand([3]).to_event(1))
    sigma = ai.sample("sigma", dist.HalfNormal(5.0).expand([3]).to_event(1))
    corr = ai.sample("corr", dist.LKJCholesky(3, concentration=2.0))
    with numpyro.plate("stars", stars.batch_shape[0]):
        v = ai.sample("velocity", dist.MultivariateNormal(mu, scale_tril=sigma[:, None] * corr),
                      marginalize=True)
        ai.sample("d", kinematics, velocity=Velocity(v=v), star=stars, obs=data)

velocities = ai.infer(kinematics_model, star_info, motions, **SETTINGS)
velocities.summary(["mu", "sigma"], expand=True)
```

The mean and spread of the 40 simulated velocities, which the data can constrain, rather than
those of the population they were drawn from:

```{code-cell} python
np.round(np.asarray(v_true.mean(0)), 2), np.round(np.asarray(v_true.std(0)), 2)
```

The posterior means of $\vec\mu$ are $(-5.08, 11.32, 2.96)$ km/s with standard deviations of
0.15, 0.30 and 0.26 km/s, against $(-5.02, 11.63, 3.25)$ for the simulated stars, and the
dispersions are $(0.86, 1.37, 1.43)$ km/s against $(0.77, 1.70, 1.44)$; each is within about one
standard deviation. The sampler explores the nine parameters of the population, rather than those
nine and the 120 components of the stars' velocities.

## A value shared across a plate

A marginalized value outside a plate is measured by every element of the plate. Here five
measurements of one quantity are combined, with a Normal prior on the quantity whose width is
itself a parameter:

```{code-cell} python
repeats = Measurement(y=jnp.array([1.2, 0.7, 1.9, 1.1, 0.9]),
                      error=jnp.array([0.5, 0.3, 0.8, 0.4, 0.6]))

def shared_model(data):
    width = ai.sample("width", dist.HalfNormal(2.0))
    x = ai.sample("x", dist.Normal(0.0, width), marginalize=True)
    with numpyro.plate("repeats", data.batch_shape[0]):
        ai.sample("y", measured, TrueValue(x=x), obs=data)

shared = ai.infer(shared_model, repeats, **SETTINGS)
shared.theta.x.shape, float(shared.theta.x.mean()), float(shared.theta.x.std())
```

The five measurements are combined. The posterior of $x$ has a mean of 0.94 and a standard
deviation of 0.20, close to the error-weighted mean of the measurements, 0.97 with an error of
0.20, and pulled slightly towards the prior's centre at 0. Since the value is outside the plate,
its likelihood's description has one entry per measurement along a leading axis, and
`NormalNormal` combines them.

## Writing a family

A family is a subclass of `ConjugateFamily` with up to three methods, each for one element of the
value's batch:

| method | what it returns |
|---|---|
| `log_marginal(self, prior, likelihood)` | the log of the integral over the value of the prior times the likelihood |
| `draw(self, key, prior, likelihood)` | one draw of the value given the likelihood, as a flat array (one element for a scalar) |
| `from_numpyro(self, distribution)` | optional: the prior's description from a NumPyro distribution, with the distribution's batch axes leading |

`prior` and `likelihood` are what the classes' `conjugate_prior_single` and
`conjugate_likelihood_single` return, so the family decides their form and documents it. When a
value outside a plate is measured inside it, the likelihood carries extra leading axes, one entry
per measurement, which the family combines.

Here is a Beta prior on a fraction, measured by binomial counts. Each of 25 open clusters has a
fraction $f_i$ of its stars in binaries, drawn from a Beta population, and a survey finds $k_i$
binaries among $n_i$ stars. The Beta–binomial integral is a ratio of beta functions:

```{code-cell} python
from jax.scipy.special import betaln, gammaln

class BetaBinomial(ai.marginal.ConjugateFamily):
    """Prior: (a, b) or dist.Beta(a, b). Likelihood: (found, examined), found ~ Binomial(examined, f)."""

    def from_numpyro(self, d):
        if not isinstance(d, dist.Beta):
            return super().from_numpyro(d)
        shape = tuple(d.batch_shape)
        return jnp.broadcast_to(d.concentration1, shape), jnp.broadcast_to(d.concentration0, shape)

    def _posterior(self, prior, likelihood):
        a, b = prior
        k, n = likelihood
        return a + jnp.sum(k), b + jnp.sum(n - k)

    def log_marginal(self, prior, likelihood):
        k, n = likelihood
        log_choose = jnp.sum(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))
        return log_choose + betaln(*self._posterior(prior, likelihood)) - betaln(*prior)

    def draw(self, key, prior, likelihood):
        return jnp.reshape(jax.random.beta(key, *self._posterior(prior, likelihood)), (1,))
```

For one cluster, the integral should be NumPyro's beta-binomial distribution:

```{code-cell} python
float(BetaBinomial().log_marginal((2.0, 5.0), (jnp.array(7.0), jnp.array(30.0)))), \
    float(dist.BetaBinomial(2.0, 5.0, total_count=30).log_prob(7))
```

The family is then used like the built-in ones:

```{code-cell} python
@parameter_dataclass(name="Binary fraction", latex="f")
class BinaryFraction:
    f: float = parameter(name="Fraction of stars in binaries", latex="f",
                         support=constraints.unit_interval)

@parameter_dataclass(name="Binary survey", latex="k")
class BinarySurvey:
    found: float = parameter(name="Binaries found", latex="k")
    examined: float = parameter(name="Stars examined", latex="n")

class BinarySearch(ai.ConditionalDistribution):
    data_type = BinarySurvey
    parameter_type = BinaryFraction
    conjugate_likelihood = {"f": BetaBinomial}

    def logpdf_single(self, d, fraction):
        return dist.Binomial(d.examined, fraction.f).log_prob(d.found)

    def conjugate_likelihood_single(self, path, d, fraction):
        return d.found, d.examined

search = BinarySearch()
k1, k2, k3 = jax.random.split(jax.random.key(5), 3)
true_fractions = dist.Beta(6.0, 9.0).sample(k1, (25,))
examined = jax.random.randint(k2, (25,), 20, 120)
surveys = BinarySurvey(found=dist.Binomial(examined, true_fractions).sample(k3).astype(float),
                       examined=examined.astype(float))

def binary_model(surveys):
    mean = ai.sample("mean", dist.Beta(2.0, 2.0))
    concentration = ai.sample("concentration", dist.LogNormal(2.0, 1.0))
    with numpyro.plate("clusters", surveys.batch_shape[0]):
        f = ai.sample("f", dist.Beta(mean * concentration, (1 - mean) * concentration),
                      marginalize=True)
        ai.sample("survey", search, BinaryFraction(f=f), obs=surveys)

binaries = ai.infer(binary_model, surveys, **SETTINGS)
binaries.summary(["mean", "concentration"])
```

The clusters were simulated from a Beta(6, 9) population, with a mean fraction of 0.40 and a
concentration $a + b$ of 15. The posterior mean of the mean fraction is 0.372 with a standard
deviation of 0.025, and that of the concentration is 21.9 with a standard deviation of 8.6.

## Rules and limits

These rules apply in this version:

1. A marginalized value is used by exactly one observed site, and an observed site integrates out
   one value. A second site that uses the value raises an error naming both sites.
2. The prior is a NumPyro distribution, or a conditional distribution other than a structured
   distribution. A structured distribution as the prior raises an error; write the prior as a
   NumPyro distribution or a small class with `conjugate_prior_single`.
3. The stand-in is NaN, so anything other than the conjugate likelihood that reads the value reads
   NaN. An `ai.sample` site with a NumPyro distribution that reads it raises an error, and NumPyro
   itself refuses a NaN parameter when it validates its arguments.
4. The value's batch shape comes from the prior's batch shape and the plates around
   `ai.sample`. The observed site's batch shape must start with it; any further axes are
   measurements of the same value.
5. `ai.infer` draws the values after sampling. With your own `MCMC`, the sampler gives the other
   parameters only; to draw the values, run `numpyro.infer.Predictive` on the posterior samples
   inside `with ai.marginal.drawing():`, with the values' names in `return_sites`.

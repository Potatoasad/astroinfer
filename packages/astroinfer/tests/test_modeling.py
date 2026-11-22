"""astroinfer.sample, factor and infer, and the NumPyro runner."""
import importlib.util
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from astroinfer import ConditionalDistribution, PosteriorData, parameter, parameter_dataclass
from astroinfer.inference.runner import RunResult, run_mcmc
from astroinfer.modeling import JointLayout, factor, infer, sample, unpack
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.infer import Predictive
from numpyro.infer.util import log_density, potential_energy

QUIET = {"summary": False, "mcmc_kwargs": {"progress_bar": False}}


@parameter_dataclass(latex="p")
class Point:
    a: float = parameter(support=constraints.positive)
    b: float = parameter(support=constraints.unit_interval)


class Correlated(ConditionalDistribution):
    """a ~ Gamma(3, 1), b | a ~ Beta(1 + a, 2): one joint density."""

    data_type = Point

    def sample_single(self, key):
        k1, k2 = jax.random.split(key)
        a = dist.Gamma(3.0, 1.0).sample(k1)
        return Point(a=a, b=dist.Beta(1 + a, 2.0).sample(k2))

    def logpdf_single(self, x):
        return dist.Gamma(3.0, 1.0).log_prob(x.a) + dist.Beta(1 + x.a, 2.0).log_prob(x.b)


@parameter_dataclass
class Obs:
    y: float


@parameter_dataclass
class Loc:
    mu: float


class Noise(ConditionalDistribution):
    data_type = Obs
    parameter_type = Loc

    def sample_single(self, loc, key):
        return Obs(y=loc.mu + jax.random.normal(key))

    def logpdf_single(self, d, loc):
        return dist.Normal(loc.mu, 1.0).log_prob(d.y)


def joint_model():
    sample("p", Correlated())


def hand_model():
    a = numpyro.sample("a", dist.ImproperUniform(constraints.positive, (), ()))
    b = numpyro.sample("b", dist.ImproperUniform(constraints.unit_interval, (), ()))
    numpyro.factor("lp", dist.Gamma(3.0, 1.0).log_prob(a) + dist.Beta(1 + a, 2.0).log_prob(b))


def get_trace(model, *args, **kwargs):
    return handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(*args, **kwargs)


# ============================================================
# Sites
# ============================================================

def test_numpyro_distribution_passes_through():
    def model():
        return sample("x", dist.Normal(0.0, 1.0)), sample("y", dist.Normal(0.0, 1.0), obs=0.5)

    tr = get_trace(model)
    assert list(tr) == ["x", "y"]
    assert isinstance(tr["x"]["fn"], dist.Normal)
    assert tr["x"]["infer"]["astroinfer"] == {
        "root": "x", "path": None, "kind": "numpyro", "type": None,
    }
    assert tr["y"]["is_observed"] and tr["y"]["value"] == 0.5
    with handlers.seed(rng_seed=0):
        x, y = model()
    assert jnp.shape(x) == () and y == 0.5
    with pytest.raises(TypeError, match="no conditions"):
        with handlers.seed(rng_seed=0):
            sample("x", dist.Normal(0.0, 1.0), 1.0)


def test_joint_site_is_one_latent_site():
    tr = get_trace(joint_model)
    assert list(tr) == ["p"]
    site = tr["p"]
    assert not site["is_observed"] and jnp.shape(site["value"]) == (2,)
    meta = site["infer"]["astroinfer"]
    assert meta["root"] == "p" and meta["path"] is None and meta["kind"] == "joint"
    assert meta["type"] is Point
    layout = meta["layout"]
    assert isinstance(layout, JointLayout)
    assert layout.paths == ("a", "b") and layout.size == 2
    assert isinstance(layout.unravel(site["value"]), Point)

    with handlers.seed(rng_seed=0):
        p = sample("p", Correlated())
    assert isinstance(p, Point) and p.batch_shape == ()
    assert p.a > 0 and 0 < p.b < 1


def test_potential_energy_matches_hand_written_model():
    for z in [(0.3, -0.2), (1.5, 2.0), (-1.0, 0.0)]:
        joint = potential_energy(joint_model, (), {}, {"p": jnp.array(z)})
        hand = potential_energy(hand_model, (), {}, {"a": jnp.array(z[0]), "b": jnp.array(z[1])})
        np.testing.assert_allclose(joint, hand, rtol=1e-6)


def test_potential_gradient_matches_direct_jax():
    def direct(z):
        a, b = jnp.exp(z[0]), jax.nn.sigmoid(z[1])
        log_jac = z[0] + jnp.log(b) + jnp.log1p(-b)
        return -(Correlated().logpdf_single(Point(a=a, b=b)) + log_jac)

    z = jnp.array([0.4, -0.7])
    grad = jax.grad(lambda z: potential_energy(joint_model, (), {}, {"p": z}))(z)
    np.testing.assert_allclose(grad, jax.grad(direct)(z), rtol=1e-5)


def test_nuts_means_match_hand_written_model():
    settings = dict(seed=1, num_warmup=300, num_samples=1500, **QUIET)
    joint = run_mcmc(joint_model, **settings).sites["p"]
    hand = run_mcmc(hand_model, **settings).sites
    means = np.mean(joint, axis=(0, 1))
    # E[a] = 3 exactly; b from a large direct sample.
    np.testing.assert_allclose(means[0], np.mean(hand["a"]), atol=0.35)
    np.testing.assert_allclose(means[1], np.mean(hand["b"]), atol=0.04)
    np.testing.assert_allclose(means[0], 3.0, atol=0.35)
    direct = Correlated().sample(jax.random.key(0), N=100_000)
    np.testing.assert_allclose(means[1], np.mean(direct.b), atol=0.04)


def test_observed_joint_site_in_plate():
    E = 4
    data = Obs(y=jnp.array([0.1, -0.5, 1.2, 0.3]))

    def shared(data):
        mu = numpyro.sample("mu", dist.Normal(0.0, 3.0))
        with numpyro.plate("events", E):
            out = sample("d", Noise(), Loc(mu=mu), obs=data)
        return out

    def per_event(data):
        with numpyro.plate("events", E):
            mu = numpyro.sample("mu", dist.Normal(0.0, 3.0))
            sample("d", Noise(), Loc(mu=mu), obs={"y": data.y})

    def by_hand(data):
        mu = numpyro.sample("mu", dist.Normal(0.0, 3.0))
        with numpyro.plate("events", E):
            numpyro.sample("d", dist.Normal(mu, 1.0), obs=data.y)

    def by_hand_per_event(data):
        with numpyro.plate("events", E):
            mu = numpyro.sample("mu", dist.Normal(0.0, 3.0))
            numpyro.sample("d", dist.Normal(mu, 1.0), obs=data.y)

    tr = get_trace(shared, data)
    assert tr["d"]["is_observed"] and jnp.shape(tr["d"]["value"]) == (E, 1)
    assert tr["d"]["infer"]["astroinfer"]["kind"] == "joint"
    with handlers.seed(rng_seed=0):
        assert shared(data) is data

    for model, ref, mu in [
        (shared, by_hand, {"mu": jnp.array(0.4)}),
        (per_event, by_hand_per_event, {"mu": jnp.array([0.4, -1.0, 2.0, 0.0])}),
    ]:
        got = log_density(model, (data,), {}, mu)[0]
        np.testing.assert_allclose(got, log_density(ref, (data,), {}, mu)[0], rtol=1e-6)

    post = infer(per_event, data, seed=0, num_warmup=100, num_samples=100, **QUIET)
    assert post.parameter_names == ["mu"]
    assert post.theta.mu.shape == (1, 100, E)
    assert type(post.theta).field_specs()["mu"].index == ("events",)


def test_latent_joint_site_in_plate():
    def model():
        with numpyro.plate("events", 3):
            return sample("p", Correlated())

    tr = get_trace(model)
    assert jnp.shape(tr["p"]["value"]) == (3, 2)
    with handlers.seed(rng_seed=0):
        p = model()
    assert isinstance(p, Point) and p.batch_shape == (3,)

    draws = Predictive(model, num_samples=5)(jax.random.PRNGKey(0))
    assert draws["p"].shape == (5, 3, 2)

    post = infer(model, seed=0, num_warmup=100, num_samples=50, **QUIET)
    assert post.parameter_names == ["p.a", "p.b"]
    assert post.theta.p.a.shape == (1, 50, 3)
    assert type(post.theta.p).field_specs()["a"].index == ("events",)
    assert type(post.theta.p).field_specs()["a"].support is constraints.positive


def test_bound_distribution_and_real_line_support():
    def model():
        return sample("d", Noise().condition(Loc(mu=2.0)))

    tr = get_trace(model)
    assert tr["d"]["fn"].support(jnp.array([-5.0]))
    with handlers.seed(rng_seed=0):
        assert isinstance(model(), Obs)
    np.testing.assert_allclose(
        potential_energy(model, (), {}, {"d": jnp.array([0.5])}),
        -dist.Normal(2.0, 1.0).log_prob(0.5),
        rtol=1e-6,
    )


def test_factor():
    def model():
        x = sample("x", dist.Normal(0.0, 1.0))
        factor("extra", -0.5 * x**2)

    tr = get_trace(model)
    assert tr["extra"]["is_observed"]
    np.testing.assert_allclose(
        potential_energy(model, (), {}, {"x": jnp.array(1.5)}),
        -dist.Normal(0.0, 1.0).log_prob(1.5) + 0.5 * 1.5**2,
        rtol=1e-6,
    )


def test_predictive_draws_unpack_to_typed_objects():
    def model():
        p = sample("p", Correlated())
        sample("s", dist.Normal(p.a, 1.0))

    draws = Predictive(model, num_samples=200)(jax.random.PRNGKey(3))
    assert draws["p"].shape == (200, 2)
    values = unpack(model, draws)
    p = values["p"]
    assert isinstance(p, Point) and p.batch_shape == (200,)
    assert values["s"].shape == (200,)
    assert jnp.all(p.a > 0) and jnp.all((p.b > 0) & (p.b < 1))
    # The draws come from Correlated.sample, so b follows a.
    assert np.corrcoef(p.a, p.b)[0, 1] > 0.2


def test_latent_leaves_need_continuous_supports():
    @parameter_dataclass
    class Count:
        n: int

    class Poisson(ConditionalDistribution):
        data_type = Count

        def sample_single(self, key):
            return Count(n=jax.random.poisson(key, 3.0))

        def logpdf_single(self, x):
            return dist.Poisson(3.0).log_prob(x.n)

    with handlers.seed(rng_seed=0), pytest.raises(TypeError, match="continuous"):
        sample("c", Poisson())

    def observed():
        sample("c", Poisson(), obs=Count(n=jnp.array([1, 4, 2])))

    np.testing.assert_allclose(
        log_density(observed, (), {}, {})[0],
        dist.Poisson(3.0).log_prob(jnp.array([1, 4, 2])).sum(),
        rtol=1e-6,
    )


def test_untyped_output_needs_a_support():
    class Vector(ConditionalDistribution):
        data_type = Any

        def sample_single(self, key):
            return jnp.exp(jax.random.normal(key, (2,)))

        def logpdf_single(self, x):
            return dist.LogNormal(0.0, 1.0).log_prob(x).sum()

    class PositiveVector(Vector):
        support = constraints.positive

    with handlers.seed(rng_seed=0), pytest.raises(TypeError, match="no support"):
        sample("x", Vector())
    with handlers.seed(rng_seed=0):
        x = sample("x", PositiveVector())
    assert jnp.shape(x) == (2,)
    np.testing.assert_allclose(
        potential_energy(lambda: sample("x", PositiveVector()), (), {}, {"x": jnp.zeros(2)}),
        -2 * dist.LogNormal(0.0, 1.0).log_prob(1.0),
        rtol=1e-6,
    )


@parameter_dataclass
class Pair:
    point: Point
    c: float


class PairDistribution(ConditionalDistribution):
    data_type = Pair

    def sample_single(self, key):
        k1, k2 = jax.random.split(key)
        return Pair(point=Correlated().sample_single(k1), c=jax.random.normal(k2))

    def logpdf_single(self, x):
        return Correlated().logpdf_single(x.point) + dist.Normal(0.0, 1.0).log_prob(x.c)


def test_nested_output_type_uses_nested_supports():
    def model():
        return sample("pair", PairDistribution())

    layout = get_trace(model)["pair"]["infer"]["astroinfer"]["layout"]
    assert layout.paths == ("point.a", "point.b", "c")
    z = jnp.array([0.2, -0.3, 0.7])
    a, b = jnp.exp(z[0]), jax.nn.sigmoid(z[1])
    log_jac = z[0] + jnp.log(b) + jnp.log1p(-b)
    expected = -(Correlated().logpdf_single(Point(a=a, b=b)) + log_jac
                 + dist.Normal(0.0, 1.0).log_prob(z[2]))
    np.testing.assert_allclose(potential_energy(model, (), {}, {"pair": z}), expected, rtol=1e-5)

    post = infer(model, seed=0, num_warmup=50, num_samples=30, **QUIET)
    assert post.parameter_names == ["pair.point.a", "pair.point.b", "pair.c"]
    assert isinstance(post.theta.pair, Pair) and isinstance(post.theta.pair.point, Point)


# ============================================================
# infer and the runner
# ============================================================

def test_infer_builds_posterior_with_nested_theta():
    y = jnp.array([0.4, 0.9, 1.3])

    def model(y):
        p = sample("p", Correlated())
        s = sample("s", dist.Normal(0.0, 1.0))
        numpyro.deterministic("ab", p.a * p.b)
        log_like = dist.Normal(p.b + s, 1.0).log_prob(y).sum()
        numpyro.deterministic("log_likelihood", log_like)
        numpyro.sample("y", dist.Normal(p.b + s, 1.0), obs=y)

    post = infer(
        model, y, seed=2, num_warmup=100, num_samples=80, num_chains=2,
        summary=False, mcmc_kwargs={"progress_bar": False, "chain_method": "sequential"},
    )
    assert isinstance(post, PosteriorData)
    assert post.sample_dims == ("chain", "draw") and post.batch_shape == (2, 80)
    assert isinstance(post.theta.p, Point)
    assert post.parameter_names == ["p.a", "p.b", "s"]
    assert list(post.extras) == ["ab"]
    np.testing.assert_allclose(post.extras["ab"], post.theta.p.a * post.theta.p.b, rtol=1e-5)
    assert {"log_likelihood", "potential_energy", "diverging"} <= set(post.sample_stats)
    assert "y" not in post.extras and "y" not in post.parameter_names
    run = post.metadata["run"]
    assert isinstance(run, RunResult) and run.seed == 2
    np.testing.assert_array_equal(run.sites["s"], post.theta.s)


def test_run_mcmc_matches_gravpop_sampler():
    gp = pytest.importorskip("gravpop")
    y = jnp.array([0.3, 1.2, -0.4, 0.8, 2.0])

    class Likelihood:
        def logpdf(self, x):
            return dist.Normal(x["mu"], x["sigma"]).log_prob(y).sum()

    priors = {"mu": dist.Normal(0.0, 5.0), "sigma": dist.HalfNormal(2.0)}
    settings = dict(num_warmup=150, num_samples=100, target_accept_prob=0.8, max_tree_depth=5)
    sampler = gp.Sampler(priors=priors, likelihood=Likelihood(), seed=7, summary=False, **settings)
    sampler.sample()

    model = gp.Sampler(priors=priors, likelihood=Likelihood()).model
    run = run_mcmc(model, seed=7, **settings, **QUIET)
    assert run.sample_dims == ("chain", "draw")
    expected = sampler.get_samples(group_by_chain=True)
    assert set(run.sites) == set(expected) == {"mu", "sigma", "log_likelihood"}
    for name, value in run.sites.items():
        np.testing.assert_array_equal(value, expected[name])
    stats = sampler.get_sample_stats(group_by_chain=True)
    assert set(run.sample_stats) == set(stats)
    for name, value in run.sample_stats.items():
        np.testing.assert_array_equal(value, stats[name])
    np.testing.assert_array_equal(run.get_samples()["mu"], sampler.get_samples()["mu"])


@pytest.mark.skipif(importlib.util.find_spec("jaxns") is None, reason="needs jaxns")
def test_nested_sampling():
    def model():
        sample("p", Correlated())

    run = run_mcmc(model, method="nested", seed=0, num_samples=200, summary=False)
    assert run.sample_dims == ("draw",) and run.sites["p"].shape == (200, 2)
    post = infer(model, method="nested", seed=0, num_samples=200, summary=False)
    assert post.sample_dims == ("draw",) and isinstance(post.theta.p, Point)

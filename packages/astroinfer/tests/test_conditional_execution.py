"""Structured distributions and mixtures inside NumPyro models (phase 5)."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from astroinfer import (
    MeasurementProcess,
    MixtureDistribution,
    PosteriorData,
    conditional,
    mixture,
    parameter_dataclass,
)
from astroinfer.inference.runner import run_mcmc
from astroinfer.modeling import infer, sample, unpack
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.infer import Predictive
from numpyro.infer.util import log_density, potential_energy

QUIET = {"summary": False, "mcmc_kwargs": {"progress_bar": False}}


@parameter_dataclass(latex="m")
class Masses:
    m1: float
    q: float


@parameter_dataclass(latex="s")
class Spins:
    chi: float


@parameter_dataclass(latex=r"\theta")
class BBH:
    masses: Masses
    spins: Spins


@parameter_dataclass(latex=r"\alpha")
class Alpha:
    alpha: float


@parameter_dataclass
class SpinObs:
    chi_obs: float


MASS_PRIOR = Masses.distribution(m1=dist.Uniform(5.0, 50.0), q=dist.Uniform(0.1, 1.0))
LOW = Spins.distribution(chi=dist.Beta(1.5, 6.0))
HIGH = Spins.distribution(chi=dist.Beta(4.0, 2.0))


def fraction(m1, alpha):
    """Weight of the high-spin component, rising with m1."""
    return jax.nn.sigmoid((m1 - 30.0) / 5.0 + alpha)


def weights(masses, alpha):
    f = fraction(masses.m1, alpha)
    return jnp.stack([1.0 - f, f])


SPIN_MIXTURE = mixture([LOW, HIGH], weights=weights)


def mixture_log_density(chi, m1, alpha):
    f = fraction(m1, alpha)
    return logsumexp(
        jnp.stack([dist.Beta(1.5, 6.0).log_prob(chi), dist.Beta(4.0, 2.0).log_prob(chi)]),
        b=jnp.stack([1.0 - f, f]),
        axis=0,
    )


def mixture_model():
    masses = sample("masses", MASS_PRIOR)
    alpha = sample("alpha", dist.Normal(0.0, 1.0))
    spins = sample("spins", SPIN_MIXTURE, masses=masses, alpha=alpha)
    return BBH(masses=masses, spins=spins)


def hand_mixture_model():
    m1 = numpyro.sample("m1", dist.Uniform(5.0, 50.0))
    numpyro.sample("q", dist.Uniform(0.1, 1.0))
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    chi = numpyro.sample("chi", dist.ImproperUniform(constraints.unit_interval, (), ()))
    numpyro.factor("spins", mixture_log_density(chi, m1, alpha))


def get_trace(model, *args):
    return handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(*args)


def joint_point(z):
    return {
        "masses.m1": z[0], "masses.q": z[1], "alpha": z[2], "spins": z[3:4],
    }


def hand_point(z):
    return {"m1": z[0], "q": z[1], "alpha": z[2], "chi": z[3]}


POINTS = [jnp.array(z) for z in ([0.1, -0.4, 0.3, 0.8], [1.2, 0.5, -1.0, -2.0], [-0.7, 2.0, 0.0, 0.1])]


# ============================================================
# The mass-dependent spin mixture
# ============================================================

def test_mixture_model_sites():
    tr = get_trace(mixture_model)
    assert [k for k, s in tr.items() if s["type"] == "sample"] == [
        "masses.m1", "masses.q", "alpha", "spins",
    ]
    assert jnp.shape(tr["masses.m1"]["value"]) == ()
    assert tr["masses.m1"]["infer"]["astroinfer"] == {
        "root": "masses", "path": "m1", "kind": "leaf", "type": Masses,
    }
    # One joint site: the label is never a latent variable.
    spins = tr["spins"]
    assert spins["infer"]["astroinfer"]["kind"] == "joint"
    assert jnp.shape(spins["value"]) == (1,) and not spins["is_observed"]
    assert isinstance(spins["fn"].distribution, MixtureDistribution)
    with handlers.seed(rng_seed=0):
        theta = mixture_model()
    assert isinstance(theta, BBH) and 0 < theta.spins.chi < 1


def test_mixture_potential_matches_hand_marginalized_model():
    for z in POINTS:
        joint = potential_energy(mixture_model, (), {}, joint_point(z))
        hand = potential_energy(hand_mixture_model, (), {}, hand_point(z))
        np.testing.assert_allclose(joint, hand, rtol=1e-6)


def test_mixture_gradients_are_finite_and_match():
    z = POINTS[0]
    grad = jax.grad(lambda z: potential_energy(mixture_model, (), {}, joint_point(z)))(z)
    hand = jax.grad(lambda z: potential_energy(hand_mixture_model, (), {}, hand_point(z)))(z)
    assert np.all(np.isfinite(grad))
    # The masses and alpha enter through the mixture weights.
    assert grad[0] != 0 and grad[2] != 0
    np.testing.assert_allclose(grad, hand, rtol=1e-5)


def test_mixture_nuts_means_match_hand_written_model():
    settings = dict(seed=0, num_warmup=300, num_samples=1000, **QUIET)
    joint = run_mcmc(mixture_model, **settings).sites
    hand = run_mcmc(hand_mixture_model, **settings).sites
    assert np.all(np.isfinite(joint["spins"]))
    np.testing.assert_allclose(np.mean(joint["spins"]), np.mean(hand["chi"]), atol=0.05)
    np.testing.assert_allclose(np.mean(joint["masses.m1"]), np.mean(hand["m1"]), atol=2.5)
    np.testing.assert_allclose(np.mean(joint["alpha"]), np.mean(hand["alpha"]), atol=0.2)


def test_structured_population_gives_same_log_density():
    def spin_leaf(x, a):
        f = fraction(x["masses.m1"], a.alpha)
        return dist.MixtureSameFamily(
            dist.Categorical(probs=jnp.stack([1.0 - f, f])),
            dist.Beta(jnp.array([1.5, 4.0]), jnp.array([6.0, 2.0])),
        )

    population = BBH.distribution(
        masses=MASS_PRIOR,
        spins=Spins.distribution(chi=conditional(spin_leaf, on="masses.m1")),
    ).given(Alpha)

    def structured_model():
        alpha = sample("alpha", dist.Normal(0.0, 1.0))
        return sample("bbh", population, Alpha(alpha=alpha))

    tr = get_trace(structured_model)
    assert [k for k, s in tr.items() if s["type"] == "sample"] == [
        "alpha", "bbh.masses.m1", "bbh.masses.q", "bbh.spins.chi",
    ]
    for z in POINTS:
        point = {
            "alpha": z[2], "bbh.masses.m1": z[0], "bbh.masses.q": z[1], "bbh.spins.chi": z[3],
        }
        np.testing.assert_allclose(
            potential_energy(structured_model, (), {}, point),
            potential_energy(mixture_model, (), {}, joint_point(z)),
            rtol=1e-6,
        )
    theta = BBH(masses=Masses(m1=33.0, q=0.4), spins=Spins(chi=0.6))
    np.testing.assert_allclose(
        population.logpdf(theta, Alpha(alpha=0.3)),
        MASS_PRIOR.logpdf(theta.masses)
        + SPIN_MIXTURE.logpdf(theta.spins, masses=theta.masses, alpha=0.3),
        rtol=1e-6,
    )


# ============================================================
# Observed likelihoods and plates
# ============================================================

class SpinNoise(MeasurementProcess):
    parameter_type = BBH
    data_type = SpinObs

    def sample_single(self, theta, key):
        return SpinObs(chi_obs=theta.spins.chi + 0.1 * jax.random.normal(key))

    def logpdf_single(self, d, theta):
        return dist.Normal(theta.spins.chi, 0.1).log_prob(d.chi_obs)


E = 3
DATA = SpinObs(chi_obs=jnp.array([0.2, 0.7, 0.5]))


def hierarchical_model(data):
    alpha = sample("alpha", dist.Normal(0.0, 1.0))
    with numpyro.plate("events", E):
        masses = sample("masses", MASS_PRIOR)
        spins = sample("spins", SPIN_MIXTURE, masses=masses, alpha=alpha)
        sample("d", SpinNoise(), BBH(masses=masses, spins=spins), obs=data)


def hand_hierarchical_model(data):
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    with numpyro.plate("events", E):
        m1 = numpyro.sample("masses.m1", dist.Uniform(5.0, 50.0))
        numpyro.sample("masses.q", dist.Uniform(0.1, 1.0))
        chi = numpyro.sample("chi", dist.ImproperUniform(constraints.unit_interval, (), ()))
        numpyro.factor("spins", mixture_log_density(chi, m1, alpha))
        numpyro.sample("d", dist.Normal(chi, 0.1), obs=data.chi_obs)


def test_observed_measurement_in_plate():
    tr = get_trace(hierarchical_model, DATA)
    assert jnp.shape(tr["masses.m1"]["value"]) == (E,)
    assert jnp.shape(tr["spins"]["value"]) == (E, 1)
    assert tr["d"]["is_observed"] and tr["d"]["infer"]["astroinfer"]["kind"] == "joint"
    key = jax.random.PRNGKey(4)
    for z in [jax.random.normal(k, (1 + 3 * E,)) for k in jax.random.split(key, 3)]:
        joint = {
            "alpha": z[0], "masses.m1": z[1:4], "masses.q": z[4:7], "spins": z[7:10, None],
        }
        hand = {"alpha": z[0], "masses.m1": z[1:4], "masses.q": z[4:7], "chi": z[7:10]}
        np.testing.assert_allclose(
            potential_energy(hierarchical_model, (DATA,), {}, joint),
            potential_energy(hand_hierarchical_model, (DATA,), {}, hand),
            rtol=1e-6,
        )

    post = infer(hierarchical_model, DATA, seed=1, num_warmup=100, num_samples=60, **QUIET)
    assert post.parameter_names == ["alpha", "masses.m1", "masses.q", "spins.chi"]
    assert post.theta.masses.m1.shape == (1, 60, E)
    assert type(post.theta.spins).field_specs()["chi"].index == ("events",)


def test_structured_observed_leaves():
    def model(theta):
        return sample("bbh", BBH.distribution(masses=MASS_PRIOR, spins=LOW), obs=theta)

    theta = BBH(masses=Masses(m1=20.0, q=0.5), spins=Spins(chi=0.3))
    tr = get_trace(model, theta)
    assert all(tr[k]["is_observed"] for k in ("bbh.masses.m1", "bbh.masses.q", "bbh.spins.chi"))
    np.testing.assert_allclose(
        log_density(model, (theta,), {}, {})[0],
        BBH.distribution(masses=MASS_PRIOR, spins=LOW).logpdf(theta),
        rtol=1e-6,
    )
    with handlers.seed(rng_seed=0):
        assert model({"masses": {"m1": 20.0, "q": 0.5}, "spins": {"chi": 0.3}}).masses.m1 == 20.0


@parameter_dataclass
class Bounds:
    lo: float
    hi: float


BOUNDED = Masses.distribution(
    m1=lambda b: dist.Uniform(b.lo, b.hi),
    q=conditional(lambda x, b: dist.Uniform(b.lo / x["m1"], 1.0), on="m1"),
).given(Bounds)
BOUNDS = Bounds(lo=jnp.array([5.0, 6.0, 8.0]), hi=jnp.array([20.0, 30.0, 50.0]))


def test_structured_prior_in_plate_with_per_event_condition():
    def model(b):
        with numpyro.plate("events", 3):
            return sample("m", BOUNDED, b)

    def hand(b):
        with numpyro.plate("events", 3):
            m1 = numpyro.sample("m.m1", dist.Uniform(b.lo, b.hi))
            numpyro.sample("m.q", dist.Uniform(b.lo / m1, 1.0))

    tr = get_trace(model, BOUNDS)
    assert tr["m.m1"]["fn"].batch_shape == (3,) and tr["m.q"]["fn"].batch_shape == (3,)
    with handlers.seed(rng_seed=0):
        m = model(BOUNDS)
    assert isinstance(m, Masses) and m.batch_shape == (3,)
    assert jnp.all((m.m1 > BOUNDS.lo) & (m.m1 < BOUNDS.hi)) and jnp.all(m.q > BOUNDS.lo / m.m1)
    for z in [jnp.array([0.1, -0.3, 1.0, 0.5, 0.2, -1.0]), jnp.zeros(6)]:
        point = {"m.m1": z[:3], "m.q": z[3:]}
        np.testing.assert_allclose(
            potential_energy(model, (BOUNDS,), {}, point),
            potential_energy(hand, (BOUNDS,), {}, point),
            rtol=1e-6,
        )

    post = infer(model, BOUNDS, seed=0, num_warmup=100, num_samples=100, **QUIET)
    m = post.theta.m
    assert m.m1.shape == (1, 100, 3)
    assert jnp.all((m.m1 > BOUNDS.lo) & (m.m1 < BOUNDS.hi)) and jnp.all(m.q > BOUNDS.lo / m.m1)


# ============================================================
# infer and Predictive on structured priors
# ============================================================

BBH_PRIOR = BBH.distribution(
    masses=MASS_PRIOR, spins=Spins.distribution(chi=0.0)
)


def test_infer_structured_prior_keeps_fixed_fields():
    def model():
        bbh = sample("bbh", BBH_PRIOR)
        numpyro.deterministic("m2", bbh.masses.m1 * bbh.masses.q)

    tr = get_trace(model)
    assert tr["bbh.spins.chi"]["type"] == "deterministic"
    post = infer(model, seed=0, num_warmup=100, num_samples=100, **QUIET)
    assert isinstance(post, PosteriorData) and post.sample_dims == ("chain", "draw")
    assert isinstance(post.theta.bbh, BBH)
    assert post.parameter_names == ["bbh.masses.m1", "bbh.masses.q", "bbh.spins.chi"]
    np.testing.assert_array_equal(post.theta.bbh.spins.chi, 0.0)
    assert post.theta.bbh.spins.chi.shape == (1, 100)
    assert list(post.extras) == ["m2"]


def test_predictive_on_structured_prior():
    def model():
        sample("bbh", BBH_PRIOR)

    draws = Predictive(model, num_samples=50)(jax.random.PRNGKey(0))
    assert set(draws) == {"bbh.masses.m1", "bbh.masses.q", "bbh.spins.chi"}
    bbh = unpack(model, draws)["bbh"]
    assert isinstance(bbh, BBH) and bbh.batch_shape == (50,)
    assert jnp.all((bbh.masses.m1 > 5) & (bbh.masses.m1 < 50))
    np.testing.assert_array_equal(bbh.spins.chi, 0.0)


def test_block_field_inside_structured_root():
    population = BBH.distribution(masses=MASS_PRIOR, spins=mixture([LOW, HIGH], weights=jnp.array([0.3, 0.7])))

    def model():
        return sample("bbh", population)

    tr = get_trace(model)
    assert tr["bbh.spins"]["infer"]["astroinfer"]["kind"] == "joint"
    assert tr["bbh.spins"]["infer"]["astroinfer"]["path"] == "spins"
    with handlers.seed(rng_seed=0):
        assert isinstance(model(), BBH)
    z = jnp.array([0.3, -0.2, 0.5])
    point = {"bbh.masses.m1": z[0], "bbh.masses.q": z[1], "bbh.spins": z[2:]}
    chi = jax.nn.sigmoid(z[2])
    theta = BBH(masses=Masses(m1=5 + 45 * jax.nn.sigmoid(z[0]), q=0.1 + 0.9 * jax.nn.sigmoid(z[1])),
                spins=Spins(chi=chi))
    log_jac = sum(jnp.log(w) + jax.nn.log_sigmoid(v) + jax.nn.log_sigmoid(-v)
                  for w, v in [(45.0, z[0]), (0.9, z[1]), (1.0, z[2])])
    np.testing.assert_allclose(
        potential_energy(model, (), {}, point), -(population.logpdf(theta) + log_jac), rtol=1e-5
    )
    draws = Predictive(model, num_samples=20)(jax.random.PRNGKey(1))
    bbh = unpack(model, draws)["bbh"]
    assert isinstance(bbh.spins, Spins) and bbh.batch_shape == (20,)
    post = infer(model, seed=0, num_warmup=50, num_samples=40, **QUIET)
    assert post.parameter_names == ["bbh.masses.m1", "bbh.masses.q", "bbh.spins.chi"]


def test_structured_rejects_unknown_keywords():
    with handlers.seed(rng_seed=0), pytest.raises(TypeError, match="unexpected keyword"):
        sample("bbh", BBH_PRIOR, extra=1.0)

"""Run gravpop's NUTS sampler on astroinfer likelihoods.

gravpop is imported lazily so the rest of astroinfer (parameters,
measurement processes, generative models) works without it.

Every entry point builds a ``PosteriorData`` straight from the sampler's
arrays. NUTS draws keep their chains, ``sample_dims=("chain", "draw")``;
nested-sampling draws have ``sample_dims=("draw",)``.
"""
from dataclasses import fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..parameters import coerce, parameter, parameter_dataclass
from ..population import as_population_model
from ..posterior import PosteriorData, _is_sample_stat
from .likelihoods import (
    FixedDataLikelihood,
    JointPopulationLikelihood,
    PopulationLikelihood,
)

# gravpop site names of the per-observation parameters in joint inference.
LOCAL_PREFIX = "theta."
# numpyro scope for the internal sites of prior blocks used as local priors.
_BLOCK_SCOPE = "_local"


def _resolve_seed(seed) -> int:
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    return int(np.random.default_rng(seed).integers(2**30))


def _prior_variables(key, prior) -> tuple[str, ...]:
    """The fields a PE prior entry has a density for."""
    names = getattr(prior, "var_names", None)
    return (key,) if names is None else tuple(names)


def _factor_name(names) -> str:
    return f"log_prior[{', '.join(names)}]"


def _prior_factors(priors) -> dict[tuple[str, ...], Any]:
    """``{variable names: prior}`` for the entries that have a density."""
    return {
        _prior_variables(k, prior): prior
        for k, prior in dict(priors).items()
        if hasattr(prior, "log_prob")
    }


def _log_prior_factors(factors, values) -> dict[str, Any]:
    """``{"log_prior[x, y]": log density}`` of every factor at ``values``."""
    out = {}
    for names, prior in factors.items():
        if hasattr(prior, "var_names"):
            log_prob = prior.log_prob({v: jnp.asarray(values[v]) for v in names})
        else:
            log_prob = prior.log_prob(jnp.asarray(values[names[0]]))
        out[_factor_name(names)] = log_prob
    return out


def run_sampler(likelihood, priors, *, seed=None, **sampler_kwargs):
    """Build and run a ``gravpop.Sampler``; return it after sampling."""
    import gravpop as gp

    sampler = gp.Sampler(
        likelihood=likelihood,
        priors=dict(priors),
        seed=_resolve_seed(seed),
        **sampler_kwargs,
    )
    sampler.sample()
    return sampler


def _sampler_arrays(sampler):
    """``(sites, sample_stats, sample_dims)`` as arrays from a run sampler."""
    if sampler.sampler_type == "nuts":
        return (
            sampler.get_samples(group_by_chain=True),
            sampler.get_sample_stats(group_by_chain=True),
            ("chain", "draw"),
        )
    return sampler.get_samples(), sampler.get_sample_stats(), ("draw",)


def _type_from_arrays(type_name, values, n_batch):
    """A parameter dataclass with one field per array in ``values``.

    Axes after the first ``n_batch`` become core axes named
    ``<field>_dim_<i>``.
    """
    namespace = {"__annotations__": {}, "__module__": __name__}
    for name, value in values.items():
        extra = np.ndim(value) - n_batch
        namespace["__annotations__"][name] = Any
        namespace[name] = parameter(
            index=tuple(f"{name}_dim_{i}" for i in range(extra))
        )
    return parameter_dataclass(type(type_name, (), namespace))


def _hyperparameter_posterior(
    sites, sample_stats, sample_dims, population, *, extras=None, **kwargs
):
    """PosteriorData with Lambda as theta.

    With ``population.hyperparameter_type`` its fields become theta and every
    other site an extra. Without one, theta gets a type built from every
    sampled hyperparameter (including variables set by prior blocks).
    """
    stats = dict(sample_stats)
    others = {}
    for name, value in sites.items():
        if _is_sample_stat(name):
            stats[name] = value
        else:
            others[name] = value

    cls = population.hyperparameter_type
    if cls is None:
        names = [k for k in others if k.isidentifier()]
        cls = _type_from_arrays(
            "Hyperparameters", {k: others[k] for k in names}, len(sample_dims)
        )
    names = [f.name for f in fields(cls)]
    missing = [k for k in names if k not in others]
    if missing:
        raise KeyError(
            f"{cls.__name__} fields {missing} were not sampled; "
            "give every hyperparameter a prior."
        )

    theta = cls.from_dict({k: jnp.asarray(others.pop(k)) for k in names})
    return PosteriorData(
        theta=theta,
        sample_stats=stats,
        extras={**others, **(extras or {})},
        sample_dims=sample_dims,
        **kwargs,
    )


# ============================================================
# Parameter estimation
# ============================================================

def parameter_estimation(
    measurement,
    d,
    pe_priors,
    seed=None,
    *,
    as_posterior=False,
    **sampler_kwargs,
):
    """
    Posterior samples of theta given one observation d.

    Uses L(d | theta) as the likelihood.
    Detection selection is handled separately.

    ``pe_priors`` maps names to numpyro distributions or gravpop
    prior blocks. A prior block with ``var_names`` and
    ``log_prob(theta)`` gives the joint density of those fields;
    blocks without ``log_prob`` (e.g. ``gravpop.Deterministic``)
    only set variables. Every field needs a density.

    Returns a PosteriorData when ``as_posterior=True``, with sample dims
    ``("chain", "draw")`` for NUTS. Its sample_stats hold the log
    likelihood, the PE prior density ``prior``, log prior densities per
    distribution (``log_prior[x]``) or block (``log_prior[x, y]``), and the
    sampler's per-draw statistics; its extras hold the derived parameters.

    Otherwise returns the same draws as a pandas DataFrame
    (``posterior.to_dataframe()``), one row per draw.

    The fitted sampler is stored on ``measurement.sampler``.
    """
    pe_priors = dict(pe_priors)
    factors = _prior_factors(pe_priors)
    covered = {v for names in factors for v in names}
    missing = [
        k for k in measurement.parameter_names
        if k not in covered
    ]
    if missing:
        raise KeyError(f"pe_priors has no density for {missing}")

    if measurement.data_type is not None:
        d = coerce(d, measurement.data_type)
    else:
        d = {k: float(v) for k, v in dict(d).items()}

    seed = _resolve_seed(seed)

    sampler = run_sampler(
        FixedDataLikelihood(measurement, d),
        pe_priors,
        seed=seed,
        **{**measurement.sampler_kwargs, **sampler_kwargs},
    )
    measurement.sampler = sampler

    sites, sample_stats, sample_dims = _sampler_arrays(sampler)
    names = list(measurement.parameter_names)

    parameter_type = measurement.parameter_type
    if parameter_type is None:
        parameter_type = _type_from_arrays(
            "Parameters", {k: sites[k] for k in names}, len(sample_dims)
        )
    theta = parameter_type.from_dict({k: jnp.asarray(sites[k]) for k in names})

    stats = dict(sample_stats)
    extras = {}
    for name, value in sites.items():
        if name in names:
            continue
        (stats if _is_sample_stat(name) else extras)[name] = value

    log_priors = _log_prior_factors(factors, sites)
    stats.update(log_priors)
    stats["prior"] = jnp.broadcast_to(
        jnp.exp(sum(log_priors.values())), theta.batch_shape
    )

    taken = set(names) | set(stats) | set(extras)
    for name, value in measurement.derived_parameters(theta).items():
        if name not in taken:
            extras[name] = value

    posterior = PosteriorData(
        theta=theta,
        sample_stats=stats,
        extras=extras,
        metadata={
            "inference": "PE",
            "sampler": "gravpop",
            "seed": seed,
            "num_samples": int(np.prod(theta.batch_shape)),
        },
        observed_data=d,
        sample_dims=sample_dims,
    )
    return posterior if as_posterior else posterior.to_dataframe()


# ============================================================
# Population inference, theta_i marginalized
# ============================================================

def population_inference(
    measurement,
    data,
    pop_model,
    pop_priors,
    *,
    N=2000,
    seed=None,
    conditioned_on=None,
    proposal=None,
    likelihood_cls=PopulationLikelihood,
    **sampler_kwargs,
):
    """
    Infer population hyperparameters Lambda from a batch
    of independently observed systems, with each theta_i
    integrated out by Monte Carlo.

    data:
        Typed data object with batch_shape=(E,).

    pop_model:
        PopulationModel p(theta | Lambda), or a GenerativeModel
        (wrapped with PopulationModel.from_generative).

    conditioned_on:
        Values of a GenerativeModel's external variables.

    pop_priors:
        gravpop-compatible hyperparameter prior dict.

    N:
        Monte Carlo population draws per observation.

    proposal:
        Optional Proposal q_Lambda(theta | d) for importance-sampled
        per-event integrals (see PopulationLikelihood).

    likelihood_cls:
        PopulationLikelihood or a subclass of it, e.g. one overriding
        ``log_weights``.

    Returns:
        PosteriorData whose theta is Lambda (typed by the population's
        ``hyperparameter_type`` if it declares one), with sample dims
        ``("chain", "draw")``. The Monte Carlo diagnostics (``N_eff_min``,
        ...) are sample stats. The likelihood and sampler are stored on
        ``measurement.population_likelihood`` /
        ``measurement.population_sampler``.
    """
    seed = _resolve_seed(seed)

    likelihood = likelihood_cls(
        measurement=measurement,
        data=data,
        pop_model=pop_model,
        N=N,
        seed=seed,
        conditioned_on=conditioned_on,
        proposal=proposal,
    )

    sampler = run_sampler(
        likelihood,
        pop_priors,
        seed=seed,
        **{**measurement.sampler_kwargs, **sampler_kwargs},
    )

    measurement.population_likelihood = likelihood
    measurement.population_sampler = sampler

    sites, sample_stats, sample_dims = _sampler_arrays(sampler)
    return _hyperparameter_posterior(
        sites,
        sample_stats,
        sample_dims,
        likelihood.population,
        observed_data=likelihood.data,
        metadata={
            "inference": "population",
            "sampler": "gravpop",
            "seed": seed,
            "N": likelihood.N,
            "num_observations": likelihood.E,
        },
    )


# ============================================================
# Joint population inference: Lambda and every theta_i sampled
# ============================================================

def _hyperprior_point(pop_priors, seed, num=201):
    """A central Lambda: the per-site median of hyperprior draws.

    Variables set by prior blocks are recomputed at those medians.
    """
    import gravpop as gp
    from numpyro import handlers
    from numpyro.infer import Predictive

    prior_model = gp.Sampler(priors=dict(pop_priors), just_prior=True, summary=False)
    draws = Predictive(prior_model.model, num_samples=num)(jax.random.key(seed))
    point = {k: jnp.median(jnp.asarray(v), axis=0) for k, v in draws.items()}
    prior_model.x = {}
    with handlers.seed(rng_seed=seed), handlers.substitute(data=point):
        prior_model.model()
    return dict(prior_model.x)


def _core_shape(parameter_type, name, template):
    n = len(parameter_type.field_specs()[name].index)
    if not n:
        return ()
    if template is None:
        raise ValueError(
            f"{parameter_type.__name__}.{name} has core axes; their lengths "
            "come from init, so pass init= a Proposal or a typed theta."
        )
    return tuple(np.shape(getattr(template, name))[-n:])


def _local_prior_block(local_priors, factors, num_obs):
    """A gravpop prior block that draws theta_i from per-event PE priors.

    The PE prior's density is cancelled by a factor, so it only sets the
    coordinates, support and starting distribution of each theta_i site;
    the joint density is unchanged.
    """
    import numpyro
    from gravpop.sampler.sampler import PriorBlock
    from numpyro import handlers

    class LocalPriors(PriorBlock):
        def sample(self, x):
            local = {}
            with numpyro.plate("observation", num_obs):
                for name, prior in local_priors.items():
                    if isinstance(prior, PriorBlock):
                        with handlers.scope(prefix=_BLOCK_SCOPE):
                            prior.sample(local)
                    else:
                        local[name] = numpyro.sample(LOCAL_PREFIX + name, prior)
                for name, value in local.items():
                    if name not in local_priors:
                        numpyro.deterministic(LOCAL_PREFIX + name, value)
                log_prior = _log_prior_factors(factors, local)
            numpyro.factor(
                "local_prior_cancellation",
                -sum(jnp.sum(v) for v in log_prior.values()),
            )
            x.update({LOCAL_PREFIX + k: v for k, v in local.items()})

    return LocalPriors()



def _feasible_start(theta, Lambda, data, measurement, population, supports,
                    *, redraw=None, key, attempts=20):
    """theta_i with a finite starting density for every observation.

    An observation is feasible when every started field lies in its site's
    support and log L(d_i | theta_i) + log p(theta_i | Lambda) is finite.
    Infeasible observations are redrawn from ``redraw`` (a Proposal) up to
    ``attempts`` times; any left over raise an error.
    """
    num_obs = data.batch_shape[0]

    def feasible(theta):
        ok = jnp.isfinite(
            measurement.logpdf(data, theta) + population.logpdf(theta, Lambda)
        )
        for name, support in supports.items():
            inside = jnp.asarray(support.check(jnp.asarray(getattr(theta, name))))
            ok &= inside.reshape(num_obs, -1).all(axis=1)
        return np.asarray(ok)

    ok = feasible(theta)
    for _ in range(attempts if redraw is not None else 0):
        if ok.all():
            break
        key, sub = jax.random.split(key)
        fresh = redraw.sample(data, sub, Lambda=Lambda)

        def keep_feasible(old, new, ok=ok):
            mask = ok.reshape((num_obs,) + (1,) * (jnp.ndim(old) - 1))
            return jnp.where(mask, old, new)

        theta = jax.tree_util.tree_map(keep_feasible, theta, fresh)
        ok = feasible(theta)

    if not ok.all():
        bad = np.flatnonzero(~ok)
        raise ValueError(
            f"No valid starting point for observations {bad[:10].tolist()}"
            + (" ..." if len(bad) > 10 else "")
            + ": theta_i lies outside its support, or L(d_i | theta_i) "
            "p(theta_i | Lambda) is zero at init_Lambda. Pass a feasible init "
            "or init_Lambda."
        )
    return theta


def joint_population_inference(
    measurement,
    data,
    pop_model,
    pop_priors,
    *,
    local_priors=None,
    init=None,
    init_Lambda=None,
    seed=None,
    conditioned_on=None,
    **sampler_kwargs,
):
    """
    Sample the hyperparameters Lambda and every observation's theta_i
    together from

        p(Lambda, {theta_i} | {d_i})
            ∝ p(Lambda) prod_i L(d_i | theta_i) p(theta_i | Lambda).

    Ignoring the theta_i draws gives the same p(Lambda | {d_i}) as
    ``population_inference``, with no Monte Carlo integral. The population
    must provide a ``logpdf`` normalized over theta for every Lambda; its
    ``sample`` is not used.

    data:
        Typed data object with batch_shape=(E,).

    pop_model, pop_priors, conditioned_on:
        As for ``population_inference``.

    local_priors:
        Optional per-event PE priors, in the same form as for ``PE``
        (numpyro distributions or gravpop prior blocks with ``log_prob``),
        applied to every theta_i. Their density is divided back out, so the
        target is unchanged: they set each site's support and
        coordinates. Their support must contain the population's, or the
        population is truncated. Fields without a local prior are sampled
        on their declared ``parameter(support=...)`` with a flat density.
        Prior blocks with a ``cleanup`` step are not supported.

    init:
        None: numpyro's default start (uniform in (-2, 2) on the
        unconstrained scale).
        A Proposal q(theta | d): theta_i starts at one draw from q at
        ``init_Lambda``.
        A typed theta with batch shape (E,): theta_i starts there.
        Lambda always starts at numpyro's default.

    init_Lambda:
        Lambda at which a Proposal init is drawn. Default: the median of
        hyperprior draws.

    sampler_kwargs:
        Passed to ``gravpop.Sampler``. NUTS settings such as
        ``dense_mass`` go in its ``sampler_kwargs`` dict;
        ``max_tree_depth`` defaults to 10 here.

    Returns:
        PosteriorData with theta = Lambda, sample dims ("chain", "draw"),
        and ``local`` = the theta_i draws, batch (chain, draw, E). extras hold
        the population's ``derived_parameters(theta_i, Lambda)`` and the
        measurement's ``derived_parameters(theta_i)``, per draw and
        observation.
    """
    from gravpop.sampler.sampler import PriorBlock
    from numpyro.distributions import Distribution, ImproperUniform
    from numpyro.infer import init_to_value

    from ..latent import LatentSpace
    from ..measurement import Proposal

    seed = _resolve_seed(seed)
    parameter_type = measurement.parameter_type
    if parameter_type is None or measurement.data_type is None:
        raise TypeError("MeasurementProcess must define parameter_type and data_type")

    data = coerce(data, measurement.data_type)
    if len(data.batch_shape) != 1 or data.batch_shape[0] == 0:
        raise ValueError(f"Expected data.batch_shape == (E,), got {data.batch_shape}")
    num_obs = data.batch_shape[0]

    population = as_population_model(pop_model, parameter_type, conditioned_on)
    if population.parameter_type is not parameter_type:
        raise TypeError(
            "pop_model.parameter_type must be the measurement's parameter_type "
            f"({parameter_type.__name__})"
        )

    pop_priors = dict(pop_priors)
    clash = [k for k in pop_priors if k.startswith((LOCAL_PREFIX, _BLOCK_SCOPE))]
    if clash:
        raise ValueError(f"Hyperprior names may not start with {LOCAL_PREFIX!r}: {clash}")

    field_names = [f.name for f in fields(parameter_type)]

    # ---------------------------------------------- starting point of theta_i
    key = jax.random.key(seed)
    Lambda0 = None
    if init is not None:
        Lambda0 = (
            dict(init_Lambda) if init_Lambda is not None
            else _hyperprior_point(pop_priors, seed)
        )
    if isinstance(init, Proposal):
        theta0 = init.sample(data, key, Lambda=Lambda0)
        init_kind = "proposal"
    elif init is not None:
        theta0 = coerce(init, parameter_type)
        if theta0.batch_shape != (num_obs,):
            raise ValueError(f"init must have batch shape ({num_obs},), got {theta0.batch_shape}")
        init_kind = "theta"
    else:
        theta0 = None
        init_kind = "default"

    # ---------------------------------------------- theta_i sites
    local_priors = dict(local_priors or {})
    unknown = {
        v for k, p in local_priors.items() for v in _prior_variables(k, p)
    } - set(field_names)
    if unknown:
        raise KeyError(f"local_priors names {sorted(unknown)} are not fields of {parameter_type.__name__}")
    for name, prior in local_priors.items():
        if hasattr(prior, "cleanup"):
            raise TypeError(f"local prior {name!r} has a cleanup step, which joint inference does not support")
    factors = _prior_factors(local_priors)
    covered = {v for names in factors for v in names}
    set_only = {
        v for k, p in local_priors.items() for v in _prior_variables(k, p)
    } - covered
    if set_only:
        raise KeyError(f"local_priors set {sorted(set_only)} without a density")

    supports = LatentSpace.default(parameter_type).supports
    local_sites = {}
    if local_priors:
        local_sites["_local_priors"] = _local_prior_block(local_priors, factors, num_obs)
    for name in field_names:
        if name in covered:
            continue
        core = _core_shape(parameter_type, name, theta0)
        local_sites[LOCAL_PREFIX + name] = ImproperUniform(
            supports[name], batch_shape=(num_obs,), event_shape=core
        )

    # ---------------------------------------------- sampler settings
    sampler_kwargs = {**measurement.sampler_kwargs, **sampler_kwargs}
    sampler_kwargs.setdefault("max_tree_depth", 10)
    if theta0 is not None:
        # Only plain sample sites can be started at a value; sites inside
        # prior blocks keep numpyro's default.
        block_vars = {
            v for k, p in local_priors.items()
            if isinstance(p, PriorBlock)
            for v in _prior_variables(k, p)
        }
        site_supports = {
            k: (local_priors[k].support if k in local_priors else supports[k])
            for k in field_names
            if k not in block_vars
        }
        theta0 = _feasible_start(
            theta0, Lambda0, data, measurement, population, site_supports,
            redraw=init if isinstance(init, Proposal) else None, key=key,
        )
        values = {
            LOCAL_PREFIX + k: jnp.asarray(getattr(theta0, k)) for k in site_supports
        }
        # Lambda starts where theta_i was drawn, for hyperpriors that are
        # plain distributions (sites of prior blocks keep the default).
        values.update({
            k: jnp.asarray(Lambda0[k])
            for k, prior in pop_priors.items()
            if isinstance(prior, Distribution) and k in Lambda0
        })
        nuts = dict(sampler_kwargs.get("sampler_kwargs", {}))
        nuts.setdefault("init_strategy", init_to_value(values=values))
        sampler_kwargs["sampler_kwargs"] = nuts

    likelihood = JointPopulationLikelihood(
        measurement, data, population, prefix=LOCAL_PREFIX
    )
    sampler = run_sampler(
        likelihood, {**pop_priors, **local_sites}, seed=seed, **sampler_kwargs
    )
    measurement.population_likelihood = likelihood
    measurement.population_sampler = sampler

    # ---------------------------------------------- assemble the posterior
    sites, sample_stats, sample_dims = _sampler_arrays(sampler)
    local = parameter_type.from_dict(
        {k: jnp.asarray(sites[LOCAL_PREFIX + k]) for k in field_names}
    )
    hyper_sites = {
        k: v for k, v in sites.items()
        if not k.startswith((LOCAL_PREFIX, _BLOCK_SCOPE))
    }

    posterior = _hyperparameter_posterior(
        hyper_sites,
        sample_stats,
        sample_dims,
        population,
        observed_data=data,
        local=local,
        metadata={
            "inference": "joint population",
            "sampler": "gravpop",
            "seed": seed,
            "num_observations": num_obs,
            "init": init_kind,
        },
    )

    derived = dict(population.derived_parameters(local, posterior.theta))
    for name, value in measurement.derived_parameters(local).items():
        if name in derived:
            raise ValueError(
                f"Derived quantity {name!r} is defined by both the population "
                "and the measurement"
            )
        derived[name] = value
    taken = set(posterior.variable_names) | set(posterior.sample_stats)
    clash = taken & set(derived)
    if clash:
        raise ValueError(f"Derived quantities {sorted(clash)} clash with sampled names")

    return PosteriorData(
        theta=posterior.theta,
        sample_stats=posterior.sample_stats,
        extras={**posterior.extras, **derived},
        metadata=posterior.metadata,
        observed_data=posterior.observed_data,
        local=local,
        sample_dims=posterior.sample_dims,
    )

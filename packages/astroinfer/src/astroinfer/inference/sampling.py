"""Run gravpop's NUTS sampler on astroinfer likelihoods.

gravpop is imported lazily so the rest of astroinfer (parameters,
measurement processes, generative models) works without it.
"""
import jax.numpy as jnp
import numpy as np

from ..parameters import coerce
from ..posterior import PosteriorData
from .likelihoods import FixedDataLikelihood, PopulationLikelihood


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

    Returns a pandas DataFrame with parameter samples,
    PE prior density, and log prior densities per distribution
    (``log_prior[x]``) or block (``log_prior[x, y]``),
    or a PosteriorData when ``as_posterior=True``.
    The fitted sampler is stored on ``measurement.sampler``.
    """
    pe_priors = dict(pe_priors)
    factors = {
        _prior_variables(k, prior): prior
        for k, prior in pe_priors.items()
        if hasattr(prior, "log_prob")
    }
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
    samples = sampler.samples

    for names, prior in factors.items():
        if hasattr(prior, "var_names"):
            log_prob = prior.log_prob({v: jnp.asarray(samples[v].to_numpy()) for v in names})
        else:
            log_prob = prior.log_prob(jnp.asarray(samples[names[0]].to_numpy()))
        samples[_factor_name(names)] = np.asarray(log_prob, dtype=np.float64)

    samples["prior"] = np.exp(
        sum(
            samples[_factor_name(names)]
            for names in factors
        )
    )

    samples = measurement.with_derived(samples)

    if not as_posterior:
        return samples

    return PosteriorData.from_dataframe(
        samples,
        parameter_type=measurement.parameter_type,
        metadata={
            "sampler": "gravpop",
            "seed": seed,
            "num_samples": len(samples),
        },
        observed_data=d,
    )


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
    of independently observed systems.

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
        The fitted gravpop Sampler. The likelihood and sampler are also
        stored on ``measurement.population_likelihood`` /
        ``measurement.population_sampler``.
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(2**30))

    likelihood = likelihood_cls(
        measurement=measurement,
        data=data,
        pop_model=pop_model,
        N=N,
        seed=int(seed),
        conditioned_on=conditioned_on,
        proposal=proposal,
    )

    sampler = run_sampler(
        likelihood,
        pop_priors,
        seed=int(seed),
        **{**measurement.sampler_kwargs, **sampler_kwargs},
    )

    measurement.population_likelihood = likelihood
    measurement.population_sampler = sampler

    return sampler

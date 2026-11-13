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

    Uses K(d | theta) as the likelihood.
    Detection selection is handled separately.

    Returns a pandas DataFrame with parameter samples,
    PE prior density, and per-parameter log prior densities,
    or a PosteriorData when ``as_posterior=True``.
    The fitted sampler is stored on ``measurement.sampler``.
    """
    missing = [
        k for k in measurement.parameter_names
        if k not in pe_priors
    ]
    if missing:
        raise KeyError(f"pe_priors has no {missing}")

    pe_priors = dict(pe_priors)

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

    for k, prior in pe_priors.items():
        samples[f"log_prior[{k}]"] = np.asarray(
            prior.log_prob(
                jnp.asarray(samples[k].to_numpy())
            ),
            dtype=np.float64,
        )

    samples["prior"] = np.exp(
        sum(
            samples[f"log_prior[{k}]"]
            for k in pe_priors
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
        GenerativeModel transforming fixed uniform
        seeds and Lambda into physical parameters.

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

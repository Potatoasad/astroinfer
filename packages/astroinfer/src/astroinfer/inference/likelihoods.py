"""Likelihood objects in the ``.logpdf(params)`` form ``gravpop.Sampler`` expects.

    FixedDataLikelihood        L(d | theta) for fixed d, a function of theta  single-event PE
    PopulationLikelihood       L(Lambda) = prod_i  E_{theta~p(.|Lambda)} L(d_i | theta)
    JointPopulationLikelihood  prod_i L(d_i | theta_i) p(theta_i | Lambda), a function
                               of Lambda and every theta_i

With a Proposal q_Lambda(theta | d), the per-event integral is estimated by
importance sampling instead:

    Z_i(Lambda) ~= (1/N) sum_j  L(d_i | theta_ij) p(theta_ij | Lambda)
                                / q_Lambda(theta_ij | d_i),
    theta_ij ~ q_Lambda(. | d_i)
"""

import jax
import jax.numpy as jnp

from ..parameters import coerce
from ..population import as_population_model


class FixedDataLikelihood:
    """L(d | theta) as a function of theta, for fixed data d, in the form gp.Sampler expects."""

    def __init__(self, measurement, d):
        self.measurement = measurement
        self.d = d

    def logpdf(self, params):
        return self.measurement.logpdf(self.d, params)


class PopulationLikelihood:
    """
    Population-marginalized likelihood for independent observations.

    L(d | theta) is provided by MeasurementProcess.
    p(theta | Lambda) is provided by a PopulationModel. A GenerativeModel
    is wrapped with PopulationModel.from_generative (using
    ``conditioned_on`` for its external variables). ``self.pop_model`` is
    the object passed in; ``self.population`` is the PopulationModel used.

    Uses a fixed key for differentiable common-random-number Monte Carlo
    integration: draws from p (or from the proposal) are the same function
    of Lambda on every call.

    proposal:
        Optional Proposal q_Lambda(theta | d). If given, draws come from q
        with a fixed key and are importance-weighted by p / q. If None,
        draws come from p itself (weights are just L).

    ``log_weights(Lambda)`` -> (E, N) is the single override point: subclass
    and replace it to use a different weight, e.g. one where factors shared
    by p and q cancel analytically (see its docstring).
    """

    # gravpop records `self.diagnostics` as deterministic sites.
    return_diagnostics = True

    def __init__(
        self,
        measurement,
        data,
        pop_model,
        *,
        N=2000,
        seed=42,
        conditioned_on=None,
        proposal=None,
    ):
        self.measurement = measurement
        self.pop_model = pop_model
        self.proposal = proposal
        self.N = int(N)

        if self.N <= 0:
            raise ValueError("N must be positive")

        if (
            measurement.parameter_type is None
            or measurement.data_type is None
        ):
            raise TypeError(
                "MeasurementProcess must define "
                "parameter_type and data_type"
            )

        # Data is a Parameters object with one batch axis
        # indexing independent observations.
        self.data = coerce(data, measurement.data_type)

        if len(self.data.batch_shape) != 1:
            raise ValueError(
                "Expected data.batch_shape == (E,), "
                f"got {self.data.batch_shape}"
            )

        self.E = self.data.batch_shape[0]

        if self.E == 0:
            raise ValueError("No observations provided")

        self.conditioned_on = dict(conditioned_on or {})
        # pop_model stays as given (subclasses may call its own methods);
        # population is the PopulationModel interface the likelihood uses.
        self.population = as_population_model(
            pop_model, measurement.parameter_type, self.conditioned_on
        )

        if self.population.parameter_type is not measurement.parameter_type:
            raise TypeError(
                "pop_model.parameter_type must be the measurement's "
                f"parameter_type ({measurement.parameter_type.__name__}), "
                f"got {getattr(self.population.parameter_type, '__name__', None)}."
            )

        if proposal is None and not self.population.differentiable_sample:
            raise ValueError(
                f"{type(self.population).__name__} does not draw samples "
                "that are differentiable in Lambda "
                "(differentiable_sample = False), so it needs a proposal."
            )

        if proposal is not None and (
            proposal.parameter_type is not measurement.data_type
            or proposal.data_type is not measurement.parameter_type
        ):
            raise TypeError(
                "Proposal types must mirror the measurement process: "
                "proposal.parameter_type is measurement.data_type and "
                "proposal.data_type is measurement.parameter_type."
            )

        # One fixed key: it drives either the proposal or the population.
        self.key = jax.random.key(int(seed))

        # Add a singleton Monte Carlo axis to observations.
        # Batch shape: (E,) -> (E,1)
        self._data_for_samples = self.data.reshape_batch(
            (self.E, 1)
        )

    def population_draws(self, Lambda):
        """theta_ij ~ p(theta | Lambda) from the fixed key, batch (E, N)."""
        theta = self.population.sample(
            Lambda, key=self.key, N=(self.E, self.N)
        )

        if theta.batch_shape != (self.E, self.N):
            theta = theta.broadcast((self.E, self.N))

        return theta

    def proposal_draws(self, Lambda):
        """theta_ij ~ q_Lambda(theta | d_i) from the fixed key, batch (E, N)."""
        if self.proposal is None:
            raise ValueError("No proposal was given to this likelihood.")
        return self.proposal.sample(
            self.data, self.key, N=self.N, Lambda=Lambda
        )

    def log_likelihood_at(self, theta):
        """log L(d_i | theta_ij) for draws theta of batch shape (E, N)."""
        return self.measurement.logpdf(self._data_for_samples, theta)

    def log_likelihood_draws(self, Lambda):
        """log L(d_i | theta_ij) at population draws, shape (E, N)."""
        return self.log_likelihood_at(self.population_draws(Lambda))

    def log_weights(self, Lambda):
        """
        Log importance weights, shape (E, N); Z_i is their mean.

        Default:
            no proposal : log L(d_i | theta_ij),                theta ~ p
            proposal    : log L + log p(theta | Lambda) - log q, theta ~ q

        Override in a subclass for a custom weight. Example: q draws masses
        from an event-informed q(m, z | d) and spins from the population's
        own p(s | m, z, Lambda); that conditional then cancels, leaving
        log L + log p(m, z | Lambda) - log q(m, z | d).
        """
        if self.proposal is None:
            return self.log_likelihood_draws(Lambda)

        theta = self.proposal_draws(Lambda)
        log_q = self.proposal.logpdf(
            theta, self._data_for_samples, Lambda=Lambda
        )
        log_p = self.population.logpdf(theta, Lambda)
        return self.log_likelihood_at(theta) + log_p - log_q

    def event_logpdf_and_neff(self, Lambda):
        """
        Returns:
            logp_events: (E,)
            neff_events: (E,)
        """
        log_w = self.log_weights(Lambda)

        log_sum_w = jax.scipy.special.logsumexp(log_w, axis=-1)

        # Per-event Monte Carlo log likelihoods
        logp_events = log_sum_w - jnp.log(self.N)

        # N_eff = (sum w)^2 / sum w^2
        log_neff = (
            2 * log_sum_w
            - jax.scipy.special.logsumexp(2 * log_w, axis=-1)
        )

        neff_events = jnp.exp(log_neff)

        # An event with all zero weights has zero ESS.
        neff_events = jnp.where(
            jnp.isneginf(log_sum_w),
            0.0,
            neff_events,
        )

        return logp_events, neff_events

    def event_logpdf(self, Lambda):
        """Log marginal likelihood for each observation, shape (E,)."""
        logp_events, _ = self.event_logpdf_and_neff(Lambda)
        return logp_events

    def logpdf(self, Lambda):
        """Scalar population log likelihood expected by gravpop."""
        logp_events, neff_events = (
            self.event_logpdf_and_neff(Lambda)
        )

        # gravpop reads this dictionary after logpdf().
        self.diagnostics = {
            "N_eff_min": jnp.min(neff_events),
            "N_eff_mean": jnp.mean(neff_events),
            "N_eff_min_fraction": (
                jnp.min(neff_events) / self.N
            ),
        }

        return jnp.sum(logp_events)


class JointPopulationLikelihood:
    """
    log of prod_i L(d_i | theta_i) p(theta_i | Lambda), for joint sampling.

    gravpop passes one dict holding the hyperparameters and, under names
    ``prefix + field``, each field of the theta_i with batch shape (E,)
    (nested fields by dotted path, ``prefix + "masses.m1"``).
    Together with the hyperprior this is the joint posterior
    p(Lambda, {theta_i} | {d_i}).
    """

    def __init__(self, measurement, data, population, *, prefix="theta."):
        self.measurement = measurement
        self.population = population
        self.data = coerce(data, measurement.data_type)
        self.parameter_type = measurement.parameter_type
        self.prefix = prefix
        self.field_names = list(self.parameter_type.flat_field_specs())

    def split(self, x):
        """``(Lambda dict, theta_i)`` from gravpop's variable dict."""
        Lambda = {k: v for k, v in x.items() if not k.startswith(self.prefix)}
        theta = self.parameter_type.from_dict(
            {k: x[self.prefix + k] for k in self.field_names}
        )
        return Lambda, theta

    def logpdf(self, x):
        Lambda, theta = self.split(x)
        return (
            jnp.sum(self.measurement.logpdf(self.data, theta))
            + jnp.sum(self.population.logpdf(theta, Lambda))
        )

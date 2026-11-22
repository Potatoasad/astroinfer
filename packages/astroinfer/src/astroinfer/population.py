"""The population model p(theta | Lambda).

Users subclass ``PopulationModel`` and write the unbatched model:
``sample_single(Lambda, key)`` and ``logpdf_single(theta, Lambda)``. It is a
:class:`~astroinfer.distributions.ConditionalDistribution` with theta as the
output and Lambda as the condition, so batching, keys and the display are
shared with ``MeasurementProcess``.

Lambda, the hyperparameter dict, is passed unbatched to every element. Under
``vmap``, terms that depend only on Lambda (normalisation constants, CDF
grids) are computed once per call, not once per element.

``PopulationLikelihood`` calls the batched ``sample`` and ``logpdf``. When the
code being wrapped is already batched, override those directly instead of the
single versions.

A ``GenerativeModel`` is adapted with ``PopulationModel.from_generative``;
``PopulationLikelihood`` does this automatically.
"""
import inspect
import contextvars
import warnings
from dataclasses import dataclass
from functools import wraps
from math import prod

import jax

from .distributions import ConditionalDistribution
from .distributions.base import _vmapped, sample_shape
from .distributions.schema import is_parameter_type, or_untyped
from .generative import GenerativeModel
from .parameters import Parameters, coerce

# Seeds for GenerativeModel draws stay strictly inside (0, 1), so the
# inverse CDF never lands on a support edge.
_SEED_EPS = 1e-7


def uniform_seeds(order, key, shape):
    """``{name: uniforms in (eps, 1 - eps)}``, one key per name in ``order``."""
    keys = jax.random.split(key, len(order))
    return {
        name: jax.random.uniform(k, shape=shape, minval=_SEED_EPS, maxval=1.0 - _SEED_EPS)
        for name, k in zip(order, keys)
    }


def _is_generative(model):
    """GenerativeModel, or an object with its interface (duck typing)."""
    if isinstance(model, GenerativeModel):
        return True
    return all(hasattr(model, a) for a in ("order", "sample", "log_prob"))


def _lambda_batch_shape(Lambda):
    """Batch axes of Lambda. Only a typed Lambda has any; a dict is one point."""
    return Lambda.batch_shape if isinstance(Lambda, Parameters) else ()


# Set while an old-style override (``sample(..., shape=())``) runs with a
# ``shape`` that the wrapper derived from ``N``, so that the override passing
# it on to ``super().sample(shape=...)`` does not warn.
_SHAPE_FROM_N = contextvars.ContextVar("_SHAPE_FROM_N", default=False)


def _resolve_n(N, shape):
    """``N``, or the deprecated ``shape=`` in its place."""
    if shape is None:
        return N
    if N is not None:
        raise TypeError("Pass N or shape, not both")
    if _SHAPE_FROM_N.get():
        return shape
    warnings.warn(
        "PopulationModel.sample(shape=) is deprecated; use N= instead.",
        DeprecationWarning, stacklevel=3,
    )
    return shape


def _over_lambda_sample(fn):
    """Let an overridden ``sample`` take a typed Lambda with batch axes.

    Each Lambda gets its own key; the output batch shape is
    ``Lambda.batch_shape + N``. Overrides written against the old
    ``sample(Lambda, key, shape=())`` signature receive ``shape=``. Other
    keywords of the override (such as ``seeds=``) are passed through, the
    same for every Lambda.
    """
    if getattr(fn, "_over_lambda", False):
        return fn
    takes_n = "N" in inspect.signature(fn).parameters

    @wraps(fn)
    def sample(self, Lambda, key=None, *, N=None, shape=None, **kwargs):
        N = _resolve_n(N, shape)
        size = {"N": N} if takes_n else {"shape": sample_shape(N)}
        token = _SHAPE_FROM_N.set(not takes_n)
        try:
            batch = _lambda_batch_shape(Lambda)
            if not batch:
                return fn(self, self._user_lambda(Lambda), key, **size, **kwargs)
            rng = self._resolve_key(key)
            keys = jax.random.split(rng, prod(batch)).reshape(batch + rng.shape)

            def one(L, k):
                return fn(self, self._user_lambda(L), k, **size, **kwargs)

            return _vmapped(one, batch)(Lambda.broadcast(), keys)
        finally:
            _SHAPE_FROM_N.reset(token)

    sample._over_lambda = True
    return sample


def _over_lambda_density(fn):
    """Let an overridden ``fn(theta, Lambda)`` take a typed Lambda with batch axes.

    theta's batch shape must start with Lambda's: Lambda draw k applies to
    theta[k, ...].
    """
    if getattr(fn, "_over_lambda", False):
        return fn

    @wraps(fn)
    def density(self, theta, Lambda):
        batch = _lambda_batch_shape(Lambda)
        if not batch:
            return fn(self, theta, self._user_lambda(Lambda))
        theta = coerce(theta, self.parameter_type).broadcast()
        if theta.batch_shape[:len(batch)] != batch:
            raise ValueError(
                f"theta batch shape {theta.batch_shape} must start with "
                f"the Lambda batch shape {batch}"
            )

        def one(t, L):
            return fn(self, t, self._user_lambda(L))

        return _vmapped(one, batch)(theta, Lambda.broadcast())

    density._over_lambda = True
    return density


@dataclass(eq=False)
class PopulationModel(ConditionalDistribution):
    """
    Population model p(theta | Lambda).

    Subclasses define:
        parameter_type                      the theta type, e.g. GWParams
        logpdf_single(theta, Lambda)        -> log p(theta | Lambda), a scalar
        sample_single(Lambda, key)          -> one theta ~ p(. | Lambda)

    Optionally:
        hyperparameter_type                 a parameter dataclass for Lambda
        derived_parameters_single(theta, Lambda)
                                            -> dict of quantities that depend
                                               on theta and Lambda, e.g.
                                               subpopulation membership

    The base class provides the batched ``logpdf(theta, Lambda)``,
    ``sample(Lambda, key=None, N=None)`` and
    ``derived_parameters(theta, Lambda)``. As a ``ConditionalDistribution``
    its output type is ``parameter_type`` (also readable as ``data_type``)
    and its condition type is ``hyperparameter_type``.

    Lambda reaches the ``_single`` methods as a dict of floats, or as a
    ``hyperparameter_type`` object when one is declared. ``sample``,
    ``logpdf`` and ``derived_parameters`` also accept a typed Lambda with
    batch axes (e.g. ``posterior.theta`` from population inference) and vmap
    over them; this holds for subclasses that override those methods too.

    PopulationLikelihood without a proposal estimates each event's integral
    with draws from ``sample`` under a fixed key, and differentiates through
    them. That needs ``sample_single`` to be a smooth function of
    (key, Lambda): location-scale transforms or inverse CDFs of uniforms, not
    rejection sampling or a hard categorical choice of subpopulation. If
    ``sample_single`` is not smooth in Lambda, set
    ``differentiable_sample = False``. The likelihood then refuses to run
    without a proposal, and ``sample`` remains usable at fixed Lambda, for
    instance for injections or predictive checks.

    ``logpdf_single`` should return -inf outside the support without an
    infinite derivative: evaluate at a safe point and mask with
    ``jnp.where``, otherwise out-of-support proposal draws give NaN
    gradients for the whole batch.
    """

    parameter_type = None
    hyperparameter_type = None
    differentiable_sample = True

    density_latex = r"\pi"
    _output_symbol = r"\theta"
    _condition_symbol = r"\Lambda"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Overrides of the batched methods also accept batched Lambda.
        for name, wrap in (
            ("sample", _over_lambda_sample),
            ("logpdf", _over_lambda_density),
            ("derived_parameters", _over_lambda_density),
        ):
            if name in cls.__dict__:
                setattr(cls, name, wrap(cls.__dict__[name]))

    @property
    def data_type(self):
        """The output type, theta; the same as ``parameter_type``."""
        return self.parameter_type

    def _output_type(self):
        return or_untyped(self.parameter_type)

    def _condition_type(self):
        return or_untyped(self.hyperparameter_type)

    # A dict Lambda stays whole and a typed Lambda is batched; each element
    # is converted by _user_lambda.
    def _coerce_conditions(self, values):
        return values

    def _single_conditions(self, values):
        return (self._user_lambda(values[0]),)

    def _user_lambda(self, Lambda):
        """Lambda as the user's code expects it: typed if declared, else a dict."""
        cls = self.hyperparameter_type
        if cls is None:
            return Lambda.to_flat_dict() if isinstance(Lambda, Parameters) else Lambda
        if isinstance(Lambda, cls):
            return Lambda
        if isinstance(Lambda, Parameters):
            Lambda = Lambda.to_dict()
        return cls.from_dict(Lambda)

    # ========================================================
    # Unbatched model: implemented by the user
    # ========================================================

    def sample_single(self, Lambda, key):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sample_single"
        )

    def logpdf_single(self, theta, Lambda):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement logpdf_single"
        )

    def derived_parameters_single(self, theta, Lambda):
        """Quantities of one theta under one Lambda; none by default."""
        return {}

    # ========================================================
    # Batched API
    # ========================================================

    def sample(self, Lambda, key=None, *, N=None, shape=None):
        """
        Draw theta ~ p(theta | Lambda) with batch shape ``N``.

        A typed Lambda with batch axes gives batch shape
        ``Lambda.batch_shape + N``. ``shape=`` is a deprecated name for ``N``.

        key:
            None (default): use the internal key and advance it, so every
            call draws fresh values.
            JAX key, int seed, or ``np.random.Generator``: use it as given;
            the internal key is not changed. The same key and N give the
            same underlying randomness for every Lambda (common random
            numbers).
        """
        return super().sample(Lambda, key, N=_resolve_n(N, shape))

    def logpdf(self, theta, Lambda):
        """log p(theta | Lambda), with theta's batch shape."""
        return super().logpdf(theta, Lambda)

    def derived_parameters(self, theta, Lambda):
        """``derived_parameters_single`` over theta's batch shape."""
        if (
            type(self).derived_parameters_single
            is PopulationModel.derived_parameters_single
        ):
            return {}
        return self._evaluate(self.derived_parameters_single, (theta,), (Lambda,), {})

    # ========================================================
    # Adapters
    # ========================================================

    @staticmethod
    def from_generative(model, parameter_type, *, conditioned_on=None):
        """A ``GenerativeModel`` as a population model over ``parameter_type``."""
        return GenerativePopulation(
            model, parameter_type, conditioned_on=conditioned_on
        )

    # ========================================================
    # Display
    # ========================================================

    def _display_methods(self, cond_tex, out_tex):
        text = {
            ".sample": "Draw from the population" + (
                "" if self.differentiable_sample
                else " (not differentiable in Lambda; use a proposal)"
            ),
            ".logpdf": "Log population density",
        }
        return [
            (name, sig, text.get(name, desc))
            for name, sig, desc in super()._display_methods(cond_tex, out_tex)
        ]


class GenerativePopulation(PopulationModel):
    """
    A ``GenerativeModel`` as a ``PopulationModel``.

    ``sample`` turns the key into one uniform seed array per variable in the
    model's order and runs ``GenerativeModel.sample``, so draws are
    differentiable in Lambda and fixed for a fixed key. External variables
    are supplied once, through ``conditioned_on``.
    """

    def __init__(self, model, parameter_type, *, conditioned_on=None, key=None):
        super().__init__(**({} if key is None else {"key": key}))
        if not _is_generative(model):
            raise TypeError(
                "Expected a GenerativeModel or an object with its interface "
                f"(order, sample, log_prob), got {type(model).__name__}"
            )
        if not model.order:
            raise ValueError("Population model must draw at least one variable")

        self.model = model
        self.external = set(getattr(model, "external", ()))
        self.parameter_type = parameter_type
        self.conditioned_on = dict(conditioned_on or {})

        missing_ext = self.external - set(self.conditioned_on)
        if missing_ext:
            raise ValueError(
                f"The model conditions on external variables "
                f"{sorted(missing_ext)}; supply them in conditioned_on."
            )

        required = set(parameter_type.flat_field_specs())
        missing = required - set(model.order) - set(self.conditioned_on)
        if missing:
            raise ValueError(
                f"Population model is missing parameters {sorted(missing)}"
            )

    def sample(self, Lambda, key=None, *, N=None):
        shape = sample_shape(N)
        seeds = uniform_seeds(self.model.order, self._resolve_key(key), shape)
        drawn = self.model.sample(
            Lambda, seeds=seeds, conditioned_on=self.conditioned_on
        )
        theta = self.parameter_type.from_dict(drawn)
        if theta.batch_shape != shape:
            theta = theta.broadcast(shape)
        return theta

    def logpdf(self, theta, Lambda):
        if self.external:
            raise ValueError(
                "logpdf needs p(theta | Lambda) for every parameter, but the "
                "model conditions on external variables "
                f"{sorted(self.external)}, so it only has the "
                "conditional density. Override log_weights, or write a "
                "PopulationModel with the full density."
            )
        theta = coerce(theta, self.parameter_type)
        return self.model.log_prob(theta.to_flat_dict(), Lambda)

    def _display_config(self):
        config = {"Order": ", ".join(self.model.order)}
        if self.conditioned_on:
            config["Conditioned on"] = ", ".join(self.conditioned_on)
        return config


class DistributionPopulation(PopulationModel):
    """
    A ``ConditionalDistribution`` over theta, conditioned on Lambda or on
    nothing, as a ``PopulationModel``.

    ``sample`` and ``logpdf`` call the distribution's own. Its draws are
    treated as differentiable in Lambda only if it implements
    ``Reparameterizable`` (see :mod:`astroinfer.distributions.transport`);
    otherwise ``differentiable_sample`` is False and ``PopulationLikelihood``
    asks for a proposal. ``joint_population_inference`` uses only
    ``logpdf`` and works either way. For a structured distribution, convert
    it with ``prior.as_generative(...)`` to draw differentiably instead.
    """

    def __init__(self, distribution, parameter_type=None, *, key=None):
        from .distributions.transport import Reparameterizable

        super().__init__(**({} if key is None else {"key": key}))
        out = distribution._output_type()
        if parameter_type is not None and out is not parameter_type:
            raise TypeError(
                f"The distribution's output type is {getattr(out, '__name__', out)}, "
                f"not the parameter type {getattr(parameter_type, '__name__', parameter_type)}"
            )
        schema = distribution._condition_schema()
        if schema.kind not in ("none", "single"):
            raise TypeError(
                "A population model takes one condition (Lambda) or none; the "
                f"distribution's conditions are {schema}"
            )
        self.distribution = distribution
        self.parameter_type = out
        cond = schema.types[0] if schema.kind == "single" else None
        self.hyperparameter_type = cond if is_parameter_type(cond) else None
        self._unconditioned = schema.kind == "none"
        self.differentiable_sample = isinstance(distribution, Reparameterizable)

    def _lambda_args(self, Lambda):
        return () if self._unconditioned else (Lambda,)

    def sample(self, Lambda, key=None, *, N=None):
        return self.distribution.sample(
            *self._lambda_args(Lambda), self._resolve_key(key), N=N
        )

    def logpdf(self, theta, Lambda):
        return self.distribution.logpdf(theta, *self._lambda_args(Lambda))

    def _display_title(self):
        return f"{type(self).__name__}[{self.distribution._display_title()}]"

    def _display_model(self):
        return self.distribution._display_model()

    def _display_config(self):
        return self.distribution._display_config()


def as_population_model(pop_model, parameter_type, conditioned_on=None):
    """
    A ``PopulationModel`` for ``pop_model``.

    A PopulationModel (including ``prior.as_generative(...)``) passes
    through. Any other ``ConditionalDistribution`` over ``parameter_type``
    with one condition or none is wrapped in ``DistributionPopulation``. A
    GenerativeModel (or duck type) is wrapped in ``GenerativePopulation``.
    """
    if isinstance(pop_model, ConditionalDistribution):
        if conditioned_on:
            raise TypeError(
                "conditioned_on applies only to a GenerativeModel; a "
                "PopulationModel handles its own conditioning."
            )
        if isinstance(pop_model, PopulationModel):
            return pop_model
        return DistributionPopulation(pop_model, parameter_type)
    if _is_generative(pop_model):
        return PopulationModel.from_generative(
            pop_model, parameter_type, conditioned_on=conditioned_on
        )
    raise TypeError(
        "pop_model must be a PopulationModel, a ConditionalDistribution, or "
        "a GenerativeModel (or an object with its order/sample/log_prob "
        f"interface), got {type(pop_model).__name__}"
    )

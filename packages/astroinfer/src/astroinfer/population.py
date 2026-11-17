"""The population model p(theta | Lambda).

Users subclass ``PopulationModel`` and write the unbatched model:
``sample_single(Lambda, key)`` and ``logpdf_single(theta, Lambda)``. The base
class vmaps them over batch shapes, as ``MeasurementProcess`` does for the
measurement model, and owns a JAX key that ``sample`` advances on every call.

Lambda, the hyperparameter dict, is passed unbatched to every element. Under
``vmap``, terms that depend only on Lambda (normalisation constants, CDF
grids) are computed once per call, not once per element.

``PopulationLikelihood`` calls the batched ``sample`` and ``logpdf``. When the
code being wrapped is already batched, override those directly instead of the
single versions.

A ``GenerativeModel`` is adapted with ``PopulationModel.from_generative``;
``PopulationLikelihood`` does this automatically.
"""
from dataclasses import dataclass, field, fields
from functools import wraps
from html import escape
from math import prod
from typing import Any

import jax
import numpy as np

from .display import RichDisplay, html_table, inline_math
from .generative import GenerativeModel
from .parameters import Parameters, coerce

# Seeds for GenerativeModel draws stay strictly inside (0, 1), so the
# inverse CDF never lands on a support edge.
_SEED_EPS = 1e-7


def _default_key():
    return jax.random.key(0)


def _vmapped(fn, batch_shape):
    for _ in batch_shape:
        fn = jax.vmap(fn)
    return fn


def _is_generative(model):
    """GenerativeModel, or an object with its interface (duck typing)."""
    if isinstance(model, GenerativeModel):
        return True
    return all(hasattr(model, a) for a in ("order", "sample", "log_prob"))


def _lambda_batch_shape(Lambda):
    """Batch axes of Lambda. Only a typed Lambda has any; a dict is one point."""
    return Lambda.batch_shape if isinstance(Lambda, Parameters) else ()


def _over_lambda_sample(fn):
    """Let ``sample(Lambda, key, shape=)`` take a typed Lambda with batch axes.

    Each Lambda gets its own key; the output batch shape is
    ``Lambda.batch_shape + shape``.
    """
    if getattr(fn, "_over_lambda", False):
        return fn

    @wraps(fn)
    def sample(self, Lambda, key=None, *, shape=()):
        batch = _lambda_batch_shape(Lambda)
        if not batch:
            return fn(self, self._user_lambda(Lambda), key, shape=shape)
        rng = self._resolve_key(key)
        keys = jax.random.split(rng, prod(batch)).reshape(batch + rng.shape)

        def one(L, k):
            return fn(self, self._user_lambda(L), k, shape=shape)

        return _vmapped(one, batch)(Lambda.broadcast(), keys)

    sample._over_lambda = True
    return sample


def _over_lambda_density(fn):
    """Let ``fn(theta, Lambda)`` take a typed Lambda with batch axes.

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


def _as_shape(shape):
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(s) for s in shape)


@dataclass(eq=False)
class PopulationModel(RichDisplay):
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
    ``sample(Lambda, key=None, shape=())`` and
    ``derived_parameters(theta, Lambda)``.

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

    key: Any = field(
        default_factory=_default_key, kw_only=True, repr=False, compare=False
    )

    parameter_type = None
    hyperparameter_type = None
    differentiable_sample = True

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

    def _user_lambda(self, Lambda):
        """Lambda as the user's code expects it: typed if declared, else a dict."""
        cls = self.hyperparameter_type
        if cls is None:
            return Lambda.to_dict() if isinstance(Lambda, Parameters) else Lambda
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
    # Keys
    # ========================================================

    def seed(self, seed: int):
        """Reset the internal key to ``jax.random.key(seed)``."""
        self.key = jax.random.key(int(seed))

    def _next_key(self):
        """Split the internal key: store one half, return the other."""
        # Subclasses with their own __init__ may skip the dataclass one.
        if "key" not in self.__dict__:
            self.key = _default_key()
        self.key, sub = jax.random.split(self.key)
        return sub

    def _resolve_key(self, key):
        if key is None:
            return self._next_key()
        if isinstance(key, np.random.Generator):
            return jax.random.key(int(key.integers(0, 2**31)))
        if isinstance(key, (int, np.integer)):
            return jax.random.key(int(key))
        return key

    # ========================================================
    # Batched API
    # ========================================================

    @_over_lambda_sample
    def sample(self, Lambda, key=None, *, shape=()):
        """
        Draw theta ~ p(theta | Lambda) with batch shape ``shape``.

        A typed Lambda with batch axes gives batch shape
        ``Lambda.batch_shape + shape``.

        key:
            None (default): use the internal key and advance it, so every
            call draws fresh values.
            JAX key, int seed, or ``np.random.Generator``: use it as given;
            the internal key is not changed. The same key and shape give the
            same underlying randomness for every Lambda (common random
            numbers).
        """
        rng = self._resolve_key(key)
        shape = _as_shape(shape)

        if not shape:
            return self.sample_single(Lambda, rng)

        keys = jax.random.split(rng, prod(shape)).reshape(shape + rng.shape)
        return _vmapped(lambda k: self.sample_single(Lambda, k), shape)(keys)

    @_over_lambda_density
    def logpdf(self, theta, Lambda):
        """log p(theta | Lambda), with theta's batch shape."""
        theta = coerce(theta, self.parameter_type).broadcast()
        return _vmapped(
            lambda t: self.logpdf_single(t, Lambda), theta.batch_shape
        )(theta)

    @_over_lambda_density
    def derived_parameters(self, theta, Lambda):
        """``derived_parameters_single`` over theta's batch shape."""
        if (
            type(self).derived_parameters_single
            is PopulationModel.derived_parameters_single
        ):
            return {}
        theta = coerce(theta, self.parameter_type).broadcast()
        return _vmapped(
            lambda t: self.derived_parameters_single(t, Lambda),
            theta.batch_shape,
        )(theta)

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

    def _theta_tex(self):
        return getattr(self.parameter_type, "_latex", r"\theta")

    def _display_math(self):
        return r"p\!\left(" + self._theta_tex() + r"\mid\Lambda\right)"

    def _display_body(self):
        theta = self._theta_tex()
        rows = [
            [
                "<code>.sample</code>",
                inline_math(rf"(\Lambda)\longrightarrow {theta}"),
                "Draw from the population",
            ],
            [
                "<code>.logpdf</code>",
                inline_math(rf"({theta},\,\Lambda)\longrightarrow \mathrm{{jax.Array}}"),
                "Log population density",
            ],
        ]
        if not self.differentiable_sample:
            rows[0][2] += " (not differentiable in Lambda; use a proposal)"
        return html_table(["Method", "Signature", ""], rows)


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

        required = {f.name for f in fields(parameter_type)}
        missing = required - set(model.order) - set(self.conditioned_on)
        if missing:
            raise ValueError(
                f"Population model is missing parameters {sorted(missing)}"
            )

    def sample(self, Lambda, key=None, *, shape=()):
        rng = self._resolve_key(key)
        shape = _as_shape(shape)
        keys = jax.random.split(rng, len(self.model.order))
        seeds = {
            name: jax.random.uniform(
                k, shape=shape, minval=_SEED_EPS, maxval=1.0 - _SEED_EPS
            )
            for name, k in zip(self.model.order, keys)
        }
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
        return self.model.log_prob(theta.to_dict(), Lambda)

    def _display_body(self):
        order = ", ".join(escape(v) for v in self.model.order)
        body = f"<div>Order: <code>{order}</code></div>"
        if self.conditioned_on:
            ext = ", ".join(escape(v) for v in self.conditioned_on)
            body += f"<div>Conditioned on: <code>{ext}</code></div>"
        return body + super()._display_body()


def as_population_model(pop_model, parameter_type, conditioned_on=None):
    """Pass a PopulationModel through; wrap a GenerativeModel (or duck type)."""
    if isinstance(pop_model, PopulationModel):
        if conditioned_on:
            raise TypeError(
                "conditioned_on applies only to a GenerativeModel; a "
                "PopulationModel handles its own conditioning."
            )
        return pop_model
    if _is_generative(pop_model):
        return PopulationModel.from_generative(
            pop_model, parameter_type, conditioned_on=conditioned_on
        )
    raise TypeError(
        "pop_model must be a PopulationModel or a GenerativeModel "
        "(or an object with its order/sample/log_prob interface), "
        f"got {type(pop_model).__name__}"
    )

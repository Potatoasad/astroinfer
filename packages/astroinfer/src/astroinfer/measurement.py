"""The measurement model L(d | theta).

Users subclass ``MeasurementProcess`` and write the *unbatched* physics:
``sample_single(theta, key)`` and ``logpdf_single(d, theta)``. Batching,
keys and the display come from
:class:`~astroinfer.distributions.ConditionalDistribution`; this module adds
parameter estimation / population inference as convenience methods
(implemented in :mod:`astroinfer.inference`).
"""
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import numpy as np
import pandas as pd

from .distributions import ConditionalDistribution
from .distributions.base import _vmapped
from .distributions.schema import or_untyped
from .inference.sampling import (
    joint_population_inference,
    parameter_estimation,
    population_inference,
)
from .parameters import Parameters, coerce


@dataclass(eq=False)
class MeasurementProcess(ConditionalDistribution):
    """
    Measurement model L(d | theta).

    Subclasses define:
        parameter_type
        data_type
        sample_single(theta, key)
        logpdf_single(d, theta)

    Optionally:
        derived_parameters_single(theta)  -> dict of quantities computed
                                             from one theta

    The base class handles batching, JAX keys, PE and display.

    Every instance carries a JAX key, ``self.key`` (default
    ``jax.random.key(0)``; keyword-only, so dataclass subclasses may declare
    non-default fields). ``sample(theta)`` splits it and stores the new key,
    so repeated calls give fresh noise; ``sample(theta, key)`` uses the given
    key and leaves ``self.key`` untouched.
    """

    parameters = ()

    density_latex = r"\mathcal{L}"
    _output_symbol = "d"

    sampler_kwargs: ClassVar[dict[str, Any]] = {
        "num_warmup": 500,
        "num_samples": 8_000,
        "summary": False,
    }

    # A measurement always has a theta and a d, typed or not.
    def _output_type(self):
        return or_untyped(self.data_type)

    def _condition_type(self):
        return or_untyped(self.parameter_type)

    # ========================================================
    # Parameter metadata and conversion
    # ========================================================

    @property
    def parameter_names(self):
        if self.parameters:
            return tuple(self.parameters)

        if self.parameter_type is None:
            return ()

        # Dotted paths for nested types, e.g. "masses.m1".
        return tuple(self.parameter_type.flat_field_specs())

    _coerce = staticmethod(coerce)
    _vmapped = staticmethod(_vmapped)

    # ========================================================
    # Batched API
    # ========================================================

    def sample(self, theta, key=None, *, N=None, **context):
        """
        Draw observations d ~ L(d | theta).

        **context:
            Extra keyword arguments forwarded unchanged to every
            ``sample_single`` call (not batched). Plain measurement
            processes take none.

        key:
            None (default): use the internal key, ``self.key``, and advance it,
            so every call draws fresh noise.
            JAX key, int seed, or ``np.random.Generator``: use it as given;
            ``self.key`` is not changed.

        If N is None:
            One observation per theta.
            Output batch_shape = theta.batch_shape.

        If N is provided (an int or a tuple of ints):
            N independent observations per theta.
            Output batch_shape = theta.batch_shape + N.
        """
        return super().sample(theta, key, N=N, **context)

    def logpdf(self, d, theta, *, outer=False, **context):
        """
        Compute log L(d | theta).

        **context is forwarded unchanged to every ``logpdf_single`` call.

        Accepts parameter objects or parameter dictionaries.

        Default: broadcast parameter and data batch shapes; a theta with
        fewer batch axes than d lines up with the leading axes of d.
        outer=True: evaluate every parameter-data combination.
        """
        # Untyped d or theta would reach every logpdf_single whole, giving a
        # wrongly shaped result, so require parameter types as before.
        d, theta = coerce(d, self.data_type), coerce(theta, self.parameter_type)
        if not (isinstance(d, Parameters) and isinstance(theta, Parameters)):
            raise TypeError(
                "Expected parameter_dataclass objects "
                "or dictionaries convertible to them"
            )
        return super().logpdf(d, theta, outer=outer, **context)

    # ========================================================
    # Truths, injections, and derived parameters
    # ========================================================

    def check(self, df):
        out = pd.DataFrame(df).reset_index(drop=True).copy()

        missing = [
            k for k in self.parameter_names
            if k not in out.columns
        ]
        if missing:
            raise KeyError(f"Missing parameters {missing}")

        return out

    def theta(self, df):
        df = pd.DataFrame(df)

        return {
            k: df[k].to_numpy(dtype=np.float64)
            for k in self.parameter_names
        }

    def with_derived(self, df):
        """``df`` with the derived parameters added as columns."""
        out = pd.DataFrame(df).copy()

        theta = {
            k: out[k].to_numpy(dtype=np.float64)
            for k in self.parameter_names
            if k in out.columns
        }
        if self.parameter_type is not None and len(theta) == len(self.parameter_names):
            theta = self.parameter_type.from_dict(theta)

        for k, v in self.derived_parameters(theta).items():
            if k not in out.columns:
                out[k] = np.asarray(v, dtype=np.float64)

        return out

    # ========================================================
    # Inference (see astroinfer.inference)
    # ========================================================

    def PE(self, d, pe_priors, seed=None, *, as_posterior=False, **sampler_kwargs):
        """Posterior samples of theta given one observation d.

        See :func:`astroinfer.inference.parameter_estimation`.
        """
        return parameter_estimation(
            self, d, pe_priors, seed,
            as_posterior=as_posterior, **sampler_kwargs,
        )

    def population_inference(self, data, pop_model, pop_priors, **kwargs):
        """Infer population hyperparameters Lambda from a batch of observations.

        See :func:`astroinfer.inference.population_inference`.
        """
        return population_inference(
            self, data, pop_model, pop_priors, **kwargs
        )

    def joint_population_inference(self, data, pop_model, pop_priors, **kwargs):
        """Sample Lambda and every observation's theta_i together.

        See :func:`astroinfer.inference.joint_population_inference`.
        """
        return joint_population_inference(
            self, data, pop_model, pop_priors, **kwargs
        )

    def posterior_predictive(self, posterior, *, draws=None, key=None):
        """
        Replicated data d_rep ~ L(d | theta), one per posterior draw.

        Uses ``posterior.local`` (the theta_i of a joint population
        posterior) when present, otherwise ``posterior.theta``; either must
        be this measurement's ``parameter_type``.

        draws:
            None: use every draw; the result keeps the posterior's sample
            axes (and the observation axis of ``local``).
            int: pick this many draws at random, without replacement, after
            stacking the sample axes; the result has batch shape
            ``(draws,)`` (``(draws, E)`` with ``local``).
        """
        theta = posterior.local if posterior.local is not None else posterior.theta
        if type(theta) is not self.parameter_type:
            raise TypeError(
                f"The posterior holds {type(theta).__name__}, but this "
                f"measurement takes {getattr(self.parameter_type, '__name__', None)}."
            )

        rng = self._resolve_key(key)
        if draws is not None:
            pick, rng = jax.random.split(rng)
            flat = posterior.stack_samples()
            n = flat.batch_shape[0]
            if not 0 < int(draws) <= n:
                raise ValueError(f"draws must be between 1 and {n}")
            idx = np.asarray(
                jax.random.choice(pick, n, (int(draws),), replace=False)
            )
            flat = flat[idx]
            theta = flat.local if flat.local is not None else flat.theta

        return self.sample(theta, rng)

    # ========================================================
    # Rich notebook display
    # ========================================================

    def _display_methods(self, theta_tex, data_tex) -> list[tuple[str, str, str]]:
        """``(name, LaTeX signature, description)`` rows under "Methods"."""
        condition = [
            row for row in super()._display_methods(theta_tex, data_tex)
            if row[0] == ".condition"
        ]
        return [
            (
                ".sample",
                rf"({theta_tex})\longrightarrow {data_tex}",
                "Generate one or more noisy observations",
            ),
            (
                ".logpdf",
                rf"({data_tex},\,{theta_tex})"
                r"\longrightarrow "
                r"\mathrm{float}\;|\;\mathrm{jax.Array}",
                "Compute log likelihood",
            ),
            *condition,
            (
                ".PE",
                rf"({data_tex},\,\pi_{{\mathrm{{PE}}}})"
                r"\longrightarrow \mathrm{DataFrame}\;|\;\mathrm{PosteriorData}",
                "Infer posterior samples of the physical parameters",
            ),
            (
                ".posterior_predictive",
                rf"(\mathrm{{PosteriorData}})\longrightarrow {data_tex}",
                "Replicate data from posterior draws",
            ),
        ]


@dataclass(eq=False)
class Proposal(ConditionalDistribution):
    """
    Importance-sampling proposal q_Lambda(theta | d): a measurement process
    run backwards.

    The roles of the two types swap relative to the measurement process it
    serves:

        parameter_type = the measurement's data_type        (what q reads, d)
        data_type      = the measurement's parameter_type   (what q draws, theta)

    Subclasses define:
        sample_single(d, key, Lambda)       -> one theta ~ q_Lambda(. | d)
        logpdf_single(theta, d, Lambda)     -> log q_Lambda(theta | d)

    Lambda is the population hyperparameter dict. It is passed unbatched to
    every element; a Lambda-independent proposal simply ignores it.

    For common random numbers across Lambda, ``sample_single`` must be a
    smooth function of (key, Lambda): use location-scale transforms, icdf of
    uniforms, or GenerativeModel.sample(key=...), not rejection sampling.

    A Proposal is a ``ConditionalDistribution`` but not a measurement
    model: the inference methods of ``MeasurementProcess`` raise TypeError.
    """

    _output_symbol = r"\theta"
    _condition_symbol = "d"

    def _output_type(self):
        return or_untyped(self.data_type)

    def _condition_type(self):
        return or_untyped(self.parameter_type)

    def sample(self, d, key=None, *, N=None, Lambda=None):
        """Draw theta ~ q_Lambda(theta | d); batch shape d.batch_shape (+ (N,))."""
        return super().sample(d, key, N=N, Lambda=Lambda)

    def logpdf(self, theta, d, *, outer=False, Lambda=None):
        """log q_Lambda(theta | d), broadcast over the batch shapes."""
        return super().logpdf(theta, d, outer=outer, Lambda=Lambda)

    def sample_single(self, d, key, Lambda):
        raise NotImplementedError

    def logpdf_single(self, theta, d, Lambda):
        raise NotImplementedError

    def PE(self, *args, **kwargs):
        raise TypeError("A Proposal is not a measurement model; PE does not apply.")

    def population_inference(self, *args, **kwargs):
        raise TypeError(
            "A Proposal is not a measurement model; pass it as "
            "MeasurementProcess.population_inference(..., proposal=...)."
        )

    def joint_population_inference(self, *args, **kwargs):
        raise TypeError(
            "A Proposal is not a measurement model; pass it as "
            "MeasurementProcess.joint_population_inference(..., init=...)."
        )

    def posterior_predictive(self, *args, **kwargs):
        raise TypeError("A Proposal is not a measurement model.")

    def _display_density(self, d_tex, theta_tex) -> str:
        # The condition is the event data d; the output is the drawn theta.
        return r"q_{\Lambda}\!\left(" + theta_tex + r"\mid " + d_tex + r"\right)"

    def _display_methods(self, d_tex, theta_tex):
        return [
            (
                ".sample",
                rf"({d_tex},\,\Lambda)\longrightarrow {theta_tex}",
                "Draw proposal samples (N= per event)",
            ),
            (
                ".logpdf",
                rf"({theta_tex},\,{d_tex},\,\Lambda)"
                r"\longrightarrow \mathrm{jax.Array}",
                "Log proposal density",
            ),
        ]

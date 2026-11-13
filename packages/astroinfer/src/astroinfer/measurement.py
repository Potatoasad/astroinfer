"""The measurement model K(d | theta).

Users subclass ``MeasurementProcess`` and write the *unbatched* physics:
``sample_single(theta, key)`` and ``logpdf_single(d, theta)``. The base class
vmaps them over arbitrary batch shapes, owns a JAX key (``self.key``) that
``sample(theta)`` advances on every call, and exposes
parameter estimation / population inference as convenience methods
(implemented in :mod:`astroinfer.inference`).
"""
from dataclasses import dataclass, field, fields
from functools import partial
from html import escape
from math import prod
from typing import Any, ClassVar

import jax
import numpy as np
import pandas as pd

from .display import RichDisplay, inline_math, kv_table, math_block
from .inference.sampling import parameter_estimation, population_inference
from .parameters import coerce


def _vmapped(fn, batch_shape):
    for _ in batch_shape:
        fn = jax.vmap(fn)
    return fn


def _default_key():
    return jax.random.key(0)


@dataclass(eq=False)
class MeasurementProcess(RichDisplay):
    """
    Measurement model K(d | theta).

    Subclasses define:
        parameter_type
        data_type
        sample_single(theta, key)
        logpdf_single(d, theta)

    The base class handles batching, JAX keys, PE and display.

    Every instance carries a JAX key, ``self.key`` (default
    ``jax.random.key(0)``; keyword-only, so dataclass subclasses may declare
    non-default fields). ``sample(theta)`` splits it and stores the new key,
    so repeated calls give fresh noise; ``sample(theta, key)`` uses the given
    key and leaves ``self.key`` untouched.
    """

    key: Any = field(
        default_factory=_default_key, kw_only=True, repr=False, compare=False
    )

    parameters = ()
    parameter_type = None
    data_type = None

    sampler_kwargs: ClassVar[dict[str, Any]] = {
        "num_warmup": 500,
        "num_samples": 8_000,
        "summary": False,
    }

    # ========================================================
    # Parameter metadata and conversion
    # ========================================================

    @property
    def parameter_names(self):
        if self.parameters:
            return tuple(self.parameters)

        if self.parameter_type is None:
            return ()

        return tuple(
            f.name for f in fields(self.parameter_type)
        )

    _coerce = staticmethod(coerce)
    _vmapped = staticmethod(_vmapped)

    # ========================================================
    # Scalar physics methods: implemented by user
    # ========================================================

    def sample_single(self, theta, key):
        raise NotImplementedError

    def logpdf_single(self, d, theta):
        raise NotImplementedError

    # ========================================================
    # Public sampling API
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

    def sample(self, theta, key=None, *, N=None, **context):
        """
        Draw observations d ~ K(d | theta).

        **context:
            Extra keyword arguments forwarded unchanged to every
            ``sample_single`` call (not batched), e.g. ``Lambda`` for a
            Proposal. Plain measurement processes take none.

        key:
            None (default): use the internal key, ``self.key``, and advance it,
            so every call draws fresh noise.
            JAX key, int seed, or ``np.random.Generator``: use it as given;
            ``self.key`` is not changed.

        If N is None:
            One observation per theta.
            Output batch_shape = theta.batch_shape.

        If N is provided:
            N independent observations per theta.
            Output batch_shape = theta.batch_shape + (N,).
        """
        # --------------------------------------------------
        # RNG handling
        # --------------------------------------------------
        if key is None:
            rng = self._next_key()
        elif isinstance(key, np.random.Generator):
            rng = jax.random.key(int(key.integers(0, 2**31)))
        elif isinstance(key, (int, np.integer)):
            rng = jax.random.key(int(key))
        else:
            rng = key

        # --------------------------------------------------
        # Parameter batching
        # --------------------------------------------------
        theta = coerce(theta, self.parameter_type).broadcast()

        if N is not None:
            if not isinstance(N, (int, np.integer)) or N < 1:
                raise ValueError("N must be a positive integer")

            shape = theta.batch_shape

            # Add a new singleton batch axis.
            theta = theta.reshape_batch(shape + (1,))

            # Broadcast across N independent realizations.
            theta = theta.broadcast(shape + (int(N),))

        # --------------------------------------------------
        # Draw independent observations
        # --------------------------------------------------
        shape = theta.batch_shape
        single = partial(self.sample_single, **context)

        if not shape:
            return single(theta, rng)

        keys = jax.random.split(
            rng, prod(shape)
        ).reshape(shape + rng.shape)

        return _vmapped(single, shape)(theta, keys)

    # ========================================================
    # Public likelihood API
    # ========================================================

    def logpdf(self, d, theta, *, outer=False, **context):
        """
        Compute log K(d | theta).

        **context is forwarded unchanged to every ``logpdf_single`` call.

        Accepts parameter objects or parameter dictionaries.

        Default: broadcast parameter and data batch shapes.
        outer=True: evaluate every parameter-data combination.
        """
        d = coerce(d, self.data_type)
        theta = coerce(theta, self.parameter_type)

        if not (
            hasattr(d, "batch_shape")
            and hasattr(theta, "batch_shape")
        ):
            raise TypeError(
                "Expected parameter_dataclass objects "
                "or dictionaries convertible to them"
            )

        if outer:
            ts = theta.batch_shape
            ds = d.batch_shape

            theta = theta.reshape_batch(
                ts + (1,) * len(ds)
            )
            d = d.reshape_batch(
                (1,) * len(ts) + ds
            )

        shape = np.broadcast_shapes(
            d.batch_shape,
            theta.batch_shape,
        )

        d = d.broadcast(shape)
        theta = theta.broadcast(shape)

        return _vmapped(
            partial(self.logpdf_single, **context), shape
        )(d, theta)

    # ========================================================
    # Truths, injections, and derived parameters
    # ========================================================

    def derived_parameters(self, theta):
        return {}

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
        out = pd.DataFrame(df).copy()

        theta = {
            k: out[k].to_numpy(dtype=np.float64)
            for k in self.parameter_names
            if k in out.columns
        }

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

    # ========================================================
    # Rich notebook display
    # ========================================================

    def _display_equations(self) -> list[str]:
        """LaTeX lines describing K(d | theta), shown under "Model"."""
        return []

    def _display_config(self) -> dict:
        """``{name: value}`` shown under "Configuration"."""
        return {}

    def _display_density(self, theta_tex, data_tex) -> str:
        """LaTeX symbol for the density, shown in the summary line."""
        return (
            r"\mathcal{L}\!\left(" + data_tex + r"\mid " + theta_tex + r"\right)"
        )

    def _display_methods(self, theta_tex, data_tex) -> list[tuple[str, str, str]]:
        """``(name, LaTeX signature, description)`` rows under "Methods"."""
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
            (
                ".PE",
                rf"({data_tex},\,\pi_{{\mathrm{{PE}}}})"
                r"\longrightarrow \mathrm{DataFrame}",
                "Infer posterior samples of the physical parameters",
            ),
        ]

    def _repr_html_(self):
        theta_tex = getattr(
            self.parameter_type, "_latex", r"\theta"
        )
        data_tex = getattr(
            self.data_type, "_latex", "d"
        )

        likelihood_tex = self._display_density(theta_tex, data_tex)

        display_name = escape(
            getattr(
                self, "display_name", type(self).__name__
            )
        )

        lines = [
            '<details style="font-family:inherit;line-height:1.6">',
            '<summary style="cursor:pointer;padding:0.25em 0">',
            (
                '<span style="display:inline-flex;'
                'align-items:center;gap:0.45em">'
            ),
            '<strong>' + display_name + ':</strong>',
            inline_math(likelihood_tex),
            '</span></summary>',
            '<div style="padding:0.45em 0 0.5em 1.3em">',
        ]

        equations = self._display_equations()
        if equations:
            lines.append(
                '<div style="font-weight:600;margin-bottom:0.35em">Model</div>'
                + math_block(equations)
            )

        lines.append(
            '<div style="font-weight:600;margin:0.5em 0 0.35em 0">Methods</div>'
        )

        def method(name, signature, description):
            lines.append(
                '<div style="display:flex;align-items:center;'
                'flex-wrap:wrap;gap:0.45em;margin:0.25em 0">'
                '<code>' + escape(name) + '</code>'
                + inline_math(signature)
                + '<span style="opacity:0.7;font-size:0.9em">'
                + escape(description)
                + '</span></div>'
            )

        for row in self._display_methods(theta_tex, data_tex):
            method(*row)

        config = self._display_config()
        if config:
            lines.append(
                '<div style="border-top:1px solid #8884;'
                'padding-top:0.45em;margin-top:0.5em">'
                '<strong>Configuration</strong></div>'
                + kv_table(config)
            )
        elif shown := [f for f in fields(self) if f.name != "key"]:
            lines.append(
                '<div style="border-top:1px solid #8884;'
                'padding-top:0.45em;margin-top:0.5em">'
                '<strong>Configuration</strong></div>'
            )

            for f in shown:
                value = repr(getattr(self, f.name))

                lines.append(
                    '<div style="margin:0.15em 0">'
                    '<code>'
                    + escape(f.name)
                    + " = "
                    + escape(value)
                    + '</code></div>'
                )

        lines.append("</div></details>")
        return "".join(lines)


@dataclass(eq=False)
class Proposal(MeasurementProcess):
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
    """

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

    def _display_density(self, theta_tex, data_tex) -> str:
        # Here theta_tex is the event data d and data_tex is the drawn theta.
        return r"q_{\Lambda}\!\left(" + data_tex + r"\mid " + theta_tex + r"\right)"

    def _display_methods(self, theta_tex, data_tex):
        return [
            (
                ".sample",
                rf"({theta_tex},\,\Lambda)\longrightarrow {data_tex}",
                "Draw proposal samples (N= per event)",
            ),
            (
                ".logpdf",
                rf"({data_tex},\,{theta_tex},\,\Lambda)"
                r"\longrightarrow \mathrm{jax.Array}",
                "Log proposal density",
            ),
        ]

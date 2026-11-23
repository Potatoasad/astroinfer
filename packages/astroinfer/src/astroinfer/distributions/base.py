"""Conditional distributions p(x | c) written from unbatched methods.

Users subclass ``ConditionalDistribution``, declare ``data_type`` (the
output x) and ``parameter_type`` (the conditions c), and write the
unbatched law:

    sample_single(*conditions, key=key, **context)  -> one x
    logpdf_single(*data, *conditions, **context)    -> one scalar

where the arguments follow the schema of each type (see
:mod:`astroinfer.distributions.schema`); mapping parts arrive as keywords.
The base class batches them, owns a JAX key, binds conditions
(``.condition``) and draws the notebook display.

Batching rule: an argument is batched if it is a ``Parameters`` instance
(dicts and DataFrames are first coerced for arguments declared with a
parameter type). Every other argument, such as a float, a plain array or a
dict of hyperparameters, reaches every single call whole. Conditions
broadcast among themselves as in NumPy. A condition with fewer batch axes
than the data is padded with trailing length-1 axes, so its axes line up
with the leading data axes, and the result then broadcasts with the data.
``sample`` appends its sample axes ``N`` after the condition batch axes, so
``d.logpdf(d.sample(c, N=n), c)`` always lines up.
"""
import inspect
from dataclasses import dataclass, field, fields
from html import escape
from math import prod
from typing import Any

import jax
import numpy as np

from ..display import (
    RichDisplay,
    details_open,
    html_value,
    inline_math,
    kv_table,
    math_block,
    method_row,
)
from ..parameters import Parameters
from .schema import Schema, batch_shape


def _default_key():
    return jax.random.key(0)


def _vmapped(fn, batch_shape, in_axes=0):
    for _ in batch_shape:
        fn = jax.vmap(fn, in_axes=in_axes)
    return fn


# ============================================================
# Batching helpers (shared by every distribution)
# ============================================================

def call_sample_single(fn, schema, values, key, **context):
    """``fn(*conditions, key=key, **context)`` for one element.

    The key goes by keyword when ``fn`` has a parameter named ``key`` (or the
    conditions are keywords). Otherwise it goes positionally after the
    conditions, so subclasses written as ``sample_single(self, theta, rng)``
    keep working.
    """
    args, kwargs = schema.arguments(values)
    try:
        by_name = "key" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        by_name = True
    if by_name or kwargs:
        return fn(*args, **kwargs, key=key, **context)
    return fn(*args, key, **context)


def sample_shape(N):
    """The sample axes for ``N``: None -> (), int -> (N,), tuple as given."""
    if N is None:
        return ()
    shape = (N,) if isinstance(N, (int, np.integer)) else tuple(N)
    if not all(isinstance(n, (int, np.integer)) and n >= 1 for n in shape):
        raise ValueError("N must be a positive integer or a tuple of them")
    return tuple(int(n) for n in shape)


def broadcast_batch(values):
    """Broadcast batch shape of the batched values; () if there are none."""
    shapes = [s for s in map(batch_shape, values) if s is not None]
    return np.broadcast_shapes(*shapes) if shapes else ()


def broadcast_values(values, shape):
    """Broadcast every batched value to ``shape``; others are unchanged."""
    return tuple(
        v.broadcast(shape) if isinstance(v, Parameters) else v for v in values
    )


def _reshape_values(values, own, shape):
    return tuple(
        v.broadcast(own).reshape_batch(shape) if isinstance(v, Parameters) else v
        for v in values
    )


def combine_batches(data, conditions, *, outer=False):
    """
    Align data and condition values under the batching rule.

    Returns ``(data, conditions, shape)`` with every batched value
    broadcast to ``shape``. With ``outer=True`` every data element is
    paired with every condition element, condition axes first.
    """
    ds, cs = broadcast_batch(data), broadcast_batch(conditions)
    if outer:
        data = _reshape_values(data, ds, (1,) * len(cs) + ds)
        conditions = _reshape_values(conditions, cs, cs + (1,) * len(ds))
    elif len(cs) < len(ds):
        conditions = _reshape_values(
            conditions, cs, cs + (1,) * (len(ds) - len(cs))
        )
    try:
        shape = np.broadcast_shapes(
            broadcast_batch(data), broadcast_batch(conditions)
        )
    except ValueError:
        raise ValueError(
            f"The data batch shape {ds} must start with the condition batch "
            f"shape {cs}, up to broadcasting"
        ) from None
    return (
        broadcast_values(data, shape),
        broadcast_values(conditions, shape),
        shape,
    )


def over_batch(fn, values, shape, keys=None, sample_axes=()):
    """
    Call ``fn(values, key)`` once per batch element.

    Batched values must have batch shape ``shape``; they are mapped over
    and the others are closed over, so ``jax.vmap`` only ever sees
    positional arguments. ``keys`` (if given) has shape
    ``shape + sample_axes``; the trailing sample axes map over the keys
    only.
    """
    idx = [i for i, v in enumerate(values) if isinstance(v, Parameters)]

    def one(batched, key):
        vals = list(values)
        for i, v in zip(idx, batched):
            vals[i] = v
        return fn(tuple(vals), key)

    f = _vmapped(one, sample_axes, in_axes=(None, 0))
    return _vmapped(f, shape)(tuple(values[i] for i in idx), keys)


def _check_names(out, cond):
    if out.kind == cond.kind == "mapping":
        shared = sorted(set(out.names) & set(cond.names))
        if shared:
            raise TypeError(
                f"Output and condition types share the names {shared}; both "
                "reach logpdf_single as keywords, so they must differ."
            )


@dataclass(eq=False)
class ConditionalDistribution(RichDisplay):
    """
    Conditional distribution p(x | c).

    Subclasses define:
        data_type                   the output x (None, type, tuple or mapping)
        parameter_type              the conditions c (None for no conditions)
        sample_single(*conditions, key=key)  -> one x
        logpdf_single(*data, *conditions)    -> log p(x | c), a scalar

    Optionally:
        derived_parameters_single(*conditions)
                                    -> dict of quantities computed from one
                                       set of conditions

    The base class provides the batched ``sample``, ``logpdf`` (also
    ``log_prob``), ``derived_parameters`` and ``condition``, owns a JAX key
    and draws the notebook display. Extra keyword arguments that are not
    schema names are forwarded unchanged (not batched) to the single
    methods as context.

    Every instance carries a JAX key, ``self.key`` (default
    ``jax.random.key(0)``; keyword-only, so dataclass subclasses may declare
    non-default fields). ``sample`` without a key splits it and stores the
    new key, so repeated calls give fresh draws; ``sample(..., key)`` uses
    the given key and leaves ``self.key`` untouched.
    """

    key: Any = field(
        default_factory=_default_key, kw_only=True, repr=False, compare=False
    )

    data_type = None
    parameter_type = None

    density_latex = "p"
    _output_symbol = "x"
    _condition_symbol = r"\theta"

    def graph(self):
        """The DAG of this distribution: its conditions and outputs (see astroinfer.graph)."""
        from ..graph import distribution_graph
        return distribution_graph(self)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            out, cond = cls._output_type(cls), cls._condition_type(cls)
        except AttributeError:  # types that live on the instance
            return
        _check_names(Schema.from_type(out), Schema.from_type(cond))

    # ========================================================
    # Types and schemas
    # ========================================================

    def _output_type(self):
        return self.data_type

    def _condition_type(self):
        return self.parameter_type

    def _output_schema(self):
        return Schema.from_type(self._output_type())

    def _condition_schema(self):
        return Schema.from_type(self._condition_type())

    def _coerce_conditions(self, values):
        """Conditions as the batching code sees them."""
        return self._condition_schema().coerce(values)

    def _single_conditions(self, values):
        """One element's conditions as the single methods receive them."""
        return values

    # ========================================================
    # Unbatched law: implemented by the user
    # ========================================================

    def sample_single(self, *conditions, key, **context):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sample_single"
        )

    def logpdf_single(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement logpdf_single"
        )

    def derived_parameters_single(self, *conditions, **context):
        """Quantities computed from one set of conditions; none by default."""
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

    @staticmethod
    def _split_key(rest, key):
        if len(rest) > 1 or (rest and key is not None):
            raise TypeError(
                "Expected at most one positional argument (the key) after "
                "the conditions"
            )
        return rest[0] if rest else key

    # ========================================================
    # Batched API
    # ========================================================

    def sample(self, *args, N=None, **kwargs):
        """
        Draw x ~ p(x | c).

        Conditions follow the condition schema: positionally for a single
        type or a tuple, as keywords (or one dict) for a mapping. A
        positional argument after them is the key.

        key:
            None (default): use the internal key, ``self.key``, and advance
            it, so every call draws fresh values.
            JAX key, int seed, or ``np.random.Generator``: use it as given;
            ``self.key`` is not changed.

        N:
            None, an int or a tuple of ints. The output batch shape is the
            broadcast batch shape of the conditions followed by ``N``.
        """
        key = kwargs.pop("key", None)
        conds, context, rest = self._condition_schema().bind(args, kwargs)
        return self._sample(conds, self._split_key(rest, key), N, context)

    def _sample(self, conds, key, N, context):
        rng = self._resolve_key(key)
        schema = self._condition_schema()
        conds = self._coerce_conditions(conds)
        shape, axes = broadcast_batch(conds), sample_shape(N)
        full = shape + axes
        keys = (
            jax.random.split(rng, prod(full)).reshape(full + rng.shape)
            if full else rng
        )

        def one(values, k):
            return call_sample_single(
                self.sample_single, schema, self._single_conditions(values),
                k, **context,
            )

        return over_batch(one, broadcast_values(conds, shape), shape, keys, axes)

    def logpdf(self, *args, outer=False, **kwargs):
        """
        log p(x | c), broadcast over the batch shapes.

        Data come first, then the conditions, each following its schema.
        Accepts parameter objects or dictionaries for typed arguments.

        outer=True: evaluate every data-condition combination, with the
        condition axes first.
        """
        data, kwargs, args = self._output_schema().bind(args, kwargs)
        conds, context, rest = self._condition_schema().bind(args, kwargs)
        if rest:
            raise TypeError(
                f"logpdf got {len(rest)} unexpected positional argument(s)"
            )
        return self._evaluate(self.logpdf_single, data, conds, context, outer=outer)

    def log_prob(self, *args, **kwargs):
        """Another name for ``logpdf``."""
        return self.logpdf(*args, **kwargs)

    def _evaluate(self, fn, data, conds, context, *, outer=False):
        """``fn(*data, *conds, **context)`` over the combined batch shape."""
        out, cond = self._output_schema(), self._condition_schema()
        data, conds, shape = combine_batches(
            out.coerce(data), self._coerce_conditions(conds), outer=outer
        )
        n = len(data)

        def one(values, _):
            d_args, d_kwargs = out.arguments(values[:n])
            c_args, c_kwargs = cond.arguments(
                self._single_conditions(values[n:])
            )
            return fn(*d_args, *c_args, **d_kwargs, **c_kwargs, **context)

        return over_batch(one, data + conds, shape)

    def derived_parameters(self, *args, **kwargs):
        """
        ``{name: array}`` of derived quantities, with the conditions' batch
        shape.

        By default this vmaps ``derived_parameters_single`` over every batch
        axis. Override it instead when your code is already batched; it
        receives typed conditions.
        """
        if (
            type(self).derived_parameters_single
            is ConditionalDistribution.derived_parameters_single
        ):
            return {}
        conds, context, rest = self._condition_schema().bind(args, kwargs)
        if rest:
            raise TypeError(
                f"derived_parameters got {len(rest)} unexpected positional "
                "argument(s)"
            )
        return self._evaluate(self.derived_parameters_single, (), conds, context)

    def condition(self, *args, **kwargs):
        """
        p(x | c) with some or all conditions fixed.

        Takes the conditions as ``sample`` does. A mapping schema may be
        bound partially; the result is then conditional on the remaining
        names. Other keywords are kept as context. The bound distribution
        gets its own key, split from this one's.
        """
        values, context, rest = self._condition_schema().bind(
            args, kwargs, partial=True
        )
        if rest:
            raise TypeError("condition takes only the conditions")
        return BoundConditionalDistribution(
            self, values, context, key=self._next_key()
        )

    # ========================================================
    # Rich notebook display
    # ========================================================

    def _display_tex(self):
        """``(condition LaTeX, output LaTeX)`` from the schemas."""
        return (
            self._condition_schema().latex(self._condition_symbol),
            self._output_schema().latex(self._output_symbol),
        )

    def _display_equations(self) -> list[str]:
        """LaTeX lines describing the distribution, shown under "Model"."""
        return []

    def _display_model(self) -> str:
        """HTML shown under "Model"; by default the ``_display_equations``."""
        equations = self._display_equations()
        return math_block(equations) if equations else ""

    def _display_config(self) -> dict:
        """``{name: value}`` shown under "Configuration"."""
        return {}

    def _display_density(self, cond_tex, out_tex) -> str:
        """LaTeX for the density, shown in the summary line."""
        given = r"\mid " + cond_tex if cond_tex else ""
        return self.density_latex + r"\!\left(" + out_tex + given + r"\right)"

    def _display_methods(self, cond_tex, out_tex) -> list[tuple[str, str, str]]:
        """``(name, LaTeX signature, description)`` rows under "Methods"."""
        both = out_tex + (r",\," + cond_tex if cond_tex else "")
        rows = [
            (
                ".sample",
                rf"({cond_tex})\longrightarrow {out_tex}",
                "Draw samples",
            ),
            (
                ".logpdf",
                rf"({both})\longrightarrow \mathrm{{jax.Array}}",
                "Compute the log density",
            ),
        ]
        if cond_tex:
            rows.append((
                ".condition",
                rf"({cond_tex})\longrightarrow "
                + self.density_latex + rf"\!\left({out_tex}\right)",
                "Fix the conditions",
            ))
        return rows

    def _display_title(self) -> str:
        return getattr(self, "display_name", type(self).__name__)

    def _display_math(self) -> str:
        return self._display_density(*self._display_tex())

    def _repr_html_(self):
        cond_tex, out_tex = self._display_tex()

        lines = [
            '<details' + details_open() + ' style="font-family:inherit;line-height:1.6">',
            '<summary style="cursor:pointer;padding:0.25em 0">',
            (
                '<span style="display:inline-flex;'
                'align-items:center;gap:0.45em">'
            ),
            '<strong>' + escape(self._display_title()) + ':</strong>',
            inline_math(self._display_density(cond_tex, out_tex)),
            '</span></summary>',
            '<div style="padding:0.45em 0 0.5em 1.3em">',
        ]

        model = self._display_model()
        if model:
            lines.append(
                '<div style="font-weight:600;margin-bottom:0.35em">Model</div>'
                + model
            )

        lines.append(
            '<div style="font-weight:600;margin:0.5em 0 0.35em 0">Methods</div>'
        )
        lines.extend(
            method_row(*row) for row in self._display_methods(cond_tex, out_tex)
        )

        heading = (
            '<div style="border-top:1px solid #8884;'
            'padding-top:0.45em;margin-top:0.5em">'
            '<strong>Configuration</strong></div>'
        )
        config = self._display_config()
        if config:
            lines.append(heading + kv_table(config))
        elif shown := [f for f in fields(self) if f.name != "key"]:
            lines.append(heading)
            for f in shown:
                lines.append(
                    '<div style="margin:0.15em 0"><code>'
                    + escape(f.name) + " = " + escape(repr(getattr(self, f.name)))
                    + '</code></div>'
                )

        lines.append("</div></details>")
        return "".join(lines)


@dataclass(eq=False)
class BoundConditionalDistribution(ConditionalDistribution):
    """
    A conditional distribution with some or all of its conditions fixed.

    Made by ``ConditionalDistribution.condition``. ``bound`` holds the
    fixed values as given: a tuple for positional schemas, a dict of names
    for a mapping schema. The remaining names of a partially bound mapping
    schema are the conditions of this distribution. Calls go through the
    parent's public methods, so overrides of its batched methods apply.
    The bound distribution has its own key.
    """

    parent: Any
    bound: Any = ()
    context: dict = field(default_factory=dict)

    def graph(self):
        """The parent's DAG, with the bound conditions marked as given."""
        from ..graph import bound_graph
        return bound_graph(self)

    def __post_init__(self):
        if isinstance(self.bound, dict):
            self.bound = dict(self.bound)
        else:
            self.bound = tuple(self.bound)

    def _output_type(self):
        return self.parent._output_type()

    def _condition_type(self):
        schema = self.parent._condition_schema()
        if schema.kind != "mapping":
            return None
        types = dict(zip(schema.names, schema.types))
        return {n: t for n, t in types.items() if n not in self.bound} or None

    def _parent_arguments(self, conds):
        """``(args, kwargs)`` of the parent's conditions."""
        if isinstance(self.bound, dict):
            names = self._condition_schema().names
            return (), {**self.bound, **dict(zip(names, conds))}
        return self.bound, {}

    def sample(self, *args, N=None, **kwargs):
        key = kwargs.pop("key", None)
        conds, context, rest = self._condition_schema().bind(args, kwargs)
        rng = self._resolve_key(self._split_key(rest, key))
        c_args, c_kwargs = self._parent_arguments(conds)
        return self.parent.sample(
            *c_args, rng, N=N, **c_kwargs, **self.context, **context
        )

    def logpdf(self, *args, outer=False, **kwargs):
        out = self._output_schema()
        data, kwargs, args = out.bind(args, kwargs)
        conds, context, rest = self._condition_schema().bind(args, kwargs)
        if rest:
            raise TypeError(
                f"logpdf got {len(rest)} unexpected positional argument(s)"
            )
        d_args, d_kwargs = out.arguments(data)
        c_args, c_kwargs = self._parent_arguments(conds)
        # Only pass outer when asked: PopulationModel.logpdf has no outer.
        extra = {"outer": True} if outer else {}
        return self.parent.logpdf(
            *d_args, *c_args, **extra,
            **d_kwargs, **c_kwargs, **self.context, **context,
        )

    def derived_parameters(self, *args, **kwargs):
        raise TypeError(
            "Call derived_parameters on the parent distribution."
        )

    def _display_title(self):
        return self.parent._display_title()

    def _display_density(self, cond_tex, out_tex):
        return self.parent._display_density(*self.parent._display_tex())

    def _display_equations(self):
        return self.parent._display_equations()

    def _display_model(self):
        return self.parent._display_model()

    def _display_config(self):
        if isinstance(self.bound, dict):
            given = self.bound
        else:
            labels = self.parent._condition_schema().labels()
            given = dict(zip(labels, self.bound))
        shown = {
            f"{name} (bound)": (
                html_value(v._repr_html_()) if hasattr(v, "_repr_html_") else v
            )
            for name, v in given.items()
        }
        return {**shown, **self.context, **self.parent._display_config()}

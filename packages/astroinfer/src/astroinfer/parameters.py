"""Typed, batched, JAX-aware parameter containers.

A ``@parameter_dataclass`` is a frozen dataclass that is also a JAX pytree.
Each field may declare *core* named axes (``index=("times", "ifo")``); any
leading axes beyond those are *batch* axes, shared and broadcast across
fields. The same machinery describes physical parameters (theta) and
observed data (d).

A field annotated with another parameter dataclass is a *nested* child:
the child object is a pytree node of its own, and the flat helpers
(``flat_field_specs``, ``to_flat_dict``) name its leaves by dotted path,
e.g. ``"masses.m1"``.
"""
import copy as copy_module
import inspect
import typing
from collections.abc import Callable, Mapping
from numbers import Number
from dataclasses import MISSING, Field, dataclass, field, fields, replace
from functools import wraps
from math import prod
from typing import Any, dataclass_transform, overload

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .display import RichDisplay, inline_math, inline_row, latex_name, parameter_row, section

_SPEC_KEY = "dataobject.parameter"


# ============================================================
# 1. Parameter metadata
# ============================================================

@dataclass(frozen=True)
class ParameterSpec:
    name: str | None = None
    latex: str | None = None
    index: tuple[str, ...] = ()
    index_latex: tuple[str, ...] = ()
    # numpyro constraint for the field's domain (None means the real line).
    support: Any = None

    @property
    def latex_display(self) -> str:
        return f"${self.latex}$"


def parameter(
    *,
    name: str | None = None,
    latex: str | None = None,
    index: tuple[str, ...] = (),
    index_latex: tuple[str, ...] = (),
    support: Any = None,
    default: Any = MISSING,
    default_factory: Any = MISSING,
) -> Any:
    index = tuple(index)
    index_latex = tuple(index_latex)

    if index_latex and len(index) != len(index_latex):
        raise ValueError("index and index_latex must have equal lengths")

    spec = ParameterSpec(name, latex, index, index_latex, support)

    return field(
        default=default,
        default_factory=default_factory,
        metadata={_SPEC_KEY: spec},
    )


# ============================================================
# 2. Generic JAX-aware parameter base class
# ============================================================

class _type_property:
    """Read-only attribute computed from the class, on the class or an instance."""

    def __init__(self, fn):
        self.fn = fn
        self.__doc__ = fn.__doc__

    def __get__(self, obj, cls):
        return self.fn(cls)


class Parameters(RichDisplay):
    """JAX pytree with named core dimensions and broadcast batch axes."""

    def tree_flatten(self):
        return tuple(
            getattr(self, f.name) for f in fields(self)
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def field_specs(cls) -> dict[str, ParameterSpec]:
        nested = cls.nested_types()
        out = {}
        for f in fields(cls):
            if f.name in nested:
                spec = f.metadata.get(_SPEC_KEY, ParameterSpec())
                out[f.name] = replace(spec, latex=spec.latex or nested[f.name]._latex)
            else:
                out[f.name] = f.metadata.get(_SPEC_KEY, ParameterSpec(latex=f.name))
        return out

    @classmethod
    def nested_types(cls) -> dict[str, type["Parameters"]]:
        """``{field: parameter type}`` for the nested fields, in field order."""
        if "_nested_fields" in cls.__dict__:
            return cls._nested_fields
        nested, complete = _resolve_nested(cls)
        # Cache only a complete answer; an unresolved string annotation may
        # name a class defined later, so try again on the next call.
        if complete:
            cls._nested_fields = nested
        return nested

    @classmethod
    def flat_field_specs(cls) -> dict[str, ParameterSpec]:
        """``{path: spec}`` for every array field, nested ones by dotted path.

        For a type without nested fields this is ``field_specs()``.
        """
        nested = cls.nested_types()
        out = {}
        for name, spec in cls.field_specs().items():
            if name in nested:
                for path, sub in nested[name].flat_field_specs().items():
                    out[f"{name}.{path}"] = sub
            else:
                out[name] = spec
        return out

    def _flat_items(self):
        """``(path, spec, value)`` for every array field, depth first."""
        nested = self.nested_types()
        for name, spec in self.field_specs().items():
            value = getattr(self, name)
            if name in nested:
                for path, sub, v in value._flat_items():
                    yield f"{name}.{path}", sub, v
            else:
                yield name, spec, value

    def to_flat_dict(self) -> dict[str, Any]:
        """``{path: array}`` with nested fields under dotted keys."""
        return {path: value for path, _, value in self._flat_items()}

    def _dimensions(self):
        shapes = []
        axes: dict[str, int] = {}

        # Named axes share one namespace across nested fields.
        for path, spec, value in self._flat_items():
            shape = jnp.shape(value)
            n = len(spec.index)

            if len(shape) < n:
                raise ValueError(
                    f"{path}: expected {n} core axes, got {shape}"
                )

            shapes.append(shape[:-n] if n else shape)

            # Require shared named axes to have consistent sizes.
            core = shape[-n:] if n else ()

            for name, length in zip(spec.index, core):
                if name in axes and axes[name] != length:
                    raise ValueError(
                        f"Named axis {name!r} conflicts: "
                        f"{axes[name]} vs {length}"
                    )

                axes[name] = length

        return np.broadcast_shapes(*shapes), axes

    # ========================================================
    # Batch-shape manipulation
    # ========================================================

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._dimensions()[0]

    def is_batched(self) -> bool:
        return bool(self.batch_shape)

    def broadcast(self, batch_shape=None):
        own_shape = self.batch_shape

        shape = (
            own_shape
            if batch_shape is None
            else np.broadcast_shapes(
                own_shape, tuple(batch_shape)
            )
        )

        values = {}
        nested = self.nested_types()

        for f in fields(self):
            if f.name in nested:
                values[f.name] = getattr(self, f.name).broadcast(shape)
                continue
            v = jnp.asarray(getattr(self, f.name))
            n = len(self.field_specs()[f.name].index)

            core_shape = v.shape[-n:] if n else ()

            values[f.name] = jnp.broadcast_to(
                v, shape + core_shape
            )

        return type(self)(**values)

    def reshape_batch(self, shape):
        shape = tuple(shape)

        if prod(shape) != prod(self.batch_shape):
            raise ValueError(
                f"Cannot reshape batch {self.batch_shape} into {shape}"
            )

        values = {}
        obj = self.broadcast()
        nested = self.nested_types()

        for f in fields(obj):
            v = getattr(obj, f.name)
            if f.name in nested:
                values[f.name] = v.reshape_batch(shape)
                continue
            n = len(self.field_specs()[f.name].index)
            core_shape = v.shape[-n:] if n else ()

            values[f.name] = jnp.reshape(
                v, shape + core_shape
            )

        return type(self)(**values)

    def __getitem__(self, key):
        if isinstance(key, str):
            # A dotted path such as "masses.m1" reaches into nested fields.
            name, _, rest = key.partition(".")
            if name not in self.field_specs() or (
                rest and name not in self.nested_types()
            ):
                raise KeyError(key)

            value = getattr(self, name)
            if not rest:
                return value
            try:
                return value[rest]
            except KeyError:
                raise KeyError(key) from None

        indices = key if isinstance(key, tuple) else (key,)

        if not self.batch_shape:
            raise IndexError("Object has no batch axes")

        def allowed(i):
            if isinstance(i, (int, np.integer, slice)):
                return True
            # Boolean masks / integer index arrays along one batch axis.
            if isinstance(i, (list, np.ndarray, jax.Array)):
                return np.asarray(i).dtype.kind in "biu" and np.ndim(i) == 1
            return False

        if (
            len(indices) > len(self.batch_shape)
            or not all(allowed(i) for i in indices)
        ):
            raise IndexError(
                "Index only batch axes with integers, slices, "
                "or 1-D boolean / integer arrays"
            )
        indices = tuple(
            np.asarray(i) if isinstance(i, list) else i for i in indices
        )

        return jax.tree_util.tree_map(
            lambda a: a[indices],
            self.broadcast(),
        )

    # ========================================================
    # Arithmetic: field-wise +/- between objects of the same type,
    # and scaling by a number
    # ========================================================
    #
    # jax.grad / jax.jacobian with respect to a Parameters object return the
    # same type, so a step is `theta + g` / `theta - g`. Batch axes broadcast
    # as usual (core axes are trailing, so NumPy broadcasting lines them up).

    def _fieldwise(self, other, op):
        if type(other) is not type(self):
            return NotImplemented
        return jax.tree_util.tree_map(op, self, other)

    def __add__(self, other):
        return self._fieldwise(other, jnp.add)

    def __sub__(self, other):
        return self._fieldwise(other, jnp.subtract)

    def __neg__(self):
        return jax.tree_util.tree_map(jnp.negative, self)

    # Scaling by a number (Python / NumPy scalar or 0-d array, including a
    # traced step size under jit), e.g. `theta - lr * g`.

    # Make NumPy scalars on the left (np.float64(0.1) * theta) defer to
    # __rmul__ instead of building an object array.
    __array_ufunc__ = None

    @staticmethod
    def _is_scalar(value):
        return (
            isinstance(value, (Number, np.ndarray, np.generic, jax.Array))
            and jnp.ndim(value) == 0
        )

    def _scaled(self, other, op):
        if not self._is_scalar(other):
            return NotImplemented
        return jax.tree_util.tree_map(lambda v: op(v, other), self)

    def __mul__(self, other):
        return self._scaled(other, jnp.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._scaled(other, jnp.true_divide)

    # ========================================================
    # Unconstrained (latent) coordinates
    # ========================================================

    def to_unconstrained(self, latent=None):
        """This point mapped to the real line, field by field.

        Returns an object of the same type holding unconstrained values,
        so ``+``, ``* lr``, ``jax.grad`` and batching work on it. Uses the
        fields' declared ``support`` unless a ``LatentSpace`` is given.
        Values on a boundary map to +/-inf.
        """
        from .latent import LatentSpace

        latent = LatentSpace.default(type(self)) if latent is None else latent
        return latent.from_parameter(self)

    @classmethod
    def from_unconstrained(cls, z, latent=None):
        """Inverse of ``to_unconstrained``: map ``z`` back into the supports."""
        from .latent import LatentSpace

        latent = LatentSpace.default(cls) if latent is None else latent
        if latent.parameter_type is not cls:
            raise TypeError(
                f"LatentSpace is for {latent.parameter_type.__name__}, "
                f"not {cls.__name__}"
            )
        return latent.to_parameter(z)

    # ========================================================
    # Copying
    # ========================================================

    def copy(self):
        """A new object of the same type whose arrays are all fresh copies.

        Fields may hold mutable NumPy arrays (e.g. from ``from_dict``), so
        editing ``theta.x[0] = ...`` would otherwise also change every object
        sharing that array. Nested containers in fields are rebuilt too.
        """
        def leaf(v):
            return v.copy() if hasattr(v, "copy") else copy_module.deepcopy(v)

        return jax.tree_util.tree_map(leaf, self)

    # ========================================================
    # Conversion
    # ========================================================

    def to_dict(self):
        """Only dataclass fields, preserving JAX arrays.

        Nested fields become nested dictionaries; see ``to_flat_dict`` for
        dotted keys.
        """
        nested = self.nested_types()
        return {
            f.name: (
                getattr(self, f.name).to_dict() if f.name in nested
                else getattr(self, f.name)
            )
            for f in fields(self)
        }

    @classmethod
    def from_dict(cls, values):
        """Accept a dict or DataFrame; ignore extra columns.

        A nested field may be given as a child object, a nested dict, or
        flat dotted keys such as ``"masses.m1"`` (so a DataFrame with dotted
        columns converts directly).
        """
        missing = []
        obj = cls._from_values(values, "", missing)
        if missing:
            raise KeyError(
                f"Missing fields for {cls.__name__}: {missing}"
            )
        return obj

    @classmethod
    def _from_values(cls, values, prefix, missing):
        """``cls`` from ``values``, appending absent paths to ``missing``."""
        nested = cls.nested_types()
        data = {}
        for name in (f.name for f in fields(cls)):
            child = nested.get(name)
            if name in values:
                v = values[name]
                if child is None:
                    # Convert pandas Series to NumPy arrays.
                    # Leave JAX arrays, tracers and scalars alone.
                    data[name] = v.to_numpy() if hasattr(v, "to_numpy") else v
                else:
                    if not isinstance(v, (child, Mapping, pd.DataFrame)):
                        raise TypeError(
                            f"{prefix}{name}: expected {child.__name__} or a "
                            f"dictionary/DataFrame, got {type(v).__name__}"
                        )
                    data[name] = v if isinstance(v, child) else child._from_values(
                        v, f"{prefix}{name}.", missing
                    )
            elif child is not None:
                dotted = name + "."
                data[name] = child._from_values(
                    {
                        k[len(dotted):]: values[k]
                        for k in values.keys()
                        if isinstance(k, str) and k.startswith(dotted)
                    },
                    f"{prefix}{dotted}",
                    missing,
                )
            else:
                missing.append(prefix + name)
        return None if missing else cls(**data)

    def __repr__(self):
        nested = self.nested_types()
        attrs = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}" if f.name in nested
            else f"{f.name}={jnp.shape(getattr(self, f.name))}"
            for f in fields(self)
        )

        return (
            f"{type(self).__name__}("
            f"batch_shape={self.batch_shape}, {attrs})"
        )

    # ========================================================
    # LaTeX representation
    # ========================================================

    def _latex_description(self):
        batch, axis_sizes = self._dimensions()
        specs = self.field_specs()

        if len(batch) == 0:
            batch_symbols = []
        elif len(batch) == 1:
            batch_symbols = ["n"]
        else:
            batch_symbols = [
                f"n_{k}" for k in range(len(batch))
            ]

        rhs = []
        axis_latex = _axis_latex(type(self))
        nested = self.nested_types()

        for f in fields(self):
            spec = specs[f.name]

            # A nested field shows the child's symbol, without core axes.
            index_symbols = () if f.name in nested else (
                spec.index_latex
                or tuple(latex_name(name) for name in spec.index)
            )

            variable = "{" + (spec.latex or f.name) + "}"

            indices = batch_symbols + list(index_symbols)

            if indices:
                variable += "_{" + ",".join(indices) + "}"

            rhs.append(variable)

        root = "{" + type(self)._latex + "}"

        if batch_symbols:
            lhs = (
                r"\{"
                + root
                + "_{"
                + ",".join(batch_symbols)
                + "}"
                + r"\}"
            )
        else:
            lhs = root

        expr = (
            lhs
            + " = "
            + r"\{"
            + r",\; ".join(rhs)
            + r"\}"
        )

        where = [
            f"{symbol} \\in 1\\ldots {length}"
            for symbol, length in zip(batch_symbols, batch)
        ]

        for name, length in axis_sizes.items():
            symbol = axis_latex[name]

            where.append(
                f"{latex_name(name)}={symbol} "
                f"\\in 1\\ldots {length}"
            )

        return expr, where

    # ========================================================
    # Rich HTML display for Marimo / Jupyter
    # ========================================================

    def _display_math(self):
        """LaTeX for the collapsed object header."""
        expr, _ = self._latex_description()
        return expr.partition(" = ")[0]

    def _display_fields(self):
        """One row per field; a nested field is a section of its own rows."""
        batch = self.batch_shape

        symbols = (
            ["n"] if len(batch) == 1
            else [f"n_{i}" for i in range(len(batch))]
        )

        rows = []
        nested = self.nested_types()

        for f in fields(self):
            spec = self.field_specs()[f.name]

            if f.name in nested:
                rows.append(section(
                    spec.name or f.name.replace("_", " ").title(),
                    getattr(self, f.name)._display_fields(),
                    opened=True,
                ))
                continue

            index_latex = spec.index_latex or tuple(
                latex_name(k) for k in spec.index
            )

            var = "{" + (spec.latex or f.name) + "}"
            indices = symbols + list(index_latex)

            if indices:
                var += "_{" + ",".join(indices) + "}"

            rows.append(
                parameter_row(
                    spec.name or f.name.replace("_", " ").title(),
                    var,
                    getattr(self, f.name),
                )
            )

        return "".join(rows)

    def _display_body(self):
        """Parameter fields, values and named dimensions."""
        batch, named_axes = self._dimensions()

        symbols = (
            ["n"] if len(batch) == 1
            else [f"n_{i}" for i in range(len(batch))]
        )

        rows = [self._display_fields()]
        axis_latex = _axis_latex(type(self))

        # Batch dimension display.
        if batch:
            tex = r",\quad ".join(
                rf"{symbol} \in 1\ldots {length}"
                for symbol, length in zip(symbols, batch)
            )

            rows.append(
                inline_row(
                    "<strong>Batch:</strong>",
                    inline_math(tex),
                )
            )

        # Physical named dimensions.
        if named_axes:
            names = ",".join(latex_name(k) for k in named_axes)

            symbols_tex = ",".join(
                axis_latex[k] for k in named_axes
            )

            rows.append(
                inline_row(
                    "<strong>Dimensions:</strong>",
                    inline_math(
                        f"({names})=({symbols_tex})"
                    ),
                    "<span style='opacity:0.7'>"
                    "shape=<code>"
                    + str(tuple(named_axes.values()))
                    + "</code></span>",
                    separated=True,
                )
            )

        return "".join(rows)

    def _repr_markdown_(self) -> str:
        expr, where = self._latex_description()

        result = f"${expr}$"

        if where:
            result += (
                " where $"
                + r",\quad ".join(where)
                + "$"
            )

        return result

    @property
    def latex(self) -> str:
        return type(self)._latex

    @property
    def name(self) -> str:
        return type(self)._display_name

    @property
    def parameter_specs(self) -> dict[str, ParameterSpec]:
        return type(self).field_specs()

    @_type_property
    def parameters(cls) -> list[str]:
        """Field names in definition order, e.g. ``['x', 'y']``.

        Works on the class (``Thing.parameters``) and on instances.
        """
        return [f.name for f in fields(cls)]


def _axis_latex(cls) -> dict[str, str]:
    """``{named axis: LaTeX symbol}`` over every array field, nested included."""
    out = {}
    for spec in cls.flat_field_specs().values():
        symbols = spec.index_latex or tuple(latex_name(k) for k in spec.index)
        for name, symbol in zip(spec.index, symbols):
            out.setdefault(name, symbol)
    return out


def _resolve_nested(cls):
    """``({field: parameter type}, complete)`` for the nested fields of ``cls``.

    String annotations are resolved with ``typing.get_type_hints``; when
    that fails (e.g. a forward reference not yet defined, or a class local
    to a function) the raw annotation is used, and ``complete`` is False if
    any annotation is still a string. A nested field may not set ``index``
    or ``support``.
    """
    try:
        hints = typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001 - fall back to the raw annotations
        hints = {}
    nested, complete = {}, True
    for f in fields(cls):
        t = hints.get(f.name, f.type)
        complete &= not isinstance(t, str)
        if isinstance(t, type) and issubclass(t, Parameters):
            spec = f.metadata.get(_SPEC_KEY)
            if spec is not None and (spec.index or spec.index_latex or spec.support is not None):
                raise TypeError(
                    f"{cls.__name__}.{f.name} is a nested {t.__name__}; "
                    "it cannot set index or support (set them on its fields)"
                )
            nested[f.name] = t
    return nested, complete


def coerce(obj, cls):
    """Turn a dict / DataFrame into ``cls``; pass instances through."""
    if cls is None:
        return obj

    if isinstance(obj, cls):
        return obj

    if isinstance(obj, (Mapping, pd.DataFrame)):
        return cls.from_dict(obj)

    raise TypeError(
        f"Expected {cls.__name__} or a dictionary/DataFrame, "
        f"got {type(obj).__name__}"
    )


# ============================================================
# 3. The decorator
# ============================================================

def _decorate(
    cls: type,
    *,
    name: str | None,
    latex: str | None,
    frozen: bool,
):
    annotations = dict(inspect.get_annotations(cls))

    # Allow `x: parameter(...)` as well as `x: T = parameter(...)`.
    for field_name, annotation in annotations.items():
        if isinstance(annotation, Field):
            if field_name not in cls.__dict__:
                setattr(cls, field_name, annotation)
            annotations[field_name] = Any

    # Allow bare `x = parameter(...)` without an annotation.
    for field_name, value in tuple(vars(cls).items()):
        if isinstance(value, Field) and field_name not in annotations:
            annotations[field_name] = Any

    cls.__annotations__ = annotations

    cls = dataclass(cls, frozen=frozen, repr=False)

    if not issubclass(cls, Parameters):
        cls = type(
            cls.__name__,
            (Parameters, cls),
            {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__doc__": cls.__doc__,
            },
        )

    cls._display_name = name or cls.__name__
    cls._latex = latex or cls.__name__

    # Find nested fields now so that misuse raises here; string annotations
    # that cannot be resolved yet are retried on first use.
    nested, complete = _resolve_nested(cls)
    if complete:
        cls._nested_fields = nested

    return jax.tree_util.register_pytree_node_class(cls)


@overload
def parameter_dataclass[T](cls: type[T], /) -> type[T]:
    ...


@overload
def parameter_dataclass[T](
    *,
    name: str | None = None,
    latex: str | None = None,
    frozen: bool = True,
) -> Callable[[type[T]], type[T]]:
    ...


@dataclass_transform(
    field_specifiers=(parameter,),
    frozen_default=True,
)
def parameter_dataclass[T](
    cls: type[T] | None = None,
    /,
    *,
    name: str | None = None,
    latex: str | None = None,
    frozen: bool = True,
) -> type[T] | Callable[[type[T]], type[T]]:

    def deco(candidate):
        return _decorate(
            candidate,
            name=name,
            latex=latex,
            frozen=frozen,
        )

    return deco if cls is None else deco(cls)


def batched_method(fn):
    """
    Automatically vmap a method over all batch dimensions of self.

    Additional arguments are shared across the batch.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.batch_shape:
            return fn(self, *args, **kwargs)

        def single(obj):
            return fn(obj, *args, **kwargs)

        # Automatically handle arbitrarily many batch axes.
        for _ in self.batch_shape:
            single = jax.vmap(single)

        return single(self.broadcast())

    return wrapper

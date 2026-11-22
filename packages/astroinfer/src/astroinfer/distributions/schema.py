"""Argument schemas: how a declared type maps onto call arguments.

A distribution declares its output and condition types as one of

    None                    no arguments
    T                       one positional argument
    (T1, T2, ...)           several positional arguments
    {"a": T1, "b": T2}      keyword arguments

A parameter dataclass is always one object; it is never expanded into its
fields. A type that is not a parameter dataclass (``float``, ``Any``, or
``None`` as a mapping value) marks an untyped argument, which is passed
through unchanged.

Values bound to a schema are kept as a tuple in schema order, whatever the
form, so batching code can treat every form alike.
"""
from collections.abc import Mapping
from typing import Any

from ..display import latex_name
from ..parameters import Parameters, coerce


def is_parameter_type(t):
    return isinstance(t, type) and issubclass(t, Parameters)


def or_untyped(t):
    """``t``, or ``Any`` (one untyped argument) when ``t`` is None."""
    return Any if t is None else t


def batch_shape(value):
    """Batch axes of a bound value; None if it is not batched."""
    return value.batch_shape if isinstance(value, Parameters) else None


class Schema:
    """One of the four argument forms, with the types in call order."""

    def __init__(self, kind, types=(), names=()):
        self.kind = kind
        self.types = tuple(types)
        self.names = tuple(names)

    @classmethod
    def from_type(cls, t):
        if t is None or (isinstance(t, (tuple, Mapping)) and not t):
            return cls("none")
        if isinstance(t, Mapping):
            return cls("mapping", t.values(), t.keys())
        if isinstance(t, tuple):
            return cls("tuple", t)
        return cls("single", (t,))

    def __repr__(self):
        if self.kind == "mapping":
            return f"Schema(mapping, {dict(zip(self.names, self.types))})"
        return f"Schema({self.kind}, {self.types})"

    @property
    def positional_arity(self):
        return 0 if self.kind == "mapping" else len(self.types)

    def labels(self):
        """A name per argument: the mapping keys, else the type names."""
        if self.kind == "mapping":
            return self.names
        return tuple(
            getattr(t, "__name__", f"arg{i}") for i, t in enumerate(self.types)
        )

    # ========================================================
    # Binding call arguments
    # ========================================================

    def bind(self, args, kwargs, *, partial=False):
        """
        Split a call into ``(values, context, remaining_args)``.

        values:
            Tuple in schema order. With ``partial=True`` a mapping schema
            may leave names unbound; values is then a dict of the bound ones.
        context:
            Keywords that are not schema names, forwarded unchanged.
        remaining_args:
            Positional arguments after the schema's own (e.g. the key).
        """
        args, kwargs = tuple(args), dict(kwargs)
        if self.kind != "mapping":
            n = self.positional_arity
            if len(args) < n:
                raise TypeError(
                    f"Expected {n} positional argument(s) for {self.labels()}, "
                    f"got {len(args)}"
                )
            return args[:n], kwargs, args[n:]

        given = {}
        if args and isinstance(args[0], Mapping) and set(args[0]) == set(self.names):
            given, args = dict(args[0]), args[1:]
        for name in self.names:
            if name in kwargs:
                if name in given:
                    raise TypeError(f"{name!r} given twice")
                given[name] = kwargs.pop(name)
        if partial:
            return given, kwargs, args
        missing = [name for name in self.names if name not in given]
        if missing:
            raise TypeError(f"Missing arguments {missing}")
        return tuple(given[name] for name in self.names), kwargs, args

    def arguments(self, values):
        """``(args, kwargs)`` for a call with these values."""
        values = tuple(values)
        if self.kind == "mapping":
            return (), dict(zip(self.names, values))
        return values, {}

    def call(self, fn, values, **extra):
        args, kwargs = self.arguments(values)
        return fn(*args, **kwargs, **extra)

    def coerce(self, values):
        """Turn dicts and DataFrames into the declared parameter types."""
        return tuple(
            coerce(v, t) if is_parameter_type(t) else v
            for v, t in zip(values, self.types)
        )

    # ========================================================
    # Display
    # ========================================================

    def latex(self, default):
        """LaTeX for the arguments: ``m, s`` for a tuple; ``default`` if untyped."""
        if self.kind == "none":
            return ""
        if self.kind == "mapping":
            parts = [
                getattr(t, "_latex", None) or latex_name(n)
                for n, t in zip(self.names, self.types)
            ]
        else:
            parts = [getattr(t, "_latex", default) for t in self.types]
        return ", ".join(parts)

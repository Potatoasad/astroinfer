"""Unconstrained coordinates for a parameter type.

Each field of a ``@parameter_dataclass`` may declare its domain as a numpyro
constraint, ``parameter(..., support=constraints.interval(0, 1))``. A
``LatentSpace`` maps every field to the real line with
``numpyro.distributions.transforms.biject_to(support)`` and back. Supports
can also be given (or overridden) when building the space, as constraints or
as numpyro distributions (whose ``.support`` is used).

The latent point is stored in the same parameter type, so arithmetic,
``jax.grad`` and batching work on it unchanged::

    z = d.to_unconstrained()
    for _ in range(200):
        z = z + lr * jax.grad(lambda z: f(type(d).from_unconstrained(z)))(z)
    d_opt = type(d).from_unconstrained(z)
"""
from dataclasses import fields
from functools import cache
from html import escape

import jax.numpy as jnp

from .display import RichDisplay, html_table, inline_math, latex_name
from .parameters import coerce


def _as_constraint(value, name):
    from numpyro.distributions import Distribution, constraints

    if value is None:
        return constraints.real
    if isinstance(value, constraints.Constraint):
        return value
    if isinstance(value, Distribution):
        return value.support
    raise TypeError(
        f"{name}: support must be a numpyro constraint or distribution, "
        f"got {type(value).__name__}"
    )


class LatentSpace(RichDisplay):
    """Bijection between a parameter type and the real line, field by field.

    LatentSpace(GWMockCatData)                         # declared supports
    LatentSpace(GWMockCatData, {"snr": constraints.positive,
                                "chi_eff": dist.Uniform(-1, 1)})

    ``supports`` overrides the declared support of the fields it names.
    Fields of nested parameter types are named by dotted path
    (``"masses.m1"``), and ``supports`` and ``transforms`` use those keys.
    ``children`` holds the space of each nested field.
    """

    def __init__(self, parameter_type, supports=None):
        from numpyro.distributions.transforms import biject_to

        specs = parameter_type.flat_field_specs()
        supports = dict(supports or {})
        unknown = set(supports) - set(specs)
        if unknown:
            raise KeyError(
                f"{parameter_type.__name__} has no fields {sorted(unknown)}"
            )

        self.parameter_type = parameter_type
        self.supports = {
            k: _as_constraint(supports.get(k, spec.support), k)
            for k, spec in specs.items()
        }
        self.transforms = {k: biject_to(c) for k, c in self.supports.items()}
        self.children = {
            name: LatentSpace(child, {
                k[len(name) + 1:]: c for k, c in self.supports.items()
                if k.startswith(name + ".")
            })
            for name, child in parameter_type.nested_types().items()
        }

    @staticmethod
    @cache
    def default(parameter_type):
        """The space from the type's declared supports (cached per type)."""
        return LatentSpace(parameter_type)

    def _coerce(self, obj):
        obj = coerce(obj, self.parameter_type)
        if type(obj) is not self.parameter_type:
            raise TypeError(
                f"Expected {self.parameter_type.__name__}, "
                f"got {type(obj).__name__}"
            )
        return obj

    def _map(self, obj, inverse):
        values = {}
        for f in fields(self.parameter_type):
            v = getattr(obj, f.name)
            if f.name in self.children:
                values[f.name] = self.children[f.name]._map(v, inverse)
            else:
                t = self.transforms[f.name]
                values[f.name] = (t.inv if inverse else t)(jnp.asarray(v))
        return self.parameter_type(**values)

    def from_parameter(self, d):
        """Physical point d -> latent point z (same type, real-valued)."""
        return self._map(self._coerce(d), inverse=True)

    def to_parameter(self, z):
        """Latent point z -> physical point d inside the supports."""
        return self._map(self._coerce(z), inverse=False)

    def log_abs_det_jacobian(self, z):
        """log |d to_parameter(z) / dz|, summed over fields and core axes.

        Not needed for optimization; add it to a log density when sampling or
        when moving a density into latent space. Shape: z's batch shape.
        Nested fields are included.
        """
        z = self._coerce(z)
        total = 0.0
        for k, spec, zk in z._flat_items():
            t = self.transforms[k]
            zk = jnp.asarray(zk)
            ladj = jnp.broadcast_to(t.log_abs_det_jacobian(zk, t(zk)), zk.shape)
            n = len(spec.index)
            total = total + (ladj.sum(axis=tuple(range(-n, 0))) if n else ladj)
        return jnp.broadcast_to(total, z.batch_shape)

    # ========================================================
    # Display
    # ========================================================

    def __repr__(self):
        inner = ", ".join(f"{k}={c!r}" for k, c in self.supports.items())
        return f"LatentSpace({self.parameter_type.__name__}: {inner})"

    def _display_title(self):
        return "Latent space"

    def _display_math(self):
        root = getattr(self.parameter_type, "_latex", self.parameter_type.__name__)
        return r"z = T^{-1}(" + root + r")\in\mathbb{R}^n"

    def _display_body(self):
        specs = self.parameter_type.flat_field_specs()
        rows = [
            [
                inline_math(specs[k].latex or latex_name(k))
                # Nested fields also show their path.
                + (" <code>" + escape(k) + "</code>" if "." in k else ""),
                "<code>" + escape(repr(c)) + "</code>",
                "<code>" + escape(type(self.transforms[k]).__name__) + "</code>",
            ]
            for k, c in self.supports.items()
        ]
        return html_table(["Field", "Support", "Transform"], rows)

"""Products and mixtures of conditional distributions.

    product(a, b)                   p(x, y | c) = p_a(x | c) p_b(y | c), tuple output
    product(masses=a, spins=b)      the same with a mapping output
    mixture([a, b], weights=w)      p(x | c) = sum_k w_k p_k(x | c)

Both are ``ConditionalDistribution`` objects, so they batch, bind
conditions and display like any other. They run their components' single
methods (``sample_single`` and ``logpdf_single``) inside their own, so a
component has to implement those; a bound component is resolved to its
parent with the bound values, which must then be unbatched.

The condition schema of a product or mixture is the union of the
components' schemas. Mapping schemas merge by name and shared names must
have the same type (an untyped name agrees with any type); single and
tuple schemas must be equal; unconditional components accept anything; a
mapping schema cannot be joined with a single or tuple one. Each component
receives only the conditions its own schema names, and every component
receives the extra keywords (context) of the call.
"""
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from ..display import math_block, section
from ..parameters import Parameters
from .base import (
    BoundConditionalDistribution,
    ConditionalDistribution,
    _check_names,
    call_sample_single,
)
from .schema import Schema, is_parameter_type


# ============================================================
# Schemas
# ============================================================

def _type_name(t):
    if isinstance(t, tuple):
        return "(" + ", ".join(map(_type_name, t)) + ")"
    return getattr(t, "__name__", repr(t))


def _union(schemas):
    """
    The condition type joining ``[(label, Schema), ...]``.

    Returns None, a type, a tuple of types or a dict, following the rules in
    the module docstring.
    """
    named, positional = {}, None
    for label, s in schemas:
        if s.kind == "none":
            continue
        if s.kind == "mapping":
            for n, t in zip(s.names, s.types):
                old = named.get(n)
                if n in named and is_parameter_type(old) and is_parameter_type(t) and old is not t:
                    raise TypeError(
                        f"Condition {n!r} is {_type_name(old)} in one component "
                        f"and {_type_name(t)} in {label}"
                    )
                if n not in named or not is_parameter_type(old):
                    named[n] = t
            continue
        if positional is not None and positional[1].types != s.types:
            raise TypeError(
                f"Positional conditions must be equal: {positional[0]} takes "
                f"{_type_name(positional[1].types)}, {label} takes "
                f"{_type_name(s.types)}"
            )
        positional = positional or (label, s)
    if named and positional:
        raise TypeError(
            "Cannot join a mapping condition schema with a single or tuple "
            f"one ({positional[0]} takes positional conditions)"
        )
    if named:
        return named
    if positional:
        s = positional[1]
        return s.types[0] if s.kind == "single" else s.types
    return None


def _component_conditions(c, conds):
    """The part of ``conds`` (a dict or tuple) that component ``c`` names."""
    s = c._condition_schema()
    if s.kind == "none":
        return ()
    if s.kind == "mapping":
        return tuple(conds[n] for n in s.names)
    return conds


def _split_conditions(schema, args, kwargs):
    """``(conditions, context)`` of one single call; a dict for mappings."""
    if schema.kind == "mapping":
        if args:
            raise TypeError("Mapping conditions must be keywords")
        conds = {n: kwargs.pop(n) for n in schema.names}
        return conds, kwargs
    if len(args) != schema.positional_arity:
        raise TypeError(
            f"Expected {schema.positional_arity} condition(s), got {len(args)}"
        )
    return tuple(args), kwargs


def _split_data(schema, args, kwargs):
    """``(data, args, kwargs)`` of one ``logpdf_single`` call."""
    if schema.kind == "mapping":
        return tuple(kwargs.pop(n) for n in schema.names), args, kwargs
    n = schema.positional_arity
    return tuple(args[:n]), args[n:], kwargs


# ============================================================
# Running a component's single methods
# ============================================================

def _resolve(c, values, context):
    """The distribution whose single methods run, its conditions and context."""
    while isinstance(c, BoundConditionalDistribution):
        if isinstance(c.bound, dict):
            given = {**c.bound, **dict(zip(c._condition_schema().names, values))}
            values = tuple(given[n] for n in c.parent._condition_schema().names)
        else:
            values = c.bound
        context = {**c.context, **context}
        c = c.parent
    return c, c._single_conditions(c._coerce_conditions(values)), context


def _sample_component(c, values, key, context):
    c, values, context = _resolve(c, values, context)
    return call_sample_single(c.sample_single, c._condition_schema(), values, key, **context)


def _logpdf_component(c, data, values, context):
    c, values, context = _resolve(c, values, context)
    d_args, d_kwargs = c._output_schema().arguments(data)
    c_args, c_kwargs = c._condition_schema().arguments(values)
    return c.logpdf_single(*d_args, *c_args, **d_kwargs, **c_kwargs, **context)


def _check_component(label, c):
    if not isinstance(c, ConditionalDistribution):
        raise TypeError(
            f"{label} must be a ConditionalDistribution, got {type(c).__name__}"
        )
    while isinstance(c, BoundConditionalDistribution):
        bound = c.bound.values() if isinstance(c.bound, dict) else c.bound
        if any(isinstance(v, Parameters) and v.batch_shape for v in bound):
            raise ValueError(
                f"{label} is bound to batched conditions; inside a product or "
                "mixture the bound conditions must be one object each"
            )
        c = c.parent


def _dummy_conditions(c):
    """Placeholder conditions for ``jax.eval_shape``; None if not possible."""
    values = []
    for t in c._condition_schema().types:
        specs = t.flat_field_specs() if is_parameter_type(t) else None
        if specs is None or any(s.index for s in specs.values()):
            return None
        values.append(t.from_dict({p: jnp.zeros(()) for p in specs}))
    return tuple(values)


# ============================================================
# Supports
# ============================================================

_BOUNDS = ("lower_bound", "upper_bound", "base_constraint", "reinterpreted_batch_ndims")


def _canonical(c):
    """Constraint class with named singletons (``positive``) mapped to their parent."""
    from numpyro.distributions.constraints import _SingletonConstraint

    mro = type(c).__mro__
    return mro[1] if _SingletonConstraint in mro and mro[1] is not _SingletonConstraint else mro[0]


def _same_constraint(a, b):
    """Whether two NumPyro constraints have the same type and bounds."""
    if a is b:
        return True
    if _canonical(a) is not _canonical(b):
        return False
    for name in _BOUNDS:
        x, y = getattr(a, name, None), getattr(b, name, None)
        if isinstance(x, jax.core.Tracer) or isinstance(y, jax.core.Tracer):
            if x is not y:  # traced bounds compare equal only to themselves
                return False
        elif hasattr(x, "check") or hasattr(y, "check"):
            if x is None or y is None or not _same_constraint(x, y):
                return False
        elif (x is None) != (y is None) or (
            x is not None
            and (np.shape(x) != np.shape(y) or not np.all(np.asarray(x) == np.asarray(y)))
        ):
            return False
    return True


def _supports_of(c):
    """``c.leaf_supports()`` if it gives a dict, else None. Bound wrappers are skipped."""
    while isinstance(c, BoundConditionalDistribution):
        c = c.parent
    fn = getattr(c, "leaf_supports", None)
    out = fn() if callable(fn) else None
    return dict(out) if isinstance(out, dict) else None


def _check_supports(supports):
    """Raise unless every dict in ``supports`` agrees at every path."""
    first = supports[0]
    for k, other in enumerate(supports[1:], start=1):
        for path in sorted(set(first) | set(other)):
            if path not in first or path not in other:
                raise ValueError(
                    f"Mixture components have different leaves: {path!r} is "
                    f"missing from component {k if path in first else 0}"
                )
            if not _same_constraint(first[path], other[path]):
                raise ValueError(
                    f"Mixture components have different supports at {path!r}: "
                    f"component 0 has {first[path]}, component {k} has "
                    f"{other[path]}. Pass support= to the mixture to set one."
                )


# ============================================================
# Display
# ============================================================

def _component_sections(items, equations=()):
    """The "Model" block: equations, then one collapsible section per component."""
    return math_block(equations) + "".join(
        section(f"Component {label}: {c._display_title()}", c._repr_html_())
        for label, c in items
    )


def _factor_tex(c):
    # Only the left-hand side: a mixture's density carries its own "= \\sum ...".
    return c._display_density(*c._display_tex()).partition(" = ")[0]


def _fn_name(fn):
    return getattr(fn, "__name__", type(fn).__name__)


# ============================================================
# Product
# ============================================================

@dataclass(eq=False)
class ProductDistribution(ConditionalDistribution):
    """
    Independent product p(x_1, ..., x_n | c) = prod_i p_i(x_i | c).

    ``components`` is a tuple (tuple-valued output, in component order) or
    a dict (mapping-valued output, keyed by name). Each component must have
    a single output type; for several outputs, use a parameter type, since
    nested tuples or mappings would not be batched. The log density is the
    sum of the components' log densities. Component ``i`` samples with
    ``jax.random.fold_in(key, i)``.
    """

    components: Any

    display_name = "Product"

    def __post_init__(self):
        named = isinstance(self.components, Mapping)
        self.components = dict(self.components) if named else tuple(self.components)
        if not self.components:
            raise ValueError("A product needs at least one component")
        for label, c in self._items():
            _check_component(f"Component {label}", c)
            if c._output_schema().kind != "single":
                raise TypeError(
                    f"Component {label} has output type "
                    f"{_type_name(c._output_type())}; product components need "
                    "a single output type (use a parameter type to group several)"
                )
        self._conditions = _union(
            [(f"component {label}", c._condition_schema()) for label, c in self._items()]
        )
        _check_names(self._output_schema(), self._condition_schema())

    def _items(self):
        if isinstance(self.components, dict):
            return list(self.components.items())
        return list(enumerate(self.components))

    def _output_type(self):
        if isinstance(self.components, dict):
            return {n: c._output_type() for n, c in self.components.items()}
        return tuple(c._output_type() for c in self.components)

    def _condition_type(self):
        return self._conditions

    def sample_single(self, *args, key, **kwargs):
        conds, context = _split_conditions(self._condition_schema(), args, kwargs)
        draws = [
            _sample_component(
                c, _component_conditions(c, conds), jax.random.fold_in(key, i), context
            )
            for i, (_, c) in enumerate(self._items())
        ]
        if isinstance(self.components, dict):
            return dict(zip(self.components, draws))
        return tuple(draws)

    def logpdf_single(self, *args, **kwargs):
        data, args, kwargs = _split_data(self._output_schema(), args, kwargs)
        conds, context = _split_conditions(self._condition_schema(), args, kwargs)
        return sum(
            _logpdf_component(c, (x,), _component_conditions(c, conds), context)
            for x, (_, c) in zip(data, self._items())
        )

    def leaf_supports(self):
        """
        ``{path: constraint}`` with every component's leaf supports under the
        component's index or name (``"0.chi"``, ``"spins.chi"``), or None if
        some component does not give them.
        """
        out = {}
        for label, c in self._items():
            supports = _supports_of(c)
            if supports is None:
                return None
            out.update({f"{label}.{path}": v for path, v in supports.items()})
        return out

    def _display_density(self, cond_tex, out_tex):
        lhs = super()._display_density(cond_tex, out_tex)
        return lhs + " = " + r"\, ".join(_factor_tex(c) for _, c in self._items())

    def _display_model(self):
        return _component_sections(self._items())

    def _display_config(self):
        return {"components": len(self.components)}


def product(*components, **named_components):
    """
    The independent product of several distributions.

    ``product(a, b)`` has a tuple-valued output ``(x_a, x_b)``;
    ``product(masses=a, spins=b)`` has a mapping-valued output
    ``{"masses": x_a, "spins": x_b}``. Positional and keyword components
    cannot be mixed. The conditions are the union of the components'
    conditions (see :mod:`astroinfer.distributions.algebra`).
    """
    if components and named_components:
        raise TypeError("Give the components either positionally or as keywords")
    return ProductDistribution(named_components or components)


# ============================================================
# Mixture
# ============================================================

def _signature(fn):
    """``(required argument names, takes **kwargs)``; None if unreadable."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    kinds = inspect.Parameter
    names = tuple(
        p.name for p in params
        if p.default is p.empty
        and p.kind in (kinds.POSITIONAL_ONLY, kinds.POSITIONAL_OR_KEYWORD, kinds.KEYWORD_ONLY)
    )
    return names, any(p.kind is kinds.VAR_KEYWORD for p in params)


@dataclass(eq=False)
class MixtureDistribution(ConditionalDistribution):
    """
    Mixture p(x | c) = sum_k w_k(c) p_k(x | c) over one output type.

    weights:
        Probabilities, an array of shape (K,) or a callable of the
        conditions returning one. A fixed array is checked to be finite,
        nonnegative and normalized (within 1e-6); a callable's result is
        used as returned.
    logits:
        Unnormalized log weights, an array or a callable as above, turned
        into weights by a softmax. Exactly one of ``weights`` and ``logits``
        is given.
    parameter_type:
        The condition type. By default it is the union of the components'
        condition types; when that is a mapping or empty, the required
        argument names of a weight or logit callable are added as untyped
        conditions. A callable is called with the conditions it names
        (positionally for single and tuple schemas).
    support:
        ``{path: constraint}`` returned by ``leaf_supports``. By default the
        components' own leaf supports are used when they all give them; they
        must then be equal.

    Every component must have the same output type and draw objects of the
    same pytree structure and leaf shapes. Sampling draws one label per
    object and returns that component's complete draw; every component is
    sampled (component k with ``fold_in``), so the cost is K draws. The log
    density marginalizes the label with ``logsumexp``, so it is
    differentiable with respect to the weights, the logits and the
    conditions; the label is not a NUTS variable.
    """

    components: Any
    weights: Any = None
    logits: Any = None
    parameter_type: Any = None
    support: Any = None

    display_name = "Mixture"

    def __post_init__(self):
        self.components = tuple(self.components)
        K = len(self.components)
        if not K:
            raise ValueError("A mixture needs at least one component")
        for k, c in enumerate(self.components):
            _check_component(f"Component {k}", c)
            if k and c._output_type() != self.components[0]._output_type():
                raise TypeError(
                    f"Mixture components must have the same output type: "
                    f"component 0 has {_type_name(self.components[0]._output_type())}, "
                    f"component {k} has {_type_name(c._output_type())}"
                )
        if (self.weights is None) == (self.logits is None):
            raise ValueError("Give exactly one of weights and logits")
        name = "weights" if self.logits is None else "logits"
        fn = self.weights if self.logits is None else self.logits
        if callable(fn):
            self._fn_signature = _signature(fn)
        else:
            fn = jnp.asarray(fn)
            if fn.shape != (K,):
                raise ValueError(
                    f"{name} must have shape ({K},) for {K} components, got "
                    f"{fn.shape}; pass a callable for weights that vary"
                )
            if name == "weights" and not isinstance(fn, jax.core.Tracer):
                w = np.asarray(fn)
                if not np.all(np.isfinite(w)) or np.any(w < 0):
                    raise ValueError(f"weights must be finite and nonnegative, got {w}")
                if abs(w.sum() - 1) > 1e-6:
                    raise ValueError(f"weights must sum to one, got {w.sum()}")
            setattr(self, name, fn)
        self._conditions = self._join_conditions(fn)
        _check_names(self._output_schema(), self._condition_schema())
        if self.support is not None:
            self._supports = dict(self.support)
        else:
            supports = [_supports_of(c) for c in self.components]
            self._supports = None if any(s is None for s in supports) else supports[0]
            if self._supports is not None:
                _check_supports(supports)
        self._check_shapes_now()

    # --------------------------------------------------------
    # Conditions and weights
    # --------------------------------------------------------

    def _join_conditions(self, fn):
        parts = [(f"component {k}", c._condition_schema()) for k, c in enumerate(self.components)]
        joined = _union(parts)
        sig = self._fn_signature if callable(fn) else None
        if self.parameter_type is not None:
            given = Schema.from_type(self.parameter_type)
            merged = Schema.from_type(_union([("parameter_type", given), *parts]))
            if merged.names != given.names or merged.positional_arity != given.positional_arity:
                raise TypeError(
                    "parameter_type must name every condition of the components"
                )
            joined = self.parameter_type
        elif sig is not None and Schema.from_type(joined).kind in ("none", "mapping"):
            extra = {n: Any for n in sig[0] if n not in (joined or {})}
            joined = {**(joined or {}), **extra} or None
        schema = Schema.from_type(joined)
        if sig is not None:
            names = sig[0]
            if schema.kind == "mapping":
                missing = [n for n in names if n not in schema.names]
            else:
                missing = list(names[schema.positional_arity:])
            if missing:
                raise TypeError(
                    f"The weight function needs {missing}, which are not "
                    "conditions of the mixture"
                )
        return joined

    def _condition_type(self):
        return self._conditions

    def _output_type(self):
        return self.components[0]._output_type()

    def _call_fn(self, fn, conds):
        if not isinstance(conds, dict):
            return fn(*conds)
        sig = self._fn_signature
        if sig is None or sig[1]:
            return fn(**conds)
        return fn(**{n: conds[n] for n in sig[0]})

    def _log_weights(self, conds):
        """``(log weights, weights or None)`` for one object."""
        K = len(self.components)
        if self.logits is not None:
            v = self._call_fn(self.logits, conds) if callable(self.logits) else self.logits
        else:
            v = self._call_fn(self.weights, conds) if callable(self.weights) else self.weights
        v = jnp.asarray(v)
        if v.shape != (K,):
            raise ValueError(
                f"The mixture weights for one object must have shape ({K},), got {v.shape}"
            )
        if self.logits is not None:
            return jax.nn.log_softmax(v), None
        return jnp.log(v), v

    # --------------------------------------------------------
    # Draw shapes
    # --------------------------------------------------------

    @staticmethod
    def _check_draws(draws):
        def describe(d):
            leaves, tree = jax.tree.flatten(d)
            return tree, [tuple(getattr(x, "shape", np.shape(x))) for x in leaves]

        first = describe(draws[0])
        for k, d in enumerate(draws[1:], start=1):
            this = describe(d)
            if this != first:
                raise TypeError(
                    "Mixture components must draw objects with the same "
                    f"structure and leaf shapes: component 0 draws {first[0]} "
                    f"with shapes {first[1]}, component {k} draws {this[0]} "
                    f"with shapes {this[1]}"
                )

    def _check_shapes_now(self):
        """Compare the draws with ``jax.eval_shape`` when the conditions allow."""
        shapes = []
        for c in self.components:
            dummies = _dummy_conditions(c)
            if dummies is None:
                return
            try:
                shapes.append(jax.eval_shape(
                    lambda k, c=c, d=dummies: _sample_component(c, d, k, {}),
                    jax.random.key(0),
                ))
            except Exception:  # needs context or other inputs: checked when sampling
                return
        self._check_draws(shapes)

    # --------------------------------------------------------
    # Single methods
    # --------------------------------------------------------

    def sample_single(self, *args, key, **kwargs):
        conds, context = _split_conditions(self._condition_schema(), args, kwargs)
        k_label, k_draw = jax.random.split(key)
        draws = [
            _sample_component(
                c, _component_conditions(c, conds), jax.random.fold_in(k_draw, k), context
            )
            for k, c in enumerate(self.components)
        ]
        self._check_draws(draws)
        label = jax.random.categorical(k_label, self._log_weights(conds)[0])
        return jax.tree.map(lambda *xs: jnp.stack(xs)[label], *draws)

    def logpdf_single(self, *args, **kwargs):
        data, args, kwargs = _split_data(self._output_schema(), args, kwargs)
        conds, context = _split_conditions(self._condition_schema(), args, kwargs)
        lp = jnp.stack([
            _logpdf_component(c, data, _component_conditions(c, conds), context)
            for c in self.components
        ])
        log_w, w = self._log_weights(conds)
        # With weights, b=w keeps the gradient finite at a zero weight.
        return logsumexp(lp, b=w) if w is not None else logsumexp(lp + log_w)

    def as_generative(self, hyperparameter_type=None, external=(), *,
                      conditioned_on=None):
        """
        A mixture of structured distributions over the same fields, drawn by
        fixed-bank transport. See :mod:`astroinfer.distributions.transport`.
        """
        from .transport import as_generative

        return as_generative(self, hyperparameter_type, external,
                             conditioned_on=conditioned_on)

    def leaf_supports(self):
        """
        ``{path: constraint}`` shared by every component, or None.

        This is ``support`` if given; otherwise the components' common leaf
        supports when every component has a ``leaf_supports`` method giving a
        dict (checked to be equal at construction); otherwise None.
        """
        return None if self._supports is None else dict(self._supports)

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    def _display_density(self, cond_tex, out_tex):
        lhs = super()._display_density(cond_tex, out_tex)
        given = r"\mid " + cond_tex if cond_tex else ""
        fn = self.weights if self.logits is None else self.logits
        w = r"w_k(" + cond_tex + ")" if cond_tex and callable(fn) else "w_k"
        return (
            lhs + rf" = \sum_{{k=1}}^{{{len(self.components)}}} {w}\, "
            r"p_k\!\left(" + out_tex + given + r"\right)"
        )

    def _display_equations(self):
        fn = self.weights if self.logits is None else self.logits
        if callable(fn):
            name = _fn_name(fn)
            value = "f" if name == "<lambda>" else r"\mathtt{" + name.replace("_", r"\_") + "}"
            value += "(" + self._display_tex()[0] + ")"
            if fn is self.logits:
                value = r"\mathrm{softmax}(" + value + ")"
            return [r"w = " + value]
        values = ", ".join(f"{v:.4g}" for v in np.asarray(fn))
        if fn is self.logits:
            return [r"w = \mathrm{softmax}(" + values + ")"]
        return [r"w = (" + values + ")"]

    def _display_model(self):
        return _component_sections(
            enumerate(self.components, start=1), self._display_equations()
        )

    def _display_config(self):
        fn = self.weights if self.logits is None else self.logits
        name = "weights" if self.logits is None else "logits"
        out = {"components": len(self.components), name: _fn_name(fn) if callable(fn) else np.asarray(fn)}
        if self.support is not None:
            out["support"] = ", ".join(f"{p}: {v}" for p, v in self.support.items())
        return out


def mixture(components, weights=None, logits=None, parameter_type=None, *, support=None):
    """
    The mixture sum_k w_k p_k(x | c) of distributions over one output type.

    Give exactly one of ``weights`` (probabilities) and ``logits``, each an
    array of shape (K,) or a callable of the conditions. See
    :class:`MixtureDistribution` for the condition schema and supports.
    """
    return MixtureDistribution(
        components, weights=weights, logits=logits,
        parameter_type=parameter_type, support=support,
    )

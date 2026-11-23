"""Distributions over parameter dataclasses, built field by field.

``BBH.distribution(masses=..., spins=...)`` returns a
``StructuredDistribution`` whose output is a ``BBH``. Each field holds one
of the records below; ``StructuredDistribution.fields`` maps field names to
them, in dataclass field order.

    NumPyroLeaf(distribution)   a NumPyro distribution for an array field
    CallableLeaf(fn)            fn(*conditions) -> NumPyro distribution
    ConditionalLeaf(fn, on)     fn(x, *conditions) -> NumPyro distribution;
                                made by ``conditional(fn, on)``
    Fixed(value)                a constant, inserted when the object is built
    Nested(distribution)        a StructuredDistribution over a nested field
    Block(distribution)         any other ConditionalDistribution, one unit

Leaves (the first three) and blocks are random *units*. A ``Nested`` child
is not a unit: its units are flattened into the root, under the child's
path, so that a field in one nested block may depend on a field in another
(``spins.chi`` on ``masses.m1``). The draw order (``order``) and the
dictionary ``x`` of values drawn so far therefore live at the root, keyed by
dotted paths relative to the root (``"masses.m1"``).

Conditions. The callables receive the conditions according to the
condition schema, set with ``.given(parameter_type)`` (a mapping schema
passes them as keywords). ``.given`` is passed down to nested children that
have no condition type of their own. Without ``.given``, a distribution
takes the condition type of its first nested child or block that has one.
A nested child or block with its own condition type receives the root's
conditions projected onto its schema: all of them for an equal single or
tuple schema, the named subset for mapping schemas.

Dependencies follow ``astroinfer.generative``. ``on`` names the dotted
paths a conditional leaf reads and is the only source of ordering
information. ``x`` holds, at each point of the draw, every fixed leaf, every
fixed nested object (at its own path), every random leaf drawn so far, and
every block value drawn so far at its path, together with the block's
leaves at their dotted paths when the value is a parameter object. Nested
structured children are not entries of ``x``; read their leaves.

Paths in ``on`` and the keys of the ``x`` a function receives are lexically
scoped: they are relative to the structured distribution that declared the
field, and a name not found there is looked up in the enclosing ones, out
to the root. ``Masses.distribution(m1=..., q=conditional(f, on="m1"))`` so
keeps working when nested under ``BBH`` (where the leaf is ``masses.m1``),
and a field of ``Spins`` may name ``"masses.m1"``, which exists only once
the spins are nested beside the masses. A path that names a field of some
enclosing type but does not exist raises at construction; a path that
cannot be resolved yet raises when the distribution is first evaluated.

The default order is depth first over the fields (a nested child keeps its
own order), with any unit moved after the units it depends on.
``.with_order([...])`` or ``order=`` replace it; an explicit order must list
every random unit once, and each must come after every unit it depends on.
"""
from collections.abc import Callable, Mapping
from dataclasses import MISSING, fields as dataclass_fields
from functools import partial
from html import escape
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints

from ..display import latex_name, math_block, section
from ..parameters import Parameters, ParameterSpec
from .base import (
    BoundConditionalDistribution,
    ConditionalDistribution,
    call_sample_single,
)
from .schema import Schema, is_parameter_type


# ============================================================
# Field records
# ============================================================

class Field:
    """Base class of the field records; ``kind`` names the record."""

    kind = None
    random = True

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class NumPyroLeaf(Field):
    """A NumPyro distribution for one array field."""

    kind = "numpyro"

    def __init__(self, distribution):
        self.distribution = distribution


class CallableLeaf(Field):
    """``fn(*conditions)`` returns the NumPyro distribution of the field."""

    kind = "callable"

    def __init__(self, fn):
        self.fn = fn


class ConditionalLeaf(Field):
    """``fn(x, *conditions)`` returns the NumPyro distribution of the field.

    ``on`` lists the dotted paths ``fn`` reads from ``x``.
    """

    kind = "conditional"

    def __init__(self, fn, on):
        self.fn = fn
        self.on = [on] if isinstance(on, str) else list(on)
        if not all(isinstance(p, str) for p in self.on):
            raise TypeError("on must be a dotted path or a list of them")


class Fixed(Field):
    """A constant field value: an array, or a parameter object for a nested field."""

    kind = "fixed"
    random = False

    def __init__(self, value):
        self.value = value


class Nested(Field):
    """A ``StructuredDistribution`` over a nested field, flattened into the root."""

    kind = "nested"

    def __init__(self, distribution):
        self.distribution = distribution


class Block(Field):
    """Any other ``ConditionalDistribution``; one random unit at its path."""

    kind = "block"

    def __init__(self, distribution):
        self.distribution = distribution


def conditional(fn: Callable, on) -> ConditionalLeaf:
    """
    A field whose distribution depends on fields drawn before it.

    ``fn(x, *conditions)`` receives ``x``, the dictionary of values drawn so
    far keyed by dotted path (see the module docstring), and the conditions
    of the distribution, and returns a NumPyro distribution. ``on`` (a
    dotted path or a list of them) names the paths ``fn`` reads; it fixes
    the draw order and is checked against it.

        q=conditional(lambda x: dist.Uniform(5 / x["m1"], 1), on="m1")
    """
    return ConditionalLeaf(fn, on)


# ============================================================
# Helpers
# ============================================================

class _Unit(NamedTuple):
    path: str                   # dotted path from the root
    record: Field
    owner: Any                  # the StructuredDistribution declaring the field
    chain: tuple                # ((prefix, type), ...), innermost first
    index: int                  # fold-in index: position in the flat field order
    spec: ParameterSpec
    schema: Schema              # condition schema the record's callables expect


def _event_support(c, k):
    """``c`` with at least ``k`` event dimensions."""
    if c.event_dim < k:
        return constraints.independent(c, k - c.event_dim)
    return c


def _compatible(a, b):
    return a is b or not is_parameter_type(a) or not is_parameter_type(b)


def _project(src, dst, values):
    """The values of schema ``src`` as arguments of schema ``dst``."""
    if dst.kind == "none":
        return ()
    if src.kind == dst.kind and src.kind in ("single", "tuple"):
        if len(src.types) == len(dst.types) and all(
            map(_compatible, src.types, dst.types)
        ):
            return tuple(values)
    elif src.kind == dst.kind == "mapping":
        types = dict(zip(src.names, src.types))
        if all(n in types and _compatible(types[n], t)
               for n, t in zip(dst.names, dst.types)):
            given = dict(zip(src.names, values))
            return tuple(given[n] for n in dst.names)
    raise TypeError(f"cannot pass conditions {src} to {dst}")


def _single_conditions(d, values):
    """One element's conditions as ``d``'s single methods receive them."""
    return d._single_conditions(d._coerce_conditions(tuple(values)))


def _block_call(block, conds, method, *data, key=None):
    """``block.sample_single`` / ``logpdf_single`` for one element."""
    context = {}
    if isinstance(block, BoundConditionalDistribution):
        c_args, c_kwargs = block._parent_arguments(conds)
        parent = block.parent
        conds, context, _ = parent._condition_schema().bind(
            c_args, {**c_kwargs, **block.context}
        )
        block = parent
        if isinstance(block, BoundConditionalDistribution):
            return _block_call(block, conds, method, *data, key=key)
    values = _single_conditions(block, conds)
    if method == "sample_single":
        return call_sample_single(
            block.sample_single, block._condition_schema(), values, key, **context
        )
    args, kwargs = block._condition_schema().arguments(values)
    return getattr(block, method)(*data, *args, **kwargs, **context)


def _check_leaf(d, spec, where):
    if not isinstance(d, dist.Distribution):
        raise TypeError(
            f"{where}: expected a NumPyro distribution, got {type(d).__name__}"
        )
    k = len(spec.index)
    batch, event = tuple(d.batch_shape), tuple(d.event_shape)
    if batch == () and len(event) == k:
        return
    axes = f"{k} core axes {spec.index}" if k else "no core axes"
    msg = (
        f"{where}: the field has {axes}, so its distribution needs an empty "
        f"batch_shape and {k} event dimension(s); got batch_shape {batch} "
        f"and event_shape {event}"
    )
    if batch and len(batch) + len(event) == k:
        msg += f". Use .to_event({len(batch)}) to make the batch axes event axes"
    raise ValueError(msg)


def _fixed_array(value, spec, where):
    if isinstance(value, Parameters) or isinstance(value, ConditionalDistribution):
        raise TypeError(
            f"{where}: expected an array value, got {type(value).__name__}"
        )
    k = len(spec.index)
    if np.ndim(value) != k:
        raise ValueError(
            f"{where}: a fixed value of a field with {k} core axes must have "
            f"{k} dimension(s), got shape {np.shape(value)}"
        )
    return value


def _flat_or_value(prefix, value):
    """``{path: value}`` with a parameter object's leaves under ``prefix``."""
    out = {prefix: value}
    if isinstance(value, Parameters):
        out.update({f"{prefix}.{p}": v for p, v in value.to_flat_dict().items()})
    return out


def _view(x, chain):
    """``x`` as seen from a scope: inner paths shadow outer ones."""
    out = {}
    for prefix, _ in reversed(chain):
        if prefix:
            n = len(prefix)
            out.update({k[n:]: v for k, v in x.items() if k.startswith(prefix)})
        else:
            out.update(x)
    return out


def _isclose(a, b):
    a, b = jnp.asarray(a), jnp.asarray(b)
    return jnp.all(jnp.isclose(a, b, rtol=1e-6, atol=0.0) | (a == b))


# ============================================================
# The structured distribution
# ============================================================

class StructuredDistribution(ConditionalDistribution):
    """
    A distribution over a parameter dataclass, given field by field.

    ``StructuredDistribution(data_type, fields, *, parameter_type=None,
    order=None, check_fixed=True)``. ``fields`` maps every field name to a
    NumPyro distribution, a callable returning one, a ``conditional(...)``,
    a fixed value, a nested ``StructuredDistribution`` or another
    ``ConditionalDistribution`` (see the module docstring). A field left out
    that has a dataclass default is fixed at that default. Usually made with
    ``data_type.distribution(**fields)``.

    The density is the chain rule along ``order``: the sum of
    ``log p(unit | earlier units, conditions)`` over the random units. With
    ``check_fixed=True``, ``logpdf`` of an object whose fixed fields differ
    from the declared values is ``-inf``.

    Adapter-facing API (names are stable):

        fields                      {name: field record}, this level only
        order                       [dotted path], random units in draw order
        random_paths()              [dotted path], random units in field order
        fixed_paths()               [dotted path], every fixed leaf
        fixed_values()              {path: value}, the initial ``x``
        field_at(path)              the record of a random unit
        unit_index(path)            its fold-in index
        leaf_distribution(path, x, conditions)
                                    the NumPyro distribution of a leaf
        dependencies(path)          (units read, unresolved on paths)
        block_conditions(path, conditions)
                                    the conditions to pass to a block
        update_values(x, path, value)
                                    add a unit's value to ``x``
        leaf_supports()             {leaf path: constraint} or None
        assemble(x)                 the ``data_type`` object built from ``x``

    ``conditions`` is always the tuple of condition values in schema order
    (as ``Schema.bind`` returns it), and ``x`` a dictionary keyed by dotted
    paths from the root.
    """

    density_latex = r"\pi"

    def graph(self):
        """The DAG of the fields: conditions, field dependencies, fixed values."""
        from ..graph import structured_graph
        return structured_graph(self)

    def __init__(self, data_type, fields, *, parameter_type=None, order=None,
                 check_fixed=True, key=None):
        if not is_parameter_type(data_type):
            raise TypeError(
                f"data_type must be a parameter dataclass, got {data_type!r}"
            )
        if not isinstance(fields, Mapping):
            raise TypeError("fields must be a mapping of field names")
        self.data_type = data_type
        self.parameter_type = parameter_type
        self.check_fixed = bool(check_fixed)
        self.key = jax.random.key(0) if key is None else key
        self._supplied = dict(fields)
        self._explicit_order = None if order is None else list(order)
        self.fields = self._make_fields(self._supplied)
        self._adopted = None if parameter_type is not None else self._adopt()
        self._build()

    # ========================================================
    # Construction
    # ========================================================

    def _where(self, name):
        return f"{self.data_type.__name__}.{name}"

    def _make_fields(self, given):
        T = self.data_type
        names = [f.name for f in dataclass_fields(T)]
        unknown = [n for n in given if n not in names]
        if unknown:
            raise TypeError(
                "Unknown field(s) "
                + ", ".join(self._where(n) for n in unknown)
                + f"; {T.__name__} has fields {names}"
            )
        specs, nested = T.field_specs(), T.nested_types()
        out = {}
        for f in dataclass_fields(T):
            name, where = f.name, self._where(f.name)
            if name in given:
                value = given[name]
            elif f.default is not MISSING:
                value = f.default
            elif f.default_factory is not MISSING:
                value = f.default_factory()
            else:
                raise TypeError(
                    f"{where}: missing field (give a distribution or a fixed value)"
                )
            out[name] = self._make_field(value, specs[name], nested.get(name), where)
        return out

    def _make_field(self, value, spec, child, where):
        if isinstance(value, Field):
            if not isinstance(value, ConditionalLeaf):
                raise TypeError(f"{where}: field records cannot be passed directly")
            if child is not None:
                raise TypeError(
                    f"{where}: a conditional leaf cannot describe the nested "
                    f"{child.__name__}"
                )
            return value
        if isinstance(value, dist.Distribution):
            if child is not None:
                raise TypeError(
                    f"{where}: a NumPyro distribution cannot describe the nested "
                    f"{child.__name__}; use {child.__name__}.distribution(...)"
                )
            _check_leaf(value, spec, where)
            return NumPyroLeaf(value)
        if isinstance(value, StructuredDistribution):
            if child is None or not issubclass(value.data_type, child):
                expected = child.__name__ if child else "an array field"
                raise TypeError(
                    f"{where}: expected {expected}, got a distribution over "
                    f"{value.data_type.__name__}"
                )
            if value.parameter_type is None and self.parameter_type is not None:
                value = value.given(self.parameter_type)
            return Nested(value)
        if isinstance(value, ConditionalDistribution):
            out = value._output_type()
            if child is not None:
                if not (is_parameter_type(out) and issubclass(out, child)):
                    raise TypeError(
                        f"{where}: expected a distribution over {child.__name__}, "
                        f"got one over {getattr(out, '__name__', out)}"
                    )
            elif is_parameter_type(out) or Schema.from_type(out).kind not in (
                "single", "none"
            ):
                raise TypeError(
                    f"{where}: an array field cannot hold a distribution over "
                    f"{getattr(out, '__name__', out)}"
                )
            return Block(value)
        if child is not None:
            if isinstance(value, Mapping):
                value = child.from_dict(value)
            if not isinstance(value, child):
                raise TypeError(
                    f"{where}: expected {child.__name__}, a distribution over it, "
                    f"or a fixed {child.__name__} value; got {type(value).__name__}"
                )
            if value.batch_shape:
                raise ValueError(
                    f"{where}: a fixed value must be one object, got batch "
                    f"shape {value.batch_shape}"
                )
            return Fixed(value)
        if callable(value) and not isinstance(value, (Parameters, type)):
            return CallableLeaf(value)
        return Fixed(_fixed_array(value, spec, where))

    def _adopt(self):
        """The condition type of the first child or block that has one."""
        for rec in self.fields.values():
            if isinstance(rec, (Nested, Block)):
                t = rec.distribution._condition_type()
                if Schema.from_type(t).kind != "none":
                    return t
        return None

    def _walk(self, prefix, chain):
        """``(path, record, owner, chain, spec)`` for every non-nested field."""
        chain = ((prefix, self.data_type),) + chain
        specs = self.data_type.field_specs()
        for name, rec in self.fields.items():
            if isinstance(rec, Nested):
                yield from rec.distribution._walk(prefix + name + ".", chain)
            else:
                yield prefix + name, rec, self, chain, specs[name]

    def _default_order(self, prefix=""):
        out = []
        for name, rec in self.fields.items():
            if isinstance(rec, Nested):
                out += [prefix + name + "." + p for p in rec.distribution.order]
            elif rec.random:
                out.append(prefix + name)
        return out

    def _build(self):
        root = self._condition_schema()
        flat = list(self.data_type.flat_field_specs())
        self._units, self._fixed, provides = {}, {}, {}

        for path, rec, owner, chain, spec in self._walk("", ()):
            if isinstance(rec, Fixed):
                if isinstance(rec.value, Parameters):
                    values = _flat_or_value(path, rec.value)
                    self._fixed.update({
                        p: v if isinstance(v, Parameters) else jnp.asarray(v)
                        for p, v in values.items()
                    })
                else:
                    self._fixed[path] = jnp.asarray(rec.value)
                provides.update(dict.fromkeys(self._fixed, None))
                continue
            if isinstance(rec, Block):
                schema = rec.distribution._condition_schema()
                out = rec.distribution._output_type()
                provides[path] = path
                if is_parameter_type(out):
                    provides.update({
                        f"{path}.{p}": path for p in out.flat_field_specs()
                    })
                inside = [i for i, p in enumerate(flat) if p.startswith(path + ".")]
                index = flat.index(path) if path in flat else (
                    inside[0] if inside else len(flat)
                )
            else:
                own = owner._condition_type()
                schema = Schema.from_type(own) if own is not None else root
                provides[path] = path
                index = flat.index(path)
            if self.parameter_type is not None and not isinstance(rec, NumPyroLeaf):
                try:
                    _project(root, schema, (None,) * len(root.types))
                except TypeError as e:
                    raise TypeError(f"{self._where(path)}: {e}") from None
            self._units[path] = _Unit(path, rec, owner, chain, index, spec, schema)

        # Resolve the dependencies of conditional leaves.
        self._deps, self._unresolved = {}, []
        for path, u in self._units.items():
            deps = []
            for dep in getattr(u.record, "on", ()):
                ref = next(
                    (pre + dep for pre, _ in u.chain if pre + dep in provides), None
                )
                if ref is not None:
                    if provides[ref] is not None:
                        deps.append(provides[ref])
                    continue
                head = dep.partition(".")[0]
                if any(head in t.field_specs() for _, t in u.chain):
                    raise ValueError(
                        f"{self._where(path)}: conditioned on unknown path {dep!r}"
                    )
                self._unresolved.append((path, dep))
            self._deps[path] = deps

        if self._explicit_order is None:
            self._order = self._sorted(self._default_order())
        else:
            self._order = self._check_order(self._explicit_order)

    def _sorted(self, order):
        """``order`` with every unit moved after the units it depends on."""
        out, state = [], {}

        def visit(p, stack):
            if state.get(p) == "done":
                return
            if state.get(p) == "active":
                cycle = stack[stack.index(p):] + [p]
                raise ValueError(
                    "Conditional fields depend on each other in a cycle: "
                    + " -> ".join(cycle)
                )
            state[p] = "active"
            for d in self._deps[p]:
                visit(d, stack + [p])
            state[p] = "done"
            out.append(p)

        for p in order:
            visit(p, [])
        return out

    def _check_order(self, order):
        order = list(order)
        units = set(self._units)
        unknown = [p for p in order if p not in units]
        if unknown:
            raise ValueError(
                f"order names {unknown}, which are not random fields of "
                f"{self.data_type.__name__}; the random fields are "
                f"{self.random_paths()}"
            )
        dupes = sorted({p for p in order if order.count(p) > 1})
        if dupes:
            raise ValueError(f"order lists {dupes} more than once")
        missing = [p for p in self.random_paths() if p not in order]
        if missing:
            raise ValueError(f"order is missing the random fields {missing}")
        pos = {p: i for i, p in enumerate(order)}
        for p in order:
            for d in self._deps[p]:
                if pos[d] >= pos[p]:
                    raise ValueError(
                        f"{self._where(p)} is conditioned on {d!r}, which does "
                        f"not precede it in the order {order}"
                    )
        return order

    def _check_resolved(self):
        if self._unresolved:
            path, dep = self._unresolved[0]
            raise ValueError(
                f"{self._where(path)}: conditioned on unknown path {dep!r}"
            )

    # ========================================================
    # Copies
    # ========================================================

    def with_order(self, order):
        """A copy drawing the random units in ``order`` (dotted paths)."""
        return StructuredDistribution(
            self.data_type, self._supplied, parameter_type=self.parameter_type,
            order=order, check_fixed=self.check_fixed, key=self.key,
        )

    def given(self, parameter_type):
        """
        A copy with condition type ``parameter_type`` (None, a type, a tuple
        or a mapping, as for any ``ConditionalDistribution``). Nested
        structured children without a condition type of their own get it too.
        """
        return StructuredDistribution(
            self.data_type, self._supplied, parameter_type=parameter_type,
            order=self._explicit_order, check_fixed=self.check_fixed,
            key=self.key,
        )

    # ========================================================
    # Types
    # ========================================================

    def _output_type(self):
        return self.data_type

    def _condition_type(self):
        if self.parameter_type is not None:
            return self.parameter_type
        return getattr(self, "_adopted", None)

    # ========================================================
    # Adapter-facing API
    # ========================================================

    @property
    def order(self) -> list[str]:
        """Dotted paths of the random units, in draw order."""
        return list(self._order)

    def random_paths(self) -> list[str]:
        """Dotted paths of the random units (leaves and blocks), in field order."""
        return sorted(self._units, key=lambda p: self._units[p].index)

    def fixed_paths(self) -> list[str]:
        """Dotted paths of every fixed leaf, nested fixed objects expanded."""
        return [p for p, v in self._fixed.items() if not isinstance(v, Parameters)]

    def fixed_values(self) -> dict:
        """``{path: value}`` of the fixed fields: the ``x`` a draw starts from.

        Holds every fixed leaf, and every fixed nested object at its path.
        """
        return dict(self._fixed)

    def field_at(self, path) -> Field:
        """The record of the random unit at ``path`` (a leaf or a ``Block``)."""
        return self._unit(path).record

    def unit_index(self, path) -> int:
        """The fold-in index of a unit: its position in the flat field order."""
        return self._unit(path).index

    def _unit(self, path):
        try:
            return self._units[path]
        except KeyError:
            raise KeyError(
                f"{path!r} is not a random field of {self.data_type.__name__}"
            ) from None

    def leaf_distribution(self, path, x, conditions=(), *, check=True):
        """
        The NumPyro distribution of the leaf at ``path``.

        ``x`` holds the values drawn so far, keyed by dotted path from the
        root (start from ``fixed_values()``); ``conditions`` is the tuple of
        condition values in schema order. Callable and conditional leaves are
        shape-checked here. ``check=False`` skips the shape check and the
        check that every ``on`` path resolves, for adapters that pass
        batched values or supply unresolved paths in ``x`` themselves.
        """
        if check:
            self._check_resolved()
        u = self._unit(path)
        if isinstance(u.record, Block):
            raise TypeError(f"{self._where(path)} is a block, not a leaf")
        conds = self._condition_schema().coerce(tuple(conditions))
        return self._leaf(u, x, conds, check=check)

    def dependencies(self, path):
        """
        ``(units, unresolved)`` for the unit at ``path``: the dotted paths of
        the random units its ``on`` paths resolve to (a block for a path
        inside it; fixed fields are left out), and the ``on`` paths that
        resolve to no field.
        """
        self._unit(path)
        return list(self._deps[path]), [d for p, d in self._unresolved if p == path]

    def update_values(self, x, path, value):
        """Add the value of the unit at ``path`` to ``x`` and return ``x``.

        A block's parameter-object value also adds its leaves at their
        dotted paths, as during ``sample_single``.
        """
        if isinstance(self._unit(path).record, Block):
            x.update(_flat_or_value(path, value))
        else:
            x[path] = value
        return x

    def block_conditions(self, path, conditions=()):
        """The conditions of the block at ``path``, in its schema order."""
        u = self._unit(path)
        if not isinstance(u.record, Block):
            raise TypeError(f"{self._where(path)} is a leaf, not a block")
        return _project(self._condition_schema(), u.schema, tuple(conditions))

    def leaf_supports(self):
        """
        ``{leaf path: NumPyro constraint}`` for every random leaf.

        NumPyro leaves give their ``support``. Callable and conditional
        leaves use the field's declared ``parameter(support=...)``, else the
        real line, with one event dimension per core axis. A block gives its
        own ``leaf_supports()`` or ``support`` (a constraint or a dict of
        them) if it has one, else the declared supports of its output type,
        else the support declared on the field. Returns None when a block's
        ``leaf_supports()`` returns None.
        """
        out = {}
        for path in self.random_paths():
            u = self._units[path]
            rec = u.record
            if isinstance(rec, NumPyroLeaf):
                out[path] = rec.distribution.support
            elif not isinstance(rec, Block):
                out[path] = _event_support(
                    u.spec.support or constraints.real, len(u.spec.index)
                )
            else:
                sub = self._block_supports(u)
                if sub is None:
                    return None
                out.update(sub)
        return out

    def _block_supports(self, u):
        d, path = u.record.distribution, u.path
        if callable(getattr(d, "leaf_supports", None)):
            sub = d.leaf_supports()
            return None if sub is None else {f"{path}.{p}": c for p, c in sub.items()}
        support = getattr(d, "support", None)
        if isinstance(support, Mapping):
            return {f"{path}.{p}": c for p, c in support.items()}
        if isinstance(support, constraints.Constraint):
            return {path: support}
        out = d._output_type()
        if is_parameter_type(out):
            return {
                f"{path}.{p}": _event_support(s.support or constraints.real, len(s.index))
                for p, s in out.flat_field_specs().items()
            }
        return {path: _event_support(u.spec.support or constraints.real, len(u.spec.index))}

    def assemble(self, x):
        """The ``data_type`` object with random units from ``x`` and fixed values."""
        return self._assemble(x, "")

    def _assemble(self, x, prefix):
        values = {}
        for name, rec in self.fields.items():
            path = prefix + name
            if isinstance(rec, Nested):
                values[name] = rec.distribution._assemble(x, path + ".")
            elif isinstance(rec, Fixed):
                values[name] = (
                    rec.value if isinstance(rec.value, Parameters)
                    else jnp.asarray(rec.value)
                )
            else:
                values[name] = x[path]
        return self.data_type(**values)

    # ========================================================
    # Single-object law
    # ========================================================

    def _conditions(self, args, kwargs):
        schema = self._condition_schema()
        if schema.kind == "mapping":
            return tuple(kwargs[n] for n in schema.names)
        if len(args) != schema.positional_arity:
            raise TypeError(
                f"expected {schema.positional_arity} condition(s), got {len(args)}"
            )
        return tuple(args)

    def _leaf(self, u, x, conds, check=True):
        rec = u.record
        if isinstance(rec, NumPyroLeaf):
            return rec.distribution
        vals = u.schema.coerce(_project(self._condition_schema(), u.schema, conds))
        try:
            if isinstance(rec, CallableLeaf):
                d = u.schema.call(rec.fn, vals)
            else:
                d = u.schema.call(partial(rec.fn, _view(x, u.chain)), vals)
        except TypeError as e:
            form = "fn(*conditions)" if isinstance(rec, CallableLeaf) else "fn(x, *conditions)"
            raise TypeError(
                f"{self._where(u.path)}: calling the field function failed ({e}). "
                f"This field is called as {form}, with the conditions set by .given."
            ) from e
        if check:
            _check_leaf(d, u.spec, self._where(u.path))
        return d

    def _block_conditions(self, u, conds):
        return _project(self._condition_schema(), u.schema, conds)

    def sample_single(self, *args, key, **kwargs):
        self._check_resolved()
        conds = self._conditions(args, kwargs)
        x = dict(self._fixed)
        for path in self._order:
            u = self._units[path]
            k = jax.random.fold_in(key, u.index)
            if isinstance(u.record, Block):
                value = _block_call(
                    u.record.distribution, self._block_conditions(u, conds),
                    "sample_single", key=k,
                )
            else:
                value = self._leaf(u, x, conds).sample(k)
            self.update_values(x, path, value)
        return self.assemble(x)

    def logpdf_single(self, value, *args, **kwargs):
        self._check_resolved()
        conds = self._conditions(args, kwargs)
        x = dict(self._fixed)
        total = 0.0
        for path in self._order:
            u = self._units[path]
            v = value[path]
            if isinstance(u.record, Block):
                total = total + _block_call(
                    u.record.distribution, self._block_conditions(u, conds),
                    "logpdf_single", v,
                )
            else:
                total = total + self._leaf(u, x, conds).log_prob(v)
            self.update_values(x, path, v)
        ok = self._fixed_ok(value, "")
        return total if ok is True else jnp.where(ok, total, -jnp.inf)

    def _fixed_ok(self, value, prefix):
        ok = True
        for name, rec in self.fields.items():
            path = prefix + name
            if isinstance(rec, Nested):
                sub = rec.distribution._fixed_ok(value, path + ".")
            elif isinstance(rec, Fixed) and self.check_fixed:
                if isinstance(rec.value, Parameters):
                    got = value[path].to_flat_dict()
                    checks = [_isclose(got[p], v)
                              for p, v in rec.value.to_flat_dict().items()]
                    sub = jnp.all(jnp.stack(checks)) if checks else True
                else:
                    sub = _isclose(value[path], rec.value)
            else:
                continue
            if sub is not True:
                ok = sub if ok is True else ok & sub
        return ok

    # ========================================================
    # Display
    # ========================================================

    def _display_model(self):
        cond_tex = self._display_tex()[0]
        return self._level_html(self, "", cond_tex)

    def _display_equations(self):
        return [line for line, _ in self._level_lines(self, "", self._display_tex()[0])
                if line is not None]

    def _symbol(self, name):
        spec = self.data_type.field_specs()[name]
        return spec.latex or latex_name(name)

    def _level_lines(self, root, prefix, cond_tex):
        """``(LaTeX line or None, section HTML or None)`` per field, in draw order."""
        pos = {p: i for i, p in enumerate(root.order)}
        flat = root.data_type.flat_field_specs()
        inf = len(pos)
        items = []
        for i, (name, rec) in enumerate(self.fields.items()):
            path = prefix + name
            if isinstance(rec, Nested):
                key = min((pos[p] for p in pos if p.startswith(path + ".")), default=inf)
            else:
                key = pos.get(path, inf)
            items.append((key, i, name, rec))

        out = []
        for _, _, name, rec in sorted(items, key=lambda t: t[:2]):
            path, sym = prefix + name, self._symbol(name)
            title = self.data_type.field_specs()[name].name or name
            if isinstance(rec, NumPyroLeaf):
                out.append((sym + r" \sim " + format_distribution(rec.distribution), None))
            elif isinstance(rec, CallableLeaf):
                given = r"\mid " + cond_tex if cond_tex else ""
                out.append((sym + rf" \sim p\!\left({sym}{given}\right)", None))
            elif isinstance(rec, ConditionalLeaf):
                deps = []
                for dep in rec.on:
                    u = root._units.get(path)
                    ref = next(
                        (pre + dep for pre, _ in u.chain if pre + dep in flat), None
                    ) if u else None
                    spec = flat.get(ref) if ref else None
                    deps.append(
                        (spec.latex if spec and spec.latex else None)
                        or latex_name(dep.rpartition(".")[2])
                    )
                deps += [cond_tex] if cond_tex else []
                out.append((
                    sym + rf" \sim p\!\left({sym}\mid " + ", ".join(deps) + r"\right)",
                    None,
                ))
            elif isinstance(rec, Fixed):
                if isinstance(rec.value, Parameters):
                    specs = rec.value.flat_field_specs()
                    lines = [
                        (specs[p].latex or latex_name(p)) + " = " + _fmt(v)
                        for p, v in rec.value.to_flat_dict().items()
                    ]
                    out.append((None, section(title + " (fixed)", math_block(lines))))
                else:
                    out.append((sym + " = " + _fmt(rec.value), None))
            elif isinstance(rec, Nested):
                body = rec.distribution._level_html(root, path + ".", cond_tex)
                out.append((None, section(title, body, opened=True)))
            else:
                d = rec.distribution
                line = sym + r" \sim " + d._display_math()
                out.append((None, section(title, math_block([line]) + d._repr_html_())))
        return out

    def _level_html(self, root, prefix, cond_tex):
        html, pending = [], []
        for line, sec in self._level_lines(root, prefix, cond_tex):
            if line is not None:
                pending.append(line)
                continue
            if pending:
                html.append(math_block(pending))
                pending = []
            html.append(sec)
        if pending:
            html.append(math_block(pending))
        return "".join(html)

    def _display_title(self):
        return f"{type(self).__name__}[{self.data_type.__name__}]"

    def _display_methods(self, cond_tex, out_tex):
        rows = super()._display_methods(cond_tex, out_tex)
        rows.append((".with_order", r"([\mathrm{paths}])", "Change the draw order"))
        rows.append((".given", r"(\mathrm{type})", "Set the condition type"))
        return rows

    def _display_config(self):
        config = {"order": ", ".join(self._order) or "(no random fields)"}
        if not self.check_fixed:
            config["check_fixed"] = False
        return config

    def as_generative(self, hyperparameter_type=None, external=(), *,
                      conditioned_on=None):
        """
        This distribution drawn by fixed-bank transport on the generative
        backend: a ``StructuredGenerativePopulation`` (a ``PopulationModel``
        conditioned on Lambda). See :mod:`astroinfer.distributions.transport`.
        """
        from .transport import as_generative

        return as_generative(self, hyperparameter_type, external,
                             conditioned_on=conditioned_on)

    def __repr__(self):
        inner = ", ".join(f"{n}={r!r}" for n, r in self.fields.items())
        return f"StructuredDistribution({self.data_type.__name__}, {inner})"


# ============================================================
# Formatting NumPyro distributions as LaTeX
# ============================================================

def _fmt(v):
    """A short LaTeX rendering of a number or a small array."""
    try:
        a = np.asarray(v)
    except Exception:  # noqa: BLE001 - tracers and other non-concrete values
        return r"\cdot"
    if a.dtype == object:
        return r"\cdot"
    if a.size == 1:  # NumPyro promotes scalar arguments to shape (1,)
        a = a.reshape(())
    if a.ndim == 0:
        if a.dtype.kind in "biu":
            return str(int(a))
        x = float(a)
        if np.isinf(x):
            return r"\infty" if x > 0 else r"-\infty"
        s = f"{x:.4g}"
        if "e" in s:
            m, e = s.split("e")
            s = rf"{m} \times 10^{{{int(e)}}}"
        return s
    if a.size <= 4:
        return "[" + ", ".join(_fmt(e) for e in a.reshape(-1)) + "]"
    return r"\mathrm{array}" + escape(str(a.shape))


_ARGS = {
    "Uniform": ("low", "high"),
    "Normal": ("loc", "scale"),
    "LogNormal": ("loc", "scale"),
    "Beta": ("concentration1", "concentration0"),
    "Exponential": ("rate",),
    "Gamma": ("concentration", "rate"),
    "HalfNormal": ("scale",),
}


def format_distribution(d) -> str:
    r"""LaTeX for a NumPyro distribution, e.g. ``\mathrm{Uniform}(5, 100)``.

    Knows the arguments of Uniform, Normal, TruncatedNormal, Beta,
    LogNormal, Exponential, Gamma and HalfNormal; other distributions show
    their class name. ``.to_event`` and ``.expand`` wrappers are looked
    through.
    """
    while isinstance(d, (dist.Independent, dist.ExpandedDistribution)):
        d = d.base_dist
    name = type(d).__name__
    if name in _ARGS:
        args = [_fmt(getattr(d, a)) for a in _ARGS[name]]
        return rf"\mathrm{{{name}}}(" + ", ".join(args) + ")"
    if "Truncated" in name and hasattr(d, "base_dist"):
        base = d.base_dist
        low = _fmt(getattr(d, "low", -np.inf))
        high = _fmt(getattr(d, "high", np.inf))
        if type(base).__name__ == "Normal":
            args = [_fmt(base.loc), _fmt(base.scale), low, high]
            return r"\mathrm{TruncatedNormal}(" + ", ".join(args) + ")"
        return format_distribution(base) + rf"\big|_{{[{low},\,{high}]}}"
    return rf"\mathrm{{{name}}}"


# ============================================================
# The factory on every parameter dataclass
# ============================================================

def _distribution(cls, **fields):
    """
    A ``StructuredDistribution`` over this parameter type.

    Each keyword is a field: a NumPyro distribution, a callable returning
    one from the conditions, ``conditional(fn, on=...)``, a fixed value, a
    nested ``Type.distribution(...)`` or another ``ConditionalDistribution``.
    Fields with dataclass defaults may be left out and are then fixed. Use
    ``.given(type)`` to set the condition type and ``.with_order([...])`` to
    set the draw order.
    """
    return StructuredDistribution(cls, fields)


Parameters.distribution = classmethod(_distribution)

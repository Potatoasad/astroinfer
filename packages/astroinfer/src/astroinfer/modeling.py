"""NumPyro sample sites for astroinfer distributions, and ``infer``.

Inside a NumPyro model, ``sample(name, distribution, ...)`` creates the
sample sites for one typed object and returns the object:

    NumPyro distribution            one ordinary site, ``name``
    StructuredDistribution          one site per random leaf, ``name.path``
                                    (``bbh.masses.m1``), in draw order; a
                                    block field is sampled as below under
                                    ``name.path``
    other ConditionalDistribution   one joint site, ``name``, holding one
                                    realization packed into a flat vector

``factor`` adds a log-density term. ``infer`` runs NUTS (or nested
sampling) on such a model and returns a ``PosteriorData`` whose theta has
one field per root: a nested field of the root's type for a typed object,
an array field for a plain NumPyro site.

Every site made by ``sample`` carries metadata in NumPyro's per-site
``infer`` slot, under ``site["infer"]["astroinfer"]``:

    root    the name given to ``sample``
    path    dotted path of the site within the root; None for the whole object
    kind    "numpyro" (plain site), "joint" (packed object) or "leaf"
            (one field of a structured object)
    type    the root's output type; None for a plain site
    layout  joint sites only: the ``JointLayout`` that packs and unpacks

The fixed fields of a latent structured object are recorded as
``numpyro.deterministic("name.path", value)``, so the object can be rebuilt
from the samples (``unpack``, ``infer``).

With ``marginalize=True``, ``sample`` creates no site; the value is integrated out of the
model in closed form at the observed site that uses it (see ``astroinfer.marginal``), and
``infer`` draws it back after sampling.

A latent joint site needs a support for every leaf. In order of
precedence, a leaf's support comes from a ``support`` attribute on the
distribution (a dict from dotted path to constraint, or one constraint for
every leaf), from ``distribution.leaf_supports()`` if it exists and returns
a dict, or from the ``parameter(support=...)`` declared on the output
type's field, where an undeclared support means the real line.
"""
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.flatten_util import ravel_pytree
from numpyro import handlers
from numpyro.distributions import Distribution, constraints

from . import marginal as _marginal
from .distributions.base import (
    BoundConditionalDistribution,
    ConditionalDistribution,
    broadcast_batch,
)
from .distributions.schema import is_parameter_type
from .distributions.structured import Block, StructuredDistribution
from .inference.runner import run_mcmc
from .inference.sampling import _type_from_arrays, _with_leading_axes
from .latent import _as_constraint
from .parameters import Parameters
from .posterior import PosteriorData, _is_sample_stat

META = "astroinfer"


def _meta(root, kind, type_=None, path=None, **extra):
    return {META: {"root": root, "path": path, "kind": kind, "type": type_, **extra}}


# ============================================================
# Packing one realization into a flat vector
# ============================================================

def _leaf_paths(tree):
    """Dotted path of every leaf, in ``jax.tree_util.tree_leaves`` order."""
    def walk(obj, prefix):
        if isinstance(obj, Parameters):
            for f in fields(obj):
                yield from walk(getattr(obj, f.name), prefix + (f.name,))
        elif isinstance(obj, Mapping):
            for k in sorted(obj):
                yield from walk(obj[k], prefix + (str(k),))
        elif isinstance(obj, (tuple, list)):
            for i, v in enumerate(obj):
                yield from walk(v, prefix + (str(i),))
        elif obj is not None:
            yield ".".join(prefix)

    paths = tuple(walk(tree, ()))
    if len(paths) != len(jax.tree_util.tree_leaves(tree)):
        raise TypeError(
            f"Cannot name the leaves of a {type(tree).__name__} output; use "
            "parameter dataclasses, tuples or dicts of them"
        )
    return paths


@dataclass(frozen=True, eq=False)
class JointLayout:
    """
    How one realization of a distribution's output maps to a flat vector.

    ``unravel`` (from ``jax.flatten_util.ravel_pytree`` on a template
    realization) turns one vector back into one object. ``pack`` and
    ``unpack`` do the same for arrays with any number of leading axes. The
    vector has the dtype ``ravel_pytree`` promotes the leaves to;
    ``unpack`` restores each leaf's dtype.
    """

    treedef: Any
    paths: tuple
    shapes: tuple
    dtypes: tuple
    dtype: Any
    unravel: Any

    def __repr__(self):
        leaves = ", ".join(f"{p or 'value'}{list(s)}" for p, s in zip(self.paths, self.shapes))
        return f"JointLayout({leaves}; size={self.size}, dtype={self.dtype})"

    @classmethod
    def from_template(cls, template):
        flat, unravel = ravel_pytree(template)
        leaves, treedef = jax.tree_util.tree_flatten(template)
        return cls(
            treedef,
            _leaf_paths(template),
            tuple(jnp.shape(x) for x in leaves),
            tuple(jnp.result_type(x) for x in leaves),
            flat.dtype,
            unravel,
        )

    @property
    def lengths(self):
        return tuple(prod(s) for s in self.shapes)

    @property
    def size(self):
        return sum(self.lengths)

    def pack(self, x):
        """Objects with leading axes ``lead`` -> array ``lead + (size,)``."""
        leaves, treedef = jax.tree_util.tree_flatten(x)
        if treedef != self.treedef:
            raise TypeError(f"Expected a value with structure {self.treedef}, got {treedef}")
        leads = []
        for path, leaf, shape in zip(self.paths, leaves, self.shapes):
            n = jnp.ndim(leaf) - len(shape)
            if n < 0 or jnp.shape(leaf)[n:] != shape:
                raise ValueError(
                    f"{path or 'value'}: expected trailing shape {shape}, got {jnp.shape(leaf)}"
                )
            leads.append(jnp.shape(leaf)[:n])
        lead = np.broadcast_shapes(*leads) if leads else ()
        return jnp.concatenate(
            [
                jnp.broadcast_to(leaf, lead + shape).reshape(lead + (-1,)).astype(self.dtype)
                for leaf, shape in zip(leaves, self.shapes)
            ],
            axis=-1,
        )

    def unpack(self, v):
        """Array ``lead + (size,)`` -> objects with leading axes ``lead``."""
        v = jnp.asarray(v)
        lead = v.shape[:-1]
        bounds = np.cumsum((0,) + self.lengths)
        leaves = [
            v[..., a:b].reshape(lead + shape).astype(dtype)
            for a, b, shape, dtype in zip(bounds[:-1], bounds[1:], self.shapes, self.dtypes)
        ]
        return jax.tree_util.tree_unflatten(self.treedef, leaves)


def _condition_arguments(distribution, conds):
    return distribution._condition_schema().arguments(conds)


def _layout(distribution, conds, context):
    """The layout of one realization, from the shapes ``sample`` would give."""
    c_args, c_kwargs = _condition_arguments(distribution, conds)
    nb = len(broadcast_batch(conds))
    out = jax.eval_shape(
        lambda k: distribution.sample(*c_args, key=k, **c_kwargs, **context),
        jax.random.PRNGKey(0),
    )
    template = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape[nb:], s.dtype), out)
    return JointLayout.from_template(template)


def _layout_of(distribution, obs):
    """The layout of one realization, read off observed data; None if they are untyped.

    An observed site needs no ``sample_single``: the data already have the shape of a
    realization.
    """
    schema = distribution._output_schema()
    values = _output_values(schema, obs)
    if not values or not all(isinstance(v, Parameters) for v in values):
        return None
    elems = []
    for v in values:
        b = v.batch_shape
        v = v.broadcast()
        elems.append(v[(0,) * len(b)] if b else v)
    if schema.kind == "single":
        template = elems[0]
    elif schema.kind == "mapping":
        template = dict(zip(schema.names, elems))
    else:
        template = tuple(elems)
    return JointLayout.from_template(jax.tree_util.tree_map(jnp.zeros_like, template))


def _output_values(schema, x):
    """The output ``x`` as a tuple in schema order."""
    if schema.kind == "single":
        return (x,)
    if schema.kind == "mapping":
        return tuple(x[n] for n in schema.names)
    return tuple(x)


def _coerce_output(distribution, x):
    """Dicts and DataFrames turned into the declared output types."""
    schema = distribution._output_schema()
    values = schema.coerce(_output_values(schema, x))
    if schema.kind == "single":
        return values[0]
    if schema.kind == "mapping":
        return dict(zip(schema.names, values))
    return values


def _align(conds, n):
    """Pad batched conditions with leading length-1 axes up to ``n`` axes.

    NumPyro lines batch axes up from the right; astroinfer's rule lines a
    shorter condition up with the leading data axes, so the conditions are
    given every axis explicitly.
    """
    out = []
    for c in conds:
        if isinstance(c, Parameters) and len(c.batch_shape) < n:
            b = c.batch_shape
            c = c.broadcast(b).reshape_batch((1,) * (n - len(b)) + b)
        out.append(c)
    return tuple(out)


# ============================================================
# Supports of a latent joint site
# ============================================================

def _lookup(distribution, attr):
    """``distribution.<attr>``, falling back to the parents of bound ones."""
    d = distribution
    while True:
        value = getattr(d, attr, None)
        if value is not None or not isinstance(d, BoundConditionalDistribution):
            return value
        d = d.parent


def _declared_supports(distribution):
    """``{path: support or None}`` from the output types' parameter specs."""
    schema = distribution._output_schema()
    prefixes = {
        "single": [""],
        "tuple": [f"{i}." for i in range(len(schema.types))],
        "mapping": [f"{n}." for n in schema.names],
    }.get(schema.kind, [])
    out = {}
    for prefix, t in zip(prefixes, schema.types):
        if is_parameter_type(t):
            out.update({prefix + p: s.support for p, s in t.flat_field_specs().items()})
    return out


def _joint_support(distribution, layout, name):
    """``constraints.independent(cat(...), 1)`` over the leaves of ``layout``."""
    given = {}
    leaf = _lookup(distribution, "leaf_supports")
    if callable(leaf) and isinstance(found := leaf(), Mapping):
        given.update(found)
    support = _lookup(distribution, "support")
    if isinstance(support, Mapping):
        given.update(support)
    elif support is not None:
        given.update(dict.fromkeys(layout.paths, support))
    unknown = set(given) - set(layout.paths)
    if unknown:
        raise KeyError(f"{name}: supports given for unknown leaves {sorted(unknown)}")

    declared = _declared_supports(distribution)
    cseq = []
    for path, dtype in zip(layout.paths, layout.dtypes):
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError(
                f"{name}: leaf {path or 'value'!r} has dtype {dtype}; a latent "
                "site needs continuous leaves (pass obs= for discrete data)"
            )
        if path in given:
            c = _as_constraint(given[path], path)
        elif path in declared:
            c = _as_constraint(declared[path], path)
        else:
            raise TypeError(
                f"{name}: no support for leaf {path or 'value'!r}. A latent "
                "site needs a parameter dataclass output with declared "
                "supports, or a `support` attribute on the distribution."
            )
        if c.event_dim != 0 or c.is_discrete:
            raise TypeError(
                f"{name}: leaf {path or 'value'!r} has support {c}; only "
                "continuous elementwise supports are allowed"
            )
        cseq.append(c)
    return constraints.independent(
        constraints.cat(cseq, dim=-1, lengths=list(layout.lengths)), 1
    )


# ============================================================
# The NumPyro wrapper distribution
# ============================================================

class _JointSite(Distribution):
    """
    One realization of a ``ConditionalDistribution``, packed into a vector.

    The event shape is ``(layout.size,)`` and the batch shape is that of
    the bound conditions, so the site broadcasts inside ``numpyro.plate``.
    ``log_prob`` unpacks the value and calls the distribution's ``logpdf``;
    ``sample`` calls its ``sample`` and packs the draws, so
    ``numpyro.infer.Predictive`` gives draws from the distribution itself.
    """

    def __init__(self, distribution, conds, context, layout, support=None):
        self.distribution = distribution
        self.conds = tuple(conds)
        self.context = dict(context)
        self.layout = layout
        super().__init__(
            batch_shape=tuple(broadcast_batch(self.conds)),
            event_shape=(layout.size,),
        )
        self._support = constraints.real_vector if support is None else support

    def sample(self, key, sample_shape=()):
        sample_shape = tuple(sample_shape)
        c_args, c_kwargs = _condition_arguments(self.distribution, self.conds)
        x = self.distribution.sample(
            *c_args, key=key, N=sample_shape or None, **c_kwargs, **self.context
        )
        # astroinfer puts the sample axes after the batch axes; NumPyro before.
        nb, ns = len(self.batch_shape), len(sample_shape)
        order = tuple(range(nb, nb + ns)) + tuple(range(nb))

        def move(leaf):
            return jnp.transpose(leaf, order + tuple(range(nb + ns, jnp.ndim(leaf))))

        return self.layout.pack(jax.tree_util.tree_map(move, x))

    def log_prob(self, value):
        x = self.layout.unpack(value)
        out = self.distribution._output_schema()
        d_args, d_kwargs = out.arguments(_output_values(out, x))
        conds = _align(self.conds, jnp.ndim(value) - 1)
        c_args, c_kwargs = _condition_arguments(self.distribution, conds)
        return self.distribution.logpdf(
            *d_args, *c_args, **d_kwargs, **c_kwargs, **self.context
        )


# ============================================================
# Leaves of a structured distribution over batch axes
# ============================================================

class _BatchedLeaf(Distribution):
    """
    A leaf distribution built separately for every element of ``lead``.

    ``make(inputs)`` returns the NumPyro distribution of one element
    (``StructuredDistribution.leaf_distribution`` is written for one
    object); ``inputs`` is a pytree whose parts with axis 0 in ``in_axes``
    carry the batch axes ``lead``. ``sample``, ``log_prob`` and ``support``
    vmap over those axes, so the leaf functions need not broadcast.
    """

    def __init__(self, make, inputs, in_axes, lead):
        self.make, self.inputs, self.in_axes = make, inputs, in_axes
        lead = tuple(lead)
        self._lead = lead
        shape = jax.eval_shape(
            lambda: self._map(lambda d: jnp.zeros(d.event_shape))
        ).shape
        event = shape[len(lead):]
        super().__init__(batch_shape=lead, event_shape=event)

        def pad(a):
            # Per-element parameters broadcast against the event from the right.
            s = jnp.shape(a)[len(lead):]
            if jnp.ndim(a) < len(lead) or len(s) >= len(event):
                return a
            return jnp.reshape(a, lead + (1,) * (len(event) - len(s)) + s)

        self._support = jax.tree_util.tree_map(pad, self._map(lambda d: d.support))

    def _map(self, fn, *args, n_extra=0):
        """``fn(element distribution, *args)`` over ``lead``, then ``n_extra`` outer axes of ``args``."""
        f = lambda inputs, *a: fn(self.make(inputs), *a)  # noqa: E731
        for _ in self._lead:
            f = jax.vmap(f, in_axes=(self.in_axes,) + (0,) * len(args))
        for _ in range(n_extra):
            f = jax.vmap(f, in_axes=(None,) + (0,) * len(args))
        return f(self.inputs, *args)

    def sample(self, key, sample_shape=()):
        shape = tuple(sample_shape) + self.batch_shape
        keys = jax.random.split(key, prod(shape)).reshape(shape + jnp.shape(key))
        return self._map(lambda d, k: d.sample(k), keys, n_extra=len(sample_shape))

    def log_prob(self, value):
        full = np.broadcast_shapes(jnp.shape(value), self.shape())
        value = jnp.broadcast_to(value, full)
        n_extra = len(full) - len(self.shape())
        return self._map(lambda d, v: d.log_prob(v), value, n_extra=n_extra)


def _lead_of(value, core):
    """Batch axes of ``value`` given the core ndim of each leaf."""
    if isinstance(value, Parameters):
        return value.batch_shape
    leaves = jax.tree_util.tree_leaves(value)
    core = core if isinstance(core, tuple) else (core,) * len(leaves)
    shapes = [jnp.shape(x)[:jnp.ndim(x) - c] for x, c in zip(leaves, core)]
    return np.broadcast_shapes(*shapes) if shapes else ()


def _broadcast_to_lead(value, core, lead):
    if isinstance(value, Parameters):
        return value.broadcast(lead)
    leaves, treedef = jax.tree_util.tree_flatten(value)
    core = core if isinstance(core, tuple) else (core,) * len(leaves)
    return jax.tree_util.tree_unflatten(treedef, [
        jnp.broadcast_to(x, lead + jnp.shape(x)[jnp.ndim(x) - c:])
        for x, c in zip(leaves, core)
    ])


def _structured_leaf(d, path, x, conds, lead, cores):
    """The NumPyro distribution of a leaf, batched over ``lead`` if needed.

    ``cores`` maps every drawn entry of ``x`` to the core ndim of its
    leaves (None for a parameter object); entries not in it are fixed.
    """
    if d.field_at(path).kind == "numpyro" or not lead:
        return d.leaf_distribution(path, x, conds)
    xb = {k: _broadcast_to_lead(v, cores[k], lead) if k in cores else v for k, v in x.items()}
    x_axes = {k: 0 if k in cores else None for k in x}
    cb = tuple(c.broadcast(lead) if isinstance(c, Parameters) else c for c in conds)
    c_axes = tuple(0 if isinstance(c, Parameters) else None for c in conds)
    return _BatchedLeaf(
        lambda inputs: d.leaf_distribution(path, *inputs), (xb, cb), (x_axes, c_axes), lead
    )


# ============================================================
# sample and factor
# ============================================================

def sample(name, distribution, *args, obs=None, marginalize=False, **conditions):
    """
    Create the NumPyro sample site(s) for ``distribution`` and return the value.

    distribution:
        A NumPyro distribution: one ordinary site, returning its value.
        A ``ConditionalDistribution``: one joint site holding one
        realization; returns the typed object (``obs`` when observed).

    args, conditions:
        The conditions, following the distribution's condition schema
        (positionally for a single type or a tuple, as keywords for a
        mapping). Other keywords are passed to the distribution as context.

    obs:
        Observed value. The site is then an observed site, whose density
        is the distribution's ``logpdf`` at ``obs``.

    marginalize:
        Integrate the value out of the model in closed form instead of
        sampling it. No site is created; the returned value is a stand-in
        filled with NaN, to be passed to the observed site whose likelihood
        is conjugate to this prior (see ``astroinfer.marginal``). ``infer``
        draws the value after sampling.

    A latent joint site needs a support for every leaf (see the module
    docstring); leaves must be continuous. The sites carry metadata under
    ``site["infer"]["astroinfer"]``.
    """
    if marginalize:
        if obs is not None:
            raise TypeError(f"{name}: an observed site cannot be marginalized")
        return _marginal.begin(name, distribution, args, conditions)
    if isinstance(distribution, Distribution):
        if args or conditions:
            raise TypeError("A NumPyro distribution takes no conditions")
        _marginal.check_unused(name, distribution)
        return numpyro.sample(
            name, distribution, obs=obs, infer=_meta(name, "numpyro")
        )
    if not isinstance(distribution, ConditionalDistribution):
        raise TypeError(
            "Expected a NumPyro distribution or a ConditionalDistribution, got "
            f"{type(distribution).__name__}"
        )
    values, context, rest = distribution._condition_schema().bind(args, conditions)
    if rest:
        raise TypeError(
            f"{name}: got {len(rest)} extra positional argument(s); the key "
            "comes from NumPyro"
        )
    if _marginal.stand_ins((values, context)):
        return _marginal.consume(name, distribution, values, context, obs)
    conds = tuple(distribution._coerce_conditions(values))
    return _dispatch(name, distribution, conds, context, obs, name, None, None)[0]


def _dispatch(site, distribution, conds, context, obs, root, path, root_type):
    """``(value, core ndims of its leaves)`` of a distribution's site(s).

    ``root``, ``path`` and ``root_type`` place the sites within a root;
    ``root_type`` None means this distribution is the root.
    """
    if isinstance(distribution, StructuredDistribution):
        if context:
            raise TypeError(
                f"{site}: unexpected keyword(s) {sorted(context)} for "
                f"{type(distribution).__name__}"
            )
        value = _sample_structured(
            distribution, conds, obs, root, "" if path is None else path + ".",
            root_type or distribution.data_type,
        )
        return value, None
    return _sample_joint(site, distribution, conds, context, obs, root, path, root_type)


def _sample_structured(d, conds, obs, root, prefix, root_type):
    """One site per random unit of ``d``, named ``root.prefix<path>``."""
    if obs is not None:
        obs = _coerce_output(d, obs)
    x, cores = d.fixed_values(), {}
    lead = tuple(broadcast_batch(conds))
    for path in d.order:
        name, full = f"{root}.{prefix}{path}", prefix + path
        given = None if obs is None else obs[path]
        record = d.field_at(path)
        if isinstance(record, Block):
            value, core = _dispatch(
                name, record.distribution, d.block_conditions(path, conds), {},
                given, root, full, root_type,
            )
            if isinstance(value, Parameters):
                for p, spec in type(value).flat_field_specs().items():
                    cores[f"{path}.{p}"] = len(spec.index)
        else:
            fn = _structured_leaf(d, path, x, conds, lead, cores)
            value = numpyro.sample(
                name, fn, obs=given, infer=_meta(root, "leaf", root_type, full)
            )
            core = len(fn.event_shape)
        cores[path] = core
        lead = tuple(np.broadcast_shapes(lead, _lead_of(value, core)))
        x = d.update_values(x, path, value)
    if obs is not None:
        return obs
    for path in d.fixed_paths():
        # Give fixed values the batch (and plate) axes of the random ones, so
        # the object can be rebuilt from the sites.
        v = jnp.asarray(x[path])
        numpyro.deterministic(
            f"{root}.{prefix}{path}", jnp.broadcast_to(v, lead + v.shape)
        )
    return d.assemble(x)


def _sample_joint(name, distribution, conds, context, obs, root, path, root_type):
    if distribution._output_schema().kind == "none":
        raise TypeError(
            f"{name}: {type(distribution).__name__} declares no output type "
            "(data_type); use typing.Any for an untyped output"
        )
    if obs is not None:
        obs = _coerce_output(distribution, obs)
    layout = (obs is not None and _layout_of(distribution, obs)) or _layout(
        distribution, conds, context
    )
    meta = _meta(
        root, "joint", root_type or distribution._output_type(), path, layout=layout
    )
    core = tuple(len(s) for s in layout.shapes)
    if obs is not None:
        site = _JointSite(distribution, conds, context, layout)
        numpyro.sample(name, site, obs=layout.pack(obs), infer=meta)
        return obs, core
    support = _joint_support(distribution, layout, name)
    site = _JointSite(distribution, conds, context, layout, support)
    return layout.unpack(numpyro.sample(name, site, infer=meta)), core


def factor(name, value):
    """
    Add ``value`` to the model's log density, through ``numpyro.factor``.

    Inside a plate, NumPyro sums a batched value over the plate.
    """
    return numpyro.factor(name, value)


# ============================================================
# Reading roots back from samples
# ============================================================

@dataclass
class _Root:
    """One root of a model: a typed object or a plain NumPyro site."""

    name: str
    kind: str        # "numpyro", "joint" (one packed site) or "parts"
    type: Any        # output type; None for plain sites
    layout: Any      # joint roots only
    axes: tuple      # names of the per-draw batch axes of the object
    sites: tuple     # names of the sites read
    parts: dict      # "parts" roots: {path: (site name, layout or None)}
    marginalized: bool = False   # integrated out; drawn after sampling


def _batch_axis_names(name, site, batch_shape):
    n = len(batch_shape)
    by_dim = {f.dim: f.name for f in site.get("cond_indep_stack", ()) if f.dim is not None}
    return tuple(by_dim.get(i - n, f"{name}_dim_{i}") for i in range(n))


def _parts_value(root, type_, parts, sites):
    """A structured root from its leaf and block sites and fixed-field sites."""
    flat = {}
    for path, (site, layout) in parts.items():
        if layout is None:
            flat[path] = jnp.asarray(sites[site])
            continue
        value = layout.unpack(sites[site])
        flat[path] = value
        if isinstance(value, Parameters):
            flat.update({f"{path}.{p}": v for p, v in value.to_flat_dict().items()})
    for p in type_.flat_field_specs():
        if p not in flat and f"{root}.{p}" in sites:
            flat[p] = jnp.asarray(sites[f"{root}.{p}"])
    return type_.from_dict(flat).broadcast()


def _part_sites(root, type_, parts):
    covered = set(parts)
    for path, (_, layout) in parts.items():
        if layout is not None:
            covered.update(f"{path}.{p}" for p in layout.paths)
    fixed = [
        f"{root}.{p}" for p in type_.flat_field_specs()
        if p not in covered and not any(p.startswith(c + ".") for c in covered)
    ]
    return tuple(site for site, _ in parts.values()) + tuple(fixed)


def _trace(model, args, kwargs):
    # Drawing mode, so that marginalized values appear as roots of the trace.
    with _marginal.drawing():
        return handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(*args, **kwargs)


def _roots(trace):
    """``(roots, deterministic site names)`` of a model trace, in trace order."""
    roots, deterministic, first = {}, [], {}
    for name, site in trace.items():
        if site["type"] == "deterministic":
            meta = (site.get("infer") or {}).get(META)
            if meta is not None and meta.get("kind") == "marginalized":
                roots[name] = _Root(name, meta["root_kind"], meta["type"], meta["layout"],
                                    tuple(meta["axes"]), (name,), {}, marginalized=True)
                continue
            deterministic.append(name)
            continue
        if site["type"] != "sample" or site["is_observed"]:
            continue
        meta = (site.get("infer") or {}).get(META)
        if meta is None or meta["kind"] == "numpyro":
            shape = jnp.shape(site["value"])
            batch = tuple(site["fn"].batch_shape)
            axes = _batch_axis_names(name, site, batch)
            axes += tuple(f"{name}_dim_{i}" for i in range(len(batch), len(shape)))
            roots[name] = _Root(name, "numpyro", None, None, axes, (name,), {})
            continue
        root = meta["root"]
        if meta["path"] is None:
            batch = tuple(site["fn"].batch_shape)
            axes = _batch_axis_names(root, site, batch)
            roots[root] = _Root(root, "joint", meta["type"], meta["layout"], axes, (name,), {})
            continue
        if root not in roots:
            roots[root] = _Root(root, "parts", meta["type"], None, (), (), {})
            first[root] = site
        layout = meta.get("layout") if meta["kind"] == "joint" else None
        roots[root].parts[meta["path"]] = (name, layout)

    values = {k: s["value"] for k, s in trace.items() if s["type"] != "plate"}
    for root, site in first.items():
        r = roots[root]
        r.sites = _part_sites(root, r.type, r.parts)
        batch = _parts_value(root, r.type, r.parts, values).batch_shape
        r.axes = _batch_axis_names(root, site, batch)
    return roots, deterministic


def _root_value(root, sites):
    if root.kind == "numpyro":
        return jnp.asarray(sites[root.name])
    if root.kind == "joint":
        return root.layout.unpack(sites[root.sites[0]])
    return _parts_value(root.name, root.type, root.parts, sites)


def unpack(model, samples, *args, **kwargs):
    """
    ``{root: value}`` from a dict of site samples of ``model``.

    ``samples`` is what ``numpyro.infer.Predictive`` or ``MCMC.get_samples``
    return. Typed roots come back as their objects, with the sample axes
    first; plain sites as arrays. ``args`` and ``kwargs`` are the model's
    arguments, used to trace it once.
    """
    roots, _ = _roots(_trace(model, args, kwargs))
    return {
        name: _root_value(r, samples) for name, r in roots.items()
        if all(s in samples for s in r.sites)
    }


# ============================================================
# infer
# ============================================================

_RUN_SETTINGS = frozenset(inspect.signature(run_mcmc).parameters) - {
    "model", "args", "method", "seed", "model_kwargs",
}


def _theta_field(name, out_type, value, axes, n_sample):
    """``(field value, core dims)`` of one root in the posterior's theta."""
    if is_parameter_type(out_type):
        cls = _with_leading_axes(out_type, axes)
        return (value if cls is out_type else cls.from_dict(value.to_flat_dict())), None
    if isinstance(out_type, (tuple, Mapping)) and out_type:
        names = (
            [str(k) for k in out_type] if isinstance(out_type, Mapping)
            else [f"_{i}" for i in range(len(out_type))]
        )
        types = list(out_type.values()) if isinstance(out_type, Mapping) else list(out_type)
        parts = list(value.values()) if isinstance(value, Mapping) else list(value)
        values, dims = {}, {}
        for part, t, v in zip(names, types, parts):
            values[part], d = _theta_field(part, t, v, axes, n_sample)
            if d is not None:
                dims[part] = d
        cls = _type_from_arrays(name.title().replace("_", ""), values, n_sample, dims=dims)
        return cls(**values), None
    extra = np.ndim(value) - n_sample - len(axes)
    return value, tuple(axes) + tuple(f"{name}_dim_{i}" for i in range(len(axes), len(axes) + extra))


def _draw_marginalized(model, args, kwargs, result, names):
    """``result.sites`` with a draw of every marginalized value per posterior draw."""
    from numpyro.infer import Predictive

    n = len(result.sample_dims)
    lead = next(iter(result.sites.values())).shape[:n]
    key = jax.random.PRNGKey((result.seed or 0) + 1)
    with _marginal.drawing():
        drawn = Predictive(model, posterior_samples=result.get_samples(),
                           return_sites=list(names))(key, *args, **kwargs)
    sites = dict(result.sites)
    for name in names:
        sites[name] = jnp.reshape(drawn[name], lead + jnp.shape(drawn[name])[1:])
    return sites


def infer(model, *args, method="nuts", seed=None, **kwargs):
    """
    Sample the posterior of a NumPyro model written with ``sample``.

    Keywords that are ``run_mcmc`` settings (``num_warmup``,
    ``num_samples``, ``num_chains``, ``target_accept_prob``,
    ``max_tree_depth``, ``summary``, ``sampler_kwargs``, ``mcmc_kwargs``)
    go to the runner; all other arguments go to the model.

    Returns a ``PosteriorData``. Its theta has a type built at run time
    with one field per latent root, in model order: a nested field of the
    root's type for a typed object, an array field for a plain NumPyro
    site. Axes a root has beyond the sample axes (plates, batched
    conditions) become leading core axes, named after the plate.
    Deterministic sites go to ``extras`` (or ``sample_stats`` for
    sampler statistics such as ``log_likelihood``), the NUTS extra fields
    to ``sample_stats``; observed sites are left out. The ``RunResult`` is
    kept as ``posterior.metadata["run"]``.

    A value sampled with ``marginalize=True`` is not seen by the sampler.
    After the run it is drawn from its distribution given the data, once
    per posterior draw, and is a field of theta like any other root.
    """
    settings = {k: kwargs.pop(k) for k in list(kwargs) if k in _RUN_SETTINGS}
    roots, deterministic = _roots(_trace(model, args, kwargs))
    if not roots:
        raise ValueError("The model has no latent sample sites")
    for name in roots:
        if not name.isidentifier():
            raise ValueError(
                f"Site name {name!r} is not an identifier, so it cannot be a "
                "field of theta"
            )

    result = run_mcmc(model, *args, method=method, seed=seed, **settings, **kwargs)
    n = len(result.sample_dims)
    marginalized = [name for name, root in roots.items() if root.marginalized]
    sites = _draw_marginalized(model, args, kwargs, result, marginalized) if marginalized \
        else result.sites
    values, dims = {}, {}
    for name, root in roots.items():
        value = _root_value(root, sites)
        values[name], d = _theta_field(name, root.type, value, root.axes, n)
        if d is not None:
            dims[name] = d
    cls = _type_from_arrays("Theta", values, n, latex=r"\theta", dims=dims)
    theta = cls(**values)

    used = {s for r in roots.values() for s in r.sites}
    stats, extras = dict(result.sample_stats), {}
    for name in deterministic:
        if name not in used and name in result.sites:
            (stats if _is_sample_stat(name) else extras)[name] = result.sites[name]

    return PosteriorData(
        theta=theta,
        sample_stats=stats,
        extras=extras,
        metadata={
            "inference": "infer",
            "sampler": "numpyro",
            "method": result.method,
            "seed": result.seed,
            "num_samples": int(prod(theta.batch_shape)),
            "marginalized": tuple(marginalized),
            "run": result,
        },
        sample_dims=result.sample_dims,
    )

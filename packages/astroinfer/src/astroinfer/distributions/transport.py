"""Fixed-bank transport: structured distributions on the generative backend.

``prior.as_generative(hyperparameter_type=H)`` converts a
``StructuredDistribution`` (or a ``MixtureDistribution`` of them) into a
``StructuredGenerativePopulation``: a ``PopulationModel`` whose draws are a
deterministic, differentiable function of a fixed bank of uniform numbers
and of Lambda. Reusing the bank at every Lambda gives common random numbers,
which is what ``PopulationLikelihood`` differentiates through.

The conversion maps the tree onto ``astroinfer.generative``:

    NumPyro leaf                  NumPyroLeaf with a constant distribution
    callable leaf fn(Lambda)      NumPyroLeaf calling fn with the typed Lambda
    conditional(fn, on)           NumPyroLeaf with deps = the resolved on paths
    nested structured child       flattened; its leaves keep their dotted paths
    mixture block                 WeightedMixture of one Product per component
    fixed field                   not a variable; inserted into every draw

Every random leaf becomes one scalar generative variable named by its dotted
path, and the structured draw order becomes the ``GenerativeModel`` order,
whose validation (each variable after the variables it reads) therefore
applies. A leaf is drawn with NumPyro's closed-form ``icdf`` when it is
differentiable (see ``generative.leaves.has_differentiable_icdf``), and
otherwise by bisection on its CDF, which needs finite bounds; the bounds of
a callable or conditional leaf come from its support when that is a fixed
interval (Beta, for example), intersected with the field's declared
``parameter(support=...)``. The declared support should contain the
distribution's: bisection inside a narrower one truncates the draws without
renormalizing them. JAX's ``betainc`` has no gradient with respect to the
Beta shape parameters, so draws of a Beta leaf whose shape parameters depend
on Lambda are not differentiable in Lambda (as for ``generative.Beta``).

Mixture blocks. A field holding a ``MixtureDistribution`` whose components
are structured distributions over the same random fields (and the same fixed
values) becomes a ``WeightedMixture``: no component label is drawn, the
responsibilities are recomputed from the drawn fields, and the draws are
smooth in the weights. A top-level mixture of structured distributions is
converted the same way. The weights may be

    an array or logits array      constant
    a callable of the conditions  called as the mixture calls it, with the
                                  typed Lambda
    a callable naming a field     a condition whose name is a top-level field
                                  of the root type (``masses`` in
                                  ``weights=lambda masses, hyper: ...``) is
                                  filled with the value drawn for that field,
                                  which must come earlier in the order

The callable sees one object at a time, as in the native mixture (it is
vmapped over the draws when it reads a field). The field-naming rule holds
for every condition of the converted distribution, not only for mixture
weights. It is specific to this conversion: the native ``sample`` and
``logpdf`` of the structured distribution treat such a name as an ordinary
condition supplied by the caller.

Conditions. The converted population takes one condition, Lambda. The
structured distribution may have no conditions, one (a parameter type, or an
untyped dict), or a mapping whose names are fields (as above) plus at most
one other name, which receives Lambda. Lambda reaches the leaves as
``hyperparameter_type.from_dict(L)`` when a type is known, else as the dict.

Not converted, and reported together in one error: fields with core axes,
discrete leaves, leaves without a CDF, unbounded leaves without a
differentiable ``icdf``, blocks other than such mixtures, mixtures whose
components differ in their fields, and ``on`` paths that resolve to no field
and are not declared in ``external``.
"""
from typing import NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from .. import generative as gen
from ..generative.leaves import has_differentiable_icdf, support_bounds
from ..parameters import Parameters, coerce
from ..population import _SEED_EPS, PopulationModel, uniform_seeds
from .algebra import MixtureDistribution, _component_conditions
from .base import _vmapped, sample_shape
from .schema import is_parameter_type
from .structured import Block, NumPyroLeaf, StructuredDistribution

# ============================================================
# The transport protocol
# ============================================================

class BaseNoiseSpec(NamedTuple):
    """The base noise of a fixed-bank transport: one array per variable.

    ``kind`` is ``"uniform"``: each variable in ``variables`` gets an array
    of uniform numbers in (``low``, ``high``) with the sample shape.
    """

    kind: str
    variables: tuple
    low: float
    high: float


@runtime_checkable
class Reparameterizable(Protocol):
    """A distribution whose draws are a deterministic map of base noise.

    ``base_noise_spec()`` describes the noise and
    ``sample_from_base(noise, *conditions)`` maps it to draws. For a fixed
    noise bank the draws are differentiable in the conditions, so a
    population with this capability needs no proposal in
    ``PopulationLikelihood``.
    """

    def base_noise_spec(self): ...

    def sample_from_base(self, noise, *conditions): ...


# ============================================================
# Conversion
# ============================================================

class _AnyKey(dict):
    """A probe mapping that returns the same value for every key."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __missing__(self, key):
        return self.value


def _check_distribution(d, where):
    """The reasons a scalar NumPyro distribution cannot be a generative leaf."""
    if tuple(d.batch_shape) or tuple(d.event_shape):
        return [(f"{where}: the distribution must be scalar, got batch shape "
                 f"{tuple(d.batch_shape)} and event shape {tuple(d.event_shape)}")]
    if d.support.is_discrete:
        return [f"{where}: {type(d).__name__} is discrete"]
    if has_differentiable_icdf(d):
        return []
    bounds = support_bounds(d.support)
    if bounds is None or not np.all(np.isfinite(bounds)):
        return [f"{where}: no differentiable inverse CDF and unbounded support"]
    try:
        d.cdf(d.support.feasible_like(jnp.zeros(())))
    except NotImplementedError:
        return [f"{where}: {type(d).__name__} has no CDF to invert"]
    return []


def _batch_of(v):
    return v.batch_shape if isinstance(v, Parameters) else jnp.shape(v)


def _broadcast(v, shape):
    return v.broadcast(shape) if isinstance(v, Parameters) else jnp.broadcast_to(v, shape)


class _Conversion:
    """Walks a structured distribution and builds the generative model."""

    def __init__(self, distribution, hyperparameter_type, external):
        self.errors = []
        self.external = list(external)
        self.hyperparameter_type = hyperparameter_type
        self.blocks = {}           # block path -> (component 0, its variables)
        mix = distribution if isinstance(distribution, MixtureDistribution) else None
        if mix is not None:
            comps = mix.components
            if not all(isinstance(c, StructuredDistribution) for c in comps):
                raise TypeError(
                    "as_generative converts a mixture only when its components "
                    "are structured distributions"
                )
            self.root = comps[0]
        elif isinstance(distribution, StructuredDistribution):
            self.root = distribution
        else:
            raise TypeError(
                "as_generative needs a StructuredDistribution or a mixture of "
                f"them, got {type(distribution).__name__}"
            )
        self.data_type = self.root.data_type

        schema = distribution._condition_schema()
        self.field_names = [] if mix is not None else [
            n for n in schema.names if n in self.data_type.field_specs()
        ]
        if mix is not None and any(n in self.data_type.field_specs() for n in schema.names):
            self.errors.append(
                "(weights): a top-level mixture cannot depend on the fields it mixes"
            )
        self._plan_conditions(schema)

        if mix is not None:
            self.variables = list(self.root.order)
            self.order = list(self.root.order)
            model = self._mixture(mix, "", lambda x, L: self.conditions(x, L))
            self.top = [model] if model is not None else []
        else:
            self.variables = self._variables(self.root)
            self.order, self.top = [], []
            for path in self.root.order:
                rec = self.root.field_at(path)
                if isinstance(rec, Block):
                    model = self._block(path, rec.distribution)
                    self.order += self.blocks.get(path, (None, [path]))[1]
                else:
                    model = self._leaf(self.root, path, "", self.conditions,
                                       self._schema_names(self.root, path))
                    self.order.append(path)
                if model is not None:
                    self.top.append(model)
        if self.errors:
            raise ValueError(
                f"Cannot convert the distribution over {self.data_type.__name__} "
                "to the generative backend:\n  " + "\n  ".join(self.errors)
            )
        root = self.top[0] if mix is not None else gen.Product(self.top)
        self.model = gen.GenerativeModel(root, self.order, external=self.external)

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    def _lambda_type(self, t):
        H = self.hyperparameter_type
        if is_parameter_type(t):
            if H is not None and H is not t:
                self.errors.append(
                    f"(conditions): the distribution is conditioned on "
                    f"{t.__name__}, not on hyperparameter_type {H.__name__}"
                )
            self.hyperparameter_type = t
            return t
        return H

    def _plan_conditions(self, schema):
        """Set ``self.conditions(x, L)``: the root conditions in schema order."""
        if schema.kind == "tuple":
            self.errors.append(
                "(conditions): a population takes one condition (Lambda), the "
                f"distribution takes {len(schema.types)} positional ones"
            )
        lam = [(n, t) for n, t in zip(schema.names or (None,) * len(schema.types),
                                      schema.types)
               if n not in self.field_names]
        if len(lam) > 1 and schema.kind == "mapping":
            self.errors.append(
                f"(conditions): {[n for n, _ in lam]} are not fields of "
                f"{self.data_type.__name__}, and a population takes one other "
                "condition (Lambda)"
            )
        types = {n: self._lambda_type(t) for n, t in lam[:1]}
        names = schema.names if schema.kind == "mapping" else (None,) * len(schema.types)

        def conditions(x, L):
            out = []
            for n in names:
                if n in self.field_names:
                    out.append(self._field_value(self._x_root(x), n))
                else:
                    out.append(self._typed(L, types.get(n)))
            return tuple(out)

        self.conditions = conditions

    def _typed(self, L, t):
        t = t if t is not None else self.hyperparameter_type
        return t.from_dict(L) if is_parameter_type(t) else L

    def _field_value(self, xf, name):
        if name in xf:
            return xf[name]
        child = self.data_type.nested_types()[name]
        n = len(name) + 1
        return child.from_dict({k[n:]: v for k, v in xf.items() if k.startswith(name + ".")})

    def _field_deps(self, names):
        """The variables of the fields named by conditions in ``names``."""
        fields = [n for n in names if n in self.field_names]
        return [v for v in self.variables
                if any(v == n or v.startswith(n + ".") for n in fields)]

    @staticmethod
    def _schema_names(sd, path):
        return sd._units[path].schema.names     # the leaf's own condition names

    # --------------------------------------------------------
    # Variables and the values a leaf sees
    # --------------------------------------------------------

    def _variables(self, sd):
        out = []
        for path in sd.order:
            rec = sd.field_at(path)
            comps = getattr(rec.distribution, "components", ()) if isinstance(rec, Block) else ()
            if comps and isinstance(comps[0], StructuredDistribution):
                out += [f"{path}.{p}" for p in comps[0].order]
            elif not isinstance(rec, Block):
                out.append(path)
        return out

    def _x_root(self, x):
        """The root's view of the drawn prefix: fixed values and block objects added."""
        xf = {**self.root.fixed_values(), **x}
        for path, (comp, names) in self.blocks.items():
            if path not in xf and all(n in x for n in names):
                xf[path] = comp.assemble(self._x_local(x, path + ".", comp))
        return xf

    @staticmethod
    def _x_local(x, prefix, sd):
        n = len(prefix)
        return {**sd.fixed_values(), **{k[n:]: v for k, v in x.items() if k.startswith(prefix)}}

    def _unit_vars(self, path):
        return self.blocks[path][1] if path in self.blocks else [path]

    # --------------------------------------------------------
    # Leaves
    # --------------------------------------------------------

    def _leaf(self, sd, path, prefix, conditions, cond_names):
        """A ``generative.NumPyroLeaf`` for the leaf ``path`` of ``sd`` under ``prefix``."""
        rec, where = sd.field_at(path), prefix + path
        spec = sd.data_type.flat_field_specs()[path]
        if spec.index:
            self.errors.append(
                f"{where}: has core axes {spec.index}; the generative backend "
                "draws scalar fields only"
            )
            return None
        if isinstance(rec, NumPyroLeaf):
            problems = _check_distribution(rec.distribution, where)
            self.errors += problems
            if problems:
                return None
            d = rec.distribution
            return gen.NumPyroLeaf(where, lambda x, L, d=d: d,
                                   bounds=support_bounds(d.support))

        deps = self._field_deps(cond_names)
        units, unresolved = sd.dependencies(path)
        for u in units:     # a component's units are its own leaves
            deps += [prefix + u] if prefix else self._unit_vars(u)
        for dep in unresolved:
            if not prefix and dep in self.external:
                deps.append(dep)
            else:
                self.errors.append(
                    f"{where}: conditioned on unknown path {dep!r}; list it in "
                    "external= if it is supplied through conditioned_on"
                )
        if prefix:
            def make_dist(x, L):
                return sd.leaf_distribution(
                    path, self._x_local(x, prefix, sd), conditions(x, L), check=False
                )
        else:
            def make_dist(x, L):
                return sd.leaf_distribution(path, self._x_root(x), conditions(x, L),
                                            check=False)

        declared = support_bounds(spec.support) if spec.support is not None else None
        info = self._probe(make_dist)
        probed = (info or {}).get("bounds")
        bounds = declared if probed is None else probed if declared is None else (
            max(probed[0], declared[0]), min(probed[1], declared[1]))
        if info is not None:
            if info["shape"]:
                self.errors.append(
                    f"{where}: the distribution must be scalar, got shape {info['shape']}"
                )
                return None
            if info["discrete"]:
                self.errors.append(f"{where}: {info['name']} is discrete")
                return None
            if not info["icdf"]:
                if bounds is None or not np.all(np.isfinite(bounds)):
                    self.errors.append(
                        f"{where}: no differentiable inverse CDF and unbounded support"
                    )
                    return None
                if not info["cdf"]:
                    self.errors.append(f"{where}: {info['name']} has no CDF to invert")
                    return None
        return gen.NumPyroLeaf(where, make_dist, deps, bounds=bounds)

    def _probe(self, make_dist):
        """What a callable or conditional leaf returns, from one traced call.

        None when the call fails with placeholder values (for example, it
        branches on a value in Python); the checks then happen at draw time.
        """
        info = {}
        H = self.hyperparameter_type
        specs = H.flat_field_specs() if is_parameter_type(H) else None
        if specs is not None and any(s.index for s in specs.values()):
            return None

        def run(s):
            x = {v: s for v in [*self.variables, *self.external]}
            L = {p: s for p in specs} if specs is not None else _AnyKey(s)
            d = make_dist(x, L)
            info.update(
                name=type(d).__name__,
                shape=tuple(d.batch_shape) + tuple(d.event_shape),
                discrete=d.support.is_discrete,
                icdf=has_differentiable_icdf(d),
                bounds=support_bounds(d.support),
            )
            try:
                d.cdf(d.support.feasible_like(s))
                info["cdf"] = True
            except NotImplementedError:
                info["cdf"] = False
            return s

        try:
            jax.eval_shape(run, jax.ShapeDtypeStruct((), jnp.zeros(()).dtype))
        except Exception:  # noqa: BLE001 - user code that needs real values
            return None
        return info

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    def _block(self, path, block):
        if not isinstance(block, MixtureDistribution):
            self.errors.append(
                f"{path}: a {type(block).__name__} block has no fixed-bank "
                "transport (only mixtures of structured distributions are converted)"
            )
            return None
        if not all(isinstance(c, StructuredDistribution) for c in block.components):
            self.errors.append(
                f"{path}: mixture components must be structured distributions "
                "to be converted"
            )
            return None
        self.blocks[path] = (
            block.components[0], [f"{path}.{p}" for p in block.components[0].order]
        )

        def conditions(x, L):
            return self.root.block_conditions(path, self.conditions(x, L))

        return self._mixture(block, path + ".", conditions,
                             block._condition_schema().names)

    def _mixture(self, mix, prefix, conditions, cond_names=()):
        """A ``generative.WeightedMixture``; ``conditions(x, L)`` are the mixture's."""
        where = prefix[:-1] or "(mixture)"
        comps = mix.components
        first = sorted(comps[0].random_paths())
        fixed = {p: comps[0].fixed_values()[p] for p in comps[0].fixed_paths()}
        n0 = len(self.errors)
        for k, c in enumerate(comps[1:], start=1):
            if sorted(c.random_paths()) != first:
                self.errors.append(
                    f"{where}: mixture components draw different fields "
                    f"({first} and {sorted(c.random_paths())})"
                )
            other = {p: c.fixed_values()[p] for p in c.fixed_paths()}
            if set(other) != set(fixed) or not all(
                np.array_equal(np.asarray(fixed[p]), np.asarray(other[p])) for p in fixed
            ):
                self.errors.append(f"{where}: mixture components fix different values")
        for c in comps:
            for p in c.random_paths():
                if isinstance(c.field_at(p), Block):
                    self.errors.append(
                        f"{prefix}{p}: blocks inside mixture components are not converted"
                    )
        if len(self.errors) > n0:
            return None

        schema = mix._condition_schema()

        def as_mixture(conds):
            conds = schema.coerce(conds)
            return dict(zip(schema.names, conds)) if schema.kind == "mapping" else conds

        models = []
        for c in comps:
            def comp_conditions(x, L, c=c):
                return tuple(_component_conditions(c, as_mixture(conditions(x, L))))

            leaves = [
                self._leaf(c, p, prefix, comp_conditions,
                           [n for n in self._schema_names(c, p) if n in cond_names])
                for p in c.order
            ]
            models.append(gen.Product([v for v in leaves if v is not None]))
        if len(self.errors) > n0:
            return None
        for v in (prefix + p for p in comps[0].order):
            bounds = [m.bounds(v) for m in models]
            if not np.all(np.isfinite(bounds)):
                self.errors.append(
                    f"{v}: a mixture is drawn by bisection on its CDF, which "
                    "needs finite bounds"
                )
        logits = mix.logits is not None
        fn = mix.logits if logits else mix.weights
        batched = [n for n in schema.names if n in self.field_names and n in cond_names]
        if not callable(fn):
            value = jnp.asarray(fn)
            weight_fn = lambda x, L: value
        else:
            def weight_fn(x, L):
                conds = as_mixture(conditions(x, L))
                if not batched:
                    return jnp.asarray(mix._call_fn(fn, conds))
                shape = np.broadcast_shapes(*(_batch_of(conds[n]) for n in batched))
                values = tuple(_broadcast(conds[n], shape) for n in batched)

                def one(vals):
                    return jnp.asarray(mix._call_fn(fn, {**conds, **dict(zip(batched, vals))}))

                return _vmapped(one, shape)(values)
        return gen.WeightedMixture(models, weight_fn, deps=self._field_deps(batched),
                                   logits=logits)


# ============================================================
# The population
# ============================================================

class StructuredGenerativePopulation(PopulationModel):
    """
    A structured distribution drawn by fixed-bank transport.

    Made by ``prior.as_generative(hyperparameter_type=None, external=(),
    conditioned_on=None)``; see :mod:`astroinfer.distributions.transport`
    for what converts. The output type (``parameter_type``) is the
    distribution's ``data_type`` and the condition is Lambda.

    ``make_seeds(key, N)`` gives the bank, one uniform array per variable in
    ``order``, derived from the key as ``GenerativePopulation`` derives it.
    ``sample(Lambda, key, N=N)`` draws from that bank, and
    ``sample(Lambda, seeds=bank)`` from a given one; either way the draws
    are typed objects (fixed fields broadcast to the sample shape), the same
    for the same bank and Lambda, and differentiable in Lambda. ``logpdf``
    is ``GenerativeModel.log_prob``, with ``-inf`` where a fixed field
    differs from its declared value (when the distribution checks fixed
    fields). ``model`` is the underlying ``GenerativeModel``.

    External variables (``on`` paths that are not fields) are declared in
    ``external`` and their values supplied once in ``conditioned_on``. The
    population is then conditional on them, so ``logpdf`` is unavailable,
    as for ``GenerativePopulation``.
    """

    def __init__(self, distribution, hyperparameter_type=None, external=(), *,
                 conditioned_on=None, key=None):
        super().__init__(**({} if key is None else {"key": key}))
        conversion = _Conversion(distribution, hyperparameter_type, external)
        self.distribution = distribution
        self.model = conversion.model
        self.parameter_type = conversion.data_type
        self.hyperparameter_type = conversion.hyperparameter_type
        self.external = set(external)
        self.conditioned_on = dict(conditioned_on or {})
        self._root = conversion.root
        self._blocks = dict(conversion.blocks)
        missing = self.external - set(self.conditioned_on)
        if missing:
            raise ValueError(
                f"The distribution conditions on external variables "
                f"{sorted(missing)}; supply them in conditioned_on."
            )
        self._fixed = {p: self._root.fixed_values()[p] for p in self._root.fixed_paths()}

    @property
    def order(self):
        """The generative variables (dotted paths), in draw order."""
        return list(self.model.order)

    # ========================================================
    # Base noise
    # ========================================================

    def base_noise_spec(self):
        return BaseNoiseSpec("uniform", tuple(self.model.order), _SEED_EPS, 1.0 - _SEED_EPS)

    def make_seeds(self, key=None, N=None):
        """``{variable: uniforms}`` with sample shape ``N``, one array per variable."""
        return uniform_seeds(self.model.order, self._resolve_key(key), sample_shape(N))

    def sample_from_base(self, noise, Lambda=None):
        return self.sample(Lambda, seeds=noise)

    # ========================================================
    # Batched API
    # ========================================================

    @staticmethod
    def _flat_lambda(Lambda):
        if Lambda is None:
            return {}
        return Lambda.to_flat_dict() if isinstance(Lambda, Parameters) else dict(Lambda)

    def sample(self, Lambda, key=None, *, N=None, seeds=None):
        """
        Draw theta ~ p(theta | Lambda) by fixed-bank transport.

        seeds:
            ``{variable: uniforms}`` from ``make_seeds``. When given, ``key``
            and ``N`` are ignored and the sample shape is the seeds' shape.
            Otherwise the bank is made from ``key`` (see
            ``PopulationModel.sample``) with sample shape ``N``.
        """
        if seeds is None:
            seeds = self.make_seeds(key, N)
        else:
            missing = [v for v in self.model.order if v not in seeds]
            if missing:
                raise ValueError(f"seeds are missing the variables {missing}")
        shape = np.broadcast_shapes(*(jnp.shape(seeds[v]) for v in self.model.order))
        drawn = self.model.sample(
            self._flat_lambda(Lambda), seeds=seeds, conditioned_on=self.conditioned_on
        )
        x = dict(drawn)
        for path, (comp, names) in self._blocks.items():
            n = len(path) + 1
            x[path] = comp.assemble(
                {**comp.fixed_values(), **{k[n:]: x[k] for k in names}}
            )
        return self._root.assemble(x).broadcast(shape)

    def logpdf(self, theta, Lambda):
        """log p(theta | Lambda), with theta's batch shape."""
        if self.external:
            raise ValueError(
                "logpdf needs p(theta | Lambda) for every parameter, but the "
                "population conditions on external variables "
                f"{sorted(self.external)}, so it only has the conditional density."
            )
        theta = coerce(theta, self.parameter_type)
        flat = theta.to_flat_dict()
        lp = self.model.log_prob(flat, self._flat_lambda(Lambda))
        if not (self._fixed and getattr(self.distribution, "check_fixed", True)):
            return lp
        specs = self.parameter_type.flat_field_specs()
        ok = True
        for p, v in self._fixed.items():
            a = jnp.asarray(flat[p])
            same = jnp.isclose(a, v, rtol=1e-6, atol=0.0) | (a == v)
            k = len(specs[p].index)
            ok = ok & (jnp.all(same, axis=tuple(range(-k, 0))) if k else same)
        return jnp.where(ok, lp, -jnp.inf)

    # ========================================================
    # Display
    # ========================================================

    def _display_title(self):
        return f"{type(self).__name__}[{self.parameter_type.__name__}]"

    def _display_model(self):
        return self.distribution._display_model()

    def _display_config(self):
        config = {"Order": ", ".join(self.model.order)}
        if self.conditioned_on:
            config["Conditioned on"] = ", ".join(self.conditioned_on)
        return config


def as_generative(distribution, hyperparameter_type=None, external=(), *,
                  conditioned_on=None):
    """
    ``distribution`` as a ``StructuredGenerativePopulation``.

    ``distribution`` is a ``StructuredDistribution`` or a
    ``MixtureDistribution`` of structured distributions over the same
    fields; usually spelled ``prior.as_generative(...)``. Raises one
    ``ValueError`` listing every path that cannot be converted and why.
    """
    return StructuredGenerativePopulation(
        distribution, hyperparameter_type, external, conditioned_on=conditioned_on
    )

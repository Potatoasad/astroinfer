"""Integrating latent values out of a model in closed form.

In a model, ``ai.sample(name, prior, ..., marginalize=True)`` creates no sample site. It
returns a *stand-in* for the value, an object of the right type and shape filled with NaN,
and records the prior. The observed ``ai.sample`` site that receives the stand-in among its
conditions replaces the prior and its own likelihood by their integral over the value,
computed in closed form by a *rule*, and adds it to the model as one factor per element of
the value's batch. The sampler never sees the value. After sampling, ``ai.infer`` draws the
value from its distribution given the data, once per posterior draw, and puts it in the
posterior's theta as for an ordinary site.

A rule comes from one of two places, tried in this order.

Pair rule
    A class registered with ``@pair(PriorClass, LikelihoodClass)``, with the methods::

        log_marginal(self, prior, likelihood, data, prior_conditions, likelihood_conditions)
        draw(self, key, prior, likelihood, data, prior_conditions, likelihood_conditions)

    ``prior`` and ``likelihood`` are the distribution objects; the other arguments are for one
    element of the value's batch. Conditions are passed as the schema gives them: the object
    for a single type, a tuple for a tuple schema, a dict for a mapping schema, None for no
    conditions. The value inside ``likelihood_conditions`` is the stand-in and must not be
    read. ``draw`` returns one element of the value, in its type.

Conjugate family
    A ``ConjugateFamily`` named by the classes. The likelihood class declares
    ``conjugate_likelihood = {path: Family}`` and implements
    ``conjugate_likelihood_single(path, d, *conditions)``, which describes the likelihood as a
    function of the value at ``path``. The prior is either a NumPyro distribution that
    ``Family.from_numpyro`` understands, or a ``ConditionalDistribution`` that declares
    ``conjugate_prior = Family`` and implements ``conjugate_prior_single(*conditions)``. Each
    family documents what the two methods return.

``path`` locates the value in the likelihood's conditions: its first part is a condition name
for a mapping schema or an index for a tuple schema; the rest are fields. For a single
condition, the path is a field of it, or ``""`` for the whole condition.

Limits of this version: the value must be used by exactly one observed site; the prior must
be a NumPyro distribution or a ``ConditionalDistribution`` other than a structured
distribution; anything else that reads the value sees NaN.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln
from numpyro.distributions import Distribution
from numpyro.primitives import _PYRO_STACK, apply_stack

from .distributions.base import (
    ConditionalDistribution,
    _vmapped,
    broadcast_batch,
    combine_batches,
    over_batch,
)
from .parameters import Parameters

__all__ = [
    "ConjugateFamily",
    "GammaPoisson",
    "LinearGaussian",
    "NormalNormal",
    "drawing",
    "pair",
]

META = "astroinfer"


# ============================================================
# Conjugate families
# ============================================================

class ConjugateFamily:
    """
    A family of priors and the likelihoods it is conjugate to.

    Subclasses implement ``log_marginal`` and ``draw``, and ``from_numpyro`` if a NumPyro
    distribution can be the prior. All three work on one element of the value's batch. The
    likelihood may carry extra leading axes, when the value is shared by several elements of
    the likelihood's batch (a value outside a plate observed inside it); every element of
    those axes is an independent measurement of the same value.
    """

    def from_numpyro(self, distribution):
        """The prior's description, from a NumPyro distribution.

        Returns arrays whose leading axes are ``distribution.batch_shape``.
        """
        raise TypeError(
            f"{type(self).__name__} does not take a NumPyro {type(distribution).__name__} "
            "as the prior"
        )

    def log_marginal(self, prior, likelihood):
        """log of the integral over the value of prior(value) times likelihood(value)."""
        raise NotImplementedError

    def draw(self, key, prior, likelihood):
        """One draw of the value given the likelihood, as a flat array.

        The array has the value's elements in the order of ``jax.flatten_util.ravel_pytree``
        (fields in declaration order); a scalar value is an array of one element.
        """
        raise NotImplementedError


def _normal_pair(p):
    if isinstance(p, dist.Normal):
        return p.loc, p.scale
    loc, scale = p
    return loc, scale


class NormalNormal(ConjugateFamily):
    """
    A scalar value x with a Normal prior, measured with Normal errors.

    Prior: ``(loc, scale)`` or ``dist.Normal(loc, scale)``.
    Likelihood: ``(observed, error)``, meaning observed ~ Normal(x, error), or
    ``dist.Normal(observed, error)``, the likelihood as a density in x. The arrays may have
    any shape; every element is one measurement of x.
    """

    def from_numpyro(self, distribution):
        if not isinstance(distribution, dist.Normal):
            return super().from_numpyro(distribution)
        b = tuple(distribution.batch_shape)
        return (jnp.broadcast_to(distribution.loc, b), jnp.broadcast_to(distribution.scale, b))

    @staticmethod
    def _sums(prior, likelihood):
        # The exponent of prior x likelihood is -(P x^2 - 2 h x + Q) / 2.
        loc, scale = _normal_pair(prior)
        y, e = _normal_pair(likelihood)
        P = 1 / scale**2 + jnp.sum(1 / e**2 * jnp.ones_like(y))
        h = loc / scale**2 + jnp.sum(y / e**2)
        Q = loc**2 / scale**2 + jnp.sum(y**2 / e**2)
        n = jnp.size(jnp.broadcast_to(y, jnp.broadcast_shapes(jnp.shape(y), jnp.shape(e))))
        log_e = jnp.sum(jnp.log(e) * jnp.ones_like(y))
        return P, h, Q, n, log_e, scale

    def log_marginal(self, prior, likelihood):
        P, h, Q, n, log_e, scale = self._sums(prior, likelihood)
        return (-0.5 * (Q - h**2 / P) - 0.5 * jnp.log(P) - jnp.log(scale) - log_e
                - 0.5 * n * jnp.log(2 * jnp.pi))

    def draw(self, key, prior, likelihood):
        P, h, *_ = self._sums(prior, likelihood)
        return jnp.reshape(h / P + jax.random.normal(key) / jnp.sqrt(P), (1,))


class GammaPoisson(ConjugateFamily):
    """
    A rate r with a Gamma prior, measured by Poisson counts.

    Prior: ``(concentration, rate)`` or ``dist.Gamma(concentration, rate)``.
    Likelihood: ``(counts, exposure)``, meaning counts ~ Poisson(r * exposure). The arrays may
    have any shape; every element is one count.
    """

    def from_numpyro(self, distribution):
        if not isinstance(distribution, dist.Gamma):
            return super().from_numpyro(distribution)
        b = tuple(distribution.batch_shape)
        return (jnp.broadcast_to(distribution.concentration, b),
                jnp.broadcast_to(distribution.rate, b))

    @staticmethod
    def _prior(prior):
        if isinstance(prior, dist.Gamma):
            return prior.concentration, prior.rate
        return prior

    def log_marginal(self, prior, likelihood):
        a, b = self._prior(prior)
        k, t = likelihood
        k, t = jnp.broadcast_arrays(k, t)
        K, T = jnp.sum(k), jnp.sum(t)
        const = jnp.sum(k * jnp.log(t) - gammaln(k + 1))
        return const + gammaln(a + K) - gammaln(a) + a * jnp.log(b) - (a + K) * jnp.log(b + T)

    def draw(self, key, prior, likelihood):
        a, b = self._prior(prior)
        k, t = jnp.broadcast_arrays(*likelihood)
        return jnp.reshape(jax.random.gamma(key, a + jnp.sum(k)) / (b + jnp.sum(t)), (1,))


def _pack(value):
    """A parameter object or array as a flat vector, in ravel_pytree order."""
    return ravel_pytree(value)[0]


def _pack_rows(design, n_lead):
    """``(..., K)`` from a design given as an array or as a batched parameter object."""
    if not isinstance(design, Parameters):
        return jnp.asarray(design)
    b = design.batch_shape
    flat = design.broadcast().reshape_batch((prod(b),))
    rows = jax.vmap(_pack)(flat)
    return rows.reshape(b + rows.shape[-1:])


class LinearGaussian(ConjugateFamily):
    """
    A value a with a Gaussian prior, and data that are Gaussian with a mean linear in a.

    The value is packed into a vector of K numbers in ``ravel_pytree`` order (fields in
    declaration order); the prior and the design can be given as objects of the value's
    type, so that the order never has to be written out.

    Prior: ``(mean, scale_tril)``, where ``mean`` is a vector of K numbers or an object of
    the value's type, and ``scale_tril`` is a (K, K) Cholesky factor of the covariance, or an
    object of the value's type holding standard deviations (a diagonal covariance). From
    NumPyro: ``MultivariateNormal``, ``Normal``, or ``Normal(...).to_event(n)``.

    Likelihood: a dict with the keys

        observed          S + (n,): the data
        design            S + (n, K), or an object of the value's type with batch shape S + (n,):
                          the mean of observed[j] is design[j] . a + offset[j]
        noise_cov         S + (n, n), or
        noise_scale_tril  S + (n, n): the noise, independent between the blocks along S
        offset            optional, S + (n,)
        mask              optional, S + (n,); rows with 0 are left out (needs noise_cov)
    """

    def from_numpyro(self, distribution):
        d, b = distribution, tuple(distribution.batch_shape)
        if isinstance(d, dist.MultivariateNormal):
            K = d.event_shape[-1]
            return (jnp.broadcast_to(d.loc, b + (K,)), jnp.broadcast_to(d.scale_tril, b + (K, K)))
        if isinstance(d, dist.Independent):
            d = d.base_dist
        if isinstance(d, dist.Normal):
            full = tuple(d.batch_shape)
            loc = jnp.broadcast_to(d.loc, full).reshape(b + (-1,))
            scale = jnp.broadcast_to(d.scale, full).reshape(b + (-1,))
            return loc, scale[..., :, None] * jnp.eye(scale.shape[-1])
        return super().from_numpyro(distribution)

    @staticmethod
    def _prior(prior):
        mean, scale = prior
        mean = _pack(mean) if isinstance(mean, Parameters) else jnp.reshape(mean, (-1,))
        if isinstance(scale, Parameters):
            scale = jnp.diag(_pack(scale))
        return mean, jnp.asarray(scale)

    @staticmethod
    def _information(likelihood, K):
        """``(P, h, c)``: log L(a) = c + h . a - a . P a / 2."""
        y = jnp.asarray(likelihood["observed"])
        n = y.shape[-1]
        M = _pack_rows(likelihood["design"], y.ndim - 1)
        M = jnp.broadcast_to(M, y.shape + (K,))
        y = y - jnp.broadcast_to(likelihood.get("offset", 0.0), y.shape)
        mask = likelihood.get("mask")
        if "noise_cov" in likelihood:
            C = jnp.broadcast_to(likelihood["noise_cov"], y.shape + (n,))
            m = jnp.ones_like(y) if mask is None else jnp.broadcast_to(mask, y.shape).astype(y.dtype)
            C = C * m[..., :, None] * m[..., None, :] + (1 - m)[..., :, None] * jnp.eye(n)
            L = jnp.linalg.cholesky(C)
            y, M = y * m, M * m[..., None]
            count = jnp.sum(m)
        else:
            if mask is not None:
                raise ValueError("LinearGaussian: a mask needs noise_cov, not noise_scale_tril")
            L = jnp.broadcast_to(likelihood["noise_scale_tril"], y.shape + (n,))
            count = y.size
        L, y, M = L.reshape(-1, n, n), y.reshape(-1, n), M.reshape(-1, n, K)
        solve = jax.vmap(lambda Lb, x: solve_triangular(Lb, x, lower=True))
        W, z = solve(L, M), solve(L, y)
        P = jnp.einsum("bnk,bnl->kl", W, W)
        h = jnp.einsum("bnk,bn->k", W, z)
        log_det = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)))
        c = -0.5 * jnp.sum(z**2) - log_det - 0.5 * count * jnp.log(2 * jnp.pi)
        return P, h, c

    def _solve(self, prior, likelihood):
        mu, Lp = self._prior(prior)
        P, h, c = self._information(likelihood, mu.shape[-1])
        R = jnp.linalg.cholesky(jnp.eye(mu.shape[-1]) + Lp.T @ P @ Lp)
        u = solve_triangular(R, Lp.T @ (h - P @ mu), lower=True)
        return mu, Lp, P, h, c, R, u

    def log_marginal(self, prior, likelihood):
        mu, _, P, h, c, R, u = self._solve(prior, likelihood)
        return c + h @ mu - 0.5 * mu @ P @ mu + 0.5 * u @ u - jnp.sum(jnp.log(jnp.diag(R)))

    def draw(self, key, prior, likelihood):
        mu, Lp, *_, R, u = self._solve(prior, likelihood)
        eps = jax.random.normal(key, u.shape)
        return mu + Lp @ solve_triangular(R.T, u + eps, lower=False)


# ============================================================
# Pair rules
# ============================================================

_PAIRS: dict[tuple[type, type], Any] = {}


def pair(prior_type, likelihood_type, *, overwrite=False):
    """Register a rule for one prior class and one likelihood class (a class decorator).

    The rule is used for the classes and their subclasses, in preference to any conjugate
    family the classes declare.
    """
    def deco(cls):
        key = (prior_type, likelihood_type)
        if key in _PAIRS and not overwrite and type(_PAIRS[key]).__qualname__ != cls.__qualname__:
            raise ValueError(
                f"A rule for ({prior_type.__name__}, {likelihood_type.__name__}) is already "
                f"registered: {type(_PAIRS[key]).__name__}. Pass overwrite=True to replace it."
            )
        _PAIRS[key] = cls()
        return cls
    return deco


def _pair_rule(prior, likelihood):
    for p in type(prior).__mro__:
        for q in type(likelihood).__mro__:
            if (p, q) in _PAIRS:
                return _PAIRS[(p, q)]
    return None


# ============================================================
# Stand-ins
# ============================================================

@dataclass(eq=False)
class _Pending:
    """A marginalized value waiting for the observed site that uses it."""

    name: str
    prior: Any
    conds: tuple           # coerced conditions of a ConditionalDistribution prior
    context: dict
    value: Any             # the stand-in
    template: Any          # one element of the value, zeros
    batch: tuple           # batch shape of the value
    axes: tuple            # names of the batch axes
    kind: str              # "numpyro" or "conditional"
    out_type: Any = None
    used_by: str | None = None
    leaves: list = field(default_factory=list)


# id of a stand-in leaf -> (leaf, record). Bounded, so that old traces do not accumulate.
_STAND_INS: "OrderedDict[int, tuple]" = OrderedDict()
_MAX_STAND_INS = 4096

_DRAWING: ContextVar[bool] = ContextVar("astroinfer_marginal_drawing", default=False)


@contextmanager
def drawing():
    """Within this block, observed sites also draw the marginalized values they integrate."""
    token = _DRAWING.set(True)
    try:
        yield
    finally:
        _DRAWING.reset(token)


def _plates():
    """``{dim: plate}`` of the NumPyro plates around the current site."""
    return {h.dim: h for h in _PYRO_STACK if isinstance(h, numpyro.plate) and h.dim is not None}


def _plate_shape(plates):
    if not plates:
        return ()
    n = max(-d for d in plates)
    shape = [1] * n
    for d, p in plates.items():
        shape[d] = p.subsample_size
    return tuple(shape)


def _axis_names(name, batch, plates):
    n = len(batch)
    return tuple(
        plates[i - n].name if (i - n) in plates else f"{name}_dim_{i}" for i in range(n)
    )


def begin(name, prior, args, named):
    """The stand-in that ``ai.sample(name, prior, ..., marginalize=True)`` returns."""
    from .distributions.structured import StructuredDistribution
    from .modeling import _layout

    plates = _plates()
    if isinstance(prior, Distribution):
        if args or named:
            raise TypeError("A NumPyro distribution takes no conditions")
        batch = tuple(np.broadcast_shapes(tuple(prior.batch_shape), _plate_shape(plates)))
        template = jnp.zeros(tuple(prior.event_shape))
        record = _Pending(name, prior, (), {}, None, template, batch,
                          _axis_names(name, batch, plates), "numpyro")
    elif isinstance(prior, StructuredDistribution):
        raise NotImplementedError(
            f"{name}: marginalize=True does not take a structured distribution as the prior "
            "yet. Use a NumPyro distribution, or a ConditionalDistribution with "
            "conjugate_prior_single or a pair rule."
        )
    elif isinstance(prior, ConditionalDistribution):
        values, context, rest = prior._condition_schema().bind(args, named)
        if rest:
            raise TypeError(f"{name}: got {len(rest)} extra positional argument(s)")
        conds = tuple(prior._coerce_conditions(values))
        layout = _layout(prior, conds, context)
        template = layout.unpack(jnp.zeros(layout.size))
        batch = tuple(np.broadcast_shapes(tuple(broadcast_batch(conds)), _plate_shape(plates)))
        record = _Pending(name, prior, conds, context, None, template, batch,
                          _axis_names(name, batch, plates), "conditional",
                          out_type=prior._output_type())
    else:
        raise TypeError(
            f"{name}: expected a NumPyro distribution or a ConditionalDistribution, got "
            f"{type(prior).__name__}"
        )

    record.value = jax.tree_util.tree_map(
        lambda t: jnp.full(batch + jnp.shape(t), jnp.nan), record.template
    )
    record.leaves = jax.tree_util.tree_leaves(record.value)
    for leaf in record.leaves:
        _STAND_INS[id(leaf)] = (leaf, record)
    while len(_STAND_INS) > _MAX_STAND_INS:
        _STAND_INS.popitem(last=False)
    return record.value


def stand_ins(tree):
    """The records of the stand-ins among the leaves of ``tree``, in order, without repeats."""
    found = []
    for leaf in jax.tree_util.tree_leaves(tree):
        entry = _STAND_INS.get(id(leaf))
        if entry is not None and entry[0] is leaf and entry[1] not in found:
            found.append(entry[1])
    return found


def check_unused(name, tree):
    """Raise if ``tree`` (e.g. a NumPyro distribution) reads a marginalized value."""
    found = stand_ins(tree)
    if found:
        raise TypeError(
            f"{name}: uses the marginalized value {found[0].name!r}. A marginalized value can "
            "only be passed to an observed ai.sample site whose ConditionalDistribution has a "
            "pair rule or declares conjugate_likelihood."
        )


# ============================================================
# The observed site
# ============================================================

def _at_path(schema, conds, path):
    parts = path.split(".") if path else []
    if schema.kind == "mapping":
        if not parts:
            raise ValueError("A path into a mapping schema starts with a condition name")
        obj, parts = conds[schema.names.index(parts[0])], parts[1:]
    elif schema.kind == "tuple":
        if not parts:
            raise ValueError("A path into a tuple schema starts with an index")
        obj, parts = conds[int(parts[0])], parts[1:]
    elif schema.kind == "single":
        obj = conds[0]
    else:
        raise ValueError("The likelihood has no conditions")
    for p in parts:
        obj = obj[p] if isinstance(obj, Parameters) else getattr(obj, p)
    return obj


def _same_leaves(obj, record):
    ids = {id(leaf) for leaf in jax.tree_util.tree_leaves(obj)}
    return bool(ids) and ids == {id(leaf) for leaf in record.leaves}


def _natural(schema, values):
    """Values as the user sees them: an object, a tuple, a dict, or None."""
    if schema.kind == "none":
        return None
    if schema.kind == "single":
        return values[0]
    if schema.kind == "mapping":
        return dict(zip(schema.names, values))
    return tuple(values)


def consume(name, likelihood, values, context, obs):
    """The observed site of a likelihood whose conditions hold a marginalized value."""
    from .modeling import JointLayout, _coerce_output, _meta, _output_values

    records = stand_ins(values)
    if len(records) > 1:
        raise NotImplementedError(
            f"{name}: uses {len(records)} marginalized values; an observed site can integrate "
            "out one value in this version"
        )
    record = records[0]
    if obs is None:
        raise TypeError(
            f"{name}: uses the marginalized value {record.name!r}, so it must be an observed "
            "site (pass obs=)"
        )
    if record.used_by is not None:
        raise NotImplementedError(
            f"{name}: the marginalized value {record.name!r} is already used by the site "
            f"{record.used_by!r}; a value can be used by one observed site in this version"
        )
    record.used_by = name

    cond_schema = likelihood._condition_schema()
    out_schema = likelihood._output_schema()
    raw = tuple(values)
    conds = tuple(likelihood._coerce_conditions(raw))
    # Keep the stand-in itself where the coercion copied it, so that it can be found.
    conds = tuple(r if stand_ins(r) else c for r, c in zip(raw, conds))

    rule = _pair_rule(record.prior, likelihood) if record.kind == "conditional" else None
    path, family = None, None
    if rule is None:
        declared = getattr(likelihood, "conjugate_likelihood", None) or {}
        path = next(
            (p for p in declared if _same_leaves(_at_path(cond_schema, conds, p), record)), None
        )
        if path is None:
            rule_text = (
                f"no pair rule with {type(record.prior).__name__} and "
                if record.kind == "conditional" else ""
            )
            raise TypeError(
                f"{name}: {type(likelihood).__name__} uses the marginalized value "
                f"{record.name!r} but has {rule_text}no conjugate_likelihood entry whose path "
                f"holds it (declared paths: {sorted(declared)})"
            )
        family = declared[path]()
        if record.kind == "conditional":
            prior_family = getattr(record.prior, "conjugate_prior", None)
            if prior_family is None or not isinstance(family, prior_family):
                raise TypeError(
                    f"{name}: the prior of {record.name!r}, {type(record.prior).__name__}, "
                    f"declares conjugate_prior = {getattr(prior_family, '__name__', None)}, "
                    f"but {type(likelihood).__name__} declares {type(family).__name__} for "
                    f"{path!r}"
                )

    data = out_schema.coerce(_output_values(out_schema, _coerce_output(likelihood, obs)))
    data_b, conds_b, shape = combine_batches(data, conds)
    batch = record.batch
    if tuple(shape[:len(batch)]) != tuple(batch):
        raise ValueError(
            f"{name}: the batch shape {tuple(shape)} of the data and conditions does not start "
            f"with the batch shape {batch} of the marginalized value {record.name!r}"
        )
    extra = tuple(shape[len(batch):])

    # What is mapped over the value's batch: the prior's description or conditions, and the
    # likelihood's data and conditions. Other arguments reach every element whole.
    if record.kind == "numpyro":
        n_prior = len(record.prior.batch_shape)
        prior_values = (jax.tree_util.tree_map(
            lambda a: jnp.broadcast_to(a, batch + jnp.shape(a)[n_prior:]),
            family.from_numpyro(record.prior),
        ),)
        prior_mapped = (True,)
    else:
        prior_values = tuple(
            c.broadcast(batch) if isinstance(c, Parameters) else c for c in record.conds
        )
        prior_mapped = tuple(isinstance(c, Parameters) for c in record.conds)
    lik_values = data_b + conds_b
    lik_mapped = tuple(isinstance(v, Parameters) for v in lik_values)
    n_data, n_prior_values = len(data_b), len(prior_values)

    all_values = prior_values + lik_values
    flags = prior_mapped + lik_mapped
    mapped = tuple(v for v, f in zip(all_values, flags) if f)

    def element(mapped_values, key, *, draw):
        it = iter(mapped_values)
        vals = [next(it) if f else v for v, f in zip(all_values, flags)]
        prior_e, lik_e = vals[:n_prior_values], vals[n_prior_values:]
        data_e, conds_e = tuple(lik_e[:n_data]), tuple(lik_e[n_data:])
        if rule is not None:
            args = (
                record.prior, likelihood, _natural(out_schema, data_e),
                _natural(record.prior._condition_schema(), prior_e),
                _natural(cond_schema, conds_e),
            )
            return rule.draw(key, *args) if draw else rule.log_marginal(*args)
        if record.kind == "numpyro":
            prior_low = prior_e[0]
        else:
            p_args, p_kwargs = record.prior._condition_schema().arguments(prior_e)
            prior_low = record.prior.conjugate_prior_single(*p_args, **p_kwargs, **record.context)

        def lower(vs, _):
            d_args, d_kwargs = out_schema.arguments(vs[:n_data])
            c_args, c_kwargs = cond_schema.arguments(vs[n_data:])
            return likelihood.conjugate_likelihood_single(
                path, *d_args, *c_args, **d_kwargs, **c_kwargs, **context
            )

        lik_low = over_batch(lower, data_e + conds_e, extra)
        if not draw:
            return family.log_marginal(prior_low, lik_low)
        _, unravel = ravel_pytree(record.template)
        return unravel(jnp.reshape(family.draw(key, prior_low, lik_low), (-1,)))

    log_m = _vmapped(lambda m, k: element(m, k, draw=False), batch)(mapped, None) if batch \
        else element(mapped, None, draw=False)
    log_m = jnp.asarray(log_m)
    if extra:
        index = (slice(None),) * len(batch) + (0,) * len(extra)
        log_m = jnp.zeros(tuple(shape), log_m.dtype).at[index].set(log_m)
    unit = dist.distribution.Unit(log_m)
    numpyro.sample(name, unit, obs=unit.sample(None),
                   infer={"is_auxiliary": True, **_meta(name, "marginal", None, None,
                                                         marginalized=record.name)})

    if _DRAWING.get():
        key = numpyro.prng_key()
        if batch:
            keys = jax.random.split(key, prod(batch)).reshape(batch + jnp.shape(key))
            value = _vmapped(lambda m, k: element(m, k, draw=True), batch)(mapped, keys)
        else:
            value = element(mapped, key, draw=True)
        _record_root(record, value, JointLayout)
    return obs


def _record_root(record, value, JointLayout):
    """A deterministic site holding the drawn value, marked as a root of the model."""
    if record.kind == "conditional":
        layout = JointLayout.from_template(record.template)
        stored, kind = layout.pack(value), "joint"
        axes = record.axes
    else:
        layout, stored, kind = None, jnp.asarray(value), "numpyro"
        n = len(record.batch)
        axes = record.axes + tuple(
            f"{record.name}_dim_{i}" for i in range(n, n + jnp.ndim(record.template))
        )
    msg = {
        "type": "deterministic",
        "name": record.name,
        "value": stored,
        "cond_indep_stack": [],
        "infer": {META: {
            "root": record.name, "path": None, "kind": "marginalized", "type": record.out_type,
            "layout": layout, "root_kind": kind, "axes": axes,
        }},
    }
    apply_stack(msg)

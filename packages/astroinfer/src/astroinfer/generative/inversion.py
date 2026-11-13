"""Differentiable inverse-CDF sampling by bisection.

The user supplies only a forward CDF. This module inverts it, and gives an
exact gradient by the implicit-function theorem -- NOT by differentiating
the bisection loop (that gives zero gradient, since every step is a hard
comparison). The forward solve is wrapped in stop_gradient; the backward
pass is attached by hand."""
from functools import partial

import jax
import jax.numpy as jnp


def norm_cdf(x):
    return 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))


def norm_ppf(u):
    return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u - 1.0)


def truncnorm_cdf(x, mu, sigma, a, b):
    lo, hi = (a - mu) / sigma, (b - mu) / sigma
    ca, cb = norm_cdf(lo), norm_cdf(hi)
    return jnp.clip((norm_cdf((x - mu) / sigma) - ca) / (cb - ca), 0.0, 1.0)


def truncnorm_pdf(x, mu, sigma, a, b):
    lo, hi = (a - mu) / sigma, (b - mu) / sigma
    Z = norm_cdf(hi) - norm_cdf(lo)
    phi = jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
    return jnp.where((x >= a) & (x <= b), phi / Z, 0.0)


def truncnorm_ppf(u, mu, sigma, a, b):
    """Closed-form quantile -- used when a leaf wants to skip bisection."""
    lo, hi = (a - mu) / sigma, (b - mu) / sigma
    ca, cb = norm_cdf(lo), norm_cdf(hi)
    return mu + sigma * norm_ppf(ca + u * (cb - ca))


def _bisection_root(cdf, u, a, b, n_iter=40):
    a = jnp.broadcast_to(jnp.asarray(a, u.dtype), u.shape)
    b = jnp.broadcast_to(jnp.asarray(b, u.dtype), u.shape)

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        too_high = cdf(mid) > u
        lo = jnp.where(too_high, lo, mid)
        hi = jnp.where(too_high, mid, hi)
        return (lo, hi)

    lo, hi = jax.lax.stop_gradient(jax.lax.fori_loop(0, n_iter, body, (a, b)))
    return 0.5 * (lo + hi)


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def bisection_ppf(cdf_fn, diff, u, a, b):
    """Invert cdf_fn(x, diff)=u for x, differentiable w.r.t. `diff` and `u`.

    `diff` is the pytree carrying every tracer the CDF depends on -- both the
    hyperparameters L and the drawn-prefix dict x. It MUST be the diff arg,
    never captured by closure (a nondiff arg holding live tracers is invalid
    and JAX's tracer-leak check rejects it). cdf_fn/a/b are nondiff."""
    return _bisection_root(lambda x: cdf_fn(x, diff), u, a, b)


def _fwd(cdf_fn, diff, u, a, b):
    x = bisection_ppf(cdf_fn, diff, u, a, b)
    return x, (diff, x)


def _bwd(cdf_fn, a, b, res, g):
    diff, x = res
    _, vjp_x = jax.vjp(lambda xx: cdf_fn(xx, diff), x)
    (dFdx,) = vjp_x(jnp.ones_like(x))          # elementwise dF/dx at the root
    c = g / dFdx
    _, vjp_d = jax.vjp(lambda d: cdf_fn(x, d), diff)
    (dbar,) = vjp_d(c)
    dbar = jax.tree_util.tree_map(lambda t: -t, dbar)
    return (dbar, c)                            # cotangents for (diff, u)


bisection_ppf.defvjp(_fwd, _bwd)
"""product() and mixture() for the generative sector.

product  : independent variable groups. Routes each variable to its owner.
mixture  : components share the SAME variables. ONE component choice governs
           all of a component's variables jointly (the 'both aligned or both
           isotropic' structure). Drawn by ancestral sampling: the component
           choice is never drawn as a hard index -- its uncertainty is folded
           into responsibilities recomputed from the drawn prefix, so the
           mixture-weight gradient flows and the coupling is exact at any
           nesting depth (an inner mixture recomputes ITS OWN responsibilities
           from the same prefix)."""
import jax
import jax.numpy as jnp
import numpy as np

from .base import Distribution


class Product(Distribution):
    def __init__(self, models):
        self.models = models
        self.var_names = [v for m in models for v in m.var_names]
        self._owner = {v: m for m in models for v in m.var_names}

    def cond_cdf(self, var, val, x, L):
        return self._owner[var].cond_cdf(var, val, x, L)

    def cond_pdf(self, var, val, x, L):
        return self._owner[var].cond_pdf(var, val, x, L)

    def cond_ppf(self, var, u, x, L):
        return self._owner[var].cond_ppf(var, u, x, L)

    def bounds(self, var):
        return self._owner[var].bounds(var)

    def deps(self, var):
        return self._owner[var].deps(var)


class Mixture(Distribution):
    def __init__(self, models, weights):
        assert len(weights) == len(models) - 1, (
            f"weights must have len(models)-1={len(models)-1} entries, "
            f"got {len(weights)}")
        self.models = models
        self.weight_names = list(weights)
        self.var_names = list(models[0].var_names)

    def _weights(self, L):
        w = [L[n] for n in self.weight_names]
        return w + [1.0 - sum(w)]

    def _log_resp(self, x, L):
        """Posterior over components given the ALREADY-drawn variables of this
        mixture. Computed in log space for stability when one component
        dominates. Constant across the bisection variable, so it is not
        recomputed per bisection step in effect (it reads only the prefix)."""
        logw = [jnp.log(wi) for wi in self._weights(L)]
        drawn = [v for v in self.var_names if v in x]
        lognum = []
        for lw, m in zip(logw, self.models):
            acc = lw
            for v in drawn:
                acc = acc + jnp.log(m.cond_pdf(v, x[v], x, L))
            lognum.append(acc)
        stacked = jnp.stack(lognum, axis=0)
        return stacked - jax.scipy.special.logsumexp(stacked, axis=0)

    def _q(self, x, L):
        return jnp.exp(self._log_resp(x, L))

    def cond_cdf(self, var, val, x, L):
        q = self._q(x, L)
        return sum(q[i] * m.cond_cdf(var, val, x, L)
                   for i, m in enumerate(self.models))

    def cond_pdf(self, var, val, x, L):
        q = self._q(x, L)
        return sum(q[i] * m.cond_pdf(var, val, x, L)
                   for i, m in enumerate(self.models))

    def bounds(self, var):
        return self.models[0].bounds(var)

    def deps(self, var):
        out = []
        for m in self.models:
            out += m.deps(var)
        return out


class WeightedMixture(Mixture):
    """A mixture whose weights come from a function of the prefix and L.

    ``weight_fn(x, L)`` returns the K component weights, as an array with
    the components on the last axis (leading axes broadcast against the
    sample shape, so weights may vary per draw) or as a list of K arrays.
    With ``logits=True`` it returns unnormalized log weights instead.
    ``deps`` names the drawn variables ``weight_fn`` reads; every variable
    of the mixture is then ordered after them.

    As in ``Mixture``, no component label is drawn: the responsibilities
    are recomputed from the drawn prefix, so the draws and their gradients
    with respect to the weights are continuous. ``bounds`` is the union of
    the components' bounds, and bisection on the mixture CDF needs it to be
    finite.
    """

    def __init__(self, models, weight_fn, deps=(), *, logits=False):
        self.models = models
        self.weight_fn = weight_fn
        self.weight_deps = [deps] if isinstance(deps, str) else list(deps)
        self.logits = bool(logits)
        self.var_names = list(models[0].var_names)

    def _log_weights(self, x, L):
        w = self.weight_fn(x, L)
        if isinstance(w, (list, tuple)):
            w = jnp.stack(jnp.broadcast_arrays(*w), axis=-1)
        w = jnp.asarray(w)
        if w.shape[-1:] != (len(self.models),):
            raise ValueError(
                f"weight_fn must return {len(self.models)} weights on the last "
                f"axis, got shape {w.shape}"
            )
        logw = jax.nn.log_softmax(w, axis=-1) if self.logits else jnp.log(w)
        return [logw[..., k] for k in range(len(self.models))]

    def _log_resp(self, x, L):
        drawn = [v for v in self.var_names if v in x]
        lognum = []
        for lw, m in zip(self._log_weights(x, L), self.models):
            acc = lw
            for v in drawn:
                acc = acc + jnp.log(m.cond_pdf(v, x[v], x, L))
            lognum.append(acc)
        stacked = jnp.stack(jnp.broadcast_arrays(*lognum), axis=0)
        return stacked - jax.scipy.special.logsumexp(stacked, axis=0)

    def cond_ppf(self, var, u, x, L):
        a, b = self.bounds(var)
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(
                f"{var}: a mixture is drawn by bisection on its CDF, which "
                f"needs finite bounds; the components give ({a}, {b})"
            )

    def bounds(self, var):
        bounds = [m.bounds(var) for m in self.models]
        return min(b[0] for b in bounds), max(b[1] for b in bounds)

    def deps(self, var):
        return self.weight_deps + super().deps(var)


def product(models):
    return Product(models)


def mixture(models, weights):
    return Mixture(models, weights)
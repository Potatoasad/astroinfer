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


def product(models):
    return Product(models)


def mixture(models, weights):
    return Mixture(models, weights)
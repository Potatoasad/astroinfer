"""Fundamental 1-D distributions. Conditioning enters only through the param
specs (mu, sigma, a, b may each be a constant, a hyperparameter name, or a
Conditional reading earlier variables). One TruncatedNormal class therefore
covers both the plain leaf and the 'conditional whose mean/sigma ride on
earlier draws' case."""
import jax
import jax.numpy as jnp

from .base import Distribution
from .inversion import truncnorm_cdf, truncnorm_pdf, truncnorm_ppf
from .spec import ParamSpec, resolve, spec_deps


class TruncatedNormal(Distribution):
    def __init__(self, var: str, mu: ParamSpec, sigma: ParamSpec, a, b):
        self.var_names = [var]
        self.var = var
        self.mu, self.sigma = mu, sigma
        self.a, self.b = a, b

    def _ms(self, x, L):
        return resolve(self.mu, x, L), resolve(self.sigma, x, L)

    def cond_cdf(self, var, val, x, L):
        mu, sigma = self._ms(x, L)
        return truncnorm_cdf(val, mu, sigma, self.a, self.b)

    def cond_pdf(self, var, val, x, L):
        mu, sigma = self._ms(x, L)
        return truncnorm_pdf(val, mu, sigma, self.a, self.b)

    def cond_ppf(self, var, u, x, L):
        mu, sigma = self._ms(x, L)
        return truncnorm_ppf(u, mu, sigma, self.a, self.b)

    def bounds(self, var):
        return self.a, self.b

    def deps(self, var):
        return spec_deps(self.mu) + spec_deps(self.sigma)


class Uniform(Distribution):
    def __init__(self, var: str, a, b):
        self.var_names = [var]
        self.var = var
        self.a, self.b = a, b

    def cond_cdf(self, var, val, x, L):
        return jnp.clip((val - self.a) / (self.b - self.a), 0.0, 1.0)

    def cond_pdf(self, var, val, x, L):
        d = 1.0 / (self.b - self.a)
        return jnp.where((val >= self.a) & (val <= self.b), d, 0.0)

    def cond_ppf(self, var, u, x, L):
        return self.a + u * (self.b - self.a)

    def bounds(self, var):
        return self.a, self.b

    def deps(self, var):
        return []


class Beta(Distribution):
    """Beta on [0,1]. cond_cdf uses the regularized incomplete beta. Gradients
    w.r.t. the shape params flow through betainc; if you hit a gradient gap on
    older JAX, prefer TruncatedNormal or supply a closed-form quantile."""
    def __init__(self, var: str, alpha: ParamSpec, beta: ParamSpec):
        self.var_names = [var]
        self.var = var
        self.alpha, self.beta = alpha, beta
        self.a, self.b = 0.0, 1.0

    def _ab(self, x, L):
        return resolve(self.alpha, x, L), resolve(self.beta, x, L)

    def cond_cdf(self, var, val, x, L):
        al, be = self._ab(x, L)
        return jnp.clip(jax.scipy.special.betainc(al, be, jnp.clip(val, 0.0, 1.0)),
                        0.0, 1.0)

    def cond_pdf(self, var, val, x, L):
        al, be = self._ab(x, L)
        logp = ((al - 1) * jnp.log(val) + (be - 1) * jnp.log1p(-val)
                - jax.scipy.special.betaln(al, be))
        return jnp.where((val > 0.0) & (val < 1.0), jnp.exp(logp), 0.0)

    def bounds(self, var):
        return 0.0, 1.0

    def deps(self, var):
        return spec_deps(self.alpha) + spec_deps(self.beta)
"""The generative distribution interface. A model answers, for one variable
at a time and given the earlier-drawn prefix:
  - cond_cdf(var, val, x, L) : conditional CDF at val
  - cond_pdf(var, val, x, L) : conditional PDF at val
  - bounds(var)              : (a, b) support
  - deps(var)                : drawn variables `var` reads (for validation)
Same three probability ops serve BOTH verbs: sampling (invert the CDF) and
scoring (evaluate the PDF). There is no separate joint-density method; the
joint is the product of the conditional PDFs along the order.

cond_ppf(var, u, x, L) is an OPTIONAL fourth op: a closed-form quantile
(inverse cond_cdf), for distributions that have one (e.g. TruncatedNormal,
Uniform). Returning None (the default) tells GenerativeModel.sample to fall
back to differentiable bisection against cond_cdf -- correct for anything
(a Mixture has no closed-form inverse), just slower. Implementing cond_ppf
on a leaf skips bisection entirely for that variable: no iteration, and an
ordinary (non-custom-vjp) autodiff path, since the closed form is already a
plain differentiable function of (u, x, L)."""
from typing import ClassVar


class Distribution:
    var_names: ClassVar[list[str]] = []

    def cond_cdf(self, var, val, x, L):
        raise NotImplementedError

    def cond_pdf(self, var, val, x, L):
        raise NotImplementedError

    def cond_ppf(self, var, u, x, L):
        return None

    def bounds(self, var):
        raise NotImplementedError

    def deps(self, var):
        return []
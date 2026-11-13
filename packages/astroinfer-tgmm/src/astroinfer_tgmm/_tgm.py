"""Lazy access to truncatedgaussianmixtures (importing it starts Julia)."""
from collections import OrderedDict

import numpy as np

LOGTP_CUTOFF = -19.0  # components with < e^-19 of their mass inside a sector's box are broken

_SECTOR_FN = """
using TruncatedGaussianMixtures
function astroinfer_sector_log_total_probs(gmm, indices)
    logtps = Float64[]
    for comp in gmm.components
        a = comp.a[indices]; b = comp.b[indices]
        m = comp.normal.μ[indices]
        S = comp.normal.Σ[indices, indices]
        push!(logtps, TruncatedMvNormal(MvNormal(m, S), a, b).logtp)
    end
    logtps
end
"""
_sector_fn_defined = False


def tgm():
    """The truncatedgaussianmixtures module, with 64-bit JAX enabled."""
    import truncatedgaussianmixtures
    import truncatedgaussianmixtures.jax_backend  # noqa: F401  (enables float64 in JAX)

    return truncatedgaussianmixtures


def sectors_from_block_structure(block_structure, n_dims):
    """``{label: 0-based column indices}``; one sector if there is no block structure."""
    if block_structure is None or len(set(block_structure)) <= 1:
        return OrderedDict(all=list(range(n_dims)))
    sectors = OrderedDict()
    for i, label in enumerate(block_structure):
        sectors.setdefault(f"block_{label}", []).append(i)
    return sectors


def log_interval_probability(mu, sigma, a, b) -> np.ndarray:
    """log P(a < X < b) for X ~ N(mu, sigma^2), stable far in the tails."""
    from scipy.special import log_ndtr

    za, zb = (np.asarray(a) - mu) / sigma, (np.asarray(b) - mu) / sigma
    # Work on the side of the mean where the CDF differences are not tiny.
    flip = za > 0
    lo = np.where(flip, -zb, za)
    hi = np.where(flip, -za, zb)
    log_hi, log_lo = log_ndtr(hi), log_ndtr(lo)
    with np.errstate(divide="ignore"):
        return log_hi + np.log1p(-np.exp(log_lo - log_hi))


def sector_log_total_probs(gmm, sectors) -> dict[str, np.ndarray]:
    """``{sector: log mass of each component inside the sector's box}``.

    One sector: the components' own log normalizers. A one-column sector:
    the exact normal interval probability (NumPy). Larger sectors evaluate
    the block's truncated multivariate normal in Julia.
    """
    if len(sectors) == 1:
        (name,) = sectors
        logz = gmm.log_normalizers
        if logz is None:
            logz = gmm._ensure_log_normalizers()
        return {name: np.asarray(logz, dtype=np.float64)}

    means = np.asarray(gmm.means, dtype=np.float64)
    covs = np.asarray(gmm.covariances, dtype=np.float64)
    lower, upper = np.asarray(gmm.lower), np.asarray(gmm.upper)
    out = {}
    for name, indices in sectors.items():
        if len(indices) == 1:
            (i,) = indices
            out[name] = log_interval_probability(means[:, i], np.sqrt(covs[:, i, i]), lower[i], upper[i])
        else:
            out[name] = _julia_sector_log_total_probs(gmm, indices)
    return out


def _julia_sector_log_total_probs(gmm, indices) -> np.ndarray:
    global _sector_fn_defined
    import juliacall

    jl = tgm().jl
    if not _sector_fn_defined:
        jl.seval(_SECTOR_FN)
        _sector_fn_defined = True
    idx = juliacall.convert(jl.Vector[jl.Int64], [i + 1 for i in indices])
    return np.asarray(jl.astroinfer_sector_log_total_probs(gmm.gmm, idx), dtype=np.float64)


def components_far_from_domain(gmm, sectors, cutoff=LOGTP_CUTOFF) -> dict[str, list[int]]:
    """``{sector: component indices}`` whose mass inside the box is below
    ``exp(cutoff)`` (these give NaNs downstream). Empty if the fit is fine."""
    problems = {}
    for name, logtps in sector_log_total_probs(gmm, sectors).items():
        bad = np.flatnonzero(~(logtps >= cutoff))  # also catches NaN
        if len(bad):
            problems[name] = bad.tolist()
    return problems

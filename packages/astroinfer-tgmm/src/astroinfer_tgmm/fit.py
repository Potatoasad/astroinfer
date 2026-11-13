"""One event's TGMM fit: the astroinfer ``Fit`` for the ``tgmm`` kind."""
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astroinfer.display import html_table, kv_table, section
from astroinfer.io import read_json, write_json
from astroinfer.representations import BaseFit, LoadContext, stack_products

from ._tgm import tgm
from .config import TGMMConfig


def pad_components(product: Mapping[str, Any], K: int) -> dict[str, np.ndarray]:
    """Pad a TGMM data product (every array has a leading component axis) to ``K``.

    Padded components get ``weights = 0``; every other array copies the
    heaviest component, so kernels and samples stay finite and in the box.
    (Zero-filling would give sigma = 0, and 0 * NaN = NaN poisons the
    likelihood and its gradient.)
    """
    weights = np.asarray(product["weights"])
    k = weights.shape[0]
    if k > K:
        raise ValueError(f"cannot pad {k} components down to {K}")
    out = {}
    for key, value in product.items():
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] != k:
            raise ValueError(f"{key!r} has shape {arr.shape}; expected a leading component axis of {k}")
        out[key] = arr
    if k == K:
        return out
    j = int(np.argmax(weights))
    for key, arr in out.items():
        if key == "weights":
            fill = np.zeros((K - k,) + arr.shape[1:], dtype=arr.dtype)
        else:
            fill = np.repeat(arr[j:j + 1], K - k, axis=0)
        out[key] = np.concatenate([arr, fill], axis=0)
    return out


class TGMMFit(BaseFit):
    """A fitted ``tgm.TGMM`` with the config it was fit with.

    ``dataproduct`` arrays all have a leading component axis K:
    ``weights (K,)``, ``<col>_mu_kernel / _sigma_kernel / _rho_kernel (K,)``
    for analytic columns (in the fitted, image space), and for sampled
    columns ``<col> (K, N)`` plus the carried ``prior (K, N)``.
    """

    def __init__(self, gmm, config: TGMMConfig, *, fit_key: str, diagnostics: dict | None = None):
        self.gmm = gmm
        self.config = config
        self._fit_key = fit_key
        self.diagnostics = dict(diagnostics or {})
        self._cache: dict[tuple, dict] = {}

    def __repr__(self):
        return (f"TGMMFit(K={self.K}, columns={list(self.config.columns)}, "
                f"image_columns={self.image_columns}, seed={self.diagnostics.get('seed')})")

    @property
    def fit_key(self) -> str:
        return self._fit_key

    @property
    def K(self) -> int:
        return int(np.asarray(self.gmm.weights).shape[0])

    @property
    def image_columns(self) -> list[str]:
        return list(self.gmm.image_cols)

    @property
    def domain_columns(self) -> list[str]:
        return list(self.gmm.domain_cols)

    # ========================================================
    # RichDisplay
    # ========================================================

    def _display_title(self):
        return "TGMM fit"

    def _display_math(self):
        return (rf"p(x)\approx\sum_{{k=1}}^{{{self.K}}} w_k\,"
                r"\mathcal{N}_{[a,b]}\!\left(T(x)\mid\mu_k,\Sigma_k\right)")

    def _display_body(self, max_components: int = 20):
        diag = self.diagnostics
        summary = kv_table({
            "K": self.K,
            "columns": ", ".join(self.domain_columns),
            "fitted as": ", ".join(self.image_columns),
            "seed used": diag.get("seed"),
            "attempts": diag.get("attempts"),
            "converged": diag.get("converged"),
            "iterations": diag.get("n_iterations"),
            "mean log-likelihood": diag.get("final_average_loglikelihood"),
            "draws fit": diag.get("n_draws"),
            "fit time (s)": diag.get("fit_seconds"),
        })
        weights = np.asarray(self.gmm.weights)
        means = np.asarray(self.gmm.means)
        sigmas = np.asarray(self.gmm.std_deviations)
        order = np.argsort(-weights)[:max_components]
        rows = [[str(k + 1), f"{weights[k]:.3f}"]
                + [f"{means[k, j]:.3g} ± {sigmas[k, j]:.2g}" for j in range(means.shape[1])]
                for k in order]
        header = ["k", "weight"] + [f"<code>{c}</code>  μ ± σ" for c in self.image_columns]
        components = html_table(header, rows)
        if self.K > max_components:
            components += f"<div>... {self.K - max_components} lighter components</div>"
        box = kv_table({c: f"[{lo:g}, {hi:g}]" for c, lo, hi in
                        zip(self.image_columns, np.asarray(self.gmm.lower), np.asarray(self.gmm.upper))})
        failed = diag.get("failed_attempts") or []
        parts = [summary,
                 section("Components (by weight)", components, opened=True, count=self.K),
                 section("Truncation box", box)]
        if failed:
            parts.append(section("Rejected attempts", kv_table(
                {f"seed {f['seed']}": f["problems"] for f in failed}), count=len(failed)))
        parts.append(section("Config", self.config._repr_html_()))
        return "".join(parts)

    # ========================================================
    # Data product and sampling
    # ========================================================

    def dataproduct(
        self,
        *,
        N: int,
        seed: int,
        analytic_columns: Sequence[str] = (),
        sampled_columns: Sequence[str] = (),
    ) -> dict[str, np.ndarray]:
        key = (int(N), int(seed), tuple(analytic_columns), tuple(sampled_columns))
        if key not in self._cache:
            dp = self.gmm.data_product(analytic_columns=list(analytic_columns),
                                       sampled_columns=list(sampled_columns),
                                       N=int(N), seed=int(seed))
            self._cache[key] = {k: np.asarray(v) for k, v in dp.items()}
        return dict(self._cache[key])

    def sample(self, N: int = 1000, seed: int | None = None):
        """``N`` draws from the fit in the original columns (a DataFrame)."""
        return self.gmm.sample(N, seed=seed)

    @classmethod
    def stack(cls, products):
        products = list(products)
        if not products:
            return {}
        K = max(np.asarray(p["weights"]).shape[0] for p in products)
        return stack_products([pad_components(p, K) for p in products])

    # ========================================================
    # Saving
    # ========================================================

    def save(self, directory: Path) -> None:
        self.gmm.save(str(directory / "gmm.h5"))
        write_json(directory / "tgmm.json", {
            "config": self.config.to_dict(),
            "fit_key": self._fit_key,
            "diagnostics": self.diagnostics,
        })

    @classmethod
    def load(cls, directory: Path, ctx: LoadContext) -> "TGMMFit":
        info = read_json(directory / "tgmm.json")
        gmm = tgm().TGMM.load(str(directory / "gmm.h5"))
        return cls(gmm, TGMMConfig.from_dict(info["config"]),
                   fit_key=info["fit_key"], diagnostics=info.get("diagnostics"))

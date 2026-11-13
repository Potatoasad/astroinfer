"""TGMMFitter: the astroinfer ``Fitter`` for the ``tgmm`` kind."""
import time
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from astroinfer.display import html_table, kv_table, section
from astroinfer.representations import BaseFitter, config_hash, register

from ._tgm import LOGTP_CUTOFF, components_far_from_domain, sectors_from_block_structure, tgm
from .config import TGMMConfig, TransformationSpec
from .fit import TGMMFit


class TGMMFitError(RuntimeError):
    pass


def fitting_frame(posterior, cfg: TGMMConfig) -> tuple[pd.DataFrame, np.ndarray | None]:
    """The draws the fit sees: ``columns``, ``carry_columns`` and ``prior``
    (the PE prior of ``columns`` alone), reproducibly subsampled to
    ``n_samples``; plus fit weights from ``log_weight_column`` if set."""
    values = posterior.variables(list(cfg.columns) + list(cfg.carry_columns))
    df = pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in values.items()})
    df["prior"] = np.asarray(posterior.marginal_prior(cfg.columns), dtype=np.float64)

    log_w = None
    if cfg.log_weight_column is not None:
        log_w = np.asarray(posterior.variables([cfg.log_weight_column])[cfg.log_weight_column],
                           dtype=np.float64)

    if cfg.n_samples is not None and len(df) > cfg.n_samples:
        idx = np.sort(np.random.default_rng(cfg.seed).choice(len(df), cfg.n_samples, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
        log_w = None if log_w is None else log_w[idx]

    weights = None
    if log_w is not None:
        if not np.all(np.isfinite(log_w) | (log_w == -np.inf)):
            raise ValueError(f"{cfg.log_weight_column!r} has NaN or +inf log weights")
        weights = np.exp(log_w - np.max(log_w))
    return df, weights


def build_transformation(cfg: TGMMConfig, df: pd.DataFrame):
    """The tgm.Transformation for this event (identity if none is given).

    There is always one, because ``prior`` (and any carry columns) ride
    along as ignored columns; quantile maps are fit to this event's draws.
    """
    t = tgm()
    spec = cfg.transformation or TransformationSpec()
    quantile = None
    if cfg.quantiled_columns:
        quantile = t.QuantileTransformation(df[list(cfg.columns)].reset_index(drop=True),
                                            quantiled_columns=list(cfg.quantiled_columns),
                                            n_quantiles=cfg.n_quantiles)
    return t.Transformation(input_columns=list(cfg.columns),
                            forward_transformation=spec.forward_transformation,
                            transformed_columns=list(spec.transformed_columns or cfg.columns),
                            inverse_transformation=spec.inverse_transformation,
                            ignore_columns=list(cfg.carry_columns) + ["prior"],
                            extra_funcs=list(spec.extra_funcs) or None,
                            quantile_transformation=quantile)


def truncation_box(cfg: TGMMConfig, df: pd.DataFrame, transformation) -> tuple[np.ndarray, np.ndarray]:
    """``(a, b)`` in the fitted space, ordered like the image columns."""
    image = transformation.forward(df)
    a, b = [], []
    for inp, img in zip(transformation.input_columns, transformation.image_columns):
        if inp in cfg.quantiled_columns:
            lo = image[img].min() - cfg.quantile_padding
            hi = image[img].max() + cfg.quantile_padding
        elif cfg.limit_for(inp, img) is not None:
            lo, hi = cfg.limit_for(inp, img)
        else:
            raise KeyError(f"No limits for fitted column {img!r}; add it to TGMMConfig.limits")
        a.append(lo)
        b.append(hi)
    return np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)


def _fit_diagnostics(gmm) -> dict[str, Any]:
    d = getattr(gmm, "diagnostics", None)
    if d is None:
        return {}
    try:
        dd = d.to_dict()
    except AttributeError:
        return {}
    out = {}
    for key in ("backend", "method", "converged", "n_iterations", "final_average_loglikelihood"):
        if key in dd:
            v = dd[key]
            out[key] = v.item() if isinstance(v, np.generic) else v
    return out


@register
class TGMMFitter(BaseFitter):
    """Fits a TGMM per event: a default ``TGMMConfig``, per-event overrides,
    and the retry policy for fits with components far outside the box.

    A fit with a component whose mass inside the box (per covariance
    block) is below ``exp(logtp_cutoff)`` is retried with seeds
    ``seed+1, seed+2, ...`` up to ``max_attempts`` fits in total. The seed
    actually used is recorded in ``fit.diagnostics`` and does not enter the
    fit key, so a retried fit is not reported as stale.
    """

    kind: ClassVar[str] = "tgmm"
    fit_type: ClassVar[type] = TGMMFit

    def __init__(
        self,
        config: TGMMConfig,
        event_configs: Mapping[str, TGMMConfig] | None = None,
        *,
        max_attempts: int = 3,
        logtp_cutoff: float = LOGTP_CUTOFF,
        verbose: bool = True,
    ):
        if not isinstance(config, TGMMConfig):
            raise TypeError("config must be a TGMMConfig")
        self._default = config
        self._event_configs: dict[str, TGMMConfig] = {}
        for eventname, cfg in dict(event_configs or {}).items():
            self.set_event_config(eventname, cfg)
        if int(max_attempts) < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = int(max_attempts)
        self.logtp_cutoff = float(logtp_cutoff)
        self.verbose = bool(verbose)

    def __repr__(self):
        return (f"TGMMFitter(K={self._default.K}, columns={list(self._default.columns)}, "
                f"event_configs={sorted(self._event_configs)})")

    def __eq__(self, other):
        return isinstance(other, TGMMFitter) and self.to_dict() == other.to_dict()

    # ========================================================
    # RichDisplay
    # ========================================================

    def _display_math(self):
        return self._default._display_math()

    def _display_body(self):
        default = self._default.to_dict()
        rows = []
        for eventname, cfg in sorted(self._event_configs.items()):
            changed = {k: v for k, v in cfg.to_dict().items() if default.get(k) != v}
            rows.append([f"<code>{eventname}</code>",
                         ", ".join(f"<code>{k}={v!r}</code>" for k, v in changed.items()) or "(same as default)"])
        overrides = (html_table(["event", "differs from default"], rows) if rows
                     else '<div style="opacity:0.7">none</div>')
        return (section("Default config", self._default._repr_html_(), opened=True)
                + section("Per-event configs", overrides, opened=bool(rows), count=len(rows))
                + section("Retry policy", kv_table({
                    "max_attempts": self.max_attempts,
                    "logtp_cutoff": self.logtp_cutoff,
                    "rule": "retry with seed+1, +2, ... if a component has < exp(logtp_cutoff) "
                            "of its mass inside the box"})))

    # ========================================================
    # Configs
    # ========================================================

    @property
    def config(self) -> TGMMConfig:
        """The default config, used by events without their own."""
        return self._default

    @property
    def event_configs(self) -> dict[str, TGMMConfig]:
        return dict(self._event_configs)

    def set_default_config(self, config: TGMMConfig) -> None:
        if not isinstance(config, TGMMConfig):
            raise TypeError("config must be a TGMMConfig")
        self._default = config

    def update_default_config(self, **changes) -> None:
        self._default = self._default.with_changes(**changes)

    def config_for(self, eventname: str) -> TGMMConfig:
        return self._event_configs.get(eventname, self._default)

    def set_event_config(self, eventname: str, config: TGMMConfig | None) -> None:
        """Give an event its own config, or ``None`` to go back to the default."""
        if config is None:
            self._event_configs.pop(eventname, None)
        elif isinstance(config, TGMMConfig):
            self._event_configs[eventname] = config
        else:
            raise TypeError("config must be a TGMMConfig or None")

    def update_event_config(self, eventname: str, **changes) -> None:
        """E.g. ``fitter.update_event_config("ev7", K=12)``: copy its config with changes."""
        self.set_event_config(eventname, self.config_for(eventname).with_changes(**changes))

    # ========================================================
    # Fitter protocol
    # ========================================================

    def fit_key(self, eventname: str, source_key: str) -> str:
        return config_hash({"kind": self.kind, "config": self.config_for(eventname).fit_dict(),
                            "source": source_key})

    def dataproduct_options(self, eventname: str) -> dict[str, Any]:
        cfg = self.config_for(eventname)
        return {"analytic_columns": list(cfg.analytic_columns),
                "sampled_columns": list(cfg.sampled_columns),
                "N": cfg.N}

    def fit(self, posterior, *, eventname: str, source_key: str) -> TGMMFit:
        cfg = self.config_for(eventname)
        missing = cfg.missing_limits()
        if missing:
            raise KeyError(f"{eventname}: no limits for {missing}; add them to TGMMConfig.limits")

        df, weights = fitting_frame(posterior, cfg)
        transformation = build_transformation(cfg, df)
        a, b = truncation_box(cfg, df, transformation)
        kwargs = dict(data=df, N=cfg.K, a=a, b=b, cov=cfg.cov,
                      block_structure=None if cfg.block_structure is None else list(cfg.block_structure),
                      transformation=transformation, weights=weights)
        kwargs.update(cfg.fit_kwargs)
        sectors = sectors_from_block_structure(cfg.block_structure, len(cfg.columns))

        failed = []
        start = time.perf_counter()
        for attempt in range(self.max_attempts):
            seed = cfg.seed + attempt
            gmm = tgm().fit_gmm(seed=seed, **kwargs)
            problems = components_far_from_domain(gmm, sectors, self.logtp_cutoff)
            if not problems:
                break
            failed.append({"seed": seed, "problems": problems})
            if self.verbose:
                print(f"{eventname}: fit {attempt + 1}/{self.max_attempts} (seed={seed}) has "
                      f"components far from the domain {problems}")
        else:
            raise TGMMFitError(f"{eventname}: all {self.max_attempts} fits have components far "
                               "from the domain; try changing the event's config (e.g. K or seed)")

        diagnostics = {"seed": seed, "attempts": attempt + 1, "failed_attempts": failed,
                       "n_draws": len(df), "fit_seconds": round(time.perf_counter() - start, 3),
                       **_fit_diagnostics(gmm)}
        return TGMMFit(gmm, cfg, fit_key=self.fit_key(eventname, source_key), diagnostics=diagnostics)

    # ========================================================
    # Serialization
    # ========================================================

    def to_dict(self) -> dict[str, Any]:
        return {"default": self._default.to_dict(),
                "event_configs": {e: c.to_dict() for e, c in self._event_configs.items()},
                "max_attempts": self.max_attempts,
                "logtp_cutoff": self.logtp_cutoff,
                "verbose": self.verbose}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TGMMFitter":
        return cls(TGMMConfig.from_dict(d["default"]),
                   {e: TGMMConfig.from_dict(c) for e, c in d.get("event_configs", {}).items()},
                   max_attempts=d.get("max_attempts", 3),
                   logtp_cutoff=d.get("logtp_cutoff", LOGTP_CUTOFF),
                   verbose=d.get("verbose", True))

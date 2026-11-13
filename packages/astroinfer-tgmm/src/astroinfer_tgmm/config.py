"""What determines a TGMM fit (and its data product), as JSON-able config."""
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from astroinfer.display import RichDisplay, html_table, kv_table, section
from astroinfer.io import json_clean


def _names(value, what) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{what} must be a sequence of names, not a string")
    return tuple(str(v) for v in value)


@dataclass(frozen=True)
class TransformationSpec(RichDisplay):
    """Optional coordinate change applied to ``TGMMConfig.columns`` before fitting.

    Maps take one argument per column and return a tuple with one entry per
    column. They must be ``'lambda ...'`` strings (helpers go in
    ``extra_funcs``) so the config can be saved. ``transformed_columns``
    defaults to the input column names.
    """

    forward_transformation: str | None = None
    inverse_transformation: str | None = None
    transformed_columns: tuple[str, ...] | None = None
    extra_funcs: tuple[str, ...] = ()

    def _display_title(self):
        return "Transformation"

    def _display_body(self):
        return kv_table({"forward": self.forward_transformation, "inverse": self.inverse_transformation,
                         "transformed columns": self.transformed_columns,
                         **({"extra funcs": list(self.extra_funcs)} if self.extra_funcs else {})})

    def __post_init__(self):
        for name in ("forward_transformation", "inverse_transformation"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a 'lambda ...' string (or None) so the config can be saved")
        if (self.forward_transformation is None) != (self.inverse_transformation is None):
            raise ValueError("give both forward_transformation and inverse_transformation, or neither")
        if self.transformed_columns is not None:
            object.__setattr__(self, "transformed_columns",
                               _names(self.transformed_columns, "transformed_columns"))
        extra = (self.extra_funcs,) if isinstance(self.extra_funcs, str) else self.extra_funcs
        object.__setattr__(self, "extra_funcs", tuple(extra))


# Fields that only shape the data product: changing them never makes a fit stale.
DATAPRODUCT_FIELDS = ("analytic_columns", "sampled_columns", "N")


@dataclass(frozen=True)
class TGMMConfig(RichDisplay):
    """Everything that determines one event's TGMM fit and its data product.

    Fit:
        K, cov, block_structure   mixture size, "diag"/"full", covariance blocks (one label per column)
        columns                   fitted variables (theta fields or extras), in order
        carry_columns             other per-draw variables carried through the fit unfitted.
                                  The PE prior of ``columns`` is always carried as ``prior``.
        quantiled_columns         quantile-transformed per event before fitting (subset of columns)
        n_quantiles, quantile_padding
        limits                    {fitted column: (lo, hi)} truncation box, in the fitted space
                                  (with a transformation, keyed by its transformed_columns).
                                  Not needed for quantiled columns.
        transformation            optional TransformationSpec of ``columns``
        n_samples                 subsample each posterior to this many draws first
        log_weight_column         per-draw log weights (sample_stats or extras) for a weighted fit
        seed                      subsampling and fit seed (retries use seed+1, seed+2, ...)
        fit_kwargs                passed to ``tgm.fit_gmm`` (JSON-serializable)
    Data product only (see DATAPRODUCT_FIELDS):
        analytic_columns, sampled_columns, N (draws per component for sampled columns)
    """

    columns: tuple[str, ...]
    K: int = 1
    cov: str = "diag"
    block_structure: tuple[int, ...] | None = None
    carry_columns: tuple[str, ...] = ()
    quantiled_columns: tuple[str, ...] = ()
    n_quantiles: int = 1_000
    quantile_padding: float = 1e-3
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    transformation: TransformationSpec | None = None
    n_samples: int | None = None
    log_weight_column: str | None = None
    seed: int = 0
    fit_kwargs: dict[str, Any] = field(default_factory=lambda: {"method": "hybrid", "backend": "jax"})
    analytic_columns: tuple[str, ...] = ()
    sampled_columns: tuple[str, ...] = ()
    N: int = 1000

    def __post_init__(self):
        set_ = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        for name in ("columns", "carry_columns", "quantiled_columns",
                     "analytic_columns", "sampled_columns"):
            set_(name, _names(getattr(self, name), name))
        if isinstance(self.transformation, dict):
            set_("transformation", TransformationSpec(**self.transformation))
        set_("limits", {str(k): (float(lo), float(hi)) for k, (lo, hi) in dict(self.limits).items()})
        if self.block_structure is not None:
            set_("block_structure", tuple(int(i) for i in self.block_structure))
        set_("K", int(self.K))
        set_("N", int(self.N))
        set_("seed", int(self.seed))
        set_("fit_kwargs", dict(self.fit_kwargs))

        cols = set(self.columns)
        if not self.columns:
            raise ValueError("TGMMConfig.columns is empty; list the columns to fit")
        if len(cols) != len(self.columns):
            raise ValueError("TGMMConfig.columns contains duplicates")
        if self.K < 1:
            raise ValueError("K must be at least 1")
        if self.N < 1:
            raise ValueError("N must be positive")
        if self.cov not in ("diag", "full"):
            raise ValueError("cov must be 'diag' or 'full'")
        if self.n_samples is not None and int(self.n_samples) < 1:
            raise ValueError("n_samples must be positive (or None)")
        if self.block_structure is not None and len(self.block_structure) != len(self.columns):
            raise ValueError(f"block_structure has {len(self.block_structure)} labels "
                             f"but columns has {len(self.columns)}")
        for name in ("quantiled_columns", "analytic_columns", "sampled_columns"):
            unknown = set(getattr(self, name)) - cols
            if unknown:
                raise ValueError(f"{name} {sorted(unknown)} are not in columns")
        both = set(self.analytic_columns) & set(self.sampled_columns)
        if both:
            raise ValueError(f"columns {sorted(both)} are both analytic and sampled")
        overlap = set(self.carry_columns) & cols
        if overlap:
            raise ValueError(f"columns {sorted(overlap)} are both fitted and carried")
        if "prior" in cols or "prior" in self.carry_columns:
            raise ValueError("'prior' is carried automatically; don't list it")
        spec = self.transformation
        if spec is not None and spec.transformed_columns is not None \
                and len(spec.transformed_columns) != len(self.columns):
            raise ValueError(f"transformation.transformed_columns has {len(spec.transformed_columns)} "
                             f"entries but columns has {len(self.columns)}")
        try:
            json.dumps(self.fit_kwargs)
        except TypeError as exc:
            raise TypeError(f"fit_kwargs must be JSON-serializable so the config can be saved: {exc}") from exc
        if "seed" in self.fit_kwargs:
            raise ValueError("set TGMMConfig.seed, not fit_kwargs['seed']")

    # ========================================================
    # Derived views
    # ========================================================

    @property
    def image_columns(self) -> list[str]:
        """Names of the fitted coordinates (what the truncation box and kernels refer to)."""
        spec = self.transformation
        names = list(spec.transformed_columns or self.columns) if spec else list(self.columns)
        return [f"{c}_quantile" if inp in self.quantiled_columns else c
                for inp, c in zip(self.columns, names)]

    def limit_for(self, input_column: str, image_column: str) -> tuple[float, float] | None:
        """The truncation interval of one fitted coordinate, or None if not given.

        Limits are in the fitted space, so they are looked up by the fitted
        (image) name only. Without a transformation that is the input name.
        """
        return self.limits.get(image_column)

    def missing_limits(self) -> list[str]:
        return [img for inp, img in zip(self.columns, self.image_columns)
                if inp not in self.quantiled_columns and self.limit_for(inp, img) is None]

    def fit_dict(self) -> dict[str, Any]:
        """The fit-relevant part (what the fit key hashes)."""
        d = self.to_dict()
        for name in DATAPRODUCT_FIELDS:
            d.pop(name)
        return d

    def with_changes(self, **changes) -> "TGMMConfig":
        return replace(self, **changes)

    # ========================================================
    # RichDisplay
    # ========================================================

    def _display_title(self):
        return "TGMM config"

    def _display_math(self):
        return (rf"p(x)\approx\sum_{{k=1}}^{{{self.K}}} w_k\,"
                r"\mathcal{N}_{[a,b]}\!\left(T(x)\mid\mu_k,\Sigma_k\right)")

    def _display_body(self):
        box = []
        for inp, img in zip(self.columns, self.image_columns):
            if inp in self.quantiled_columns:
                interval = "data range ± " + str(self.quantile_padding)
            else:
                lim = self.limit_for(inp, img)
                interval = "<em>missing</em>" if lim is None else f"[{lim[0]:g}, {lim[1]:g}]"
            role = ("analytic" if inp in self.analytic_columns
                    else "sampled" if inp in self.sampled_columns else "")
            box.append([f"<code>{inp}</code>", f"<code>{img}</code>", interval, role])
        fit = kv_table({
            "K": self.K, "cov": self.cov, "block_structure": self.block_structure,
            "n_samples": self.n_samples, "seed": self.seed,
            "carry_columns": list(self.carry_columns) or "()",
            "log_weight_column": self.log_weight_column,
            "fit_kwargs": self.fit_kwargs,
        })
        parts = [html_table(["column", "fitted as", "truncation", "data product"], box), fit]
        if self.transformation is not None:
            parts.append(self.transformation._repr_html_())
        parts.append(section("Data product", kv_table({
            "analytic_columns": list(self.analytic_columns), "sampled_columns": list(self.sampled_columns),
            "N (draws per component)": self.N}), opened=True))
        return "".join(parts)

    # ========================================================
    # Serialization
    # ========================================================

    def to_dict(self) -> dict[str, Any]:
        return json_clean(asdict(self))

    @classmethod
    def from_dict(cls, d) -> "TGMMConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown TGMMConfig fields {sorted(unknown)}")
        return cls(**d)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "TGMMConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

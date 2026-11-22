"""The ``raw`` representation: the posterior samples themselves.

Every Catalog has exactly one raw representation. Its fits hold the
PosteriorData that every other representation is fit from, so saving a
catalog's raw representation is what saves its posteriors.

    RawSamples(columns, N)   fitter: which columns the raw data product
                             exports, and how many draws per event
    RawFit(posterior)        one event's posterior
"""
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np

from ..display import code, kv_table, section
from ..io import json_clean, load_arrays, read_json, save_arrays, type_path, write_json
from ..parameters import Parameters
from ..posterior import PosteriorData
from .base import BaseFit, BaseFitter, LoadContext
from .registry import register

_FORMAT_VERSION = 1


class RawFit(BaseFit):
    """One event's posterior draws; ``fit_key`` is their content hash."""

    def __init__(self, posterior: PosteriorData, *, fit_key: str | None = None):
        if not isinstance(posterior, PosteriorData):
            raise TypeError("posterior must be a PosteriorData instance")
        self.posterior = posterior
        self._fit_key = fit_key

    def __repr__(self):
        return f"RawFit(n_draws={self.posterior.batch_shape[0]})"

    @property
    def fit_key(self) -> str:
        if self._fit_key is None:
            self._fit_key = self.posterior.content_hash()
        return self._fit_key

    def _display_title(self):
        return "Raw posterior"

    def _display_math(self):
        return self.posterior._display_math()

    def _display_body(self):
        return (kv_table({"draws": self.posterior.batch_shape[0],
                          "content hash": self.fit_key[:12] + "..."})
                + section("Posterior", self.posterior._repr_html_(), opened=True))

    # ========================================================
    # Data product
    # ========================================================

    def dataproduct(self, *, N: int, seed: int, columns: Sequence[str]) -> dict[str, Any]:
        """``{column: (N,), "prior": (N,)}``: draws resampled to N.

        With N equal to the number of draws they are returned in order;
        otherwise N indices are drawn (with replacement only if N is larger).
        ``prior`` is the PE prior of ``columns`` alone.
        """
        posterior = self.posterior
        size = posterior.batch_shape[0]
        if N == size:
            indices = np.arange(size)
        else:
            rng = np.random.default_rng(seed)
            indices = rng.choice(size, size=N, replace=N > size)

        variables = posterior.variables(columns)
        out = {
            name: jnp.take(jnp.asarray(variables[name]), indices, axis=0)
            for name in columns
        }
        prior = posterior.marginal_prior(columns)
        if np.shape(prior) != (size,):
            raise ValueError(f"prior has shape {np.shape(prior)}, expected {(size,)}")
        out["prior"] = jnp.take(jnp.asarray(prior), indices, axis=0)
        return out

    # ========================================================
    # Saving
    # ========================================================

    def save(self, directory: Path) -> None:
        p = self.posterior
        groups = {
            "theta": p.theta.to_flat_dict(),
            "sample_stats": dict(p.sample_stats),
            "extras": dict(p.extras),
        }

        observed = None
        if isinstance(p.observed_data, Parameters):
            od = p.observed_data
            groups["observed_data"] = od.to_flat_dict()
            observed = {"format": "parameters", "type": type_path(type(od))}
        elif p.observed_data is not None:
            try:
                observed = {"format": "json", "value": json_clean(p.observed_data)}
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    f"observed_data of type {type(p.observed_data).__name__} "
                    f"is not saved ({exc})",
                    stacklevel=2,
                )

        metadata, dropped = {}, []
        for key, value in p.metadata.items():
            try:
                metadata[str(key)] = json_clean(value)
            except (TypeError, ValueError):
                dropped.append(str(key))
        if dropped:
            warnings.warn(f"metadata keys {dropped} are not JSON-serializable and are not saved",
                          stacklevel=2)

        save_arrays(directory / "posterior.h5", groups)
        write_json(directory / "posterior.json", {
            "format_version": _FORMAT_VERSION,
            "parameter_type": type_path(p.parameter_type),
            "content_hash": self.fit_key,
            "observed_data": observed,
            "metadata": metadata,
        })

    @classmethod
    def load(cls, directory: Path, ctx: LoadContext) -> "RawFit":
        info = read_json(directory / "posterior.json")
        groups = load_arrays(directory / "posterior.h5")
        parameter_type = ctx.resolve(info["parameter_type"])

        # Arrays stay NumPy (full stored precision) until JAX uses them.
        theta = parameter_type.from_dict(groups["theta"])

        observed = info.get("observed_data")
        if observed is None:
            observed_data = None
        elif observed["format"] == "parameters":
            observed_data = ctx.resolve(observed["type"]).from_dict(groups["observed_data"])
        else:
            observed_data = observed["value"]

        posterior = PosteriorData(
            theta=theta,
            sample_stats=groups.get("sample_stats", {}),
            extras=groups.get("extras", {}),
            metadata=info.get("metadata", {}),
            observed_data=observed_data,
        )
        return cls(posterior, fit_key=info["content_hash"])


@register
@dataclass(frozen=True)
class RawSamples(BaseFitter):
    """The raw representation's settings (all data-product only).

    columns: exported variables (theta fields or extras), in order.
    N:       default draws per event in the data product.
    """

    kind: ClassVar[str] = "raw"
    fit_type: ClassVar[type] = RawFit

    columns: tuple[str, ...]
    N: int = 2000

    def __post_init__(self):
        if isinstance(self.columns, str):
            raise TypeError("columns must be a sequence of names, not a string")
        object.__setattr__(self, "columns", tuple(self.columns))
        if not self.columns:
            raise ValueError("RawSamples.columns cannot be empty")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("RawSamples.columns contains duplicates")
        if int(self.N) <= 0:
            raise ValueError("N must be positive")
        object.__setattr__(self, "N", int(self.N))

    def _display_title(self):
        return "Raw samples"

    def _display_body(self):
        return ("<div>Kind: " + code(self.kind) + "</div>"
                + kv_table({"columns": ", ".join(self.columns), "N": self.N})
                + '<div style="opacity:0.7;font-size:0.9em">Data product: '
                  "each column and the PE prior of the columns, resampled to N draws per event "
                  "(E, N).</div>")

    def check(self, posterior: PosteriorData, eventname: str) -> None:
        missing = set(self.columns) - set(posterior.variable_names)
        if missing:
            raise KeyError(f"Event {eventname!r} is missing columns {sorted(missing)}")

    def fit(self, posterior, *, eventname, source_key=None) -> RawFit:
        self.check(posterior, eventname)
        return RawFit(posterior, fit_key=source_key)

    def fit_key(self, eventname, source_key) -> str:
        return source_key

    def dataproduct_options(self, eventname) -> dict[str, Any]:
        return {"columns": list(self.columns), "N": self.N}

    def to_dict(self) -> dict[str, Any]:
        return {"columns": list(self.columns), "N": self.N}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RawSamples":
        return cls(columns=tuple(d["columns"]), N=d.get("N", 2000))

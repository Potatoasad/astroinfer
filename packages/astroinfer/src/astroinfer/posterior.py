"""Posterior draws of typed parameters, plus their inference bookkeeping."""
import hashlib
from dataclasses import dataclass, field, fields
from html import escape
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .display import RichDisplay, array_node, inline_row, shape_badge
from .parameters import Parameters

# DataFrame columns that are inference quantities rather than derived variables.
_SAMPLE_STAT_NAMES = {
    "log_likelihood",
    "log_posterior",
    "log_prior",
    "weight",
    "weights",
    "log_weight",
    "log_weights",
}


def _is_sample_stat(name: str) -> bool:
    return (
        name == "prior"
        or name.startswith("log_prior[")
        or name in _SAMPLE_STAT_NAMES
    )


@dataclass(repr=False)
class PosteriorData[P: Parameters](RichDisplay):
    """
    Posterior draws and associated inference information.

    theta:
        User-defined Parameters object, batched over posterior draws.

    sample_stats:
        Sample-aligned quantities such as prior densities,
        log likelihoods, and weights.

    extras:
        Other sample-aligned quantities, including derived parameters.

    metadata:
        Run-level information, not indexed over posterior draws.

    observed_data:
        The original observation d, shared across all draws.
    """

    theta: P
    sample_stats: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_data: Any = None

    # ========================================================
    # Validation and properties
    # ========================================================

    def __post_init__(self):
        shape = self.batch_shape

        parameter_names = {
            f.name for f in fields(self.theta)
        }
        stats_names = set(self.sample_stats)
        extra_names = set(self.extras)

        if (
            parameter_names & (stats_names | extra_names)
            or stats_names & extra_names
        ):
            raise ValueError(
                "Overlapping parameter, sample_stats, or extras names"
            )

        for name, value in {
            **self.sample_stats,
            **self.extras,
        }.items():
            actual = np.shape(value)

            if actual[:len(shape)] != shape:
                raise ValueError(
                    f"{name!r} must start with sample shape "
                    f"{shape}; got {actual}"
                )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.theta.batch_shape

    @property
    def parameter_type(self) -> type[P]:
        return type(self.theta)

    @property
    def variable_names(self) -> list[str]:
        """Names usable as exported columns: theta fields, then extras."""
        return [f.name for f in fields(self.theta)] + list(self.extras)

    def variables(self, names) -> dict[str, Any]:
        """``{name: per-draw array}`` for theta fields, extras or sample_stats."""
        theta = self.theta.broadcast()
        out = {}
        for name in names:
            if name in self.extras:
                out[name] = self.extras[name]
            elif name in self.sample_stats:
                out[name] = self.sample_stats[name]
            elif name in {f.name for f in fields(theta)}:
                out[name] = getattr(theta, name)
            else:
                raise KeyError(
                    f"{name!r} is not a parameter, extra or sample stat; "
                    f"available: {self.variable_names + list(self.sample_stats)}"
                )
        return out

    # ========================================================
    # Prior of a subset of the variables
    # ========================================================

    def marginal_prior(self, columns) -> jnp.ndarray:
        """PE prior density of ``columns`` alone (the rest marginalized out).

        Uses the stored full ``prior`` when ``columns`` covers every
        physical parameter. For a strict subset it multiplies the
        *complete* independent ``log_prior[...]`` factors over the subset;
        a joint factor that is only partly covered cannot be marginalized
        and raises.
        """
        stats = self.sample_stats
        physical = {f.name for f in fields(self.theta)}
        selected = set(columns) & physical
        extras = set(columns) - physical

        if extras and selected != physical:
            raise ValueError(
                "Exporting derived/extra columns while omitting physical "
                "parameters requires an explicit transformed prior"
            )

        if selected == physical and "prior" in stats:
            return jnp.asarray(stats["prior"])

        used: set[str] = set()
        logp = None
        for key, value in stats.items():
            if not (key.startswith("log_prior[") and key.endswith("]")):
                continue
            variables = {v.strip() for v in key[len("log_prior["):-1].split(",")}
            if "" in variables:
                raise ValueError(f"Invalid prior factor {key!r}")
            overlap = variables & selected
            if overlap and not variables <= selected:
                raise ValueError(
                    f"Cannot export {sorted(columns)}: {key!r} is a joint "
                    "prior factor and cannot be partially marginalized"
                )
            if variables <= selected:
                if used & variables:
                    raise ValueError(f"Overlapping PE prior factors at {key!r}")
                used.update(variables)
                logp = jnp.asarray(value) if logp is None else logp + value

        if used != selected:
            missing = selected - used
            raise ValueError(
                f"Missing PE prior factors for {sorted(missing)}. "
                "Provide 'prior' for all physical parameters, or complete "
                "log_prior[...] factors for the requested subset."
            )
        if logp is None:
            raise ValueError("No prior factors found")
        return jnp.exp(logp)

    # ========================================================
    # Identity
    # ========================================================

    def content_hash(self) -> str:
        """SHA-256 of every per-draw array and the parameter type.

        Fits record this as their source, so replacing an event's
        posterior makes its fits stale (also across save/load).
        """
        h = hashlib.sha256()
        h.update(
            f"{self.parameter_type.__module__}:"
            f"{self.parameter_type.__qualname__}".encode()
        )
        groups = {
            "theta": {f.name: getattr(self.theta, f.name) for f in fields(self.theta)},
            "sample_stats": self.sample_stats,
            "extras": self.extras,
        }
        # Numeric arrays are hashed at 64-bit width, so the same values give
        # the same hash whether JAX holds them as float32 or float64.
        canonical = {"f": "<f8", "i": "<i8", "u": "<u8", "b": "|b1", "c": "<c16"}
        for group, values in groups.items():
            for name in sorted(values):
                arr = np.asarray(values[name])
                kind = arr.dtype.kind
                h.update(f"{group}/{name}/{kind}/{arr.shape}".encode())
                if kind in canonical:
                    h.update(np.ascontiguousarray(arr, dtype=canonical[kind]).tobytes())
                else:
                    h.update("\x00".join(map(str, arr.ravel())).encode())
        return h.hexdigest()

    # ========================================================
    # Indexing
    # ========================================================

    def __getitem__(self, key):
        # Access individual variables by name.
        if isinstance(key, str):
            if key in {f.name for f in fields(self.theta)}:
                return self.theta[key]

            if key in self.sample_stats:
                return self.sample_stats[key]

            if key in self.extras:
                return self.extras[key]

            raise KeyError(key)

        if not self.batch_shape:
            raise IndexError(
                "Cannot index a single unbatched posterior draw"
            )

        # Index all sample-aligned quantities together.
        return type(self)(
            theta=self.theta[key],
            sample_stats={
                k: v[key] for k, v in self.sample_stats.items()
            },
            extras={
                k: v[key] for k, v in self.extras.items()
            },
            metadata=self.metadata.copy(),
            observed_data=self.observed_data,
        )

    # ========================================================
    # DataFrame conversion
    # ========================================================

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameter_type: type[P],
        *,
        metadata: dict[str, Any] | None = None,
        observed_data: Any = None,
    ) -> "PosteriorData[P]":

        df = pd.DataFrame(df)

        names = {
            f.name for f in fields(parameter_type)
        }

        missing = names - set(df.columns)
        if missing:
            raise KeyError(
                f"Missing posterior parameters: {sorted(missing)}"
            )

        # Construct the user's original Parameters class.
        theta = parameter_type.from_dict({
            name: jnp.asarray(df[name].to_numpy())
            for name in names
        })

        sample_stats = {}
        extras = {}

        for name in df.columns:
            if name in names:
                continue

            raw = df[name].to_numpy(copy=True)

            try:
                values = jnp.asarray(raw)
            except (TypeError, ValueError):
                values = raw

            if _is_sample_stat(name):
                sample_stats[name] = values
            else:
                extras[name] = values

        return cls(
            theta=theta,
            sample_stats=sample_stats,
            extras=extras,
            metadata=dict(metadata or {}),
            observed_data=observed_data,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert one-dimensional scalar-valued posterior draws
        into a Pandas DataFrame.
        """
        if len(self.batch_shape) != 1:
            raise ValueError(
                "to_dataframe() requires one draw axis; "
                f"got batch_shape={self.batch_shape}"
            )

        theta = self.theta.broadcast()

        values = {
            **{
                f.name: getattr(theta, f.name)
                for f in fields(theta)
            },
            **self.sample_stats,
            **self.extras,
        }

        for name, value in values.items():
            if np.shape(value) != self.batch_shape:
                raise ValueError(
                    f"{name!r} has shape {np.shape(value)}. "
                    "DataFrame columns must have one scalar per draw."
                )

        return pd.DataFrame({
            name: np.asarray(value)
            for name, value in values.items()
        })

    # ========================================================
    # Python representation
    # ========================================================

    def __repr__(self) -> str:
        return (
            f"PosteriorData[{self.parameter_type.__name__}]("
            f"batch_shape={self.batch_shape}, "
            f"sample_stats={list(self.sample_stats)}, "
            f"extras={list(self.extras)})"
        )

    # ========================================================
    # RichDisplay
    # ========================================================

    def _display_title(self) -> str:
        return "Posterior Samples"

    def _display_math(self) -> str:
        theta = getattr(
            self.parameter_type, "_latex", r"\theta"
        )

        data = (
            getattr(type(self.observed_data), "_latex", "d")
            if self.observed_data is not None
            else "d"
        )

        return rf"p_{{\mathrm{{PE}}}}({theta}\mid {data})"

    def _display_array_group(
        self,
        title: str,
        values: dict,
        *,
        opened: bool = False,
    ) -> str:

        if not values:
            return ""

        rows = [
            inline_row(
                "<strong>" + escape(str(name)) + ":</strong>",
                array_node(value),
                shape_badge(value),
            )
            for name, value in values.items()
        ]

        return (
            "<details"
            + (" open" if opened else "")
            + ' style="margin:0.4em 0">'

            '<summary style="cursor:pointer">'
            "<strong>"
            + escape(title)
            + "</strong>"
            f' <span style="opacity:0.65">({len(values)})</span>'
            "</summary>"

            '<div style="padding:0.3em 0 0.3em 1.2em">'
            + "".join(rows)
            + "</div></details>"
        )

    def _display_metadata(self) -> str:

        if not self.metadata:
            return ""

        rows = []

        for name, value in self.metadata.items():
            text = repr(value)

            # Short metadata values appear directly.
            if len(text) <= 100:
                content = (
                    "<code>"
                    + escape(text)
                    + "</code>"
                )

            # Long metadata values are expandable.
            else:
                content = (
                    '<details style="display:inline-block;'
                    'max-width:100%">'

                    '<summary style="cursor:pointer">'
                    "<code>"
                    + escape(text[:70])
                    + "...</code></summary>"

                    '<pre style="max-height:200px;'
                    'overflow:auto;white-space:pre">'
                    + escape(text)
                    + "</pre></details>"
                )

            rows.append(
                inline_row(
                    "<strong>"
                    + escape(str(name))
                    + ":</strong>",
                    content,
                )
            )

        return (
            '<details open style="margin:0.4em 0">'

            '<summary style="cursor:pointer">'
            "<strong>Metadata</strong>"
            f' <span style="opacity:0.65">'
            f"({len(self.metadata)})</span>"
            "</summary>"

            '<div style="padding:0.3em 0 0.3em 1.2em">'
            + "".join(rows)
            + "</div></details>"
        )

    def _display_body(self) -> str:
        body = []

        # Number of posterior draws / batch shape.
        body.append(
            '<div style="opacity:0.7;font-size:0.9em;'
            'margin-bottom:0.5em">'
            "Sample shape: <code>"
            + escape(str(self.batch_shape))
            + "</code></div>"
        )

        # Reuse the complete Parameters HTML representation.
        body.append(self.theta._repr_html_())

        # Inference quantities.
        body.append(
            self._display_array_group(
                "Sample Statistics",
                self.sample_stats,
                opened=True,
            )
        )

        # Derived parameters and other sample-aligned fields.
        body.append(
            self._display_array_group(
                "Derived / Extra Variables",
                self.extras,
                opened=False,
            )
        )

        # Run-level information.
        body.append(self._display_metadata())

        # Original observed data, if available.
        if self.observed_data is not None:

            if hasattr(self.observed_data, "_repr_html_"):
                content = self.observed_data._repr_html_()
            else:
                content = (
                    "<pre>"
                    + escape(repr(self.observed_data))
                    + "</pre>"
                )

            body.append(
                '<div style="border-top:1px solid #8884;'
                'padding-top:0.5em;margin-top:0.5em">'
                + content
                + "</div>"
            )

        return "".join(body)

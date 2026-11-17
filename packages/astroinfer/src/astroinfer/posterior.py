"""Posterior draws of typed parameters, plus their inference bookkeeping."""
import hashlib
import inspect
from dataclasses import dataclass, field, fields
from html import escape
from math import prod
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .display import RichDisplay, array_node, inline_row, shape_badge
from .parameters import Parameters

# Names of quantities that describe the inference rather than the model:
# densities, weights, and the sampler's own per-draw statistics.
_SAMPLE_STAT_NAMES = {
    "log_likelihood",
    "log_posterior",
    "log_prior",
    "weight",
    "weights",
    "log_weight",
    "log_weights",
    # NUTS extra fields reported by gravpop.
    "potential_energy",
    "energy",
    "num_steps",
    "accept_prob",
    "diverging",
    "adapt_state.step_size",
    # Population-likelihood diagnostics reported by gravpop.
    "N_eff_min",
    "N_eff_mean",
    "N_eff_min_fraction",
    "total_log_variance",
    # The chain each draw came from, added by stack_samples().
    "chain",
}

# Names ArviZ uses for NUTS statistics in its sample_stats group.
_ARVIZ_STAT_NAMES = {
    "accept_prob": "acceptance_rate",
    "num_steps": "n_steps",
    "adapt_state.step_size": "step_size",
}

OBSERVATION_DIM = "observation"


def _is_sample_stat(name: str) -> bool:
    return (
        name == "prior"
        or name.startswith("log_prior[")
        or name in _SAMPLE_STAT_NAMES
    )


def _is_scalar_index(i) -> bool:
    return isinstance(i, (int, np.integer))


def _indexed_dims(dims: tuple[str, ...], key) -> tuple[str, ...]:
    """Sample dims left after indexing with ``key``.

    An integer removes its axis; a slice or a 1-D index array keeps it.
    Axes beyond the end of ``key`` are kept.
    """
    key = key if isinstance(key, tuple) else (key,)
    kept = [d for d, k in zip(dims, key) if not _is_scalar_index(k)]
    return tuple(kept) + tuple(dims[len(key):])


def _core_dims(name, spec, value, n_batch):
    """Dim names for one parameter field after its batch axes."""
    ndim = np.ndim(value) - n_batch
    names = tuple(spec.index)
    if len(names) == ndim:
        return names
    return tuple(f"{name}_dim_{i}" for i in range(ndim))


def _summary_rows(name, values, chained, prob, expand):
    """Rows of ``summary()`` for one variable: ``[(label, row), ...]``.

    values: array with axes (chain, draw, *core) if ``chained`` else
    (draw, *core).
    """
    from numpyro.diagnostics import summary as numpyro_summary

    values = np.asarray(values)
    if values.dtype.kind == "b":
        values = values.astype(np.float64)
    grouped = values if chained else values[None]
    stats = numpyro_summary({name: grouped}, prob=prob, group_by_chain=True)[name]
    lo, hi = (k for k in stats if k.endswith("%"))
    core = grouped.shape[2:]

    columns = {
        "mean": stats["mean"],
        "sd": stats["std"],
        lo: stats[lo],
        hi: stats[hi],
        "ess": stats["n_eff"],
        "r_hat": stats["r_hat"] if chained else np.full(core, np.nan),
    }

    if not core:
        return [(name, {**{k: float(v) for k, v in columns.items()}, "size": 1})]

    if expand:
        rows = []
        for idx in np.ndindex(*core):
            label = f"{name}[{','.join(map(str, idx))}]"
            rows.append(
                (label, {**{k: float(np.asarray(v)[idx]) for k, v in columns.items()},
                         "size": 1})
            )
        return rows

    # One row for the whole field: the worst diagnostics over its elements.
    r_hat = np.asarray(columns["r_hat"])
    return [(name, {
        "mean": np.nan, "sd": np.nan, lo: np.nan, hi: np.nan,
        "ess": float(np.min(columns["ess"])),
        "r_hat": float(np.nanmax(r_hat)) if np.isfinite(r_hat).any() else np.nan,
        "size": int(prod(core)),
    })]


@dataclass(repr=False)
class PosteriorData[P: Parameters](RichDisplay):
    """
    Posterior draws and associated inference information.

    theta:
        User-defined Parameters object, batched over posterior draws.
        For parameter estimation this is theta; for population inference
        it is the hyperparameters Lambda.

    sample_stats:
        Sample-aligned quantities such as prior densities,
        log likelihoods, weights and sampler statistics.

    extras:
        Other sample-aligned quantities, including derived parameters.

    metadata:
        Run-level information, not indexed over posterior draws.

    observed_data:
        The original observation d, shared across all draws.

    local:
        Joint population inference only: the per-observation parameters
        theta_i, a Parameters object with batch shape
        ``batch_shape + (E,)``. The last axis is named "observation".

    sample_dims:
        Names of the sample axes, one per batch axis of ``theta``, e.g.
        ``("chain", "draw")``. Defaults to ``("draw",)`` for one batch axis.
    """

    theta: P
    sample_stats: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_data: Any = None
    local: Parameters | None = None
    sample_dims: tuple[str, ...] | None = None

    # ========================================================
    # Validation and properties
    # ========================================================

    def __post_init__(self):
        shape = self.batch_shape

        if self.sample_dims is None:
            if len(shape) > 1:
                raise ValueError(
                    f"theta has batch shape {shape}; name its axes with "
                    "sample_dims, e.g. sample_dims=('chain', 'draw')"
                )
            self.sample_dims = ("draw",) * len(shape)
        self.sample_dims = tuple(self.sample_dims)

        if len(self.sample_dims) != len(shape):
            raise ValueError(
                f"sample_dims {self.sample_dims} must name every batch "
                f"axis of theta, batch shape {shape}"
            )
        if len(set(self.sample_dims)) != len(self.sample_dims):
            raise ValueError(f"Repeated sample dim in {self.sample_dims}")

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

        if self.local is not None:
            local_shape = self.local.batch_shape
            if (
                len(local_shape) != len(shape) + 1
                or local_shape[:len(shape)] != shape
            ):
                raise ValueError(
                    f"local must have batch shape {shape} + (E,); "
                    f"got {local_shape}"
                )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.theta.batch_shape

    @property
    def sizes(self) -> dict[str, int]:
        """``{sample dim: length}``, e.g. ``{"chain": 4, "draw": 1000}``."""
        return dict(zip(self.sample_dims, self.batch_shape))

    @property
    def num_samples(self) -> int:
        """Total number of draws, over all sample axes."""
        return prod(self.batch_shape)

    @property
    def parameter_type(self) -> type[P]:
        return type(self.theta)

    @property
    def local_dims(self) -> tuple[str, ...] | None:
        """Batch dims of ``local``: the sample dims, then "observation"."""
        if self.local is None:
            return None
        return self.sample_dims + (OBSERVATION_DIM,)

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
        if self.local is not None:
            groups["local"] = self.local.to_dict()
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
            local=None if self.local is None else self.local[key],
            sample_dims=_indexed_dims(self.sample_dims, key),
        )

    def isel(self, **indexers):
        """Index sample axes by name, e.g. ``isel(chain=0, draw=slice(500, None))``."""
        unknown = set(indexers) - set(self.sample_dims)
        if unknown:
            raise KeyError(
                f"No sample dims {sorted(unknown)}; sample_dims={self.sample_dims}"
            )
        return self[tuple(indexers.get(d, slice(None)) for d in self.sample_dims)]

    def stack_samples(self) -> "PosteriorData[P]":
        """Merge all sample axes into one ``"draw"`` axis (C order).

        Core axes, and the "observation" axis of ``local``, are unchanged.
        When there is a "chain" axis, the chain of every draw is kept as
        ``sample_stats["chain"]``.
        """
        shape = self.batch_shape
        if self.sample_dims == ("draw",):
            return self
        n = prod(shape)
        k = len(shape)

        def flat(v):
            # NumPy and JAX arrays both have .reshape; keep whichever it is.
            v = v if hasattr(v, "reshape") else np.asarray(v)
            return v.reshape((n,) + v.shape[k:])

        sample_stats = {name: flat(v) for name, v in self.sample_stats.items()}
        if "chain" in self.sample_dims and "chain" not in sample_stats:
            axis = self.sample_dims.index("chain")
            sample_stats["chain"] = np.broadcast_to(
                np.arange(shape[axis]).reshape(
                    (-1,) + (1,) * (k - axis - 1)
                ),
                shape,
            ).reshape(n)

        local = None
        if self.local is not None:
            E = self.local.batch_shape[-1]
            local = self.local.reshape_batch((n, E))

        return type(self)(
            theta=self.theta.reshape_batch((n,)),
            sample_stats=sample_stats,
            extras={name: flat(v) for name, v in self.extras.items()},
            metadata=self.metadata.copy(),
            observed_data=self.observed_data,
            local=local,
            sample_dims=("draw",),
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    def summary(self, var_names=None, *, prob=0.9, expand=False) -> pd.DataFrame:
        """Posterior mean, sd, central interval, ESS and split R-hat.

        var_names:
            Variables to report: theta fields, extras, sample stats, or
            fields of ``local`` (as ``"local.<field>"``). Default: every
            theta field, then every field of ``local``.

        prob:
            Probability mass of the central interval.

        expand:
            A field with core axes (e.g. one distance per star) is shown as
            one row with its smallest ESS and largest R-hat, and its number
            of elements in a ``size`` column (present only when some field
            has more than one element). ``expand=True`` gives one full row
            per element.

        R-hat needs a "chain" axis; without one (e.g. after
        ``stack_samples()``) it is NaN. ESS is computed per chain and summed.
        """
        if self.sample_dims not in (("chain", "draw"), ("draw",)):
            raise ValueError(
                "summary() needs sample_dims ('chain', 'draw') or ('draw',); "
                f"got {self.sample_dims}"
            )
        chained = "chain" in self.sample_dims

        local_names = (
            [f.name for f in fields(self.local)] if self.local is not None else []
        )
        if var_names is None:
            var_names = [f.name for f in fields(self.theta)]
            var_names += [f"local.{n}" for n in local_names]

        rows = []
        for name in var_names:
            if name.startswith("local."):
                if self.local is None or name[6:] not in local_names:
                    raise KeyError(f"local has no field {name[6:]!r}")
                values = getattr(self.local.broadcast(), name[6:])
            else:
                values = self.variables([name])[name]
            rows += _summary_rows(name, values, chained, prob, expand)

        table = pd.DataFrame(
            [row for _, row in rows], index=[label for label, _ in rows]
        )
        if (table["size"] == 1).all():
            table = table.drop(columns="size")
        if "diverging" in self.sample_stats:
            table.attrs["divergences"] = int(np.sum(self.sample_stats["diverging"]))
        return table

    def to_arviz(self):
        """An ``arviz`` InferenceData (needs ``arviz``).

        Groups: posterior (theta fields, extras, and ``local`` fields
        prefixed ``local.``), sample_stats, and observed_data when it is a
        Parameters object. Dims come from ``sample_dims`` and each field's
        ``index``; the observation axis of ``local`` is "observation".
        A posterior without a "chain" axis is given one chain.
        """
        try:
            import arviz as az
        except ImportError as exc:
            raise ImportError("to_arviz() needs arviz: pip install arviz") from exc

        if self.sample_dims == ("chain", "draw"):
            post = self
        elif self.sample_dims == ("draw",):
            post = PosteriorData(
                theta=self.theta.reshape_batch((1,) + self.batch_shape),
                sample_stats={k: np.asarray(v)[None] for k, v in self.sample_stats.items()},
                extras={k: np.asarray(v)[None] for k, v in self.extras.items()},
                metadata=self.metadata,
                observed_data=self.observed_data,
                local=None if self.local is None
                else self.local.reshape_batch((1,) + self.local.batch_shape),
                sample_dims=("chain", "draw"),
            )
        else:
            raise ValueError(
                f"to_arviz() needs sample_dims ('chain', 'draw') or ('draw',); "
                f"got {self.sample_dims}"
            )

        posterior, dims = {}, {}

        theta = post.theta.broadcast()
        for name, spec in post.parameter_type.field_specs().items():
            value = np.asarray(getattr(theta, name))
            posterior[name] = value
            dims[name] = list(_core_dims(name, spec, value, 2))

        if post.local is not None:
            local = post.local.broadcast()
            for name, spec in type(local).field_specs().items():
                value = np.asarray(getattr(local, name))
                posterior[f"local.{name}"] = value
                dims[f"local.{name}"] = [OBSERVATION_DIM] + list(
                    _core_dims(name, spec, value, 3)
                )

        for name, value in post.extras.items():
            value = np.asarray(value)
            posterior[name] = value
            if post.local is not None and value.shape[:3] == post.local.batch_shape:
                dims[name] = [OBSERVATION_DIM] + [
                    f"{name}_dim_{i}" for i in range(value.ndim - 3)
                ]

        sample_stats = {}
        for name, value in post.sample_stats.items():
            value = np.asarray(value)
            if name == "potential_energy":
                sample_stats["lp"] = -value
            sample_stats[_ARVIZ_STAT_NAMES.get(name, name)] = value

        groups = {"posterior": posterior, "sample_stats": sample_stats}
        if isinstance(post.observed_data, Parameters):
            od = post.observed_data
            observed = {}
            for name, spec in type(od).field_specs().items():
                value = np.asarray(getattr(od, name))
                observed[name] = value
                # ArviZ's dims apply to every group, so a name shared with
                # a posterior variable keeps the posterior's dims and ArviZ
                # names this variable's axes itself.
                if name not in dims:
                    n_batch = len(od.batch_shape)
                    dims[name] = (
                        ([OBSERVATION_DIM] if n_batch == 1 else [])
                        + list(_core_dims(name, spec, value, n_batch))
                    )
            groups["observed_data"] = observed

        # ArviZ >= 1.0 takes one dict of groups and returns a DataTree;
        # earlier versions take the groups as keyword arguments.
        if "data" in inspect.signature(az.from_dict).parameters:
            return az.from_dict(groups, dims=dims)
        return az.from_dict(**groups, dims=dims)

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
        Scalar-valued posterior draws as a Pandas DataFrame, one row per draw.

        Sample axes are stacked first (see ``stack_samples``), so a
        ``("chain", "draw")`` posterior gives a ``chain`` column. Every
        theta field, sample stat and extra must be scalar per draw;
        ``local`` is not exported.
        """
        post = self.stack_samples()
        theta = post.theta.broadcast()

        values = {
            **{
                f.name: getattr(theta, f.name)
                for f in fields(theta)
            },
            **post.sample_stats,
            **post.extras,
        }

        for name, value in values.items():
            if np.shape(value) != post.batch_shape:
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
        local = (
            f", local={type(self.local).__name__}"
            f"{tuple(self.local.batch_shape)}"
            if self.local is not None else ""
        )
        return (
            f"PosteriorData[{self.parameter_type.__name__}]("
            f"sizes={self.sizes}, "
            f"sample_stats={list(self.sample_stats)}, "
            f"extras={list(self.extras)}{local})"
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

        inference = self.metadata.get("inference")
        if inference == "population":
            return rf"p({theta}\mid \{{{data}_i\}})"
        if inference == "joint population":
            local = getattr(type(self.local), "_latex", r"\theta")
            return rf"p({theta},\,\{{{local}_i\}}\mid \{{{data}_i\}})"
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

        # Named sample axes and, if recorded, the number of divergences.
        sizes = " &times; ".join(
            f"{escape(name)}: {size}" for name, size in self.sizes.items()
        ) or "()"
        line = "Sample dims: <code>" + sizes + "</code>"
        if "diverging" in self.sample_stats:
            line += (
                " &nbsp; Divergences: <code>"
                + str(int(np.sum(self.sample_stats["diverging"])))
                + "</code>"
            )
        body.append(
            '<div style="opacity:0.7;font-size:0.9em;'
            'margin-bottom:0.5em">' + line + "</div>"
        )

        # Reuse the complete Parameters HTML representation.
        body.append(self.theta._repr_html_())

        # Per-observation parameters of a joint population posterior.
        if self.local is not None:
            body.append(
                '<div style="margin-top:0.4em"><strong>Local parameters</strong>'
                ' <span style="opacity:0.65">(dims '
                + escape(", ".join(self.local_dims))
                + ")</span></div>"
                + self.local._repr_html_()
            )

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

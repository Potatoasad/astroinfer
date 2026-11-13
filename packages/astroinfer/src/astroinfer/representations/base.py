"""The representation protocol.

A *representation kind* (``"raw"``, ``"tgmm"``, ...) is a pair of classes:

    Fitter   builds one Fit per event from its PosteriorData. Holds a JSON
             config (default + per-event overrides) and decides staleness.
    Fit      one event's fitted object. Produces the event's data product
             and saves itself into a directory it owns.

The Catalog never looks inside either; it only calls the methods below.
That is what lets any kind be fit, checked for staleness, stacked, saved
and loaded the same way.

Staleness: every Fit carries ``fit_key``, a hash of everything that
produced it (the fit-relevant config and the source posterior's hash).
A fit is stale when ``fit.fit_key != fitter.fit_key(eventname, source_key)``.
Settings that only shape the data product are *not* part of the key; the
Catalog takes them from the current fitter via ``dataproduct_options``.
"""
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import jax.numpy as jnp
import numpy as np

from ..display import RichDisplay, code, kv_table
from ..io import json_clean, resolve_type

Array = Any


@dataclass(frozen=True)
class LoadContext:
    """What a Fit may need when it is loaded.

    eventname: the event the fit belongs to.
    types:     classes supplied by the user to resolve saved type paths
               (see ``astroinfer.io.resolve_type``).
    """

    eventname: str
    types: tuple[type, ...] = ()

    def resolve(self, path: str) -> type:
        return resolve_type(path, self.types)


@runtime_checkable
class Fit(Protocol):
    @property
    def fit_key(self) -> str: ...

    def dataproduct(self, *, N: int, seed: int, **options) -> dict[str, Array]: ...

    def save(self, directory: Path) -> None:
        """Write everything needed to rebuild the fit into ``directory``.

        The directory exists and is empty; the Catalog writes its own
        ``fit.json`` next to whatever the fit writes.
        """
        ...

    @classmethod
    def load(cls, directory: Path, ctx: LoadContext) -> Self: ...

    @classmethod
    def stack(cls, products: Sequence[Mapping[str, Array]]) -> dict[str, Array]:
        """Combine per-event products into event-major arrays."""
        ...


@runtime_checkable
class Fitter(Protocol):
    kind: ClassVar[str]
    fit_type: ClassVar[type]

    def fit(self, posterior, *, eventname: str, source_key: str) -> Fit: ...

    def fit_key(self, eventname: str, source_key: str) -> str: ...

    def dataproduct_options(self, eventname: str) -> dict[str, Any]: ...

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self: ...


# ============================================================
# Helpers for implementers
# ============================================================

def config_hash(value: Any) -> str:
    """Stable SHA-256 of a JSON-serializable value (key order ignored)."""
    text = json.dumps(json_clean(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def stack_products(
    products: Sequence[Mapping[str, Array]],
    names: Iterable[str] | None = None,
) -> dict[str, Array]:
    """Stack equal-keyed, equal-shaped products along a new leading event axis."""
    products = list(products)
    if not products:
        return {}
    names = [f"#{i}" for i in range(len(products))] if names is None else list(names)
    expected = set(products[0])
    for name, product in zip(names, products):
        if set(product) != expected:
            raise ValueError(
                f"Event {name!r} product keys do not match {names[0]!r}: "
                f"{sorted(set(product) ^ expected)}"
            )
    out = {}
    for key in products[0]:
        values = [np.asarray(p[key]) for p in products]
        shapes = {name: v.shape for name, v in zip(names, values)}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Cannot stack {key!r}; shapes differ: {shapes}")
        out[key] = jnp.asarray(np.stack(values, axis=0))
    return out


class BaseFit(RichDisplay):
    """Default ``stack`` and display; subclasses implement the rest of ``Fit``."""

    @classmethod
    def stack(cls, products):
        return stack_products(products)

    def _display_body(self):
        return kv_table({"fit_key": self.fit_key[:12] + "..."})


class BaseFitter(RichDisplay):
    """Defaults: no data-product options, JSON save/load of ``to_dict``, and
    a display of ``to_dict``."""

    def _display_body(self):
        return f"<div>Kind: {code(self.kind)}</div>" + kv_table(self.to_dict())

    def dataproduct_options(self, eventname: str) -> dict[str, Any]:
        return {}

    def save(self, path) -> None:
        from ..io import write_json

        write_json(path, {"kind": self.kind, "config": self.to_dict()})

    @classmethod
    def load(cls, path) -> Self:
        from ..io import read_json

        d = read_json(path)
        if d.get("kind") != cls.kind:
            raise ValueError(f"{path} holds a {d.get('kind')!r} fitter, not {cls.kind!r}")
        return cls.from_dict(d["config"])

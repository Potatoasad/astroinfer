"""A catalog of per-event posteriors and the representations fit to them.

Every representation (raw samples, TGMM fits, ...) is a named Fitter plus
one Fit per event; see :mod:`astroinfer.representations`. The ``raw``
representation is always present and holds the posteriors themselves.

On disk (``Catalog.save`` / ``Catalog.load``)::

    root/catalog.json                              event order, {name: kind}, seed
    root/representations/<name>/fitter.json        {"kind", "config": fitter.to_dict()}
    root/representations/<name>/events/<event>/
        fit.json                                   {"kind", "fit_key", "saved_at"}
        ...                                        whatever Fit.save writes
"""
import datetime
import hashlib
import shutil
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .display import RichDisplay, badge, code, html_table, kv_table, section
from .io import read_json, replace_directory, write_json
from .posterior import PosteriorData
from .representations import LoadContext, RawSamples, get_fitter_class

_FORMAT_VERSION = 1
RAW = "raw"


def _check_name(name: str, what: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what} must be a nonempty string")
    if "/" in name or "\\" in name or name.startswith(".") or "\x00" in name:
        raise ValueError(f"{what} {name!r} cannot be used as a folder name")
    return name


def event_seed(base_seed: int, eventname: str) -> int:
    """Per-event seed from the event's *name*, so it doesn't depend on
    which other events are requested or their order."""
    digest = hashlib.sha256(eventname.encode()).digest()
    name_entropy = int.from_bytes(digest[:8], "little")
    return int(np.random.SeedSequence([int(base_seed), name_entropy]).generate_state(1)[0])


class Catalog(RichDisplay):
    """PosteriorData per event, plus named representations fit to them.

    >>> catalog = Catalog(posteriors, raw=RawSamples(columns=["m1", "q"]))
    >>> catalog.register_representation("tgmm", TGMMFitter(config))
    >>> catalog.fit("tgmm")                       # missing or stale events only
    >>> catalog.dataproduct(representation="tgmm", N=1000)
    >>> catalog.save("catalog_dir");  Catalog.load("catalog_dir")
    """

    def __init__(
        self,
        posteriors: Sequence[PosteriorData] | Mapping[str, PosteriorData] | None = None,
        *,
        raw: RawSamples | Sequence[str],
        seed: int = 42,
    ):
        if not isinstance(raw, RawSamples):
            raw = RawSamples(columns=tuple(raw))
        self.seed = int(seed)
        self._fitters: dict[str, Any] = {RAW: raw}
        self._fits: dict[str, dict[str, Any]] = {RAW: OrderedDict()}
        self._dirty: set[tuple[str, str]] = set()
        self._saved_root: Path | None = None

        if posteriors is None:
            return
        if isinstance(posteriors, Mapping):
            items = posteriors.items()
        else:
            items = (
                (p.metadata.get("event_name", f"event_{i:04d}"), p)
                for i, p in enumerate(posteriors)
            )
        for name, posterior in items:
            self.add_event(name, posterior)

    # ========================================================
    # Basic access
    # ========================================================

    @property
    def eventnames(self) -> list[str]:
        return list(self._fits[RAW])

    @property
    def posteriors(self) -> list[PosteriorData]:
        return [fit.posterior for fit in self._fits[RAW].values()]

    @property
    def representations(self) -> list[str]:
        return list(self._fitters)

    @property
    def raw(self) -> RawSamples:
        return self._fitters[RAW]

    def __len__(self) -> int:
        return len(self._fits[RAW])

    def __iter__(self):
        return iter(self._fits[RAW])

    def __contains__(self, eventname) -> bool:
        return eventname in self._fits[RAW]

    def __getitem__(self, eventname: str) -> PosteriorData:
        return self.posterior(eventname)

    def __repr__(self) -> str:
        return (
            f"Catalog(n_events={len(self)}, columns={list(self.raw.columns)!r}, "
            f"representations={self.representations!r})"
        )

    def posterior(self, eventname: str) -> PosteriorData:
        try:
            return self._fits[RAW][eventname].posterior
        except KeyError:
            raise KeyError(
                f"{eventname!r} is not in the catalog; add it with catalog.add_event(...)"
            ) from None

    def source_key(self, eventname: str) -> str:
        """Content hash of the event's posterior (the input to every fit)."""
        self.posterior(eventname)
        return self._fits[RAW][eventname].fit_key

    # ========================================================
    # Events
    # ========================================================

    def add_event(
        self,
        eventname: str,
        posterior: PosteriorData,
        *,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add (or with ``overwrite`` replace) an event's posterior.

        Replacing a posterior keeps its other fits but makes them stale.
        """
        _check_name(eventname, "eventname")
        if not isinstance(posterior, PosteriorData):
            raise TypeError("posterior must be a PosteriorData instance")
        if eventname in self._fits[RAW] and not overwrite:
            raise ValueError(f"Event {eventname!r} already exists; pass overwrite=True")
        if posterior.local is not None:
            raise TypeError(
                f"Event {eventname!r}: a joint population posterior is not a "
                "single-event posterior"
            )
        if not posterior.batch_shape:
            raise ValueError(f"Event {eventname!r} has no draw axis")
        # A catalog keeps exchangeable draws; the chain of each draw stays
        # available as sample_stats["chain"].
        posterior = posterior.stack_samples()
        if posterior.batch_shape[0] == 0:
            raise ValueError(f"Event {eventname!r} has no posterior draws")
        if metadata is not None:
            posterior = replace(posterior, metadata={**posterior.metadata, **metadata})

        self._fits[RAW][eventname] = self.raw.fit(posterior, eventname=eventname)
        self._dirty.add((RAW, eventname))

    def remove_event(self, eventname: str) -> None:
        self.posterior(eventname)
        for fits in self._fits.values():
            fits.pop(eventname, None)

    # ========================================================
    # Representations
    # ========================================================

    def register_representation(self, name: str, fitter, *, overwrite: bool = False) -> None:
        """Attach a Fitter under ``name``. Existing fits are kept (they are
        stale wherever the new fitter would produce a different key)."""
        _check_name(name, "representation name")
        if name == RAW:
            raise ValueError("'raw' is built in; use catalog.set_raw(...) to change it")
        for attr in ("kind", "fit_type", "fit", "fit_key", "to_dict", "from_dict",
                     "dataproduct_options"):
            if not hasattr(fitter, attr):
                raise TypeError(f"fitter has no {attr!r}; see astroinfer.representations.Fitter")
        if name in self._fitters and not overwrite:
            raise ValueError(f"Representation {name!r} exists; pass overwrite=True")
        old = self._fitters.get(name)
        if old is not None and old.fit_type is not fitter.fit_type:
            self._fits[name] = {}
        self._fitters[name] = fitter
        self._fits.setdefault(name, {})

    def remove_representation(self, name: str) -> None:
        if name == RAW:
            raise ValueError("The raw representation cannot be removed")
        del self._fitters[name]
        del self._fits[name]

    def set_raw(self, raw: RawSamples | Sequence[str]) -> None:
        """Change the raw data-product settings (columns, N)."""
        if not isinstance(raw, RawSamples):
            raw = RawSamples(columns=tuple(raw))
        for eventname, fit in self._fits[RAW].items():
            raw.check(fit.posterior, eventname)
        self._fitters[RAW] = raw

    def fitter(self, name: str):
        try:
            return self._fitters[name]
        except KeyError:
            raise KeyError(
                f"No representation {name!r}; registered: {self.representations}"
            ) from None

    def fits(self, name: str) -> dict[str, Any]:
        """``{eventname: fit}`` for the events that have one, in catalog order."""
        self.fitter(name)
        return {e: self._fits[name][e] for e in self.eventnames if e in self._fits[name]}

    def get_fit(self, eventname: str, name: str):
        self.posterior(eventname)
        try:
            return self._fits[self._name(name)][eventname]
        except KeyError:
            raise KeyError(f"{eventname!r} has no {name!r} fit; call catalog.fit({name!r})") from None

    def add_representation(self, eventname: str, name: str, fit, *, overwrite: bool = False) -> None:
        """Store a fit made elsewhere (it must be the fitter's ``fit_type``)."""
        self.posterior(eventname)
        fitter = self.fitter(self._name(name))
        if name == RAW:
            raise ValueError("Use add_event(...) for raw posteriors")
        if not isinstance(fit, fitter.fit_type):
            raise TypeError(
                f"{name!r} fits must be {fitter.fit_type.__name__}, got {type(fit).__name__}"
            )
        if eventname in self._fits[name] and not overwrite:
            raise ValueError(f"Representation {name!r} exists for {eventname!r}")
        self._fits[name][eventname] = fit
        self._dirty.add((name, eventname))

    def _name(self, name: str) -> str:
        self.fitter(name)
        return name

    # ========================================================
    # Fitting and staleness
    # ========================================================

    def is_stale(self, eventname: str, name: str) -> bool:
        """True if the stored fit no longer matches its fitter or posterior."""
        fit = self.get_fit(eventname, name)
        expected = self.fitter(name).fit_key(eventname, self.source_key(eventname))
        return fit.fit_key != expected

    def stale(self, name: str) -> list[str]:
        return [e for e in self.fits(name) if self.is_stale(e, name)]

    def missing(self, name: str) -> list[str]:
        self.fitter(name)
        return [e for e in self.eventnames if e not in self._fits[name]]

    def fit(
        self,
        name: str,
        eventnames: Iterable[str] | None = None,
        *,
        refit: bool = False,
        errors: Literal["raise", "warn"] = "raise",
    ) -> list[str]:
        """Fit events that are missing a ``name`` fit or whose fit is stale.

        ``refit`` refits regardless. With ``errors="warn"`` a failing event
        is reported and skipped (its previous fit, if any, is kept).
        Returns the events that were (re)fit.
        """
        if name == RAW:
            raise ValueError("The raw representation is not fit")
        if errors not in ("raise", "warn"):
            raise ValueError("errors must be 'raise' or 'warn'")
        fitter = self.fitter(name)
        names = self.eventnames if eventnames is None else list(eventnames)
        done = []
        for eventname in names:
            source_key = self.source_key(eventname)
            if (not refit and eventname in self._fits[name]
                    and not self.is_stale(eventname, name)):
                continue
            try:
                fit = fitter.fit(self.posterior(eventname), eventname=eventname,
                                 source_key=source_key)
            except Exception as exc:
                if errors == "raise":
                    raise
                warnings.warn(f"{name!r} fit of {eventname!r} failed: {exc}", stacklevel=2)
                continue
            self._fits[name][eventname] = fit
            self._dirty.add((name, eventname))
            done.append(eventname)
        return done

    # ========================================================
    # Data products
    # ========================================================

    def dataproduct(
        self,
        *,
        representation: str = RAW,
        N: int | None = None,
        seed: int | None = None,
        eventnames: Sequence[str] | None = None,
        allow_stale: bool = False,
    ) -> dict[str, Any]:
        """Event-major arrays from the representation's fits.

        Raw: ``{column: (E, N), "prior": (E, N)}``. Fitted kinds define
        their own keys and stacking (e.g. TGMM pads components to max K).
        Data-product settings come from the *current* fitter; ``N``
        overrides its default (or the raw ``N`` if the fitter has none).
        Each event's seed depends on ``seed`` and
        the event name only.
        """
        fitter = self.fitter(representation)
        names = self.eventnames if eventnames is None else list(eventnames)
        if len(names) != len(set(names)):
            raise ValueError("eventnames contains duplicates")
        if N is not None and int(N) <= 0:
            raise ValueError("N must be positive")
        base_seed = self.seed if seed is None else int(seed)

        missing = [e for e in names if e not in self._fits[representation]]
        for e in names:
            self.posterior(e)
        if missing:
            raise KeyError(
                f"Missing {representation!r} fits for {missing}; "
                f"call catalog.fit({representation!r})"
            )
        if representation != RAW and not allow_stale:
            stale = [e for e in names if self.is_stale(e, representation)]
            if stale:
                raise ValueError(
                    f"{representation!r} fits are stale for {stale}; refit with "
                    f"catalog.fit({representation!r}) or pass allow_stale=True"
                )

        products = []
        for e in names:
            options = dict(fitter.dataproduct_options(e))
            if N is not None:
                options["N"] = int(N)
            options.setdefault("N", self.raw.N)
            products.append(
                self._fits[representation][e].dataproduct(seed=event_seed(base_seed, e), **options)
            )
        return fitter.fit_type.stack(products)

    # ========================================================
    # Saving and loading
    # ========================================================

    def save(self, root) -> Path:
        """Write the catalog to ``root``.

        Saving to the same folder again only rewrites what changed: new or
        refit fits, every fitter.json, and the manifest. Folders of
        removed events or representations are deleted.
        """
        root = Path(root).expanduser().resolve()
        manifest_path = root / "catalog.json"
        if root.exists() and any(root.iterdir()) and not manifest_path.exists():
            raise FileExistsError(f"{root} exists, is not empty and is not a catalog folder")
        incremental = root == self._saved_root
        now = datetime.datetime.now().isoformat(timespec="seconds")
        reps_dir = root / "representations"

        for name, fitter in self._fitters.items():
            rep_dir = reps_dir / name
            rep_dir.mkdir(parents=True, exist_ok=True)
            write_json(rep_dir / "fitter.json", {"kind": fitter.kind, "config": fitter.to_dict()})
            events_dir = rep_dir / "events"
            fits = self._fits[name]

            for eventname, fit in fits.items():
                target = events_dir / eventname
                if incremental and (name, eventname) not in self._dirty:
                    info_path = target / "fit.json"
                    if info_path.exists():
                        info = read_json(info_path)
                        if info.get("kind") == fitter.kind and info.get("fit_key") == fit.fit_key:
                            continue

                def write(tmp, fit=fit):
                    fit.save(tmp)
                    write_json(tmp / "fit.json",
                               {"kind": fitter.kind, "fit_key": fit.fit_key, "saved_at": now})

                replace_directory(target, write)

            if events_dir.exists():
                for d in events_dir.iterdir():
                    if (d.is_dir() and not d.name.startswith(".") and d.name not in fits
                            and (d / "fit.json").exists()):
                        shutil.rmtree(d)

        for d in reps_dir.iterdir():
            if d.is_dir() and d.name not in self._fitters and (d / "fitter.json").exists():
                shutil.rmtree(d)

        write_json(manifest_path, {
            "format_version": _FORMAT_VERSION,
            "saved_at": now,
            "seed": self.seed,
            "eventnames": self.eventnames,
            "representations": {name: f.kind for name, f in self._fitters.items()},
        })
        self._saved_root = root
        self._dirty.clear()
        return root

    @classmethod
    def load(cls, root, *, types: Iterable[type] = ()) -> "Catalog":
        """Rebuild a saved catalog: posteriors, fitters (with their per-event
        configs) and every fit.

        ``types``: classes to resolve saved type paths with, needed for
        Parameters classes defined in a notebook or script.
        Representation kinds are found through the registry and the
        ``astroinfer.representations`` entry points.
        """
        root = Path(root).expanduser().resolve()
        manifest = read_json(root / "catalog.json")
        if manifest.get("format_version") != _FORMAT_VERSION:
            raise ValueError(f"Unsupported catalog format {manifest.get('format_version')!r}")
        types = tuple(types)
        reps_dir = root / "representations"

        fitters = {}
        for name, kind in manifest["representations"].items():
            info = read_json(reps_dir / name / "fitter.json")
            if info["kind"] != kind:
                raise ValueError(f"{name}: manifest says {kind!r}, fitter.json says {info['kind']!r}")
            fitters[name] = get_fitter_class(kind).from_dict(info["config"])

        catalog = cls(raw=fitters.pop(RAW), seed=manifest["seed"])
        for name, fitter in fitters.items():
            catalog.register_representation(name, fitter)

        for name, fitter in catalog._fitters.items():
            for eventname in manifest["eventnames"]:
                directory = reps_dir / name / "events" / eventname
                info_path = directory / "fit.json"
                if not info_path.exists():
                    if name == RAW:
                        raise FileNotFoundError(f"No saved posterior for {eventname!r} in {directory}")
                    continue
                info = read_json(info_path)
                fit = fitter.fit_type.load(directory, LoadContext(eventname, types))
                if fit.fit_key != info["fit_key"]:
                    warnings.warn(
                        f"{name}/{eventname}: loaded fit_key differs from the saved one",
                        stacklevel=2,
                    )
                catalog._fits[name][eventname] = fit

        catalog._saved_root = root
        catalog._dirty.clear()
        return catalog

    # ========================================================
    # RichDisplay
    # ========================================================

    def _display_title(self):
        return "Catalog"

    def _display_math(self):
        return None

    def _event_status(self, eventname: str, name: str) -> str:
        if eventname not in self._fits[name]:
            return badge("missing", "none")
        if name != RAW and self.is_stale(eventname, name):
            return badge("stale", "warn")
        return badge("fit" if name != RAW else "ok", "ok")

    def _display_body(self, max_events: int = 50, max_posteriors: int = 10):
        summary = kv_table({
            "events": len(self),
            "raw columns": ", ".join(self.raw.columns),
            "raw N": self.raw.N,
            "seed": self.seed,
            "saved to": str(self._saved_root) if self._saved_root else "(not saved)",
        })

        rep_rows = []
        for name, fitter in self._fitters.items():
            n_fit = len(self._fits[name])
            n_stale = 0 if name == RAW else len(self.stale(name))
            rep_rows.append([
                "<strong>" + escape(name) + "</strong>",
                code(fitter.kind),
                str(n_fit),
                badge(str(n_stale), "warn") if n_stale else "0",
                str(len(self) - n_fit),
                fitter._repr_html_() if hasattr(fitter, "_repr_html_") else code(repr(fitter)),
            ])
        reps = html_table(["name", "kind", "fit", "stale", "missing", "fitter"], rep_rows)

        names = self.eventnames
        event_rows = []
        for eventname in names[:max_events]:
            row = ["<strong>" + escape(eventname) + "</strong>",
                   str(self.posterior(eventname).batch_shape[0])]
            row += [self._event_status(eventname, name) for name in self._fitters if name != RAW]
            event_rows.append(row)
        header = ["event", "draws"] + [escape(n) for n in self._fitters if n != RAW]
        events = html_table(header, event_rows)
        if len(names) > max_events:
            events += f"<div>... and {len(names) - max_events} more events</div>"

        posteriors = "".join(
            section(e, self.posterior(e)._repr_html_()) for e in names[:max_posteriors]
        )
        if len(names) > max_posteriors:
            posteriors += f"<div>... and {len(names) - max_posteriors} more</div>"

        return (summary
                + section("Representations", reps, opened=True, count=len(self._fitters))
                + section("Events", events, opened=True, count=len(self))
                + section("Posteriors", posteriors, count=len(self)))

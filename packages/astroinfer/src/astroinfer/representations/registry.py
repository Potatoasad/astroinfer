"""Map representation kinds to Fitter classes.

Extension packages advertise kinds through the entry-point group
``astroinfer.representations``::

    [project.entry-points."astroinfer.representations"]
    tgmm = "astroinfer_tgmm:TGMMFitter"

so ``Catalog.load`` finds them without astroinfer importing them up
front. ``register`` adds kinds by hand (e.g. a fitter defined in a
notebook); it also works as a class decorator.
"""
from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "astroinfer.representations"

_REGISTRY: dict[str, type] = {}


def register(fitter_cls: type | None = None, *, overwrite: bool = False):
    def deco(cls):
        kind = getattr(cls, "kind", None)
        if not isinstance(kind, str) or not kind:
            raise TypeError(f"{cls.__name__} needs a nonempty class attribute `kind`")
        if not hasattr(cls, "fit_type"):
            raise TypeError(f"{cls.__name__} needs a class attribute `fit_type`")
        existing = _REGISTRY.get(kind)
        if existing is not None and existing is not cls and not overwrite:
            if (existing.__module__, existing.__qualname__) != (cls.__module__, cls.__qualname__):
                raise ValueError(
                    f"Kind {kind!r} is already registered to "
                    f"{existing.__module__}.{existing.__qualname__}"
                )
        _REGISTRY[kind] = cls
        return cls

    return deco if fitter_cls is None else deco(fitter_cls)


def get_fitter_class(kind: str) -> type:
    if kind in _REGISTRY:
        return _REGISTRY[kind]
    matches = entry_points(group=ENTRY_POINT_GROUP, name=kind)
    for ep in matches:
        cls = ep.load()
        _REGISTRY.setdefault(kind, cls)
        return _REGISTRY[kind]
    raise LookupError(
        f"No representation kind {kind!r}. Registered: {sorted(_REGISTRY)}; "
        f"installed: {available_kinds()}. Install the extension that "
        "provides it, or astroinfer.representations.register(...) it."
    )


def available_kinds() -> list[str]:
    names = set(_REGISTRY)
    names.update(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))
    return sorted(names)

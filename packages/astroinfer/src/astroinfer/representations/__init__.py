"""Per-event representations of posteriors (raw samples, fitted densities, ...).

See :mod:`astroinfer.representations.base` for the Fit / Fitter protocol
that extensions (e.g. ``astroinfer_tgmm``) implement.
"""
from .base import (
    BaseFit,
    BaseFitter,
    Fit,
    Fitter,
    LoadContext,
    config_hash,
    stack_products,
)
from .raw import RawFit, RawSamples
from .registry import (
    ENTRY_POINT_GROUP,
    available_kinds,
    get_fitter_class,
    register,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "BaseFit",
    "BaseFitter",
    "Fit",
    "Fitter",
    "LoadContext",
    "RawFit",
    "RawSamples",
    "available_kinds",
    "config_hash",
    "get_fitter_class",
    "register",
    "stack_products",
]

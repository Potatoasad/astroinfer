"""TGMM (truncated Gaussian mixture) representations for astroinfer catalogs.

    >>> from astroinfer import Catalog, RawSamples
    >>> from astroinfer_tgmm import TGMMConfig, TGMMFitter
    >>> catalog.register_representation("tgmm", TGMMFitter(TGMMConfig(columns=[...], K=8, limits={...})))
    >>> catalog.fit("tgmm")
    >>> catalog.dataproduct(representation="tgmm")

Importing this package does not start Julia; fitting or loading a fit does.
The ``tgmm`` kind is registered on import and advertised through the
``astroinfer.representations`` entry point, so ``Catalog.load`` finds it.
"""
from ._tgm import LOGTP_CUTOFF
from .config import DATAPRODUCT_FIELDS, TGMMConfig, TransformationSpec
from .fit import TGMMFit, pad_components
from .fitter import TGMMFitError, TGMMFitter

__all__ = [
    "DATAPRODUCT_FIELDS",
    "LOGTP_CUTOFF",
    "TGMMConfig",
    "TGMMFit",
    "TGMMFitError",
    "TGMMFitter",
    "TransformationSpec",
    "pad_components",
]

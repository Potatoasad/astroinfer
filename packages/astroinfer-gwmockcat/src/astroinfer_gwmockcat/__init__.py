"""GWMockCat-style mock gravitational-wave catalogs for astroinfer.

    GWParams, GWMockCatData     typed theta and d
    GWMockCatMeasurement        K(d | theta): SNR, chirp mass, eta, projection factor, chi_eff, chi_p
    physics                     conversions, d_L(z), ToyOptimalSNR, projection factor
"""
from . import physics
from .measurement import GWMockCatData, GWMockCatMeasurement, GWParams
from .physics import (
    CHI_EFF_BOUNDS,
    CHI_P_BOUNDS,
    RedshiftToLuminosityDistance,
    ToyOptimalSNR,
    chirp_mass_from_mass_ratio,
    effective_spins,
    isotropic_projection_factor,
    projection_factor_model,
    symmetric_mass_ratio,
    total_mass_from_chirp_mass,
)

__all__ = [
    "CHI_EFF_BOUNDS",
    "CHI_P_BOUNDS",
    "GWMockCatData",
    "GWMockCatMeasurement",
    "GWParams",
    "RedshiftToLuminosityDistance",
    "ToyOptimalSNR",
    "chirp_mass_from_mass_ratio",
    "effective_spins",
    "isotropic_projection_factor",
    "physics",
    "projection_factor_model",
    "symmetric_mass_ratio",
    "total_mass_from_chirp_mass",
]

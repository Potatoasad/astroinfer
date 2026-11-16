"""GWMockCat-style mock gravitational-wave catalogs (Farah et al. 2023, App. A).

    GWParams, GWMockCatData     typed theta and d
    GWMockCatMeasurement        L(d | theta): SNR, chirp mass, eta, projection factor, chi_eff, chi_p
    ToyOptimalSNR               the toy optimal-SNR model the measurement uses by default
    pe_priors                   PE priors flat in detector-frame masses, and uniform in comoving
                                volume (pe_prior_comoving) or luminosity distance (pe_prior_luminosity)
"""
from . import pe_priors
from .measurement import GWMockCatData, GWMockCatMeasurement, GWParams
from .snr import ToyOptimalSNR

__all__ = ["GWMockCatData", "GWMockCatMeasurement", "GWParams", "ToyOptimalSNR", "pe_priors"]

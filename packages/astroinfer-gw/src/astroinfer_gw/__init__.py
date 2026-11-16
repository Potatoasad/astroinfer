"""Gravitational-wave models for astroinfer.

    gwmockcat   GWMockCat-style mock catalogs: GWParams, GWMockCatData, GWMockCatMeasurement,
                ToyOptimalSNR, pe_priors
    physics     mass and spin conversions, d_L(z), projection factor
"""
from . import gwmockcat, physics

__all__ = ["gwmockcat", "physics"]

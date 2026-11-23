"""Gravitational-wave models for astroinfer.

    gwmockcat   GWMockCat-style mock catalogs: GWParams, GWMockCatData, GWMockCatMeasurement,
                ToyOptimalSNR, pe_priors
    physics     mass and spin conversions, d_L(z), projection factor
    populations NumPyro distributions for GW populations: PowerLaw, PowerLawRedshift,
                ProjectionFactor
"""
from . import gwmockcat, physics, populations

__all__ = ["gwmockcat", "physics", "populations"]

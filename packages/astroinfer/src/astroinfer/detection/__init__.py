"""Detection triggers P(detected | d) and injection tables.

    DetectionTrigger    base: detection_probability(d), detect(d, rng), injections(measurement, draws, rng)
    ThresholdTrigger    detected if an observed statistic exceeds a threshold
"""
from .triggers import DetectionTrigger, ThresholdTrigger

__all__ = ["DetectionTrigger", "ThresholdTrigger"]

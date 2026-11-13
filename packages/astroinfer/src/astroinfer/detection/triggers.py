"""Detection: P(detected | d), and the injection table that measures it."""
import numpy as np
import pandas as pd

from ..display import RichDisplay, kv_table, math_block, method_row, section
from ._data import as_arrays, n_rows


class DetectionTrigger(RichDisplay):
    """Which data are detected.

    Subclasses implement ``detection_probability(d)`` in [0, 1], vectorized
    over a batch of data (a Parameters object or dict of arrays), and may
    implement ``selection_statistic(d)`` (e.g. the SNR), recorded with
    injections.
    """

    def detection_probability(self, d):
        raise NotImplementedError

    def selection_statistic(self, d):
        return np.full(n_rows(as_arrays(d)), np.nan)

    @staticmethod
    def _broadcast(value, n):
        return np.broadcast_to(np.asarray(value, dtype=np.float64), (n,))

    def detect(self, d, rng) -> np.ndarray:
        """D ~ Bernoulli(detection_probability(d)) for each entry of d."""
        n = n_rows(as_arrays(d))
        return rng.uniform(size=n) < self._broadcast(self.detection_probability(d), n)

    def injections(self, measurement, draws, rng) -> pd.DataFrame:
        """The injection table for ``draws`` (theta ~ g, one row each; e.g. a
        population sample with its density g as ``prior``).

        Each row is scattered once by ``measurement`` with fresh noise and
        put through this trigger. Returned: ``draws`` with the derived
        parameters, the data as ``observed_<name>``, ``selection_statistic``,
        ``detection_probability`` and the boolean ``detected``.
        """
        table = measurement.with_derived(measurement.check(draws))
        d = measurement.sample(table, rng)
        for k, v in as_arrays(d).items():
            table[f"observed_{k}"] = v
        n = len(table)
        p = self._broadcast(self.detection_probability(d), n)
        table["selection_statistic"] = self._broadcast(self.selection_statistic(d), n)
        table["detection_probability"] = p
        table["detected"] = rng.uniform(size=n) < p
        return table

    # ========================================================
    # RichDisplay (subclasses fill in _display_rule / _display_config)
    # ========================================================

    def _display_title(self):
        return getattr(self, "display_name", type(self).__name__)

    def _display_math(self):
        return r"P(\mathrm{det}\mid d)"

    def _display_rule(self) -> list[str]:
        return []

    def _display_config(self) -> dict:
        return {}

    def _display_body(self):
        parts = []
        rule = self._display_rule()
        if rule:
            parts.append(math_block(rule))
        config = self._display_config()
        if config:
            parts.append(kv_table(config))
        parts.append(section("Methods", "".join([
            method_row(".detection_probability", r"(d)\longrightarrow [0,1]",
                       "probability each datum is detected"),
            method_row(".detect", r"(d,\ \mathrm{rng})\longrightarrow D",
                       "Bernoulli draw of detection"),
            method_row(".injections", r"(K,\ \theta_i\sim g,\ \mathrm{rng})\longrightarrow\mathrm{DataFrame}",
                       "scatter injections once and record detection"),
        ]), opened=False))
        return "".join(parts)


class ThresholdTrigger(DetectionTrigger):
    """Detected if the observed ``statistic`` (e.g. ``"snr"``) exceeds ``threshold``."""

    def __init__(self, threshold=8.0, statistic="snr"):
        self.threshold = float(threshold)
        self.statistic = statistic

    def __repr__(self):
        return f"ThresholdTrigger({self.statistic} > {self.threshold})"

    def _display_rule(self):
        stat = r"\hat{\rho}" if self.statistic == "snr" else (
            r"\mathrm{" + self.statistic.replace("_", r"\_") + "}")
        return [rf"P(\mathrm{{det}}\mid d) = \mathbb{{1}}\left[{stat} > {self.threshold:g}\right]"]

    def _display_config(self):
        return {"statistic": self.statistic, "threshold": self.threshold}

    def selection_statistic(self, d):
        return np.asarray(as_arrays(d)[self.statistic], dtype=np.float64)

    def detection_probability(self, d):
        return (self.selection_statistic(d) > self.threshold).astype(np.float64)

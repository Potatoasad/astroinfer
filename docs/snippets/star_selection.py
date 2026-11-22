# Definitions for the selection effects example on the landing page (docs/index.md). It reuses
# the classes of snippets/star_population.py, which the page runs first. The page shows the
# pieces between the "# --- name" markers and runs the whole file in a hidden cell.
# --- imports
from functools import partial

import numpy as np
import pandas as pd
from jax.scipy.special import logsumexp

import astroinfer as ai
from astroinfer.detection import ThresholdTrigger
# --- trigger
trigger = ThresholdTrigger(-0.3, statistic="log_flux")
# --- likelihood
class SelectedLikelihood(ai.PopulationLikelihood):
    def __init__(self, *args, injections: pd.DataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        found = injections[injections["detected"]]
        self.found = {k: jnp.asarray(found[k].to_numpy())
                      for k in ["log_L", "distance"]}
        self.log_g = jnp.log(jnp.asarray(found["prior"].to_numpy()))
        self.n_injections = len(injections)

    def log_alpha(self, Lambda):
        log_ratio = self.pop_model.log_prob(self.found, Lambda) - self.log_g
        return logsumexp(log_ratio) - jnp.log(self.n_injections)

    def logpdf(self, Lambda):
        return super().logpdf(Lambda) - self.E * self.log_alpha(Lambda)
# --- end

# The template on the landing page (docs/index.md). The page shows it piece by piece between the
# "# --- name" markers and runs the whole file in a hidden cell.
# --- imports
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from numpyro.distributions import constraints

from astroinfer import MeasurementProcess, parameter, parameter_dataclass

jax.config.update("jax_enable_x64", True)
# --- theta
@parameter_dataclass(name="My parameters", latex=r"\theta")
class MyParameters:
    # A scalar: one value per object.
    x: float = parameter(name="A positive scalar", latex="x",
                         support=constraints.positive)
    # A vector with a named axis: theta.ys[j] for j = 0, 1, ...
    ys: jax.Array = parameter(name="A vector of fractions", latex="y",
                              index=("component",), index_latex=("j",),
                              support=constraints.unit_interval)
# --- data
@parameter_dataclass(name="My data", latex="d")
class MyData:
    # One measured value per reading k.
    values: jax.Array = parameter(name="Measured values",
                                  latex=r"\hat{v}",
                                  index=("reading",), index_latex=("k",))
# --- measurement
@dataclass
class MyMeasurement(MeasurementProcess):
    # Settings of the instrument: fixed, not inferred.
    n_readings: int = 100
    noise: float = 0.1

    parameter_type = MyParameters  # theta is a MyParameters
    data_type = MyData             # d is a MyData

    def sample_single(self, theta: MyParameters,
                      key: jax.Array) -> MyData:
        """Simulate one data set d for one theta.

        theta.x has shape () and theta.ys has shape (n_components,).
        Return a MyData whose values have shape (n_readings,).
        No batch axes: astroinfer adds them.
        """
        ...

    def logpdf_single(self, d: MyData,
                      theta: MyParameters) -> jax.Array:
        """Return log L(d | theta), a scalar, for one d and one theta."""
        ...
# --- end

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from astroinfer import MeasurementProcess, parameter_dataclass
from astroinfer.detection import DetectionTrigger, ThresholdTrigger


@parameter_dataclass
class Theta:
    x: float


@parameter_dataclass
class Obs:
    snr: float


class ToyMeasurement(MeasurementProcess):
    """snr ~ N(x, 1)."""
    parameter_type = Theta
    data_type = Obs

    def sample_single(self, theta, key):
        return Obs(snr=theta.x + jax.random.normal(key))

    def logpdf_single(self, d, theta):
        return jax.scipy.stats.norm.logpdf(d.snr, theta.x, 1.0)

    def derived_parameters(self, theta):
        return {"x2": jnp.asarray(theta["x"]) ** 2}


def test_detect_typed_data():
    rng = np.random.default_rng(0)
    d = ToyMeasurement().sample(Theta(x=np.linspace(0, 10, 1000)), rng)
    detected = ThresholdTrigger(5.0).detect(d, rng)
    np.testing.assert_array_equal(detected, np.asarray(d.snr) > 5)


def test_detect_dict_data():
    class Positive(DetectionTrigger):
        def detection_probability(self, d):
            return np.asarray(d["snr"]) > 0

    d = {"snr": np.array([-1.0, 1.0, 2.0])}
    assert Positive().detect(d, np.random.default_rng(0)).tolist() == [False, True, True]


def test_injections_table():
    rng = np.random.default_rng(1)
    draws = pd.DataFrame({"x": rng.uniform(0, 10, 5000), "prior": 0.1})
    inj = ThresholdTrigger(5.0).injections(ToyMeasurement(), draws, rng)
    assert {"x", "x2", "prior", "observed_snr", "selection_statistic",
            "detection_probability", "detected"} <= set(inj)
    np.testing.assert_array_equal(inj["detected"], inj["observed_snr"] > 5)
    assert 0.3 < inj["detected"].mean() < 0.7


def test_trigger_display():
    html = ThresholdTrigger(8.0)._repr_html_()
    assert "threshold" in html and "8" in html and ".injections" in html


def test_parameters_mask_and_index_arrays():
    t = Theta(x=np.arange(5.0))
    np.testing.assert_array_equal(t[np.array([True, False, True, False, False])].x, [0.0, 2.0])
    np.testing.assert_array_equal(t[[4, 1]].x, [4.0, 1.0])
    assert t[np.int64(3)].batch_shape == ()

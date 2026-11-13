import jax
import numpy as np
import pandas as pd
import pytest
from astroinfer.detection import ThresholdTrigger

from astroinfer_gwmockcat import (
    GWMockCatData,
    GWMockCatMeasurement,
    GWParams,
    effective_spins,
    projection_factor_model,
)

jax.config.update("jax_enable_x64", True)


def truths(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(dict(
        mass_1_source=rng.uniform(10, 60, n), mass_ratio=rng.uniform(0.3, 1, n),
        redshift=rng.uniform(0.05, 0.8, n), projection_factor=rng.beta(2, 4, n),
        chi_1=rng.uniform(0, 1, n), chi_2=rng.uniform(0, 1, n),
        cos_tilt_1=rng.uniform(-1, 1, n), cos_tilt_2=rng.uniform(-1, 1, n)))


@pytest.fixture(scope="module")
def m():
    return GWMockCatMeasurement()


def test_sample_shapes_and_bounds(m):
    d = m.sample(truths(), np.random.default_rng(1))
    assert isinstance(d, GWMockCatData) and d.batch_shape == (4000,)
    for name, (lo, hi) in dict(symmetric_mass_ratio=(0, 0.25), projection_factor_obs=(0, 1),
                               chi_eff=(-1, 1), chi_p=(0, 1)).items():
        v = np.asarray(getattr(d, name))
        assert np.all(np.isfinite(v)) and v.min() >= lo and v.max() <= hi, name


def test_noise_is_centred_on_truth(m):
    t = truths(20_000)
    d = m.sample(t, np.random.default_rng(2))
    true = m.true_values({k: t[k].to_numpy() for k in t})
    np.testing.assert_allclose(np.mean(np.asarray(d.snr) - np.asarray(true["snr"])), 0, atol=0.03)
    np.testing.assert_allclose(np.mean(np.asarray(d.log_chirp_mass_det) - np.asarray(true["log_chirp_mass_det"])),
                               0, atol=0.01)


def test_logpdf_finite_and_peaks_near_truth(m):
    t = truths(50)
    d = m.sample(t, np.random.default_rng(3))
    lp = np.asarray(m.logpdf(d, t))
    assert lp.shape == (50,) and np.all(np.isfinite(lp))
    t_off = t.assign(mass_1_source=t.mass_1_source * 1.5)
    assert np.mean(np.asarray(m.logpdf(d, t_off)) < lp) > 0.9


def test_derived_and_effective_spins(m):
    t = truths(10)
    out = m.with_derived(t)
    assert {"mass_2_source", "chirp_mass_det", "luminosity_distance", "chi_eff", "chi_p"} <= set(out)
    chi_eff, _ = effective_spins(t.iloc[0])
    np.testing.assert_allclose(out.chi_eff.iloc[0], chi_eff)


def test_detection_and_injections(m):
    rng = np.random.default_rng(4)
    t = truths(5000)
    d = m.sample(t, rng)
    detected = ThresholdTrigger(8.0).detect(d, rng)
    assert 0 < detected.mean() < 1
    inj = ThresholdTrigger(8.0).injections(m, t.assign(prior=1.0), rng)
    assert {"observed_snr", "observed_chi_eff", "detected", "chi_eff"} <= set(inj)


def test_projection_factor_model():
    model = projection_factor_model()
    df = model.sample(pd.DataFrame([{}]), oversample=1000, random_state=0)
    assert df.projection_factor.between(0, 1).all()
    assert abs(df.projection_factor.mean() - 1 / 3) < 0.03


def test_single_event_typing(m):
    d = m.sample(GWParams(**{k: v[0] for k, v in truths(1).items()}), np.random.default_rng(5))
    assert d.batch_shape == ()


def test_display(m):
    html = m._repr_html_()
    assert "Model" in html and "Configuration" in html and "u_chi_p" in html

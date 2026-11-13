"""Tests that need no Julia: config, fit keys, padding, registration."""
import sys

import numpy as np
import pytest
from astroinfer.representations import get_fitter_class

from astroinfer_tgmm import TGMMConfig, TGMMFit, TGMMFitter, TransformationSpec, pad_components


def cfg(**kw):
    base = dict(columns=["x", "y"], K=2, limits={"x": (-10, 10), "y": (0, 1)})
    return TGMMConfig(**{**base, **kw})


def test_import_does_not_start_julia():
    assert "truncatedgaussianmixtures" not in sys.modules
    assert "juliacall" not in sys.modules


def test_config_round_trip(tmp_path):
    c = cfg(transformation=TransformationSpec("lambda x, y: (x, jnp.log(y))",
                                              "lambda x, ly: (x, jnp.exp(ly))", ["x", "log_y"]),
            quantiled_columns=["x"], block_structure=[0, 1], cov="full",
            analytic_columns=["x"], sampled_columns=["y"])
    c.save(tmp_path / "c.json")
    assert TGMMConfig.load(tmp_path / "c.json") == c
    assert c.image_columns == ["x_quantile", "log_y"]
    assert c.missing_limits() == ["log_y"]


@pytest.mark.parametrize("bad, match", [
    (dict(columns=[]), "empty"),
    (dict(columns="x"), "not a string"),
    (dict(quantiled_columns=["z"]), "not in columns"),
    (dict(analytic_columns=["x"], sampled_columns=["x"]), "both analytic and sampled"),
    (dict(carry_columns=["x"]), "both fitted and carried"),
    (dict(carry_columns=["prior"]), "carried automatically"),
    (dict(block_structure=[0]), "block_structure"),
    (dict(fit_kwargs={"f": object()}), "JSON-serializable"),
    (dict(fit_kwargs={"seed": 3}), "TGMMConfig.seed"),
    (dict(cov="weird"), "cov"),
])
def test_config_validation(bad, match):
    with pytest.raises((ValueError, TypeError), match=match):
        cfg(**bad)


def test_transformation_must_be_strings():
    with pytest.raises(TypeError, match="string"):
        TransformationSpec(lambda x: x, lambda x: x)
    with pytest.raises(ValueError, match="both"):
        TransformationSpec("lambda x: x", None)


def test_fit_key_ignores_dataproduct_fields():
    f = TGMMFitter(cfg())
    key = f.fit_key("ev", "src")
    f.update_default_config(analytic_columns=["x"], sampled_columns=["y"], N=50)
    assert f.fit_key("ev", "src") == key
    assert f.dataproduct_options("ev") == {"analytic_columns": ["x"], "sampled_columns": ["y"], "N": 50}
    f.update_event_config("ev", K=5)
    assert f.fit_key("ev", "src") != key
    assert f.fit_key("other", "src") == key
    assert f.fit_key("other", "src2") != key
    f.set_event_config("ev", None)
    assert f.fit_key("ev", "src") == key


def test_fitter_round_trip():
    f = TGMMFitter(cfg(), {"ev3": cfg(K=7)}, max_attempts=5)
    g = TGMMFitter.from_dict(f.to_dict())
    assert g == f and g.config_for("ev3").K == 7 and g.max_attempts == 5


def test_tgmm_kind_registered():
    assert get_fitter_class("tgmm") is TGMMFitter


def test_padding_uses_weight_zero_and_heaviest_component():
    product = {
        "weights": np.array([0.3, 0.7]),
        "x_mu_kernel": np.array([1.0, 2.0]),
        "x_sigma_kernel": np.array([0.1, 0.2]),
        "y": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    }
    out = pad_components(product, 4)
    np.testing.assert_array_equal(out["weights"], [0.3, 0.7, 0, 0])
    np.testing.assert_array_equal(out["x_sigma_kernel"], [0.1, 0.2, 0.2, 0.2])
    np.testing.assert_array_equal(out["y"][2:], [[0.4, 0.5, 0.6]] * 2)
    with pytest.raises(ValueError, match="down"):
        pad_components(product, 1)
    with pytest.raises(ValueError, match="leading component axis"):
        pad_components({**product, "bad": np.zeros(3)}, 4)


def test_stack_pads_to_max_K():
    p2 = {"weights": np.array([0.5, 0.5]), "x_mu_kernel": np.array([0.0, 1.0])}
    p3 = {"weights": np.array([0.2, 0.3, 0.5]), "x_mu_kernel": np.array([0.0, 1.0, 2.0])}
    out = TGMMFit.stack([p2, p3])
    assert out["weights"].shape == (2, 3)
    np.testing.assert_allclose(out["weights"].sum(axis=1), 1.0)
    np.testing.assert_allclose(out["x_mu_kernel"][0], [0.0, 1.0, 0.0])


def test_displays():
    c = cfg(transformation=TransformationSpec("lambda x, y: (x, jnp.log(y))", "lambda x, ly: (x, jnp.exp(ly))",
                                              ["x", "log_y"]), analytic_columns=["x"])
    html = c._repr_html_()
    assert "TGMM config" in html and "<em>missing</em>" in html and "analytic" in html
    f = TGMMFitter(cfg(), {"ev3": cfg(K=7)})
    html = f._repr_html_()
    assert "Per-event configs" in html and "K=7" in html and "Retry policy" in html


def test_one_column_sector_probabilities_without_julia():
    from types import SimpleNamespace

    from scipy.stats import norm

    from astroinfer_tgmm._tgm import (
        components_far_from_domain,
        log_interval_probability,
        sector_log_total_probs,
        sectors_from_block_structure,
    )

    # exact in the bulk, finite and sensible far in the tail
    np.testing.assert_allclose(log_interval_probability(0.0, 1.0, -1.0, 2.0),
                               np.log(norm.cdf(2) - norm.cdf(-1)))
    far = log_interval_probability(np.array([0.0, 50.0]), 1.0, -1.0, 1.0)
    assert np.isfinite(far[1]) and far[1] < -1000

    gmm = SimpleNamespace(means=np.array([[0.5, 0.5], [0.5, 30.0]]),
                          covariances=np.array([np.diag([0.01, 0.01]), np.diag([0.01, 1.0])]),
                          lower=np.array([0.0, 0.0]), upper=np.array([1.0, 1.0]), log_normalizers=None)
    sectors = sectors_from_block_structure([0, 1], 2)
    assert list(sectors) == ["block_0", "block_1"]
    logtps = sector_log_total_probs(gmm, sectors)
    assert logtps["block_0"].shape == (2,)
    assert components_far_from_domain(gmm, sectors) == {"block_1": [1]}

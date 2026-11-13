"""Fit real TGMMs (needs Julia): fit, stale/refit, padded stacking, save/load."""
import numpy as np
import pytest
from astroinfer import Catalog, PosteriorData, RawSamples, parameter_dataclass

from astroinfer_tgmm import TGMMConfig, TGMMFit, TGMMFitter, TransformationSpec

pytestmark = pytest.mark.julia


@parameter_dataclass
class XY:
    x: float
    y: float
    z: float


def make_posterior(seed, n=3000):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.5 * seed, 1.0, n)
    y = np.clip(rng.beta(2 + seed, 5, n), 1e-6, 1 - 1e-6)
    z = rng.uniform(0, 1, n)
    return PosteriorData(
        theta=XY(x=x, y=y, z=z),
        sample_stats={"log_prior[x]": np.full(n, -np.log(20.0)), "log_prior[y]": np.zeros(n),
                      "log_prior[z]": np.zeros(n), "prior": np.full(n, 1 / 20.0)},
    )


QUIET = {"method": "hybrid", "backend": "jax", "progress": False}


@pytest.fixture(scope="module")
def catalog():
    cat = Catalog({f"ev{i}": make_posterior(i) for i in range(3)}, raw=RawSamples(["x", "y"], N=200))
    config = TGMMConfig(columns=["x", "y"], K=2, cov="full", limits={"x": (-10, 10), "y": (0, 1)},
                        n_samples=2000, analytic_columns=["x"], sampled_columns=["y"], N=50,
                        fit_kwargs=QUIET)
    cat.register_representation("tgmm", TGMMFitter(config, {"ev2": config.with_changes(K=3)}))
    cat.fit("tgmm")
    return cat


def test_fit_and_padded_dataproduct(catalog):
    assert catalog.missing("tgmm") == [] and catalog.stale("tgmm") == []
    fit = catalog.get_fit("ev0", "tgmm")
    assert isinstance(fit, TGMMFit) and fit.K == 2 and catalog.get_fit("ev2", "tgmm").K == 3
    assert fit.diagnostics["seed"] == 0 and fit.diagnostics["n_draws"] == 2000
    html = fit._repr_html_()
    assert "Components (by weight)" in html and "Truncation box" in html
    assert "tgmm" in catalog._repr_html_()

    dp = catalog.dataproduct(representation="tgmm")
    assert dp["weights"].shape == (3, 3)
    np.testing.assert_array_equal(dp["weights"][0, 2], 0.0)
    np.testing.assert_allclose(dp["weights"].sum(axis=1), 1.0, rtol=1e-6)
    assert dp["y"].shape == (3, 3, 50) and dp["prior"].shape == (3, 3, 50)
    np.testing.assert_allclose(dp["prior"], 1 / 20.0)     # marginal prior of (x, y)
    for key in ("x_mu_kernel", "x_sigma_kernel", "y", "prior"):
        assert np.all(np.isfinite(dp[key])), key

    again = catalog.dataproduct(representation="tgmm", eventnames=["ev1"])
    np.testing.assert_array_equal(again["y"][0, :2], dp["y"][1, :2])


def test_config_change_refits_only_that_event(catalog):
    fitter = catalog.fitter("tgmm")
    fitter.update_event_config("ev1", N=10)              # data product only
    assert catalog.stale("tgmm") == []
    assert catalog.dataproduct(representation="tgmm", eventnames=["ev1"])["y"].shape == (1, 2, 10)
    fitter.update_event_config("ev1", K=1)
    assert catalog.stale("tgmm") == ["ev1"]
    assert catalog.fit("tgmm") == ["ev1"]
    assert catalog.get_fit("ev1", "tgmm").K == 1
    fitter.set_event_config("ev1", None)
    catalog.fit("tgmm")


def test_save_load(tmp_path, catalog):
    root = catalog.save(tmp_path / "cat")
    assert (root / "representations/tgmm/events/ev0/gmm.h5").exists()
    loaded = Catalog.load(root)
    assert loaded.stale("tgmm") == []
    assert loaded.fitter("tgmm") == catalog.fitter("tgmm")
    a = catalog.dataproduct(representation="tgmm")
    b = loaded.dataproduct(representation="tgmm")
    for key in a:
        np.testing.assert_allclose(a[key], b[key], err_msg=key)

    loaded.add_event("ev0", make_posterior(10), overwrite=True)
    assert loaded.stale("tgmm") == ["ev0"]


def test_transformation_quantile_and_weights():
    post = make_posterior(1)
    post.extras["log_w"] = np.zeros(post.batch_shape[0])
    cat = Catalog({"a": post}, raw=["x", "y"])
    config = TGMMConfig(columns=["x", "y"], K=2, quantiled_columns=["x"],
                        transformation=TransformationSpec("lambda x, y: (x, jnp.log(y / (1 - y)))",
                                                          "lambda x, ly: (x, 1 / (1 + jnp.exp(-ly)))",
                                                          ["x", "logit_y"]),
                        limits={"logit_y": (-20, 20)}, log_weight_column="log_w",
                        carry_columns=["z"], analytic_columns=["x", "y"], fit_kwargs=QUIET)
    cat.register_representation("tgmm", TGMMFitter(config))
    cat.fit("tgmm")
    fit = cat.get_fit("a", "tgmm")
    assert fit.image_columns == ["x_quantile", "logit_y"]
    draws = fit.sample(500, seed=1)
    assert set(draws.columns) >= {"x", "y"} and draws["y"].between(0, 1).all()
    dp = cat.dataproduct(representation="tgmm")
    assert set(dp) >= {"weights", "x_mu_kernel", "y_mu_kernel"}


def test_one_column_block_structure():
    """A covariance block with a single column (z alone) must fit and pass the domain check."""
    cat = Catalog({"a": make_posterior(0)}, raw=["x", "y"])
    config = TGMMConfig(columns=["x", "y", "z"], K=2, cov="full", block_structure=[0, 0, 1],
                        limits={"x": (-10, 10), "y": (0, 1), "z": (0, 1)},
                        analytic_columns=["x", "y"], sampled_columns=["z"], fit_kwargs=QUIET)
    cat.register_representation("tgmm", TGMMFitter(config))
    assert cat.fit("tgmm") == ["a"]
    assert cat.dataproduct(representation="tgmm")["z"].shape == (1, 2, 1000)

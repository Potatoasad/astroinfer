import gravpop as gp
import jax
import numpy as np
import pandas as pd
import pytest

from astroinfer_gw.gwmockcat import GWMockCatMeasurement
from astroinfer_gw.gwmockcat.pe_priors import DetectorFramePrior, pe_prior_comoving, pe_prior_luminosity

jax.config.update("jax_enable_x64", True)

MASS_DET = (2.0, 150.0)
Z_MAX = 1.0


@pytest.fixture(scope="module", params=["comoving", "luminosity"])
def prior(request):
    return DetectorFramePrior(request.param, mass_det=MASS_DET, redshift_max=Z_MAX)


def uniform_box(n, seed=0):
    """Uniform draws over a box containing the support of the prior, and its volume."""
    rng = np.random.default_rng(seed)
    box = dict(mass_1_source=(MASS_DET[0] / (1 + Z_MAX), MASS_DET[1]), mass_ratio=(0.0, 1.0),
               redshift=(0.0, Z_MAX))
    draws = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in box.items()}
    return draws, np.prod([hi - lo for lo, hi in box.values()])


def test_log_prob_is_normalized(prior):
    draws, volume = uniform_box(2_000_000)
    density = np.exp(np.asarray(prior.log_prob(draws)))
    np.testing.assert_allclose(density.mean() * volume, 1.0, rtol=0.02)


def test_log_prob_is_minus_inf_outside_support(prior):
    theta = dict(mass_1_source=np.array([1.0, 100.0, 50.0]), mass_ratio=np.array([0.5, 0.9, 0.5]),
                 redshift=np.array([0.1, 1.0, 1.2]))
    assert np.all(np.isneginf(np.asarray(prior.log_prob(theta))))


class NoLikelihood:
    def logpdf(self, x):
        return 0.0


def test_block_samples_match_log_prob(prior):
    sampler = gp.Sampler(priors={"detector_frame": prior}, likelihood=NoLikelihood(), num_samples=20_000,
                         num_warmup=1_000, seed=1, summary=False, mcmc_kwargs={"progress_bar": False})
    sampler.sample()
    draws = sampler.samples
    assert np.all(draws.mass_2_det <= draws.mass_1_det)
    np.testing.assert_allclose(draws.mass_ratio, draws.mass_2_det / draws.mass_1_det)

    box, _ = uniform_box(2_000_000, seed=2)
    weights = np.exp(np.asarray(prior.log_prob(box)))
    for name in DetectorFramePrior.var_names:
        expected = np.average(box[name], weights=weights)
        np.testing.assert_allclose(draws[name].mean(), expected, rtol=0.03, err_msg=name)


@pytest.mark.parametrize("make_priors", [pe_prior_comoving, pe_prior_luminosity])
def test_pe_records_joint_prior_factor(make_priors):
    m = GWMockCatMeasurement()
    theta = pd.DataFrame([dict(mass_1_source=30.0, mass_ratio=0.8, redshift=0.2, projection_factor=0.6,
                               chi_1=0.3, chi_2=0.4, cos_tilt_1=0.2, cos_tilt_2=-0.1)])
    d = m.sample(theta, np.random.default_rng(0))[0]
    priors = make_priors(mass_det=MASS_DET, redshift_max=Z_MAX, cosmology=m.cosmology)
    post = m.PE(d, priors, seed=0, num_warmup=300, num_samples=500, summary=False,
                mcmc_kwargs={"progress_bar": False}, as_posterior=True)

    factor = "log_prior[mass_1_source, mass_ratio, redshift]"
    assert factor in post.sample_stats
    assert np.all(np.isfinite(np.asarray(post.sample_stats["prior"])))
    joint = np.exp(np.asarray(post.sample_stats[factor]))
    np.testing.assert_allclose(post.marginal_prior(["mass_1_source", "mass_ratio", "redshift"]), joint)
    with pytest.raises(ValueError, match="joint prior factor"):
        post.marginal_prior(["mass_1_source"])
    assert "mass_1_det" in post.extras


def test_pe_requires_a_density_for_every_field():
    m = GWMockCatMeasurement()
    priors = pe_prior_comoving()
    priors["detector_frame"] = gp.Deterministic(lambda x: None)
    with pytest.raises(KeyError, match="no density"):
        m.PE(None, priors)

"""GWMockCat-style measurement process (Farah et al. 2023, App. A) with
effective spins added, as a typed astroinfer ``MeasurementProcess``."""
import jax
import jax.numpy as jnp
from astroinfer import MeasurementProcess, parameter, parameter_dataclass
from jax.scipy.stats import norm, truncnorm
from numpyro.distributions import constraints

from .physics import (
    CHI_EFF_BOUNDS,
    CHI_P_BOUNDS,
    ETA_BOUNDS,
    PROJECTION_BOUNDS,
    RedshiftToLuminosityDistance,
    ToyOptimalSNR,
    chirp_mass_from_mass_ratio,
    effective_spins,
    symmetric_mass_ratio,
)


@parameter_dataclass(name="GW source", latex=r"\theta")
class GWParams:
    mass_1_source: float = parameter(name="Primary mass (source)", latex=r"m_1")
    mass_ratio: float = parameter(name="Mass ratio", latex=r"q")
    redshift: float = parameter(name="Redshift", latex=r"z")
    projection_factor: float = parameter(name="Projection factor", latex=r"\Theta")
    chi_1: float = parameter(name="Primary spin magnitude", latex=r"\chi_1")
    chi_2: float = parameter(name="Secondary spin magnitude", latex=r"\chi_2")
    cos_tilt_1: float = parameter(name="Primary tilt cosine", latex=r"\cos\theta_1")
    cos_tilt_2: float = parameter(name="Secondary tilt cosine", latex=r"\cos\theta_2")


@parameter_dataclass(name="GWMockCat data", latex=r"d")
class GWMockCatData:
    snr: float = parameter(name="Observed SNR", latex=r"\hat\rho")
    log_chirp_mass_det: float = parameter(name="Log detector-frame chirp mass", latex=r"\ln\hat{\mathcal{M}}_\mathrm{det}")
    symmetric_mass_ratio: float = parameter(name="Symmetric mass ratio", latex=r"\hat\eta",
                                            support=constraints.interval(*ETA_BOUNDS))
    projection_factor_obs: float = parameter(name="Observed projection factor", latex=r"\hat\Theta",
                                             support=constraints.interval(*PROJECTION_BOUNDS))
    chi_eff: float = parameter(name="Effective spin", latex=r"\hat\chi_\mathrm{eff}",
                               support=constraints.interval(*CHI_EFF_BOUNDS))
    chi_p: float = parameter(name="Precessing spin", latex=r"\hat\chi_p",
                             support=constraints.interval(*CHI_P_BOUNDS))


def _truncnorm_draw(key, loc, scale, bounds):
    lo, hi = bounds
    return loc + scale * jax.random.truncated_normal(key, (lo - loc) / scale, (hi - loc) / scale)


def _truncnorm_logpdf(x, loc, scale, bounds):
    lo, hi = bounds
    return truncnorm.logpdf(x, (lo - loc) / scale, (hi - loc) / scale, loc=loc, scale=scale)


class GWMockCatMeasurement(MeasurementProcess):
    """K(d | theta) of a GWMockCat-style catalog with effective spins.

    The data are made from the detector-frame Mc_det and d_L. With
    rho = snr_model(theta) * Theta and w = 1 / max(snr, snr_floor):

        snr                    ~ N(rho, 1)
        log_chirp_mass_det     ~ N(ln Mc_det, u_chirp_mass w)
        symmetric_mass_ratio   ~ TN(eta(q), u_eta w)               on [0, 1/4]
        projection_factor_obs  ~ TN(Theta, u_projection w)         on [0, 1]
        chi_eff                ~ TN(chi_eff(theta), u_chi_eff w)   on [-1, 1]
        chi_p                  ~ TN(chi_p(theta), u_chi_p w)       on [0, 1]

    Derived parameters (added to truths, injections and posteriors):
    mass_2_source, chirp_mass_det, luminosity_distance, chi_eff, chi_p.
    """

    parameter_type = GWParams
    data_type = GWMockCatData
    display_name = "GWMockCat measurement"

    def __init__(self, cosmology=None, snr_model=None, *, u_chirp_mass=0.08, u_eta=0.022,
                 u_projection=0.21, u_chi_eff=0.8, u_chi_p=7.2, snr_floor=1.0, sampler_kwargs=None,
                 key=None):
        super().__init__(**({} if key is None else {"key": key}))
        self.cosmology = RedshiftToLuminosityDistance() if cosmology is None else cosmology
        self.snr_model = ToyOptimalSNR() if snr_model is None else snr_model
        self.u_chirp_mass = u_chirp_mass
        self.u_eta = u_eta
        self.u_projection = u_projection
        self.u_chi_eff = u_chi_eff
        self.u_chi_p = u_chi_p
        self.snr_floor = snr_floor
        self.sampler_kwargs = {**MeasurementProcess.sampler_kwargs,
                               "num_warmup": 1_000, "sampler_kwargs": {"dense_mass": True},
                               **(sampler_kwargs or {})}

    def __repr__(self):
        return f"GWMockCatMeasurement(snr_model={self.snr_model}, cosmology={self.cosmology})"

    # ========================================================
    # RichDisplay
    # ========================================================

    def _display_equations(self):
        w = r"\,w"
        return [
            r"\rho = \rho_\mathrm{opt}(\mathcal{M}_\mathrm{det}, q, d_L)\,\Theta,\qquad "
            r"w = 1/\max(\hat\rho, " + f"{self.snr_floor:g}" + ")",
            r"\hat\rho \sim \mathcal{N}(\rho, 1)",
            rf"\ln\hat{{\mathcal{{M}}}}_\mathrm{{det}} \sim \mathcal{{N}}(\ln\mathcal{{M}}_\mathrm{{det}},\ {self.u_chirp_mass:g}{w})",
            rf"\hat\eta \sim \mathcal{{TN}}_{{[0,1/4]}}(\eta(q),\ {self.u_eta:g}{w})",
            rf"\hat\Theta \sim \mathcal{{TN}}_{{[0,1]}}(\Theta,\ {self.u_projection:g}{w})",
            rf"\hat\chi_\mathrm{{eff}} \sim \mathcal{{TN}}_{{[-1,1]}}(\chi_\mathrm{{eff}}(\theta),\ {self.u_chi_eff:g}{w})",
            rf"\hat\chi_p \sim \mathcal{{TN}}_{{[0,1]}}(\chi_p(\theta),\ {self.u_chi_p:g}{w})",
        ]

    def _display_config(self):
        return {"snr_model": repr(self.snr_model), "cosmology": repr(self.cosmology),
                "u_chirp_mass": self.u_chirp_mass, "u_eta": self.u_eta, "u_projection": self.u_projection,
                "u_chi_eff": self.u_chi_eff, "u_chi_p": self.u_chi_p, "snr_floor": self.snr_floor,
                "sampler_kwargs": self.sampler_kwargs}

    # ========================================================
    # Noise-free quantities
    # ========================================================

    def detector_frame(self, theta):
        z = theta["redshift"]
        return {"mass_2_source": theta["mass_ratio"] * theta["mass_1_source"],
                "chirp_mass_det": chirp_mass_from_mass_ratio(theta["mass_1_source"], theta["mass_ratio"]) * (1 + z),
                "luminosity_distance": self.cosmology.luminosity_distance(z)}

    def true_snr(self, theta):
        frame = self.detector_frame(theta)
        return self.snr_model({"mass_ratio": theta["mass_ratio"], **frame}) * theta["projection_factor"]

    def true_values(self, theta) -> dict:
        """theta -> the noise-free data (excluding the SNR's own noise)."""
        chi_eff, chi_p = effective_spins(theta)
        return {"snr": self.true_snr(theta),
                "log_chirp_mass_det": jnp.log(self.detector_frame(theta)["chirp_mass_det"]),
                "symmetric_mass_ratio": symmetric_mass_ratio(theta["mass_ratio"]),
                "projection_factor_obs": theta["projection_factor"],
                "chi_eff": chi_eff,
                "chi_p": chi_p}

    def derived_parameters(self, theta):
        chi_eff, chi_p = effective_spins(theta)
        return {**self.detector_frame(theta), "chi_eff": chi_eff, "chi_p": chi_p}

    def _widths(self, snr):
        w = 1.0 / jnp.maximum(snr, self.snr_floor)
        return {"log_chirp_mass_det": self.u_chirp_mass * w, "symmetric_mass_ratio": self.u_eta * w,
                "projection_factor_obs": self.u_projection * w, "chi_eff": self.u_chi_eff * w,
                "chi_p": self.u_chi_p * w}

    # ========================================================
    # MeasurementProcess
    # ========================================================

    def sample_single(self, theta, key):
        keys = jax.random.split(key, 6)
        mean = self.true_values(theta)
        snr = mean["snr"] + jax.random.normal(keys[0])
        width = self._widths(snr)
        return GWMockCatData(
            snr=snr,
            log_chirp_mass_det=mean["log_chirp_mass_det"]
            + width["log_chirp_mass_det"] * jax.random.normal(keys[1]),
            symmetric_mass_ratio=_truncnorm_draw(keys[2], mean["symmetric_mass_ratio"],
                                                 width["symmetric_mass_ratio"], ETA_BOUNDS),
            projection_factor_obs=_truncnorm_draw(keys[3], mean["projection_factor_obs"],
                                                  width["projection_factor_obs"], PROJECTION_BOUNDS),
            chi_eff=_truncnorm_draw(keys[4], mean["chi_eff"], width["chi_eff"], CHI_EFF_BOUNDS),
            chi_p=_truncnorm_draw(keys[5], mean["chi_p"], width["chi_p"], CHI_P_BOUNDS),
        )

    def logpdf_single(self, d, theta):
        mean = self.true_values(theta)
        width = self._widths(d.snr)
        return (norm.logpdf(d.snr, mean["snr"], 1.0)
                + norm.logpdf(d.log_chirp_mass_det, mean["log_chirp_mass_det"], width["log_chirp_mass_det"])
                + _truncnorm_logpdf(d.symmetric_mass_ratio, mean["symmetric_mass_ratio"],
                                    width["symmetric_mass_ratio"], ETA_BOUNDS)
                + _truncnorm_logpdf(d.projection_factor_obs, mean["projection_factor_obs"],
                                    width["projection_factor_obs"], PROJECTION_BOUNDS)
                + _truncnorm_logpdf(d.chi_eff, mean["chi_eff"], width["chi_eff"], CHI_EFF_BOUNDS)
                + _truncnorm_logpdf(d.chi_p, mean["chi_p"], width["chi_p"], CHI_P_BOUNDS))

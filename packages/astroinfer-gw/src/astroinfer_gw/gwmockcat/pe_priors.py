"""PE priors for the GWMockCat measurement.

Both priors are flat in the detector-frame component masses (m_2 <= m_1),
isotropic in the projection factor, uniform in the spin magnitudes and
isotropic in the spin tilts. They differ in the distance prior:

    pe_prior_comoving     uniform in comoving volume, p(z) proportional to dV_c/dz
    pe_prior_luminosity   uniform in luminosity distance

Each returns a dict for ``measurement.PE``. The masses and distance are one
``DetectorFramePrior`` block, a gravpop prior block that samples the
detector-frame masses and the distance, sets ``mass_1_source``,
``mass_ratio`` and ``redshift`` from them, and gives the joint prior density
of those three. The other fields have numpyro distributions.

The priors are specific to GWMockCat: the projection factor Theta stands in
for the sky position, inclination and polarization of a real PE prior.
"""
import gravpop as gp
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from astroinfer.display import RichDisplay, kv_table, math_block, method_row, section

from ..physics import RedshiftToLuminosityDistance, isotropic_projection_factor

DISTANCE_PRIORS = ("comoving", "luminosity")


class DetectorFramePrior(gp.PriorBlock, RichDisplay):
    """Prior flat in the detector-frame masses m_1 >= m_2, and uniform in
    comoving volume or in luminosity distance.

    In the sampler it draws ``mass_1_det``, ``mass_2_det`` and either
    ``redshift`` (comoving) or ``luminosity_distance`` (luminosity), then sets

        mass_1_source = mass_1_det / (1 + z),   mass_ratio = mass_2_det / mass_1_det.

    ``log_prob`` is the normalized joint density of (mass_1_source,
    mass_ratio, redshift), the coordinates the measurement uses:

        p(m_1, q, z) = p(m_1^det, m_2^det) p(z) m_1 (1 + z)^2

    with p(z) proportional to dV_c/dz, or p(z) = p(d_L) dd_L/dz for a flat d_L.

    Parameters
    ----------
    distance : {"comoving", "luminosity"}
        Uniform in comoving volume, or uniform in luminosity distance.
    mass_det : (float, float)
        Bounds of both detector-frame masses, in Msun.
    redshift_max : float
        Upper redshift bound; the flat-d_L prior runs to d_L(redshift_max).
    cosmology : RedshiftToLuminosityDistance, optional
        Use the measurement's (``measurement.cosmology``). Default: Planck15.
    """

    var_names = ("mass_1_source", "mass_ratio", "redshift")
    display_name = "Detector-frame PE prior"

    def __init__(self, distance="comoving", mass_det=(1.0, 300.0), redshift_max=1.5, cosmology=None):
        if distance not in DISTANCE_PRIORS:
            raise ValueError(f"distance must be one of {DISTANCE_PRIORS}, got {distance!r}")
        self.distance = distance
        self.mass_det = (float(mass_det[0]), float(mass_det[1]))
        if not 0 < self.mass_det[0] < self.mass_det[1]:
            raise ValueError(f"mass_det must satisfy 0 < low < high, got {mass_det}")
        self.cosmology = RedshiftToLuminosityDistance() if cosmology is None else cosmology
        self.redshift_max = float(redshift_max)
        if not 0 < self.redshift_max <= self.cosmology.z_max:
            raise ValueError(f"redshift_max must be in (0, {self.cosmology.z_max}] for {self.cosmology}")
        self.luminosity_distance_max = float(self.cosmology.luminosity_distance(self.redshift_max))

        low, high = self.mass_det
        self._log_mass_area = np.log((high - low) ** 2 / 2)
        z = np.linspace(0, self.redshift_max, 4001)
        self._log_comoving_volume = np.log(np.trapezoid(
            np.asarray(self.cosmology.differential_comoving_volume(z)), z))

    def __repr__(self):
        return (f"DetectorFramePrior(distance={self.distance!r}, mass_det={self.mass_det}, "
                f"redshift_max={self.redshift_max})")

    # ========================================================
    # gravpop prior block
    # ========================================================

    def _source_frame(self, mass_1_det, mass_2_det, redshift):
        return {"mass_1_source": mass_1_det / (1 + redshift),
                "mass_ratio": mass_2_det / mass_1_det,
                "redshift": redshift}

    def sample(self, x):
        low, high = self.mass_det
        mass_1_det = numpyro.sample("mass_1_det", dist.Uniform(low, high))
        mass_2_det = numpyro.sample("mass_2_det", dist.Uniform(low, mass_1_det))
        # p(m2 | m1) = 1 / (m1 - low); this factor makes the joint flat on m2 <= m1.
        numpyro.factor("mass_det_flat_jacobian", jnp.log(mass_1_det - low))
        if self.distance == "comoving":
            redshift = numpyro.sample("redshift", dist.Uniform(0.0, self.redshift_max))
            numpyro.factor("comoving_volume_weight",
                           jnp.log(self.cosmology.differential_comoving_volume(redshift)))
        else:
            d_L = numpyro.sample("luminosity_distance", dist.Uniform(0.0, self.luminosity_distance_max))
            redshift = self.cosmology.redshift(d_L)
        x.update(self._source_frame(mass_1_det, mass_2_det, redshift))

    def cleanup(self, samples):
        if self.distance == "comoving":
            redshift = samples["redshift"]
        else:
            redshift = self.cosmology.redshift(samples["luminosity_distance"])
        samples.update(self._source_frame(samples["mass_1_det"], samples["mass_2_det"], redshift))

    # ========================================================
    # Density in the measurement's coordinates
    # ========================================================

    def log_prob(self, theta):
        """log p(mass_1_source, mass_ratio, redshift); -inf outside the support."""
        mass_1, q, z = theta["mass_1_source"], theta["mass_ratio"], theta["redshift"]
        low, high = self.mass_det
        mass_1_det = mass_1 * (1 + z)
        mass_2_det = q * mass_1_det
        in_support = ((mass_1_det <= high) & (mass_2_det >= low) & (q > 0) & (q <= 1)
                      & (z >= 0) & (z <= self.redshift_max))
        if self.distance == "comoving":
            log_p_z = jnp.log(self.cosmology.differential_comoving_volume(z)) - self._log_comoving_volume
        else:
            log_p_z = (jnp.log(self.cosmology.luminosity_distance_derivative(z))
                       - np.log(self.luminosity_distance_max))
        log_p = -self._log_mass_area + log_p_z + jnp.log(mass_1) + 2 * jnp.log1p(z)
        return jnp.where(in_support, log_p, -jnp.inf)

    # ========================================================
    # RichDisplay
    # ========================================================

    def _display_math(self):
        return r"\pi(m_1, q, z)"

    def _display_body(self):
        low, high = self.mass_det
        if self.distance == "comoving":
            distance = r"p(z) \propto \frac{dV_c}{dz},\quad 0 \le z \le " + f"{self.redshift_max:g}"
        else:
            distance = (r"p(d_L) = \mathrm{const},\quad 0 \le d_L \le "
                        + f"{self.luminosity_distance_max:.0f}" + r"\,\mathrm{Mpc}")
        rules = [
            r"p(m_1^\mathrm{det}, m_2^\mathrm{det}) = \mathrm{const},\quad "
            + f"{low:g}" + r" \le m_2^\mathrm{det} \le m_1^\mathrm{det} \le " + f"{high:g}",
            distance,
            r"\pi(m_1, q, z) = p(m_1^\mathrm{det}, m_2^\mathrm{det})\, p(z)\, m_1 (1+z)^2",
        ]
        config = {"distance": self.distance, "mass_det": self.mass_det,
                  "redshift_max": self.redshift_max,
                  "luminosity_distance_max": round(self.luminosity_distance_max, 1),
                  "cosmology": repr(self.cosmology)}
        methods = section("Methods", "".join([
            method_row(".log_prob", r"(\theta)\longrightarrow \log\pi(m_1, q, z)",
                       "joint log density in the measurement's coordinates"),
            method_row(".sample", r"(x)", "gravpop prior block: draws the detector-frame masses and distance"),
        ]))
        return math_block(rules) + kv_table(config) + methods


def _pe_prior(distance, mass_det, redshift_max, cosmology):
    return {
        "detector_frame": DetectorFramePrior(distance, mass_det, redshift_max, cosmology),
        "projection_factor": isotropic_projection_factor(),
        "chi_1": dist.Uniform(0.0, 1.0),
        "chi_2": dist.Uniform(0.0, 1.0),
        "cos_tilt_1": dist.Uniform(-1.0, 1.0),
        "cos_tilt_2": dist.Uniform(-1.0, 1.0),
    }


def pe_prior_comoving(mass_det=(1.0, 300.0), redshift_max=1.5, cosmology=None):
    """GWMockCat PE priors uniform in comoving volume; see ``DetectorFramePrior``."""
    return _pe_prior("comoving", mass_det, redshift_max, cosmology)


def pe_prior_luminosity(mass_det=(1.0, 300.0), redshift_max=1.5, cosmology=None):
    """GWMockCat PE priors uniform in luminosity distance; see ``DetectorFramePrior``."""
    return _pe_prior("luminosity", mass_det, redshift_max, cosmology)

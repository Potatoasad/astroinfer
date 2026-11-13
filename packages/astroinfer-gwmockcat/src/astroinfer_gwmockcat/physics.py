"""GW source physics for GWMockCat-style mock catalogs.

All functions take numpy or jnp inputs (and traced JAX values), so they
work on arrays, inside vmapped measurement methods, and inside samplers.
Mappings ``theta`` may be dicts, DataFrames rows or Parameters objects
(anything with ``theta["name"]``).
"""
import jax.numpy as jnp
import numpy as np

CHI_EFF_BOUNDS = (-1.0, 1.0)
CHI_P_BOUNDS = (0.0, 1.0)
ETA_BOUNDS = (0.0, 0.25)
PROJECTION_BOUNDS = (0.0, 1.0)


# ============================================================
# Masses and spins (q = m2 / m1 <= 1)
# ============================================================

def chirp_mass_from_mass_ratio(mass_1, mass_ratio):
    return mass_1 * mass_ratio**0.6 / (1 + mass_ratio) ** 0.2


def total_mass_from_chirp_mass(chirp_mass, mass_ratio):
    return chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio**0.6


def symmetric_mass_ratio(mass_ratio):
    return mass_ratio / (1 + mass_ratio) ** 2


def effective_spins(theta):
    """``(chi_eff, chi_p)`` from spin magnitudes and tilt cosines."""
    q = theta["mass_ratio"]
    c1, c2 = theta["cos_tilt_1"], theta["cos_tilt_2"]
    chi_eff = (theta["chi_1"] * c1 + q * theta["chi_2"] * c2) / (1 + q)
    chi_p = jnp.maximum(theta["chi_1"] * jnp.sqrt(1 - c1**2),
                        (4 * q + 3) / (4 + 3 * q) * q * theta["chi_2"] * jnp.sqrt(1 - c2**2))
    return chi_eff, chi_p


# ============================================================
# Cosmology and SNR
# ============================================================

class RedshiftToLuminosityDistance:
    """Luminosity distance of an astropy cosmology, tabulated in z (fast, and
    usable inside the sampler). Default: Planck15."""

    def __init__(self, cosmology=None, z_max=5.0, n_grid=5_000):
        if cosmology is None:
            from astropy.cosmology import Planck15 as cosmology
        self.z_max = float(z_max)
        self.z = np.linspace(0, z_max, n_grid + 1)
        self.d_L = cosmology.luminosity_distance(self.z).to("Mpc").value

    def __repr__(self):
        return f"RedshiftToLuminosityDistance(z_max={self.z_max})"

    def luminosity_distance(self, z):
        """d_L(z) in Mpc."""
        return jnp.interp(z, self.z, self.d_L)


class ToyOptimalSNR:
    """rho_opt(theta): the SNR of an optimally oriented source in one detector;
    a made-up stand-in for the PSD integral GWMockCat does with the O3 LIGO
    Livingston PSD.

        rho_opt = reference_snr * (Mc_det / reference_chirp_mass)^(5/6)
                  * (bns_horizon / d_L) / (1 + (M_det / m_cut)^2)

    The first factors are the inspiral scaling, normalized so a
    Mc_det = 1.2 Msun BNS is at ``reference_snr`` at ``bns_horizon`` Mpc. The
    last is a roll-off for heavy binaries whose merger leaves the band. The
    defaults are roughly one O3 detector: a 300 Mpc BNS horizon puts a
    30+30 Msun BBH at SNR ~ 20 at 1.6 Gpc (optimally oriented).
    """

    def __init__(self, bns_horizon=300.0, reference_snr=8.0, reference_chirp_mass=1.2, m_cut=250.0):
        self.bns_horizon = float(bns_horizon)
        self.reference_snr = float(reference_snr)
        self.reference_chirp_mass = float(reference_chirp_mass)
        self.m_cut = float(m_cut)

    def __repr__(self):
        return (f"ToyOptimalSNR(bns_horizon={self.bns_horizon}, reference_snr={self.reference_snr}, "
                f"m_cut={self.m_cut})")

    def __call__(self, theta):
        """``theta`` needs chirp_mass_det, luminosity_distance and mass_ratio."""
        chirp_mass, d_L = theta["chirp_mass_det"], theta["luminosity_distance"]
        total_mass = total_mass_from_chirp_mass(chirp_mass, theta["mass_ratio"])
        return (self.reference_snr * (chirp_mass / self.reference_chirp_mass) ** (5 / 6)
                * (self.bns_horizon / d_L) / (1 + (total_mass / self.m_cut) ** 2))


# ============================================================
# Projection factor
# ============================================================

def isotropic_projection_factor():
    """Theta in [0, 1], with rho = rho_opt * Theta, for an isotropic sky
    position and orientation in one detector: Finn & Chernoff's (1993) fit
    P(Theta_FC) = 5 Theta_FC (4 - Theta_FC)^3 / 256 with Theta = Theta_FC / 4,
    i.e. Beta(2, 4). A numpyro distribution (e.g. Theta's PE prior)."""
    import numpyro.distributions as dist

    return dist.Beta(2.0, 4.0)


def projection_factor_model(var_name="projection_factor"):
    """Theta ~ Beta(2, 4) as a gravpop population model (no hyperparameters),
    to put in the product of a population that draws truths or injections."""
    import gravpop as gp

    class ProjectionFactorModel(gp.SampledPopulationModel):
        def __init__(self, var_names=(var_name,)):
            self.var_names = list(var_names)
            self.hyper_var_names = []
            self._distribution = isotropic_projection_factor()

        @property
        def limits(self):
            return {self.var_names[0]: [0.0, 1.0]}

        def __call__(self, data, params):
            return jnp.exp(self._distribution.log_prob(data[self.var_names[0]]))

        def sample(self, df_hyper_samples, oversample=1, random_state=None):
            rng = (random_state if isinstance(random_state, np.random.Generator)
                   else np.random.default_rng(random_state))
            import pandas as pd

            return pd.DataFrame({self.var_names[0]: rng.beta(2.0, 4.0, len(df_hyper_samples) * oversample)})

    return ProjectionFactorModel()

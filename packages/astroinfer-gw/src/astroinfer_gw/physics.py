"""Gravitational-wave source physics: mass and spin conversions, the luminosity
distance and comoving volume, and the projection factor.

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
# Cosmology
# ============================================================

class RedshiftToLuminosityDistance:
    """Luminosity distance of an astropy cosmology, tabulated in z (fast, and
    usable inside the sampler), with its inverse, its derivative and the
    differential comoving volume. Default: Planck15."""

    def __init__(self, cosmology=None, z_max=5.0, n_grid=5_000):
        if cosmology is None:
            from astropy.cosmology import Planck15 as cosmology
        self.z_max = float(z_max)
        self.z = np.linspace(0, z_max, n_grid + 1)
        self.d_L = cosmology.luminosity_distance(self.z).to("Mpc").value
        self.dd_L_dz = np.gradient(self.d_L, self.z, edge_order=2)
        self.dV_c_dz = 4 * np.pi * cosmology.differential_comoving_volume(self.z).to("Mpc3 / sr").value

    def __repr__(self):
        return f"RedshiftToLuminosityDistance(z_max={self.z_max})"

    def luminosity_distance(self, z):
        """d_L(z) in Mpc."""
        return jnp.interp(z, self.z, self.d_L)

    def redshift(self, luminosity_distance):
        """z(d_L), the inverse of ``luminosity_distance``; d_L in Mpc."""
        return jnp.interp(luminosity_distance, self.d_L, self.z)

    def luminosity_distance_derivative(self, z):
        """dd_L/dz in Mpc."""
        return jnp.interp(z, self.z, self.dd_L_dz)

    def differential_comoving_volume(self, z):
        """dV_c/dz over the whole sky, in Mpc^3."""
        return jnp.interp(z, self.z, self.dV_c_dz)


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

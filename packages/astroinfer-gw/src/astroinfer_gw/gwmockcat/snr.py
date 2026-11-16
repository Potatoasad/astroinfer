"""The toy optimal-SNR model used by the GWMockCat-style measurement process."""
from ..physics import total_mass_from_chirp_mass


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

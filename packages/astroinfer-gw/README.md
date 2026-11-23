# astroinfer-gw

Gravitational-wave models for astroinfer.

- `astroinfer_gw.gwmockcat` is a GWMockCat-style measurement process (Farah et al. 2023,
  App. A) with effective spins added: `GWParams` for the source parameters, `GWMockCatData` for
  the data, `GWMockCatMeasurement` for $\mathcal{L}(d\mid\theta)$, and `ToyOptimalSNR`, the toy model of
  the optimal SNR that the measurement uses by default.
- `astroinfer_gw.physics` has general source physics: chirp mass and mass ratios, effective
  spins, the luminosity distance $d_L(z)$, and the projection factor.
- `astroinfer_gw.populations` has the NumPyro distributions a population over `GWParams` needs
  that NumPyro lacks: `PowerLaw`, a truncated power law with closed-form CDF and inverse CDF;
  `PowerLawRedshift`, $p(z)\propto \frac{dV_c}{dz}(1+z)^{\lambda-1}$ tabulated on a redshift
  grid; and `ProjectionFactor`, the isotropic $\Theta\sim\mathrm{Beta}(2,4)$ with a polynomial
  CDF. They can be fields of `GWParams.distribution(...)`, and convert with `as_generative`.

```python
import astroinfer as ai
import numpyro.distributions as dist
from astroinfer import parameter_dataclass
from astroinfer_gw.gwmockcat import GWMockCatMeasurement, GWParams
from astroinfer_gw.populations import PowerLaw, PowerLawRedshift, ProjectionFactor

measurement = GWMockCatMeasurement()

@parameter_dataclass
class Hyper:
    alpha: float
    beta: float
    lamb: float

population = GWParams.distribution(
    mass_1_source=lambda L: PowerLaw(L.alpha, 5.0, 80.0),
    mass_ratio=ai.conditional(
        lambda x, L: PowerLaw(L.beta, 5.0 / x["mass_1_source"], 1.0, support=(0.0, 1.0)),
        on="mass_1_source"),
    redshift=lambda L: PowerLawRedshift(L.lamb, z_max=1.5),
    projection_factor=ProjectionFactor(),
    chi_1=dist.Uniform(0.0, 1.0), chi_2=dist.Uniform(0.0, 1.0),
    cos_tilt_1=dist.Uniform(-1.0, 1.0), cos_tilt_2=dist.Uniform(-1.0, 1.0),
).given(Hyper)
theta = population.sample(Hyper(alpha=-2.35, beta=1.1, lamb=2.7), key=0, N=1000)
```

The mass ratio's lower bound depends on $m_1$, so its `PowerLaw` is given a fixed `support=`;
`as_generative` draws these distributions by bisection on their CDF, which needs a support with
fixed bounds.

The notebooks in [`notebooks/gw/`](../../notebooks/gw) build a mock catalog with this
measurement process and run population inference on it.

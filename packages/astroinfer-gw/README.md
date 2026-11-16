# astroinfer-gw

Gravitational-wave models for astroinfer.

- `astroinfer_gw.gwmockcat` is a GWMockCat-style measurement process (Farah et al. 2023,
  App. A) with effective spins added: `GWParams` for the source parameters, `GWMockCatData` for
  the data, `GWMockCatMeasurement` for $\mathcal{L}(d\mid\theta)$, and `ToyOptimalSNR`, the toy model of
  the optimal SNR that the measurement uses by default.
- `astroinfer_gw.physics` has general source physics: chirp mass and mass ratios, effective
  spins, the luminosity distance $d_L(z)$, and the projection factor.

```python
from astroinfer_gw.gwmockcat import GWMockCatMeasurement, GWParams

measurement = GWMockCatMeasurement()
```

The notebooks in [`notebooks/gw/`](../../notebooks/gw) build a mock catalog with this
measurement process and run population inference on it.

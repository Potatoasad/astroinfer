# astroinfer-tgmm

Truncated Gaussian mixture (TGMM) representations for `astroinfer` catalogs.
This is an extension package: `astroinfer` itself does not depend on
`truncatedgaussianmixtures` or Julia.

```python
from astroinfer import Catalog, RawSamples
from astroinfer_tgmm import TGMMConfig, TGMMFitter

catalog = Catalog(posteriors, raw=RawSamples(columns=["mass_1_source", "mass_ratio", "redshift"]))
catalog.register_representation("tgmm", TGMMFitter(TGMMConfig(
    columns=["mass_1_source", "mass_ratio", "redshift"], K=8, cov="full", block_structure=[0, 0, 1],
    limits={"mass_1_source": (2, 120), "mass_ratio": (0.05, 1), "redshift": (0, 1.5)},
    analytic_columns=["mass_1_source", "mass_ratio"], sampled_columns=["redshift"],
)))
catalog.fit("tgmm")                                   # missing or stale events only
catalog.fitter("tgmm").update_event_config("event_7", K=12)
catalog.fit("tgmm")                                   # refits just event_7
dp = catalog.dataproduct(representation="tgmm")       # components padded to max K (weight 0)
catalog.save("my_catalog")                            # incremental on re-save
catalog = Catalog.load("my_catalog")                  # finds 'tgmm' via the entry point
```

Tests: `uv run pytest -m "not julia"` (fast) or `uv run pytest` (fits real TGMMs).

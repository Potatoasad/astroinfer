# astroinfer

Typed forward models and hierarchical inference in JAX.

You describe parameters and data with `@parameter_dataclass`, write a `MeasurementProcess`
(`sample_single` / `logpdf_single` for one θ and one observation), and astroinfer handles
batching, random keys, parameter estimation, population models (`astroinfer.generative`),
population inference with selection effects, and catalogs of posteriors with pluggable
representations.

```bash
pip install "astroinfer @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer"
```

See the [tutorials](https://github.com/Potatoasad/astroinfer/tree/main/tutorials) to get started.

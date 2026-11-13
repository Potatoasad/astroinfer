# Gravitational-wave notebooks

The graduate track, after the tutorials. Run from the repository root:

```bash
uv run marimo edit notebooks/gw/01_gw_mock_catalog.py
```

1. `01_gw_mock_catalog.py`: the GWMockCat measurement process and trigger, a gravpop true
   population, simulated truths -> data -> detection, PE for the detected events, the raw data
   product, and an injection table. Saves the catalog to `notebooks/data/gw_catalog` (reloaded on
   later runs; tick the box to redo PE).
2. `02_gw_tgmm_fitting.py`: TGMM configs, fitting the catalog from notebook 1, per-event
   overrides and staleness, padded data products, a second (spin) representation, transformed
   coordinates, and save / reload. Starts Julia on first use.
3. `03_proposal_population.py`: population inference with an event-informed `Proposal`, custom
   `generative` leaves, a cancelled-spin importance weight and a selection term.

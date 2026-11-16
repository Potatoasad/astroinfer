# Gravitational-wave notebooks

These notebooks apply astroinfer to mock gravitational-wave catalogs, using the measurement
process in `astroinfer_gw.gwmockcat`. They assume you have worked through the tutorials in
`tutorials/`. Run them from the repository root:

```bash
uv run marimo edit notebooks/gw/01_gw_mock_catalog.py
```

1. `01_gw_mock_catalog.py` simulates a catalog: the measurement process and detection trigger, a
   true population from gravpop, simulated detections, PE for the detected events, the raw data
   product, and an injection table. It saves the catalog to `notebooks/data/gw_catalog` and
   reloads it on later runs; tick the box in the notebook to rerun PE.
2. `02_gw_tgmm_fitting.py` fits the catalog from notebook 1 with TGMMs: configurations, per-event
   overrides and stale fits, padded data products, a second representation for the spins,
   transformed coordinates, and saving and loading. It starts Julia the first time it runs.
3. `03_proposal_population.py` runs population inference with an event-informed `Proposal`,
   custom `generative` leaves, an importance weight in which the spin factor cancels, and a
   selection term.

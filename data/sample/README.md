# Sample Data

This directory contains lightweight sample artifacts used for quickstarts and CI-safe experiments.

- `sample_demand_excerpt.csv` is a small deterministic slice of the synthetic demand generator.
- The full synthetic dataset remains at `data/sample_demand.csv`.
- Use this directory for versioned sample assets that should remain small enough for source control.

If you adopt DVC later, this directory is the intended starting point for `dvc add data/sample/`.

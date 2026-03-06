# Contributing

## Scope

This repository is maintained as an academic software project. Contributions should improve at least one of:

- reproducibility
- empirical rigor
- theoretical clarity
- software reliability
- documentation quality

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[distributed,dev]"
pytest -q
python scripts/run_verification.py
```

Conda users can start with:

```bash
conda env create -f environment.yml
conda activate marl-forecast-game
```

## Pull Requests

Please keep pull requests focused and include:

1. A concise problem statement.
2. The intended methodological or engineering impact.
3. Reproduction steps.
4. Any changed metrics, figures, or benchmark outputs.

For behavior changes, include before/after evidence from one of:

- `pytest -q`
- `python scripts/run_verification.py`
- `python benchmarks/run_benchmark.py`
- `python experiments/run_reproducibility_check.py`

## Research Changes

If you change the training loop, reward definitions, scoring, or benchmark protocol:

- document the rationale in the pull request
- update `docs/validation.md` if evaluation changes
- update `docs/theory.md` if assumptions or objectives change
- update `docs/literature.md` if new related work is introduced

## Coding Standards

- Preserve immutable state and deterministic behavior where possible.
- Prefer explicit seeds in scripts and tests.
- Avoid adding network-dependent tests.
- Keep public APIs importable from `marl_forecast_game`.
- Add or update tests with any code change that affects runtime behavior.

## Issues

Bug reports should include:

- operating system
- Python version
- installation method
- exact command run
- stack trace or failing assertion
- whether `FRED_API_KEY` or other external credentials were set

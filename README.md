# marl-forecast-game

MVP+ implementation of a multi-agent adversarial forecasting game with:

- immutable simulation state
- four core agents (forecasting, adversary, defender, refactoring)
- pluggable runtime, disturbance, defense, and LLM-refactor mock modules
- DRD-aligned source adapter interface with provenance fields
- reproducible verification script, report artifact generation, and automated tests

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/run_verification.py
```

## Docker Validation

Build and run the full validation suite in a container:

```bash
docker build -t marl-forecast-game:test .
docker run --rm marl-forecast-game:test
```

The container installs dependencies from `requirements.txt` and executes:

- `pytest -q`
- `python scripts/run_verification.py`

## Project Structure

- `framework/types.py`: immutable game state, protocol objects, and config
- `framework/strategy_runtime.py`: pluggable strategy runtime backends
- `framework/agents.py`: forecasting, adversary, defender, and refactoring agents
- `framework/disturbances.py`: disturbance model registry and implementations
- `framework/defenses.py`: defense model registry and implementations
- `framework/game.py`: round-based Markov game runner with observability artifacts
- `framework/data.py`: dataset generation, ingestion, source adapters, normalization, chronological split
- `framework/data_sources/`: DRD-style source adapter abstractions and exemplars
- `framework/metrics.py`: MAE/RMSE/MAPE/worst-case and robustness deltas/ratios
- `framework/verify.py`: end-to-end verification checks and metrics report
- `tests/test_framework.py`: unit tests for deterministic behavior and guardrails

## Verification Coverage

The current implementation verifies:

- deterministic/pure state transitions (+ property-based invariants)
- chronological train/valid/test splitting
- runtime backend selection/fallback
- enforcement of maximum simulation rounds
- trajectory, confidence interval, and message artifacts
- adversarial scenario quality checks relative to clean runs and intensity sweeps
- robustness deltas on clean and attacked scenarios
- source-adapter schema conformance

Running `python scripts/run_verification.py` emits:

- console JSON summary
- `planning/verification_report.json`

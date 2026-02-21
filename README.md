# marl-forecast-game

MVP implementation of a multi-agent adversarial forecasting game with:

- immutable simulation state
- three core agents (forecasting, adversary, defender)
- disturbance injection for adversarial evaluation
- reproducible verification script and automated tests

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

- `framework/types.py`: immutable game state and transition model
- `framework/agents.py`: forecasting, adversary, and defender policies
- `framework/game.py`: round-based Markov game runner with round caps/fallback guard
- `framework/data.py`: dataset generation, ingestion, normalization, chronological split
- `framework/metrics.py`: MAE/RMSE/MAPE/worst-case metrics
- `framework/verify.py`: end-to-end verification checks and metrics report
- `tests/test_framework.py`: unit tests for deterministic behavior and guardrails

## Verification Coverage

The current implementation verifies:

- deterministic/pure state transitions
- chronological train/valid/test splitting
- enforcement of maximum simulation rounds
- adversarial scenario quality checks relative to clean runs
- accuracy metrics on clean and attacked scenarios

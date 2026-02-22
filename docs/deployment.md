# Deployment

## Docker

### Dockerfile

The project provides a single-stage Docker image based on `python:3.10-slim`:

```dockerfile
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/validate.sh

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import json; from framework.verify import run_verification; print(json.dumps(run_verification()['checks']))" || exit 1

CMD ["scripts/validate.sh"]
```

### Key Properties

- **Base image**: `python:3.10-slim` for minimal footprint.
- **Healthcheck**: Runs `run_verification()` every 30 seconds, checking all 9 verification checks. The container is considered unhealthy if any check fails.
- **Default command**: `scripts/validate.sh`, which runs `pytest -q` and `run_verification.py`.
- **Environment**: `PYTHONDONTWRITEBYTECODE=1` prevents `.pyc` files; `PYTHONUNBUFFERED=1` ensures real-time log output.

### Building and Running

```bash
# Build
docker build -t marl-forecast-game:test .

# Run default validation
docker run --rm marl-forecast-game:test

# Run with FRED data
docker run --rm -e FRED_API_KEY=your_key marl-forecast-game:test

# Run specific command
docker run --rm marl-forecast-game:test python scripts/run_training.py --episodes 50

# Interactive shell
docker run --rm -it marl-forecast-game:test bash
```

### With Ollama

To use LLM features inside the container, the Ollama server must be accessible from within the Docker network:

```bash
docker run --rm \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  marl-forecast-game:test \
  python -c "from framework.llm.ollama_interface import OllamaInterface; print(OllamaInterface().is_available())"
```

## Container Test Harness

The `scripts/run_container_test_harness.sh` script provides a comprehensive containerized test pipeline:

1. Builds the Docker image.
2. Runs `pytest -q` inside the container.
3. Runs `python scripts/run_verification.py` inside the container.
4. Runs all 22 validation scenarios inside the container.
5. Collects logs and reports.

```bash
bash scripts/run_container_test_harness.sh
```

Output is logged to `planning/container-test-harness-*.log`.

## CI/CD (GitHub Actions)

The CI pipeline is defined in `.github/workflows/ci.yml` with three parallel jobs:

### Job: `test`

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with:
      python-version: '3.12'
  - run: pip install -r requirements.txt
  - run: pytest -q
  - run: python scripts/run_verification.py
```

Runs the full test suite (unit tests + property tests) and all 9 verification checks.

### Job: `training-smoke`

```yaml
steps:
  - run: pip install -r requirements.txt
  - run: python scripts/run_training.py --episodes 10 --horizon 20
```

Smoke tests the MARL training pipeline with minimal episodes to verify the training loop completes without errors.

### Job: `backtest-smoke`

```yaml
steps:
  - run: pip install -r requirements.txt
  - run: python scripts/run_backtest.py --windows 3 --window-size 40 --step-size 15
```

Smoke tests the walk-forward backtesting pipeline with 3 small windows.

### Triggers

All jobs run on:
- Every push to any branch.
- Every pull request.

## Distributed Execution

### ParallelGameRunner

The `ParallelGameRunner` class provides concurrent game execution using Python's `multiprocessing.Pool`:

```python
@dataclass
class ParallelGameRunner:
    n_workers: int = 4
```

### Methods

**`map_scenarios(configs, init_state, seeds, disturbed)`**: Runs different `SimulationConfig` instances in parallel, one per worker. Each config/seed pair produces an independent game run.

**`run_seeds(config, init_state, seeds, disturbed)`**: Runs the same configuration with different seeds for Monte Carlo analysis.

### Serialization

`ForecastState` uses `MappingProxyType` fields that are not directly picklable by `multiprocessing`. The runner uses helper functions to convert states to/from primitive dictionaries:

- `_state_to_primitives(state)`: Converts `ForecastState` to a plain dict with concrete `dict` values.
- `_state_from_primitives(d)`: Reconstructs `ForecastState` from the primitive dict, wrapping mappings in `frozen_mapping()`.

### Usage

```python
from framework.distributed import ParallelGameRunner
from framework.types import ForecastState, SimulationConfig

runner = ParallelGameRunner(n_workers=4)
config = SimulationConfig(horizon=100)
init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

results = runner.run_seeds(config, init, seeds=[1, 2, 3, 4, 5])
for r in results:
    print(f"Seed {r['seed']}: {len(r['forecasts'])} forecasts")
```

### Result Format

Each result dict contains:

```python
{
    "seed": int,
    "forecasts": list[float],
    "targets": list[float],
    "convergence": {"rounds_executed": int, "max_rounds": int, "round_cap_hit": bool}
}
```

## Observability

### Structured Logging

The `GameObserver` class provides structured logging via `structlog` (with a `logging` fallback):

```python
@dataclass(frozen=True)
class GameObserver:
    logger_name: str = "forecast_game"
```

The game loop logs `round_complete` events (with round index, reward, disturbance) and `round_timeout` warnings.

### Prometheus Metrics

When `prometheus_client` is installed, the framework exports two metrics:

| Metric | Type | Description |
|---|---|---|
| `marl_game_rounds_total` | Counter | Total number of game rounds executed |
| `marl_game_round_latency_seconds` | Histogram | Per-round execution latency |

Export via:

```python
from framework.observability import export_prometheus_metrics
print(export_prometheus_metrics())
```

If `prometheus_client` is not installed, metrics collection is silently disabled and `export_prometheus_metrics()` returns an empty string.

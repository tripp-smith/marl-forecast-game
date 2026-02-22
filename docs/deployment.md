# Deployment

## Docker

### Multi-Stage Dockerfile

The project uses a multi-stage Docker build with a Haskell compilation stage and a Python runtime stage:

**Stage 1 -- `haskell-build`**: Uses `haskell:9.6-slim` to compile the Haskell library and executable. Runs `cabal build all` and `cabal test all`, then copies the compiled binary to `/usr/local/bin/marl-forecast-game`.

**Stage 2 -- `base`**: Uses `python:3.10-slim` for the runtime. Copies the pre-built Haskell binary from stage 1, enabling `runtime_backend="haskell"` without requiring cabal at runtime.

```dockerfile
FROM haskell:9.6-slim AS haskell-build
WORKDIR /hs
COPY haskell/ .
RUN cabal update && cabal build all && cabal test all \
 && cp "$(cabal list-bin marl-forecast-game)" /usr/local/bin/marl-forecast-game

FROM python:3.10-slim AS base
COPY --from=haskell-build /usr/local/bin/marl-forecast-game /usr/local/bin/marl-forecast-game
# ... Python setup follows
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PYTHONDONTWRITEBYTECODE` | `1` | Prevents `.pyc` file creation |
| `PYTHONUNBUFFERED` | `1` | Real-time log output |
| `LOG_LEVEL` | `WARNING` | Python logging level |
| `METRICS_PORT` | `0` (disabled) | Port for Prometheus `/metrics` HTTP endpoint |
| `FRED_API_KEY` | (none) | FRED API key for real macroeconomic data |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL for LLM features |

### Key Properties

- **Healthcheck**: Runs `run_verification()` every 30s. Failures are logged to `/app/logs/healthcheck.log`.
- **Default command**: `scripts/validate.sh` (pytest + verification).
- **Haskell binary**: Available at `/usr/local/bin/marl-forecast-game` for `HaskellRLMRuntime`.

### Building and Running

```bash
# Build (includes Haskell compilation + tests)
docker build -t marl-forecast-game:test .

# Run default validation
docker run --rm marl-forecast-game:test

# Run with FRED data and metrics server
docker run --rm -e FRED_API_KEY=your_key -e METRICS_PORT=9090 -p 9090:9090 marl-forecast-game:test

# Run with Haskell runtime backend
docker run --rm marl-forecast-game:test \
  python -c "from framework.strategy_runtime import HaskellRLMRuntime; from framework.types import ForecastState; print(HaskellRLMRuntime().forecast_delta(ForecastState(0, 10.0, 1.0, 0.0)))"

# Interactive shell
docker run --rm -it marl-forecast-game:test bash
```

### With Ollama

```bash
docker run --rm \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  marl-forecast-game:test \
  python -c "from framework.llm.ollama_interface import OllamaInterface; print(OllamaInterface().is_available())"
```

## Container Test Harness

The `scripts/run_container_test_harness.sh` script provides a comprehensive containerized test pipeline:

1. Builds the Docker image (including Haskell stage).
2. Runs `pytest -q` inside the container.
3. Runs `python scripts/run_verification.py` inside the container.
4. Runs all validation scenarios inside the container.
5. Collects logs and reports.

```bash
bash scripts/run_container_test_harness.sh
```

## CI/CD (GitHub Actions)

The CI pipeline is defined in `.github/workflows/ci.yml` with 6 jobs:

| Job | Trigger | Purpose |
|---|---|---|
| `test` | Every push/PR | pytest + verification + artifact upload |
| `training-smoke` | Every push/PR | 10-episode MARL training smoke test |
| `backtest-smoke` | Every push/PR | 3-window walk-forward backtest |
| `typecheck` | Every push/PR | `mypy` on core framework modules |
| `property-tests` | Every push/PR | Hypothesis tests with `--hypothesis-seed=0` |
| `haskell-test` | `haskell/` changes only | `cabal build all && cabal test all` |

### Artifact Uploads

The `test` job uploads `planning/verification_report.json` as a build artifact for post-run analysis.

## Error Penalty Mitigation

Per TRD requirements, forecast errors carry financial and legal risks. The following measures mitigate these:

1. **Determinism**: Seeded RNG + 100-run determinism verification ensures reproducible outputs.
2. **Convergence threshold**: `SimulationConfig.convergence_threshold` triggers early termination if rolling MAE exceeds the limit, preventing runaway errors.
3. **Poisoning detection**: z-score + modified z-score (MAD) gates reject contaminated datasets before they enter the game loop.
4. **Defense ensemble**: Stacked defenses (dampening + clipping + bias guard) limit adversarial impact.
5. **Healthcheck monitoring**: Container healthcheck writes failures to persistent log for alerting integration.
6. **Attack cost tracking**: `convergence.attack_cost_total` and `accuracy_vs_cost` fields enable cost-benefit analysis.

## Production Checklist

- [ ] Set `FRED_API_KEY` for real macroeconomic data
- [ ] Configure `METRICS_PORT` for Prometheus scraping
- [ ] Set `LOG_LEVEL=INFO` for production logging
- [ ] Verify data freshness: cache age < 24h (check via `cache_status()`)
- [ ] Run `scripts/run_verification.py` and confirm all 9 checks pass
- [ ] Monitor `/app/logs/healthcheck.log` for failures
- [ ] Set `convergence_threshold` appropriate to use case
- [ ] Review poisoning detection thresholds for data source characteristics

## Prometheus Metrics Server

When `prometheus_client` is installed and `METRICS_PORT` is set:

```python
from framework.observability import start_metrics_server
start_metrics_server()  # reads METRICS_PORT env var
```

Exposes `/metrics` endpoint with:

| Metric | Type | Description |
|---|---|---|
| `marl_game_rounds_total` | Counter | Total game rounds executed |
| `marl_game_round_latency_seconds` | Histogram | Per-round execution latency |

## Distributed Execution

The `ParallelGameRunner` class provides concurrent game execution using `multiprocessing.Pool`:

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

## JSON Trajectory Export

```python
from framework.export import export_trajectories
from framework.game import ForecastGame
from framework.types import ForecastState, SimulationConfig

cfg = SimulationConfig(horizon=100)
init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
out = ForecastGame(cfg, seed=42).run(init)
export_trajectories(out, "planning/trajectory_export.json", config=cfg, seed=42)
```

Exports include metadata (config hash, timestamp, seed), convergence info, and full trajectory logs.

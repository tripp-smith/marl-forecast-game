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

## All-in-One Container

The all-in-one container bundles every service into a single image managed by `supervisord`. One command builds and runs the full pipeline -- tests, verification, training, backtesting, stress tests -- then keeps the Streamlit UI and Grafana alive for interactive review.

### Build and Run

```bash
docker build --target allinone -t marl-forecast-game:allinone .
docker run --rm -p 8501:8501 -p 3000:3000 -p 9090:9090 -p 9091:9091 marl-forecast-game:allinone
```

To persist results on the host:

```bash
docker run --rm \
  -p 8501:8501 -p 3000:3000 -p 9090:9090 -p 9091:9091 \
  -v ./results:/app/results \
  marl-forecast-game:allinone
```

Or via Docker Compose:

```bash
docker compose up allinone
```

### Service Map

| Port | Service | Description |
|------|---------|-------------|
| 8501 | Streamlit UI | Explainability dashboard with auto-loaded results |
| 3000 | Grafana | Cluster health and simulation metrics dashboards |
| 9090 | Prometheus | Metrics aggregation and alerting |
| 9091 | App Metrics | Prometheus `/metrics` endpoint from the application |

### What Happens on Startup

`supervisord` launches five programs in priority order:

1. **Prometheus** (priority 10) -- scrapes `localhost:9091` for application metrics
2. **Grafana** (priority 20) -- auto-provisions dashboards and datasource pointing to localhost Prometheus
3. **Metrics Server** (priority 30) -- exposes Prometheus counters/gauges from the framework
4. **Pipeline** (priority 40) -- one-shot script that runs all tests and exports results to `/app/results/`
5. **Streamlit** (priority 50) -- UI with auto-discovery of results files

The pipeline runs to completion while the other four services stay alive indefinitely. The container remains running until killed.

### Pipeline Phases

The pipeline script (`scripts/run_all_pipeline.sh`) executes sequentially:

1. `pytest` -- unit + property tests
2. `run_verification.py` -- 9 determinism/correctness checks
3. `run_training.py` -- 20-episode MARL smoke training
4. `run_backtest.py` -- 5-window walk-forward backtest
5. `run_validation_scenarios.py` -- all 24 scenarios
6. `run_stress_test.py` -- 20-game, 200-round stress test
7. Trajectory export -- generates `simulation_clean.json`, `simulation_attacked.json`, `simulation_attacked_s99.json`

All output is written to `/app/results/` and immediately available in the Streamlit UI.

### Monitoring Logs

```bash
docker logs -f <container_id>

# Or inside the container:
tail -f /app/logs/pipeline.log
tail -f /app/logs/streamlit.log
tail -f /app/logs/prometheus.log
tail -f /app/logs/grafana.log
tail -f /app/logs/metrics_server.log
```

### Configuration Files

| File | Purpose |
|------|---------|
| `infra/supervisord.conf` | Process definitions for all 5 programs |
| `infra/prometheus-standalone.yml` | Prometheus config targeting `localhost:9091` |
| `infra/grafana-standalone/provisioning/` | Grafana datasource and dashboard provisioning |
| `scripts/run_all_pipeline.sh` | Pipeline orchestrator script |

### Notes

- Ray is **not** included in the all-in-one container. The parallel runner falls back to `multiprocessing`, which works correctly.
- The multi-service `docker compose up` flow remains available for production/cluster deployments with Ray.
- Grafana is configured with anonymous access (Viewer role) -- no login required.

---

## Docker Compose Stack

The full observability stack is orchestrated via `docker-compose.yml`:

| Service | Image | Ports | Purpose |
|---|---|---|---|
| `ray-head` | `rayproject/ray:2.9.0-py310` | 8265, 6379, 8080 | Ray head node + dashboard + metrics |
| `ray-worker` | `rayproject/ray:2.9.0-py310` | -- | Scalable Ray worker |
| `prometheus` | `prom/prometheus:latest` | 9090 | Metrics aggregation + alerting |
| `grafana` | `grafana/grafana:latest` | 3000 | Dashboard visualization |
| `app` | Built from `Dockerfile` (base) | 9091 | Python application + metrics server |
| `streamlit` | Built from `Dockerfile` (streamlit) | 8501 | Explainability UI |

### Quickstart

```bash
docker compose up -d
docker compose logs -f app
docker compose down
```

Scale Ray workers:

```bash
docker compose up -d --scale ray-worker=4
```

### Access Points

| Service | URL |
|---|---|
| Ray Dashboard | http://localhost:8265 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| Streamlit UI | http://localhost:8501 |
| App Metrics | http://localhost:9091/metrics |

## Prometheus and Grafana

### Configuration Files

| File | Purpose |
|---|---|
| `infra/prometheus.yml` | Prometheus scrape config (ray-head:8080 + app:9090) |
| `infra/alert_rules.yml` | Alert rules (AttackedMAETooLow, SimulationExceedsMaxRounds) |
| `infra/grafana/provisioning/datasources/prometheus.yml` | Auto-provision Prometheus datasource |
| `infra/grafana/provisioning/dashboards/dashboard.yml` | Auto-provision dashboard directory |

### Grafana Dashboards

| Dashboard | UID | Panels |
|---|---|---|
| Cluster Health | `marl-cluster-health` | Ray node count, CPU/memory utilization, task lifecycle, actor placement |
| Simulation Metrics | `marl-sim-metrics` | MAE/RMSE/MAPE gauges, disturbance success rate, agent delta histograms, round latency heatmap, robustness delta trend |

### Alert Rules

| Alert | Condition | Severity |
|---|---|---|
| `AttackedMAETooLow` | Attacked MAE < Clean MAE * 1.05 | Warning |
| `SimulationExceedsMaxRounds` | Rounds > 200 | Critical |
| `HighDisturbanceSuccessRate` | Success rate > 80% for 2m | Warning |

## Streamlit UI

The explainability UI is accessible at `http://localhost:8501`.

### Pages

| Page | Description |
|---|---|
| Simulation Replay | Step-through round visualization with forecast/target charts |
| Agent Contributions | Delta/reward analysis, correlation heatmap |
| Metric Decomposition | Clean vs attacked comparison, error attribution waterfall |
| What-If | Interactive parameter tweaking with on-demand simulation |
| Data Lineage | Pipeline visualization, split boundaries, cache status |
| LLM Inspection | Placeholder for DSPy-REPL / HaskellRLM integration |

### Standalone Launch

```bash
streamlit run ui/app.py --server.port=8501 --server.headless=true
```

## Ray Cluster Deployment

### Local Mode

By default, `parallel_runner()` initializes a local Ray instance. No external cluster is needed.

### Cluster Mode

Connect to an existing Ray cluster by setting the `ray_address` parameter:

```python
runner = parallel_runner(backend="ray", ray_address="ray://cluster-head:10001")
```

Or via the `RAY_ADDRESS` environment variable:

```bash
export RAY_ADDRESS=ray://cluster-head:10001
```

### KubeRay / Anyscale

For production scaling, deploy a Ray cluster on Kubernetes using [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) or use [Anyscale](https://www.anyscale.com/) for managed Ray. The application requires no code changes -- only `ray_address` needs to point to the cluster head.

## CI/CD (GitHub Actions)

The CI pipeline is defined in `.github/workflows/ci.yml` with 8 jobs:

| Job | Trigger | Purpose |
|---|---|---|
| `test` | Every push/PR | pytest + verification + artifact upload |
| `test-distributed` | Every push/PR | Distributed and observability tests with `RAY_LOCAL_MODE=1` |
| `training-smoke` | Every push/PR | 10-episode MARL training smoke test |
| `backtest-smoke` | Every push/PR | 3-window walk-forward backtest |
| `typecheck` | Every push/PR | `mypy` on core framework modules including distributed/ray/rllib |
| `property-tests` | Every push/PR | Hypothesis tests with `--hypothesis-seed=0` |
| `streamlit-lint` | Every push/PR | Validates Streamlit app parses without errors |
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

| Metric | Type | Labels | Description |
|---|---|---|---|
| `marl_game_rounds_total` | Counter | -- | Total game rounds executed |
| `marl_game_round_latency_seconds` | Histogram | -- | Per-round execution latency |
| `marl_sim_mae` | Gauge | `seed`, `disturbed` | Simulation MAE |
| `marl_sim_rmse` | Gauge | `seed`, `disturbed` | Simulation RMSE |
| `marl_sim_mape` | Gauge | `seed`, `disturbed` | Simulation MAPE |
| `marl_sim_worst_case` | Gauge | `seed`, `disturbed` | Worst-case error |
| `marl_sim_duration_seconds` | Histogram | -- | Simulation wall-clock time |
| `marl_sim_rounds` | Gauge | `seed` | Rounds executed |
| `marl_agent_delta` | Histogram | `actor`, `role` | Agent delta per round |
| `marl_agent_reward_total` | Counter | `actor` | Cumulative agent reward |
| `marl_disturbance_injections_total` | Counter | -- | Disturbance injection count |
| `marl_disturbance_success_total` | Counter | -- | Disturbances that increased error |
| `marl_alert_anomaly_total` | Counter | `alert_type` | Alert threshold breaches |

## Distributed Execution

The `parallel_runner()` factory selects the appropriate backend automatically:

```python
from framework.distributed import parallel_runner
from framework.types import ForecastState, SimulationConfig

runner = parallel_runner()  # Ray if available, else multiprocessing
config = SimulationConfig(horizon=100)
init = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

results = runner.run_seeds(config, init, seeds=[1, 2, 3, 4, 5])
for r in results:
    print(f"Seed {r['seed']}: {len(r['forecasts'])} forecasts")
```

Force a specific backend:

```python
runner = parallel_runner(backend="ray", ray_address="ray-head:6379")
runner = parallel_runner(backend="multiprocessing", n_workers=8)
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

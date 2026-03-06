# Configuration Reference

## SimulationConfig

Controls the game engine behavior. All fields have sensible defaults and are validated in `__post_init__`.

```python
@dataclass(frozen=True)
class SimulationConfig:
```

| Field | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `horizon` | int | 100 | >= 0 | Target number of simulation rounds |
| `max_rounds` | int | 200 | >= 0 | Hard cap on total rounds (overrides horizon) |
| `max_round_timeout_s` | float | 1.0 | > 0 | Per-round wall-clock timeout in seconds |
| `base_noise_std` | float | 0.15 | >= 0 | Standard deviation of Gaussian noise added each round |
| `disturbance_prob` | float | 0.1 | [0, 1] | Probability of disturbance occurring each round |
| `disturbance_scale` | float | 1.0 | >= 0 | Amplitude multiplier for disturbance models |
| `adversarial_intensity` | float | 1.0 | >= 0 | Aggressiveness scale for adversary and disturbances |
| `runtime_backend` | str | `"python"` | -- | Strategy runtime backend identifier |
| `disturbance_model` | str | `"gaussian"` | -- | Disturbance model name |
| `defense_model` | str | `"dampening"` | -- | Defense model name |
| `enable_refactor` | bool | True | -- | Whether to apply inter-round refactoring bias |
| `enable_llm_refactor` | bool | False | -- | Use LLM (via Ollama) for refactoring |
| `attack_cost` | float | 0.0 | >= 0 | Cost penalty reducing adversary effectiveness |
| `convergence_threshold` | float | 0.0 | >= 0 | Rolling MAE threshold for early stopping |
| `adversary_tau_init` | float | 5.0 | > 0 | Initial Boltzmann temperature for bounded rationality curriculum |
| `adversary_tau_final` | float | 0.1 | > 0 | Terminal Boltzmann temperature |
| `tau_decay_rate` | float | 0.05 | > 0 | Exponential decay rate for tau schedule |
| `bankruptcy_threshold` | float | 0.01 | > 0 | Kelly bankroll floor below which agents are pruned |
| `wolfpack_correlation_threshold` | float | 0.7 | [0, 1] | Pearson rho cutoff for wolfpack coalition membership |
| `dynamics` | str | `"static"` | `static`/`evolutionary` | Strategy update regime used by `ForecastGame` and MNPO |
| `population_size` | int | 20 | >= 1 | Evolutionary strategy pool size |
| `evolution_rate` | float | 0.05 | >= 0 | Replicator-dynamics learning rate |
| `evolution_batch_size` | int | 100 | >= 1 | Batch size used when evolving populations |
| `equilibrium_type` | str | `"nash"` | `nash`/`correlated`/`bayesian` | Coordination mode used in the game loop |
| `prior_alpha` | tuple[float, ...] | `(1.0, 1.0)` | positive entries | Dirichlet prior over hidden agent types |
| `quarantine_threshold` | float | 0.7 | [0, 1] | Posterior threshold for adversary quarantine |
| `feedback_mode` | str | `"full"` | `full`/`bandit_uninformed`/`bandit_informed` | Training feedback regime |
| `regret_horizon` | int | 500 | >= 1 | Target horizon for bandit updates |
| `bias_check` | bool | `True` | -- | Enable LLM bias/signaling probes |
| `signal_rounds` | int | 3 | >= 1 | Number of LLM signaling probes per evaluation |
| `coalitions` | str | `"static"` | `static`/`dynamic` | Coalition topology regime |
| `sabotage_prob` | float | 0.1 | [0, 1] | Probability of sabotage-style penalty per round |
| `coalition_reform_interval` | int | 50 | >= 1 | Dynamic coalition refresh cadence |

### Validation Rules

The `__post_init__` method raises `ValueError` if:
- `horizon < 0`
- `max_rounds < 0`
- `max_round_timeout_s <= 0`
- `base_noise_std < 0`
- `disturbance_prob` outside `[0, 1]`
- `disturbance_scale < 0`
- `adversarial_intensity < 0`
- `attack_cost < 0`
- `adversary_tau_init <= 0`
- `adversary_tau_final <= 0`
- `tau_decay_rate <= 0`
- `bankruptcy_threshold <= 0`
- `wolfpack_correlation_threshold` outside `[0, 1]`

## Advanced Game-Theory Modes

### Evolutionary Dynamics

- Set `dynamics="evolutionary"` to enable `EvolutionaryAgentPopulation`.
- `population_size` and `evolution_rate` control strategy-pool breadth and replicator pressure.
- The game loop samples variants per role and returns the evolved population in `GameOutputs.evolutionary_population`.

### Equilibrium and Bayesian Coordination

- `equilibrium_type="correlated"` uses `framework/equilibria.py` to solve a welfare-maximizing correlated equilibrium from round-local payoff matrices.
- `equilibrium_type="bayesian"` maintains a Dirichlet posterior over hidden types and can quarantine high-posterior adversaries when `quarantine_threshold` is exceeded.

### Partial-Feedback Training

- `feedback_mode="bandit_uninformed"` selects `TsallisINFBandit`.
- `feedback_mode="bandit_informed"` selects `MaximinUCBBandit`.
- `regret_horizon` provides the intended online horizon for bandit exploration pressure.

### Coalitions and Sabotage

- `coalitions="dynamic"` enables `CoalitionTopologyManager` in `framework/topology.py`.
- `coalition_reform_interval` controls how often coalition graphs are recomputed.
- `sabotage_prob` injects sabotage penalties into the runtime forecast path and records them in trajectory logs.

## DataProfile

Configures the data loading pipeline. See [data-pipeline.md](data-pipeline.md) for usage details.

```python
@dataclass(frozen=True)
class DataProfile:
```

| Field | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `source` | str | `"sample_csv"` | -- | Data source identifier |
| `periods` | int | 240 | > 0 | Number of time periods to load |
| `train_ratio` | float | 0.7 | (0, 1) | Fraction of data for training split |
| `valid_ratio` | float | 0.15 | [0, 1) | Fraction for validation split |
| `normalize` | bool | True | -- | Apply z-score normalization to features |
| `realtime_refresh` | bool | False | -- | Update the latest record's `fetched_at` timestamp |
| `hybrid_weight` | float | 0.5 | [0, 1] | Weight for real data in hybrid blending |
| `fail_on_poisoning` | bool | False | -- | Raise exception on detected data poisoning |

### Validation Rules

- `periods > 0`
- `0 < train_ratio < 1`
- `0 <= valid_ratio < 1`
- `train_ratio + valid_ratio < 1`
- `0 <= hybrid_weight <= 1`

### Valid Source Identifiers

| Source | Description |
|---|---|
| `sample_csv` | Generated synthetic demand data (default) |
| `fred` | FRED single-series CPI data |
| `imf` | IMF World Economic Outlook data |
| `polymarket` | Polymarket prediction market data |
| `hybrid` | Blend of FRED real data + synthetic data |
| `fred_training` | FRED multi-series training set (5 series) |

## Runtime Backend Names

Used in `SimulationConfig.runtime_backend`, resolved by `runtime_from_name()`:

| Name | Class | Description |
|---|---|---|
| `python`, `default` | `PythonStrategyRuntime` | Deterministic local computation: `0.4 + 0.4 * exogenous` |
| `prompt`, `mock_llm`, `llm` | `PromptStrategyRuntime` | LLM-backed via `PromptRuntimeClient` (mock by default) |
| `chat`, `ollama_chat` | `ChatStrategyRuntime` | Conversation-based strategy generation via Ollama chat API |

## Disturbance Model Names

Used in `SimulationConfig.disturbance_model`, resolved by `disturbance_from_name()`:

| Name(s) | Model | Description |
|---|---|---|
| `gaussian`, `default` | `GaussianDisturbance` | Zero-mean Gaussian shocks |
| `shift`, `regime_shift_basic` | `ShiftDisturbance` | Constant-magnitude shift |
| `evasion`, `evasion_like` | `EvasionDisturbance` | Direction-dependent disturbance |
| `volatility`, `volatility_scaled`, `vol_scaled` | `VolatilityScaledDisturbance` | Scales with recent volatility |
| `regime_shift`, `regime` | `RegimeShiftDisturbance` | Periodic structural breaks |
| `volatility_burst`, `burst` | `VolatilityBurstDisturbance` | High-amplitude Gaussian bursts |
| `drift`, `systematic_drift` | `DriftDisturbance` | Time-growing systematic drift |
| `historical`, `historical_residual` | `HistoricalDisturbance` | Sampled from cached historical residuals |
| `escalating`, `escalate` | `EscalatingDisturbance` | Linearly intensifying disturbance |
| `wolfpack`, `wolf_pack` | `WolfpackDisturbance` | Dual-scale coordinated ensemble attack |

See [disturbances-and-defenses.md](disturbances-and-defenses.md) for model details.

## Defense Model Names

Used in `SimulationConfig.defense_model`, resolved by `defense_from_name()`:

| Name(s) | Model | Description |
|---|---|---|
| `dampening`, `default` | `DampeningDefense` | Proportional counter + regularization |
| `clipping`, `clip` | `ClippingDefense` | Full negation clamped to range |
| `bias_guard`, `bias` | `BiasGuardDefense` | Capped negation |
| `ensemble`, `filter_ensemble` | `EnsembleDefense` | Average of dampening + clipping + bias_guard |
| `stack:x,y` | `StackedDefense` | Sequential composition of two models |

See [disturbances-and-defenses.md](disturbances-and-defenses.md) for model details.

## Model Backend Names

Used with `ModelBackend` protocol in `strategy_runtime.py`:

| Class | Description |
|---|---|
| `PassthroughBackend` | Wraps default Python delta: `0.4 + 0.4 * exogenous` |
| `XGBoostBackend` | GradientBoostingRegressor-based (falls back to passthrough) |
| `ARIMABackend` | Linear: `trend_coeff + exo_coeff * exogenous + persistence * hidden_shift` |

## Parallel Execution

### `parallel_runner()` Factory

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `str` | `"auto"` | Backend selection: `"auto"`, `"ray"`, `"multiprocessing"` |
| `n_workers` | `int` | `4` | Number of multiprocessing workers (ignored for Ray) |
| `ray_address` | `str \| None` | `None` | Ray cluster address; `None` = local |

### `FaultToleranceConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_task_retries` | `int` | `3` | Max retries per Ray task on transient failure |
| `retry_delay_s` | `float` | `1.0` | Delay between retries (seconds) |
| `actor_max_restarts` | `int` | `3` | Max restarts for Ray actors on node loss |

### CLI Arguments

`scripts/run_verification.py` accepts:

```bash
python scripts/run_verification.py --backend auto    # default
python scripts/run_verification.py --backend ray
python scripts/run_verification.py --backend multiprocessing
```

## Observability Configuration

### Metrics Server

| Variable | Default | Description |
|---|---|---|
| `METRICS_PORT` | `0` (disabled) | Port for Prometheus `/metrics` HTTP endpoint |

Set `METRICS_PORT=9090` to enable the metrics server. The server starts on a daemon thread when `start_metrics_server()` is called.

### Structured Logging

Structured JSON logging is automatically enabled when `structlog` is installed. The processor chain includes timestamp, log level, and JSON rendering. No additional configuration is needed.

### Tracing

OpenTelemetry tracing is automatically enabled when `opentelemetry-sdk` is installed. By default, spans are exported via OTLP gRPC to `localhost:4317`. Override with the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable.

### Prometheus Scrape Targets

Configure in `infra/prometheus.yml`:

| Job | Default Target | Description |
|---|---|---|
| `ray` | `ray-head:8080` | Ray's built-in metrics endpoint |
| `marl_app` | `app:9090` | Application metrics endpoint |

### Custom Alert Rules

Add rules to `infra/alert_rules.yml`. Available metrics for alerting include all custom metrics listed in the deployment documentation.

## Explainability UI

### Streamlit Configuration

| Setting | Default | Description |
|---|---|---|
| `--server.port` | `8501` | UI server port |
| `--server.headless` | `false` | Run without browser launch |
| `--server.address` | `localhost` | Bind address |

### Data Sources

The UI loads simulation outputs from JSON files containing `trajectory_logs`, `forecasts`, `targets`, and `convergence` data. These files are produced by `_game_outputs_to_dict()` in `framework/distributed.py` or by `framework/export.py`.

## Environment Variables

| Variable | Default | Used By | Description |
|---|---|---|---|
| `FRED_API_KEY` | (none) | `FredMacroAdapter`, training scripts | FRED API authentication key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `OllamaClient`, `OllamaInterface` | Ollama server URL |
| `METRICS_PORT` | `0` (disabled) | `observability.py` | Prometheus metrics server port |
| `RAY_ADDRESS` | (none) | `distributed.py` | Ray cluster address for auto-connect |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `localhost:4317` | `observability.py` | OpenTelemetry OTLP export endpoint |

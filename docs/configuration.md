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
| `haskell`, `haskellrlm` | `HaskellRLMRuntime` | Haskell subprocess via `cabal run` (2s timeout, Python fallback) |
| `prompt`, `mock_llm`, `llm` | `PromptStrategyRuntime` | LLM-backed via `PromptRuntimeClient` (mock by default) |

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

## Environment Variables

| Variable | Default | Used By | Description |
|---|---|---|---|
| `FRED_API_KEY` | (none) | `FredMacroAdapter`, training scripts | FRED API authentication key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `OllamaClient`, `OllamaInterface` | Ollama server URL |

# Validation

## Overview

The framework provides three layers of validation: a 22-scenario validation framework, 9 automated verification checks, and 13 Hypothesis property-based tests. Together they cover data integrity, simulation correctness, determinism, robustness, and advanced feature behavior.

## Validation Scenarios (22)

Scenarios are registered in `SCENARIO_REGISTRY` and executed via `run_all_scenarios()`. Each scenario is defined by a `ValidationScenario` dataclass specifying the data source, adversarial parameters, and expected properties.

### Scenario Catalog

| # | Name | Category | Description | Handler |
|---|---|---|---|---|
| 1 | `baseline_synthetic` | Simulation | Sample CSV, no attack, MAE < 5.0 | `_run_simulation_scenario` |
| 2 | `fred_cpi_clean` | Data | FRED CPI, verifies ingestion and splitting | `_run_data_validation` |
| 3 | `imf_gdp_clean` | Data | IMF GDP, verifies adapter | `_run_data_validation` |
| 4 | `polymarket_clean` | Data | Polymarket, verifies prediction market adapter | `_run_data_validation` |
| 5 | `hybrid_blend` | Data | Hybrid real+synthetic blending | `_run_data_validation` |
| 6 | `adversarial_gaussian` | Simulation | Gaussian disturbance at intensity 1.0, MAE < 10.0 | `_run_simulation_scenario` |
| 7 | `adversarial_regime_shift` | Simulation | Regime shift disturbance, MAE < 10.0 | `_run_simulation_scenario` |
| 8 | `adversarial_drift` | Simulation | Drift disturbance, MAE < 50.0 | `_run_simulation_scenario` |
| 9 | `defense_dampening` | Robustness | Dampening defense, attack MAE > clean MAE | `_run_robustness_scenario` |
| 10 | `defense_ensemble` | Robustness | Ensemble defense, attack MAE > clean MAE | `_run_robustness_scenario` |
| 11 | `poisoning_detection` | Data | Injected outliers detected, no false positives | `_run_poisoning_scenario` |
| 12 | `stress_high_rounds` | Stress | 5000 rounds without crash | `_run_simulation_scenario` |
| 13 | `determinism_cross_seed` | Determinism | 50 identical runs produce identical output | `_run_determinism_scenario` |
| 14 | `cache_integrity_fred` | Data | FRED cache file integrity check | `_run_cache_integrity` |
| 15 | `cross_source_consistency` | Data | Schema consistency across FRED/IMF/Polymarket | `_run_cross_source_scenario` |
| 16 | `llm_mock_refactor` | Simulation | Mock LLM refactoring agent, MAE < 5.0 | `_run_simulation_scenario` |
| 17 | `hierarchical_agent_run` | Hierarchical | BottomUp + TopDown + Aggregator, MAE < 20.0 | `_run_hierarchical_scenario` |
| 18 | `bayesian_calibration` | Bayesian | Weight sum = 1.0 after updates | `_run_bayesian_calibration` |
| 19 | `marl_convergence` | MARL | WoLF-PHC converges within 200 episodes | `_run_marl_convergence` |
| 20 | `llm_refiner_stability` | LLM | RecursiveStrategyRefiner bias within clamp bounds | `_run_llm_refiner_stability` |
| 21 | `fred_training_backtest` | Backtest | Walk-forward on FRED training data (skips without key) | `_run_fred_training_backtest` |
| 22 | `parallel_determinism` | Distributed | Parallel runner matches sequential for same seeds | `_run_parallel_determinism` |

### Running Scenarios

```bash
# Run all 22 scenarios
python scripts/run_validation_scenarios.py --scenarios all

# Run specific scenarios
python scripts/run_validation_scenarios.py --scenarios baseline_synthetic,marl_convergence

# List available scenarios
python scripts/run_validation_scenarios.py --list
```

### Report Format

Results are written to:
- `planning/validation_scenarios_report.json`: Full details per scenario.
- `planning/validation_scenarios_summary.csv`: One-line-per-scenario summary.

Each `ScenarioResult` contains:

```python
@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    duration_s: float
    details: dict
    errors: list
```

### Adding a New Scenario

1. Define a `ValidationScenario` and register it with `_register()`.
2. Write a handler function matching the signature `(ValidationScenario) -> ScenarioResult`.
3. Add the scenario name and handler to the `_DISPATCH` dict.

## Verification Checks (9)

The `run_verification()` function in `framework/verify.py` performs 9 end-to-end checks:

| # | Check | Description | Criteria |
|---|---|---|---|
| 1 | `split_non_empty` | Train/valid/test splits all contain data | `len > 0` for each split |
| 2 | `pure_transition` | `evolve_state` is deterministic | Two calls with same inputs produce identical state |
| 3 | `max_rounds_respected` | Game loop respects round cap | `rounds_executed <= max_rounds` |
| 4 | `source_adapter_rows` | FRED adapter returns sufficient data | `>= 10` rows |
| 5 | `real_data_split_non_empty` | FRED data splits are non-empty | `len > 0` for train and valid |
| 6 | `hybrid_data_non_empty` | Hybrid blended data is non-empty | `len > 0` for train |
| 7 | `attack_differs_from_clean` | Adversarial run differs from clean | `|MAE_attack - MAE_clean| > 1e-9` |
| 8 | `deterministic_100_runs` | 100 runs with same seed are identical | All forecast vectors match |
| 9 | `stress_10k_rounds` | 10,000 round simulation completes | `rounds_executed == 10000` |

Additionally, the verification function runs an adversarial intensity sweep at 0.5, 1.0, and 1.5 with volatility disturbance and ensemble defense, recording robustness metrics.

### Running Verification

```bash
python scripts/run_verification.py
```

Output is written to `planning/verification_report.json` and printed to stdout. The script exits with code 1 if any check fails.

## Hypothesis Property Tests (13)

Property-based tests in `tests/test_properties.py` use the Hypothesis library to verify invariants:

| # | Property | What It Tests |
|---|---|---|
| 1 | State value is finite after evolve | No NaN/Inf from `evolve_state` |
| 2 | Evolve is deterministic | Same inputs always produce same output |
| 3 | Config rejects negative horizon | Validation constraint |
| 4 | Config rejects out-of-range disturbance_prob | Must be in [0, 1] |
| 5 | Config rejects negative noise std | Must be >= 0 |
| 6 | Forecast delta is finite | No NaN/Inf from runtime |
| 7 | MAE is non-negative | Metric correctness |
| 8 | RMSE >= MAE | Mathematical relationship |
| 9 | Agent action delta is finite | No NaN/Inf from agents |
| 10 | Adversary direction opposes trend | Adversarial behavior |
| 11 | Defense magnitude bounded | Correction is finite |
| 12 | Chronological split is disjoint | No data leakage |
| 13 | Normalize preserves row count | Feature engineering correctness |

### Running Property Tests

```bash
pytest tests/test_properties.py -v
```

Hypothesis runs 100 examples per property by default (configurable via settings).

## Unit and Integration Tests

The `tests/test_framework.py` module contains 28 tests covering:

- **Types**: `evolve_state` determinism, `ForecastState` immutability.
- **Agents**: Action correctness for all agent types, `SafeAgentExecutor` fallback.
- **Game**: Round cap enforcement, clean/disturbed parity, `GameOutputs` structure.
- **Data**: CSV loading, normalization, chronological split, poisoning detection.
- **Metrics**: MAE/RMSE/MAPE/worst-case calculation, robustness delta/ratio.
- **Integration**: End-to-end `run_verification()` check pass.

```bash
pytest tests/test_framework.py -v
```

## CI Integration

GitHub Actions runs three jobs on every push and pull request:

| Job | Command | Purpose |
|---|---|---|
| `test` | `pytest -q` + `run_verification.py` | Full test suite + 9 verification checks |
| `training-smoke` | `run_training.py --episodes 10 --horizon 20` | Training pipeline sanity |
| `backtest-smoke` | `run_backtest.py --windows 3 --window-size 40 --step-size 15` | Backtesting pipeline sanity |

All jobs use Python 3.12 on `ubuntu-latest`.

### Container Test Harness

The `scripts/run_container_test_harness.sh` script builds the Docker image and runs the full validation pipeline inside the container:

1. `docker build -t marl-forecast-game:test .`
2. Run `pytest -q` inside the container.
3. Run `python scripts/run_verification.py` inside the container.
4. Run all 22 validation scenarios inside the container.

```bash
bash scripts/run_container_test_harness.sh
```

See [deployment.md](deployment.md) for Docker details.

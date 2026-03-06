# Validation

## Evaluation Philosophy

The validation stack is designed to support academic claims with three layers of evidence:

1. **Software reliability** through unit, integration, and property-based tests.
2. **Reproducibility** through fixed-seed experiments and deterministic package-level checks.
3. **Empirical comparison** through rolling-window benchmarks, ablations, and paired significance tests.

This is intentionally stricter than a smoke-test-only workflow. A forecasting system that claims adversarial robustness should show not just that it runs, but that it runs repeatably and remains auditable.

## Programmatic Harness

The recommended local entrypoint is the root-level harness:

```bash
python run_project_tests.py --mode quick
python run_project_tests.py --mode full
```

It writes a consolidated report to `results/test-harness/<timestamp>/` by default, including:

- `summary.json`
- `summary.md`
- per-stage logs under `stages/`

Full mode runs `pytest`, verification, validation scenarios, reproducibility, benchmark, training smoke matrices, simulation smoke checks, backtest smoke, stress, mypy, and UI syntax validation.

## Benchmark Protocol

The benchmark harness lives in [benchmarks/run_benchmark.py](../benchmarks/run_benchmark.py).

### Default benchmark design

- Dataset: `sample_csv` by default, with a deterministic synthetic demand series
- Windows: rolling windows with configurable `window_size`, `step_size`, and `windows`
- Seed: fixed to `42` unless overridden
- Models compared:
  - `naive_last`
  - `moving_average_5`
  - `game_clean_ensemble`
  - `game_attack_identity`
  - `game_attack_ensemble`
- Metrics:
  - MAE
  - RMSE
  - approximate CRPS from interval-derived Gaussian width
  - interval coverage
- Significance:
  - paired Wilcoxon signed-rank tests on per-forecast absolute errors

### Run it

```bash
python benchmarks/run_benchmark.py --source sample_csv --windows 6 --window-size 60 --step-size 12
```

Outputs:

- `results/benchmarks/benchmark_summary.json`
- `results/benchmarks/benchmark_summary.md`

### Current sample-data snapshot

The latest local run used:

```bash
python benchmarks/run_benchmark.py --source sample_csv --windows 4 --window-size 40 --step-size 10
```

Result table:

| Model | MAE | RMSE | Approx. CRPS | Coverage |
|---|---:|---:|---:|---:|
| `naive_last` | 3.1749 | 3.8071 | 3.0386 | 0.0500 |
| `moving_average_5` | 2.8331 | 3.4051 | 1.9934 | 0.9250 |
| `game_clean_ensemble` | 2.9722 | 3.4965 | 2.9150 | 0.0250 |
| `game_attack_identity` | 2.9017 | 3.4725 | 2.8055 | 0.1000 |
| `game_attack_ensemble` | 2.9327 | 3.5027 | 2.8343 | 0.0750 |

Interpretation:

- The harness is reproducible and produces paired significance tests.
- The current bundled-sample benchmark does **not** support a claim that the defended game significantly outperforms the naive baseline.
- That is acceptable for now; it means future benchmark improvements can be evaluated against a transparent baseline rather than a hand-waved success story.

## Ablation Design

The built-in ablation logic is modest but useful:

- `game_clean_ensemble` isolates the underlying game without adversarial disturbance.
- `game_attack_identity` measures the attacked system without an active defense.
- `game_attack_ensemble` measures the attacked system with the ensemble defense.

That gives two immediate research questions:

1. How much does attack pressure degrade a clean forecasting configuration?
2. How much does the defense recover relative to the undefended attacked system?

If you add new components such as MNPO variants, deep RL policies, or subgroup fairness penalties, the benchmark harness should be extended rather than replaced so comparisons remain longitudinally stable.

## Reproducibility Audit

The reproducibility check lives in [experiments/run_reproducibility_check.py](../experiments/run_reproducibility_check.py).

It verifies:

- repeated same-seed runs produce identical forecast traces
- a changed seed changes the forecast trace
- the public import path `marl_forecast_game` is sufficient to run the audit

Run it with:

```bash
python experiments/run_reproducibility_check.py
```

Output:

- `results/experiments/reproducibility_check.json`

## Scenario Validation

The framework still provides the existing scenario-based validation suite for functional coverage. Scenarios are registered in `SCENARIO_REGISTRY` and exercised via `run_all_scenarios()`.

### Scenario catalog

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
| 21 | `fred_training_backtest` | Backtest | Walk-forward on FRED training data | `_run_fred_training_backtest` |
| 22 | `parallel_determinism` | Distributed | Parallel runner matches sequential for same seeds | `_run_parallel_determinism` |
| 23 | `rarl_bounded_rationality_convergence` | MARL | RARL avoids gradient starvation | `_run_rarl_bounded_rationality` |
| 24 | `wolfpack_ensemble_stress_test` | Robustness | Wolfpack adversary vs. stacked defense | `_run_wolfpack_stress_test` |

Run all 28 scenarios:

```bash
python scripts/run_validation_scenarios.py --scenarios all
```

## Verification Checks

`framework/verify.py` performs end-to-end checks on determinism, round limits, source adapters, hybrid loading, and robustness sweeps.

Run:

```bash
python scripts/run_verification.py
```

The default verification report currently includes 13 checks in non-qual mode.

Primary artifacts:

- `results/verification_report.json`

## Property-Based Tests

Property tests in `tests/test_properties.py` cover invariants such as:

- deterministic state transitions
- finite forecasts and agent actions
- split integrity without leakage
- Bayesian weight-sum preservation
- bounded adversarial coalition sizes

Run:

```bash
pytest tests/test_properties.py -q
```

## Public API Test

The academic-facing import path is verified in `tests/test_package_api.py`:

```bash
pytest tests/test_package_api.py -q
```

## CI Integration

GitHub Actions runs CI on push and pull request with jobs for:

- full tests
- distributed tests
- training smoke
- backtest smoke
- type checking
- property tests
- Streamlit parse validation

All CI jobs use Python 3.12 on `ubuntu-latest`.

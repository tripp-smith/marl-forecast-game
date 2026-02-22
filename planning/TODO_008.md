# TODO_008 — Production Evolution: Hierarchical Agents, Bayesian Aggregation, MARL Training, and FRED Data Integration

### Context

Sub-Phases A–Q (TODO_001 through TODO_007) established the deterministic Markov game core, pluggable disturbance/defense registries, OSINT adapters (FRED/IMF/Polymarket), Ollama LLM integration, 16-scenario validation framework, Hypothesis property tests, and containerized test harness. This document introduces Sub-Phases R through V, transitioning the project from proof-of-concept to a production-ready system with hierarchical multi-agent architectures, Bayesian probabilistic aggregation, multi-agent reinforcement learning, LLM-driven recursive strategy refinement, multi-series FRED data as a first-class training set, and scalable deployment infrastructure.

### Guiding Principles

- Preserve pure-functional determinism: all new state types remain frozen dataclasses, all transitions remain pure functions.
- Backward compatibility: new fields on ForecastState use defaults so existing code continues to work unchanged.
- Graceful degradation: FRED_API_KEY-dependent features fall back to synthetic proxies when the key is absent.
- Incremental validation: each sub-phase includes V&V steps that must pass before proceeding.

---

#### Sub-Phase R: Hierarchical Agent Architecture and Advanced Models
Focus: Replace the fixed 4-agent topology with a flexible multi-agent registry supporting bottom-up, top-down, and ensemble aggregation patterns, with pluggable ML model backends.

- **TODO Items:**
  - R.1 Define `BottomUpAgent` and `TopDownAgent` frozen dataclasses in `framework/agents.py`, each with `act()` returning `AgentAction`; `BottomUpAgent` reads segment-level state slices, `TopDownAgent` reads macro context.
  - R.2 Create `EnsembleAggregatorAgent` in `framework/agents.py` that receives a `list[AgentAction]` and produces a weighted combined delta using configurable strategy (equal-weight, rank-based, or reward-proportional).
  - R.3 Refactor `AgentFactory` in `framework/game.py` from the fixed 4-tuple to a new `AgentRegistry` dataclass holding named agent collections (`forecasters`, `adversaries`, `defenders`, `aggregator`).
  - R.4 Define `ModelBackend` protocol in `framework/strategy_runtime.py` with `predict(state) -> float`; implement `ARIMABackend`, `XGBoostBackend`, and `PassthroughBackend`.
  - R.5 Extend `ForecastState` in `framework/types.py` with optional `segment_id` and `segment_values` fields with backward-compatible defaults.
  - R.6 Update `game.py` `run()` loop to iterate over all agents in `AgentRegistry`, pass actions through `EnsembleAggregatorAgent`, and record per-agent reward breakdowns.

- **V&V Steps:**
  - Verify `BottomUpAgent` and `TopDownAgent` produce finite `AgentAction` deltas for 500 random states (Hypothesis property test).
  - Verify `EnsembleAggregatorAgent` output is bounded by min/max of input deltas for equal-weight mode.
  - Confirm backward compatibility: existing `default_agent_factory` continues to work and all 28 existing tests pass.
  - Run full 16-scenario validation suite and confirm no regressions.

---

#### Sub-Phase S: Bayesian Aggregation and Probabilistic Outputs
Focus: Replace point-forecast aggregation with Bayesian Model Averaging producing distributional outputs, and add calibration metrics and hyperparameter optimization.

- **TODO Items:**
  - S.1 Implement `BayesianAggregator` class in `framework/aggregation.py`: per-agent posterior weights updated each round via Gaussian log-likelihood of forecast errors.
  - S.2 Replace `forecast = state.value + sum(deltas)` in `game.py` with `BayesianAggregator.aggregate()` returning `(weighted_mean, variance)`.
  - S.3 Add `ProbabilisticForecast` frozen dataclass in `framework/types.py` with `mean`, `variance`, `quantiles`; update `StepResult` to carry it.
  - S.4 Add `pit_score()`, `crps()`, and `interval_coverage()` to `framework/metrics.py`.
  - S.5 Implement `BayesianOptimizer` in `framework/hyperopt.py` wrapping the `bayesian-optimization` library to tune `SimulationConfig` fields.
  - S.6 Add `numpy`, `scipy`, and `bayesian-optimization` to `requirements.txt`.

- **V&V Steps:**
  - Verify `BayesianAggregator` weights sum to 1.0 after every update across 1000 rounds (Hypothesis property test).
  - Verify `ProbabilisticForecast` quantiles are monotonically non-decreasing (q10 <= q25 <= q50 <= q75 <= q90).
  - Confirm CRPS is non-negative for all inputs and PIT values lie in [0, 1].
  - Run `BayesianOptimizer` for 10 iterations on synthetic data and confirm it reduces validation MAE relative to default config.

---

#### Sub-Phase T: Dynamic State and Real-Time Data Integration
Focus: Expand state representation with macroeconomic context, implement multi-series FRED download as a first-class training set using FRED_API_KEY, add geopolitical/OECD/BIS adapters, and build Monte Carlo scenario generation.

- **TODO Items:**
  - T.1 Add `macro_context: Mapping[str, float]` field to `ForecastState` with `frozen_mapping({})` default; update `evolve_state` to accept optional `coeff_map` for macro-influenced transitions.
  - T.2 Extend `FredMacroAdapter` to support multi-series download (CPIAUCSL, GDP, UNRATE, FEDFUNDS, T10YIE) via `fetch_multi_series()` when `FRED_API_KEY` is set.
  - T.3 Add `FredTrainingDataBuilder` in `framework/data_utils.py`: calls multi-series fetch, aligns timestamps, writes to `data/cache/fred_training.json`, returns `DatasetBundle`.
  - T.4 Update `load_dataset()` in `framework/data.py` to recognize `source="fred_training"` invoking `FredTrainingDataBuilder`; falls back to synthetic when `FRED_API_KEY` is absent.
  - T.5 Implement `GeopoliticalRiskAdapter`, `OECDCLIAdapter`, `BISPolicyRateAdapter` in `framework/data_sources/`.
  - T.6 Build `ScenarioGenerator` in `framework/scenarios.py`: Monte Carlo replications producing `ScenarioFan` with percentile trajectories.
  - T.7 Add `fred_training_clean` validation scenario; skips gracefully without `FRED_API_KEY`.

- **V&V Steps:**
  - Verify `FredMacroAdapter.fetch_multi_series()` returns rows with all 5 series when `FRED_API_KEY` is set (or synthetic fallback with matching schema).
  - Verify `FredTrainingDataBuilder` produces a `DatasetBundle` with non-empty train/valid/test splits and no future leakage.
  - Confirm `evolve_state` with `macro_context` is deterministic (same inputs produce same outputs) via 1000-iteration property test.
  - Verify `ScenarioGenerator` percentile trajectories are monotonic (p10 <= p25 <= ... <= p90 at each timestep).
  - Run `fred_training_clean` scenario and confirm pass (or graceful skip).

---

#### Sub-Phase U: MARL Training and LLM-Driven Strategy Refinement
Focus: Replace fixed heuristic policies with learnable agents using Q-learning and WoLF-PHC, implement adversarial RL training, activate Haskell subprocess bridge, and build LLM-driven recursive strategy refinement.

- **TODO Items:**
  - U.1 Implement `DiscreteActionSpace` in `framework/training.py`: discretizes delta range into N bins with bidirectional conversion.
  - U.2 Implement `QTableAgent` in `framework/training.py`: state-feature-hashed Q-table with epsilon-greedy policy and Q-learning updates.
  - U.3 Implement `WoLFPHCAgent` subclassing `QTableAgent`: mixed-strategy policy with variable win/lose learning rates.
  - U.4 Implement `TrainingLoop`: runs episodes, extracts experience tuples, calls agent updates, tracks convergence, serializes Q-tables.
  - U.5 Implement `RADversarialTrainer`: alternating adversary/forecaster training epochs for minimax robustness.
  - U.6 Activate `HaskellRLMRuntime` with subprocess bridge to `cabal run`, with timeout and fallback.
  - U.7 Build `RecursiveStrategyRefiner` in `framework/llm/refiner.py`: trajectory-to-prompt pipeline feeding OllamaRefactorClient.
  - U.8 Add `scripts/run_training.py` CLI using FRED training data as default when `FRED_API_KEY` is available.

- **V&V Steps:**
  - Verify `DiscreteActionSpace` round-trips: `action_to_delta(delta_to_action(d))` is within one bin width of `d` for all valid deltas.
  - Verify `QTableAgent` Q-values converge (decreasing TD error) over 200 episodes on a fixed synthetic game.
  - Verify `WoLFPHCAgent` learning rate switches correctly: uses `delta_win` when outperforming average policy.
  - Confirm `TrainingLoop` serialized Q-tables can be deserialized and produce identical actions for the same states.
  - Verify `RecursiveStrategyRefiner` clamps adjustments to `[-0.1, 0.1]` regardless of LLM response.
  - Run `scripts/run_training.py --episodes 10` and confirm exit code 0 with convergence output.

---

#### Sub-Phase V: Production Validation and Scalable Deployment
Focus: Add walk-forward backtesting, sensitivity analysis, distributed parallel execution, iterative feedback loops, and expanded CI/validation coverage.

- **TODO Items:**
  - V.1 Implement `WalkForwardBacktester` in `framework/backtesting.py`: sliding-window train/evaluate with per-window and aggregate metrics.
  - V.2 Add `scripts/run_backtest.py` CLI: loads FRED training data, runs walk-forward, writes `planning/backtest_report.json`.
  - V.3 Implement `SensitivityAnalyzer` in `framework/backtesting.py`: per-factor perturbation analysis with ranked importance scores.
  - V.4 Add `ParallelGameRunner` in `framework/distributed.py`: multiprocessing-based batch scenario execution.
  - V.5 Implement `IterativeFeedbackLoop` in `framework/training.py`: updates aggregator weights and Q-values from realized outcomes.
  - V.6 Add 6 new validation scenarios: `hierarchical_agent_run`, `bayesian_calibration`, `marl_convergence`, `llm_refiner_stability`, `fred_training_backtest`, `parallel_determinism`.
  - V.7 Update `.github/workflows/ci.yml` with `training-smoke` and `backtest-smoke` jobs.

- **V&V Steps:**
  - Verify `WalkForwardBacktester` produces monotonically advancing window boundaries with no overlap.
  - Verify `SensitivityAnalyzer` importance scores sum to approximately 1.0 (normalized).
  - Verify `ParallelGameRunner` produces identical results to sequential execution for the same seeds.
  - Run all 22 validation scenarios (16 existing + 6 new) and confirm all pass.
  - Run containerized test harness and confirm all stages complete without error.

---

### Dependency Additions

| Sub-Phase | Package | Version Constraint |
|---|---|---|
| S | numpy | >=1.26 |
| S | scipy | >=1.12 |
| S | bayesian-optimization | >=1.4 |
| R | scikit-learn | >=1.4 |

### FRED_API_KEY Integration Points

| Component | File | With FRED_API_KEY | Without |
|---|---|---|---|
| FredMacroAdapter.fetch_multi_series | data_sources/macro_fred.py | 5 real FRED series | Synthetic proxy |
| FredTrainingDataBuilder | data_utils.py | Multi-series training bundle | Synthetic fallback |
| load_dataset(source='fred_training') | data.py | Real FRED training set | Warning + synthetic |
| scripts/run_training.py | scripts/ | Trains on real FRED | Trains on synthetic |
| scripts/run_backtest.py | scripts/ | Backtests on FRED history | Backtests on synthetic |
| fred_training_clean scenario | validation_scenarios.py | Full validation | Graceful skip |


### New Task List for Next Stage of Development
#### Sub-Phase G: Stabilize and Test Deterministic Core
Focus: Refine and rigorously test the existing Python core to ensure mission-critical reliability before adding complexity.
- **TODO Items:**
  - Refactor game.py to use pluggable agent factories, integrating disturbances.py and defenses.py fully (continue from Sub-Phase B).
  - Add input validation and exception handling in data.py and game.py to prevent crashes from invalid data or configs.
  - Expand tests/test_framework.py with 10+ new unit tests for edge cases (e.g., zero disturbance, max rounds).
  - Benchmark core loop performance on larger horizons (e.g., 1000 steps) and document bottlenecks.
- **V&V Steps:**
  - Run extended pytest with coverage >90%; verify all checks in verify.py pass under varied seeds.
  - Property-based testing (using hypothesis if installable, else manual) for state immutability and reward consistency.
  - Manual review for antipatterns; simulate production errors (e.g., divide-by-zero in mape) and confirm handling.

#### Sub-Phase H: Integrate Real Public Data
Focus: Replace synthetic data with OSINT sources to achieve realistic forecasting baselines.
- **TODO Items:**
  - Implement data ingestion modules for FRED and IMF APIs (ref 2), supporting time-series fetch and caching.
  - Update data.py to handle real datasets (e.g., GDP forecasts), including chronological splitting and normalization.
  - Adapt sample_demand.csv generation to mimic real data distributions from Kaggle (address remaining from Sub-Phase D).
  - Configure SimulationConfig for data-driven exogenous factors (e.g., map macro_index to real indicators).
- **V&V Steps:**
  - Validate ingested data against known benchmarks (e.g., compare FRED GDP to published values).
  - Run verify.py on real data subsets; assert metrics (e.g., MAE <0.3) outperform synthetic baselines.
  - Integration tests for end-to-end flow: ingest, split, simulate, and metric computation.

#### Sub-Phase I: Enhance Adversarial Realism
Focus: Introduce advanced disturbance models and defenses for incremental robustness.
- **TODO Items:**
  - Implement 2-3 new disturbance models in disturbances.py (e.g., regime shifts from ref 1).
  - Tune AdversaryAgent for economic trade-offs (e.g., cost-based aggressiveness).
  - Integrate ensemble defenses in defenses.py and test combinations.
  - Update game.py to support multi-round adversarial learning loops (prepare for MARL).
- **V&V Steps:**
  - Scenario-based testing: Simulate attacks and verify degradation (e.g., attack MAE > clean MAE by 20%+).
  - Property checks for disturbance quality (e.g., variance bounds).
  - Comparative runs: Metrics on real vs. synthetic data under attacks.

#### Sub-Phase J: Prepare for LLM Integration
Focus: Add stubs and initial wiring for LLM-driven refactoring without full deployment.
- **TODO Items:**
  - Extend strategy_runtime.py with DSPy-compatible LLM calls (deferred from ref 5, start simple prompts).
  - Implement basic refactoring agent in agents.py, using LLM for delta adjustments.
  - Add trajectory logging in game.py for LLM inputs (e.g., JSON dumps of steps).
  - Document LLM hooks for future dspy-repl/HaskellRLM (ref 1).
- **V&V Steps:**
  - Mock LLM tests: Simulate responses and verify integration without real calls.
  - End-to-end verification: Run game with stubbed LLM and check for improved metrics over baselines.
  - Security audit: Ensure sandboxing for LLM-generated code.

Achieving this next stage would unlock scalable, data-driven adversarial training, paving the way for Haskell migration and production deployment.
# Filename: TODO_013_next-stage-production-readiness.md

## Next Stage Production Readiness Plan

This document outlines the detailed tasks and instructions for advancing the MARL Forecast Game toward a production-ready application. Building on previous phases (e.g., core framework establishment in Phases 1-4 and initial adversarial elements), this plan focuses on integrating real-world data, enhancing adversarial robustness, incorporating LLM capabilities, bolstering verification, and resolving technical debt. All efforts emphasize scalability, reliability, error resilience, and compliance to ensure the system can handle mission-critical operations with minimal risk of financial or legal penalties.

The plan is divided into focused sub-phases (G-K), each with a clear goal, TODO checklist, and dedicated validation & verification (V&V) steps. References to prior sub-phases (e.g., A-F from previous critiques) are included for progression tracking. Completion of these sub-phases will enable distributed execution, full LLM refactoring, and deployment readiness as per original Phases 5-12.

### Sub-Phase G: Real Data Integration and Validation

**Focus:** Seamlessly incorporate OSINT data sources to enable realistic, production-grade simulations that reflect live economic and market conditions.

**Prerequisites:** Completion of synthetic data handling from Sub-Phase C; access to API keys for FRED, IMF, and Polymarket.

**Detailed Instructions:**
- Prioritize secure API interactions with rate limiting, retries, and authentication to prevent service disruptions.
- Ensure data ingestion complies with terms of service for each source, logging all accesses for audit trails.
- Blend real and synthetic data to create hybrid datasets that maintain temporal consistency and handle missing values gracefully.

**TODO Items:**
- Develop dedicated API adapter modules in a new `data_sources/` subdirectory for FRED (using fredapi or equivalent), IMF (via their SDMX API), and Polymarket (REST endpoints), including error handling for 4xx/5xx responses and fallback to cached data.
- Extend the `load_dataset` function in `data.py` to fully support real and hybrid data profiles, incorporating automated poisoning detection algorithms (e.g., outlier rejection via IQR) and data validation schemas using pydantic.
- Implement `build_hybrid_rows` to merge synthetic and real time-series with configurable weights (e.g., 70% real, 30% synthetic), preserving chronological order and interpolating gaps using linear methods from scipy.
- Modify the core simulation loop in `game.py` to derive initial states from real dataset bundles, allowing configurable start dates and horizons based on ingested series.

**V&V Steps:**
- Write unit tests for each API adapter using pytest-mock to simulate responses, verifying that at least 100 rows are fetched and schema-matched for sample queries.
- Execute end-to-end verification in `verify.py` on hybrid datasets, asserting that chronological splits yield non-empty train/test sets and that key metrics (e.g., MAE) differ by less than 10% from synthetic baselines under identical configs.
- Use hypothesis for property-based testing of data ingestion, ensuring invariance (e.g., same seed yields identical outputs) across 200 randomized runs with varying source combinations.
- Conduct manual audit of logs for compliance, confirming no unauthorized data accesses and proper error reporting.

### Sub-Phase H: Advanced Adversarial and Defense Modules

**Focus:** Introduce modular, extensible adversarial models and defenses to achieve robust performance against realistic disturbances in production environments.

**Prerequisites:** Basic agent interactions from Sub-Phase B; familiarity with AML techniques from the original design specification.

**Detailed Instructions:**
- Design models to be pluggable via dependency injection, supporting easy extension for future threats.
- Incorporate economic constraints to simulate real-world attack costs, ensuring defenses are cost-effective.
- Update metrics to provide comprehensive robustness insights, suitable for production monitoring dashboards.

**TODO Items:**
- Define abstract base classes for `DisturbanceModel` and `DefenseModel` in `agents.py`, implementing concrete subclasses for Gaussian noise, regime shifts (using HMM from hmmlearn), and volatility bursts (GARCH-inspired).
- Integrate these models into `SimulationConfig` and the `run_simulation` loop in `game.py`, enabling dynamic selection via config flags and runtime parameters.
- Enhance `AdversaryAgent` to include attack cost calculations, reducing disturbance magnitude proportionally to a configurable budget, and log cost-benefit analyses per round.
- Expand `metrics.py` to compute a `robustness_ratio` (clean MAE / attacked MAE) for all evaluation metrics, including Continuous Ranked Probability Score (CRPS) for probabilistic outputs using properscoring.

**V&V Steps:**
- Test disturbance sampling distributions statistically (e.g., Kolmogorov-Smirnov test for Gaussian mean ≈ 0) across 1,000 samples per model.
- Run comparative scenarios in `verify.py` with and without defenses, verifying that attacked MAE exceeds clean MAE by at least 20% in undefended cases, and is mitigated to <5% delta with defenses.
- Perform stress testing with simulations of 10,000+ rounds, asserting no runtime errors, memory leaks (monitor via psutil), and strict adherence to max_rounds configurations.
- Validate economic trade-offs by asserting that higher costs reduce average disturbance impact by 30-50% in parameterized tests.

### Sub-Phase I: LLM Stub Integration and Prompt Runtime

**Focus:** Embed LLM capabilities via Ollama API to enable adaptive agent strategies, laying the foundation for intelligent, production-scale decision-making.

**Prerequisites:** Ollama server running on localhost:11434; stubs from Sub-Phase E.

**Detailed Instructions:**
- Use the API for single-turn generations initially, with keep_alive set for performance in long simulations.
- Ensure prompts are hardened against injection risks and include context from game states.
- Prepare for future Haskell migration by designing wrappers that abstract LLM calls.

**TODO Items:**
- Fully implement `OllamaInterface` in a new `llm.py` module, supporting `/api/generate` for strategy deltas, with configurable models (e.g., llama2) and keep_alive timeouts.
- Add a `PromptStrategyRuntime` option to `runtime_from_name` in `game.py`, using deterministic mock responses for offline testing and real API calls in production mode.
- Update `ForecastingAgent` and `DefenderAgent` in `agents.py` to optionally invoke LLM for action generation based on config flags, formatting state as JSON prompts.
- Create a Pythonic stub for HaskellRLM in `refactor.py`, simulating LLM-driven code suggestions for agent policies.

**V&V Steps:**
- Mock Ollama API responses in unit tests, asserting that generated deltas fall within expected numerical ranges (e.g., ±10% of baseline forecasts).
- Perform end-to-end runs with LLM enabled, verifying trajectory logs in `game.py` include LLM prompt-response pairs and that simulations complete without API timeouts.
- Test availability and error handling by simulating unreachable hosts, ensuring graceful fallbacks to rule-based agents without crashes.
- Measure latency impacts, asserting average round time <2s in benchmarks with 100 rounds.

### Sub-Phase J: Verification and Observability Enhancements

**Focus:** Strengthen the verification framework and add observability features to support production monitoring, debugging, and compliance auditing.

**Prerequisites:** Initial property tests from Sub-Phase D; Prometheus setup stubs.

**Detailed Instructions:**
- Expand properties to cover probabilistic and adversarial edge cases.
- Implement full metrics export for integration with monitoring tools like Grafana.
- Ensure all configs are versioned and loadable securely.

**TODO Items:**
- Add a `convergence` dictionary to `GameOutputs` in `types.py`, tracking metrics like rounds_executed, epsilon_convergence, and cap_hits for Nash-like stability.
- Fully implement `export_prometheus_metrics` in `metrics.py` to push key indicators (e.g., MAE, robustness_delta) with labels for agents and scenarios.
- Extend `tests/test_framework.py` with new hypothesis strategies for confidence interval invariance and data poisoning rejection under varied attack probabilities.
- Integrate static typing with mypy across all modules and use pyyaml for secure config loading in `run_verification.py`, adding schema validation.

**V&V Steps:**
- Rerun `verify.py` on expanded scenarios (including real data), asserting 100% pass rate for all defined checks.
- Use hypothesis to property-test core invariants (immutability, determinism) over 500+ randomized configurations.
- Simulate Prometheus scrapes in tests, verifying exported metrics include required labels and values match computed outputs.
- Conduct code reviews for typing compliance, fixing all mypy errors and ensuring no unsafe YAML loads.

### Sub-Phase K: Bug Fixes and Technical Debt Cleanup

**Focus:** Eliminate known issues and refactor for maintainability, ensuring the framework is stable and scalable for production use.

**Prerequisites:** Address remaining items from Sub-Phases F (e.g., config inconsistencies).

**Detailed Instructions:**
- Prioritize fixes that impact reliability, such as attribute errors and edge-case handling.
- Add comprehensive documentation to facilitate team onboarding and audits.
- Refactor for better separation of concerns, preparing for distributed scaling.

**TODO Items:**
- Resolve attribute and import errors (e.g., add `convergence` to outputs, define `DataProfile` dataclass in `data.py`, import missing modules like disturbance_from_name).
- Refactor `data.py` to encapsulate source adapters and profiles, with factory patterns for extensibility.
- Implement robust error handling throughout (e.g., validate probs ≤1, handle zero-round sims with early exits), raising custom exceptions for production logging.
- Add docstrings to all public functions and classes, referencing original phases and design specs for context.

**V&V Steps:**
- Execute full pytest suite, achieving 100% coverage and pass rate, including new tests for fixed bugs.
- Run mypy on the entire codebase, resolving all type violations and enforcing strict mode.
- Perform differential testing against previous versions (from Sub-Phase F), confirming bug resolutions via assert-equal on sample outputs.
- Manual inspection of documentation completeness, ensuring at least 80% of code is covered with meaningful comments.

## Completion Criteria
All sub-phases must pass their V&V steps with documented evidence (e.g., updated reports in planning/). This stage unlocks advanced features like distributed training (Phase 5) and full production deployment, enabling resilient forecasting in high-stakes environments.
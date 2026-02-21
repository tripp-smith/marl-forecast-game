# TODO_005: Comprehensive Implementation Plan (Sub-Phases G–J)

## 1) Purpose and Scope
This plan operationalizes `planning/TODO_004.md` using requirements and constraints from:
1. `DRD.md` (real data sources and ingestion requirements)
2. `PRD.md` (architecture and functional goals)
3. `TRD.md` (technical objectives, Haskell/LLM direction, observability)

The repository currently implements a Python MVP core. This plan prioritizes:
- Stabilizing and validating the deterministic core
- Integrating high-value public macro data
- Improving adversarial realism and defense coverage
- Preparing robust LLM integration seams without forcing production LLM dependencies

---

## 2) Guiding Principles
- **Determinism first**: preserve reproducibility for the core simulation loop.
- **Incremental realism**: add external data and adversarial complexity in measurable steps.
- **Interface stability**: extend via adapters/factories, avoid breaking existing tests.
- **Verification-driven delivery**: each milestone has explicit acceptance checks.
- **Future-proofing**: keep Python runtime modular for later Haskell and HaskellRLM integration.

---

## 3) Current-State Assessment (from static repo review)
- Core modules exist: `framework/game.py`, `framework/data.py`, `framework/agents.py`, `framework/disturbances.py`, `framework/defenses.py`, `framework/strategy_runtime.py`.
- Existing tests focus on framework behavior (`tests/test_framework.py`).
- Data-source adapters already started under `framework/data_sources/` (notably FRED and Polymarket paths).
- Verification scaffolding exists (`framework/verify.py`, `scripts/run_verification.py`, `planning/verification_report.json`).

Implication: Sub-Phases G–J can be executed as enhancement work, not a greenfield build.

---

## 4) Delivery Plan by Sub-Phase

## Sub-Phase G — Stabilize and Test Deterministic Core
### G1. Agent-factory refactor in `framework/game.py`
**Goal:** decouple game loop from concrete agent construction.

Tasks:
1. Introduce or extend a factory interface (e.g., `build_agents(config)` / dependency-injected constructors).
2. Route disturbance and defense wiring through explicit strategy/factory selection.
3. Ensure defaults preserve current behavior for existing tests and scripts.

Acceptance criteria:
- Game can run with default factories unchanged.
- Alternate factories can be injected in tests without monkeypatch-heavy setup.

### G2. Input validation and exception handling (`framework/data.py`, `framework/game.py`)
Tasks:
1. Validate configuration ranges (round counts, disturbance intensity bounds, split ratios).
2. Validate data schema assumptions before simulation starts.
3. Add typed exceptions (or clearly named domain errors) for recoverable failures.
4. Guard metric computation for pathological values (e.g., zero denominator MAPE handling).

Acceptance criteria:
- Invalid configs fail fast with actionable error messages.
- Simulations do not crash with opaque tracebacks for common user errors.

### G3. Expand deterministic and edge-case tests (`tests/test_framework.py`)
Tasks:
1. Add at least 10 focused unit tests covering:
   - zero disturbance baseline
   - max rounds boundary
   - malformed config inputs
   - empty or constant series data behavior
   - defense pass-through behavior
2. Add regression tests for bug-prone paths discovered during G1/G2.

Acceptance criteria:
- New tests are deterministic and isolated.
- Existing test behavior remains intact.

### G4. Performance benchmark scaffolding
Tasks:
1. Add lightweight benchmark utility script (e.g., under `scripts/`) for 1,000-step simulation timing.
2. Capture timings and rough bottlenecks in `planning/` notes (no premature optimization).

Acceptance criteria:
- Reproducible benchmark command exists.
- Baseline numbers documented for future optimization comparisons.

---

## Sub-Phase H — Integrate Real Public Data
### H1. Productionize FRED and IMF ingestion adapters
Tasks:
1. Finalize adapter contracts in `framework/data_sources/base.py`.
2. Harden `framework/data_sources/macro_fred.py` for pagination, date windows, and normalization.
3. Add IMF adapter module with same normalized output contract.
4. Implement response caching (filesystem cache is sufficient for MVP).

Acceptance criteria:
- FRED + IMF adapters return a common normalized time-series structure.
- Adapter errors are explicit (auth/network/shape).

### H2. Real-data flow in `framework/data.py`
Tasks:
1. Add data source selector (`synthetic`, `csv`, `fred`, `imf`).
2. Ensure chronological split + normalization are consistently applied regardless of source.
3. Preserve deterministic preprocessing semantics.

Acceptance criteria:
- A single data loading API can feed simulation from any supported source.
- Split integrity and chronology invariants hold.

### H3. Realistic sample data generation
Tasks:
1. Update `data/sample_demand.csv` generation guidance to mimic real-world distributional properties.
2. Document the generation recipe in `planning/` for reproducibility.

Acceptance criteria:
- Synthetic sample better approximates trend + seasonality + variance than flat toy data.

### H4. Exogenous factors in simulation config
Tasks:
1. Extend config schema to map macro indicators into simulation state (e.g., `macro_index`).
2. Ensure optionality: when absent, fallback behavior is deterministic.

Acceptance criteria:
- Config-driven exogenous series can influence simulation without code changes.

---

## Sub-Phase I — Enhance Adversarial Realism
### I1. New disturbance models (`framework/disturbances.py`)
Targets:
- Regime shift disturbance (level shift at randomized/specified point)
- Volatility burst disturbance (heteroskedastic window)
- Drift disturbance (gradual directional bias)

Acceptance criteria:
- Models are pluggable and parameterized.
- Each model has unit coverage and sanity bounds.

### I2. Adversary trade-off tuning (`framework/agents.py`)
Tasks:
1. Introduce cost-aware aggressiveness parameterization.
2. Add objective balancing attack strength vs cost penalties.

Acceptance criteria:
- Adversary decisions vary with configured cost profile.

### I3. Ensemble defenses (`framework/defenses.py`)
Tasks:
1. Implement combinable defense stack (e.g., clipping + smoothing + robust aggregation).
2. Add ordering and compatibility rules.

Acceptance criteria:
- Defense combinations can be declared in config.
- Combined defense paths are tested for deterministic behavior.

### I4. Multi-round adversarial learning support (`framework/game.py`)
Tasks:
1. Add loop mode where adversary/defender parameters evolve across rounds.
2. Capture per-round adaptation signals in trajectory logs.

Acceptance criteria:
- Multi-round mode is optional and backward-compatible.

---

## Sub-Phase J — Prepare for LLM Integration
### J1. LLM runtime seam in `framework/strategy_runtime.py`
Tasks:
1. Expand runtime interface to support prompt-based strategy calls and mock provider execution.
2. Add strict timeout/validation wrappers around LLM responses.

Acceptance criteria:
- LLM path is fully mockable in tests.
- No hard dependency on remote API availability for core execution.

### J2. Refactoring agent baseline (`framework/agents.py`)
Tasks:
1. Implement minimal refactoring-agent behavior that consumes trajectory summaries.
2. Produce bounded “delta adjustments” to agent strategy parameters.

Acceptance criteria:
- Refactoring agent is deterministic under mock runtime.

### J3. Trajectory logging upgrades (`framework/game.py`)
Tasks:
1. Emit structured JSON trajectory artifacts with round-wise state/action/reward summaries.
2. Include fields needed by refactoring prompts.

Acceptance criteria:
- Logs are parseable, stable, and versioned.

### J4. LLM hook documentation
Tasks:
1. Add docs describing request/response schema for runtime providers.
2. Describe extension path toward dspy-repl HaskellRLM bridge (TRD alignment).

Acceptance criteria:
- Another contributor can implement a concrete provider from docs alone.

---

## 5) Cross-Cutting Verification Matrix
For each sub-phase, implement:
1. **Unit tests** for new logic paths.
2. **Integration tests** for end-to-end simulation behavior.
3. **Determinism checks** using seeded runs.
4. **Metric checks** (MAE/RMSE/MAPE/worst-case deltas).
5. **Verification script updates** in `framework/verify.py` and `scripts/run_verification.py`.

Target thresholds:
- Coverage: minimum 90% for core modules touched in G–J.
- Runtime: benchmark reports for baseline horizon (1,000 steps).
- Robustness: attacked MAE should degrade clean MAE in controlled scenarios (explicitly tracked).

---

## 6) Proposed Work Breakdown and Sequencing
1. **Week 1:** G1–G3 (factory refactor, validation, tests)
2. **Week 2:** G4 + H1–H2 (benchmarking + external data adapters)
3. **Week 3:** H3–H4 + I1 (data realism + first advanced disturbances)
4. **Week 4:** I2–I4 (adversary/defense realism + multi-round loop)
5. **Week 5:** J1–J4 (LLM seams, mock integration, documentation)
6. **Week 6:** hardening pass, full verification, release notes

---

## 7) Risks and Mitigations
- **External API instability (FRED/IMF):** use adapter abstraction + cache + retry/backoff.
- **Determinism regressions:** enforce seeded tests and golden-path assertions.
- **Complexity growth in game loop:** isolate feature toggles and keep default path simple.
- **LLM nondeterminism/security:** use strict mocks in CI; sanitize and validate generated outputs.
- **Haskell migration drift:** maintain interface documentation and typed contracts now, defer runtime coupling.

---

## 8) Detailed Assistance Instructions (when execution is blocked)
Use this checklist if a contributor cannot complete a step due to environment/tooling gaps.

### A. Missing Python dependencies
1. Create/activate virtual environment.
2. Install `requirements.txt`.
3. Re-run targeted module test first, then full test suite.
4. Record exact package and version conflicts in `planning/IMPLEMENTATION_BLOCKERS.md`.

### B. External API access issues (FRED/IMF)
1. Confirm API key/env variable names expected by adapter.
2. Validate endpoint with a minimal curl request.
3. If blocked, run adapter tests with cached fixtures and mock responses.
4. Log outage/auth details and fallback action in blocker file.

### C. Benchmark cannot run or times out
1. Reduce horizon temporarily (1000 -> 300) to validate script path.
2. Capture stack trace or timeout context.
3. Profile only hot path function(s) with lightweight timer instrumentation.
4. Record hardware/context assumptions with observed runtime.

### D. Nondeterministic test failures
1. Pin random seeds at module and test case levels.
2. Verify no mutable shared defaults in dataclasses/config objects.
3. Compare failing run trajectory artifact against passing baseline.
4. Document diff and suspected source in blocker file.

### E. LLM runtime unavailable
1. Switch all LLM-involved tests to `framework/llm/mock.py`.
2. Validate schema-level behavior only (request/response contract).
3. Mark provider-specific checks as deferred with explicit test skip reason.
4. Add TODO owner and due date in blocker file.

### Blocker file template
Create/update: `planning/IMPLEMENTATION_BLOCKERS.md`
- Date/time
- Sub-phase + task ID (e.g., H1)
- Blocker description
- Evidence (error text or failing command)
- Mitigation attempted
- Help needed (specific)
- Owner + next checkpoint

---

## 9) Definition of Done (Program Level)
This plan is complete when:
1. Sub-Phases G–J tasks are implemented with merged tests and docs.
2. Verification scripts cover new behavior and pass in a clean environment.
3. Real-data ingestion (FRED + IMF) works with caching and graceful failure modes.
4. At least 2 advanced disturbances + ensemble defense paths are benchmarked.
5. LLM hooks are mock-verified and documented for future provider integration.


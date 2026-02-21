<<<<<<< HEAD
# TODO_002 — Implementation Backlog (Aligned to PRD/DRD/TRD)

## Scope
This backlog operationalizes the architecture and delivery gaps identified against `PRD.md`, `DRD.md`, and `TRD.md`.
It complements `planning/TODO_001.md` by converting broad phases into prioritized, testable implementation work.

## Priority and Phases

### P0 — Architecture and Protocol Foundation
**Objective:** Align core runtime and agent interactions with TRD architecture and component expectations.

**Implementation Tasks**
- [ ] Add runtime abstraction for strategy backends (Python baseline + HaskellRLM adapter placeholder).
- [ ] Introduce typed interaction protocol artifacts for round-level coordination.
- [ ] Add Refactoring Agent and optional strategy refinement stage in game loop.
- [ ] Add trajectory objects for observability and downstream refactoring.

**Validation Tasks**
- [ ] Unit tests for backend selection and fallback behavior.
- [ ] Unit tests for message ordering and deterministic replay with fixed seeds.
- [ ] Regression tests ensuring max round caps still apply.

**Dependencies**
- Existing immutable state and game runner.

**Exit Criteria**
- Runtime backends are pluggable via config.
- Refactoring phase can be enabled/disabled without breaking existing behavior.
- All tests pass.

---

### P1 — Adversarial/Defense Modularity
**Objective:** Implement pluggable disturbances and defenses with measurable robustness deltas.

**Implementation Tasks**
- [ ] Implement disturbance registry (noise, shift, evasion-like perturbation).
- [ ] Implement defense registry (dampening, clipping).
- [ ] Wire disturbance/defense selection from config into simulation rounds.
- [ ] Extend metric reporting with robustness deltas and ratios.

**Validation Tasks**
- [ ] Unit tests for registry operations and deterministic model outputs.
- [ ] Verification checks for attacked-vs-clean comparatives.

**Dependencies**
- P0 runtime abstractions.

**Exit Criteria**
- Multiple disturbance/defense combinations run from configuration.
- Verification report includes robustness comparison metrics.

---

### P2 — Data Adapters, Observability, and Delivery Hardening
**Objective:** Add DRD-aligned source adapters and richer output/reporting artifacts.

**Implementation Tasks**
- [ ] Add common data source adapter interface and normalized schema.
- [ ] Add exemplar adapters for a prediction market source and macro source.
- [ ] Attach provenance metadata to records.
- [ ] Extend game outputs with confidence intervals and convergence diagnostics.
- [ ] Emit machine-readable verification report artifact.
- [ ] Update README and implementation report.

**Validation Tasks**
- [ ] Schema conformance tests for adapter outputs.
- [ ] Confidence interval and report-shape tests.
- [ ] End-to-end verification command execution.

**Dependencies**
- P0 and P1 complete.

**Exit Criteria**
- Adapter outputs conform to normalized schema.
- Verification produces report artifact with pass/fail checks.

## Traceability Matrix

| Backlog Task | Requirement References |
|---|---|
| Runtime abstraction and backend selection | TRD §3, TRD §6.3, PRD §2 |
| Refactoring Agent + iterative update loop | TRD §4.1, TRD §4.4, PRD §3.1, PRD §3.4 |
| Disturbance and defense registries | TRD §4.3, TRD §5.2, PRD §3.3 |
| Attack/clean comparative metrics | TRD §5.2, TRD §5.1, PRD §4.2 |
| Data source adapters + provenance | DRD §2.1, DRD §2.2, DRD §3 |
| Trajectories, logs, and report artifacts | TRD §5.3, TRD §6.1, PRD §4.3 |

## TODO_001 Crosswalk

### Completed from TODO_001
- Repository structure and runnable MVP baseline.
- Immutable state transition model.
- Baseline data handling and normalization.
- Baseline adversarial simulation and verification checks.

### Deferred / Expanded into TODO_002
- Formal runtime/backend abstraction.
- Refactoring-agent orchestration loop.
- Disturbance/defense plugin system.
- DRD-specific source adapter architecture.
- Rich observability and verification artifacts.

## Out of Scope (for this backlog iteration)
- Full production Haskell runtime execution and distributed Cloud Haskell deployment.
- Live external API integration requiring credentials or paid access.
- Bayesian optimization and full MARL training algorithms beyond baseline placeholders.
- UI/dashboard development.
=======
#### Sub-Phase A: Enhance Data Integration with Public Datasets
Focus: Replace synthetic with real OSINT for testing granularity/complexity (e.g., demand/finance domains).
- **TODO Items:**
  - Add API wrappers for 2-3 public sources (e.g., FRED for macro series, Kaggle Store Item Demand CSV download).
  - Update `data.py` to handle diverse formats (e.g., JSON from APIs, time-series alignment).
  - Implement validation checks (e.g., cross-source consistency, poisoning simulation).
  - Configure caching to manage rate limits/volatility.
- **V&V Steps:**
  - Test ingestion on sample queries; verify formats and chronological splits.
  - Run verification on real data; assert metrics (MAE/RMSE/MAPE/worst-case) improve over synthetic baselines.
  - Simulate volatility (e.g., mock API failures) and check fallbacks.

#### Sub-Phase B: Advance Agent and Adversarial Capabilities
Focus: Make agents configurable; add pluggable disturbances and basic defenses.
- **TODO Items:**
  - Refactor agents for configurability (e.g., params via YAML, add Refactoring Agent stub).
  - Implement pluggable disturbance models (e.g., noise + simple evasion/shifts).
  - Integrate basic RARL wrapper (e.g., using numpy for policy updates—no full ML yet).
  - Add economic trade-offs (e.g., cost/reward in rewards).
- **V&V Steps:**
  - Test agent swaps/configs in games; verify no regressions in purity.
  - Evaluate under varied disturbances; assert "attack_is_harder" and 10-20% robustness gain vs. baselines.
  - Unit tests for new models; integration with verification script.

#### Sub-Phase C: Introduce LLM-Driven Refactoring (Stubs)
Focus: Add HaskellRLM integration stubs, deferred full execution.
- **TODO Items:**
  - Install dspy-repl via pip; add Haskell stubs (e.g., Python wrappers calling GHCI).
  - Implement basic RLM loop for strategy generation (e.g., prompt for agent delta tweaks).
  - Refactor `agents.py` to use LLM-generated policies (initially mock).
  - Add trajectory logging for refactoring inputs.
- **V&V Steps:**
  - Test RLM loops with sample prompts; verify generated code parses/runs.
  - Simulate refactoring on error logs; check improved metrics.
  - Purity checks: Ensure LLM calls are isolated IO.

#### Sub-Phase D: Haskell Migration for Core Logic
Focus: Port key pure components to Haskell for type safety.
- **TODO Items:**
  - Port `types.hs` and `evolve_state` to Haskell; add Python FFI (e.g., via inline-c).
  - Update `game.py` to call Haskell for transitions.
  - Document migration rationale and mixed-language boundaries.
- **V&V Steps:**
  - Type-check Haskell; compare outputs to Python for determinism.
  - Run full verification; assert no performance degradation.
  - Test edge cases (e.g., mutations prevented).

#### Sub-Phase E: Enhance Observability, Testing, and Deployment
Focus: Address gaps in logging, docs, and fixes.
- **TODO Items:**
  - Add verbose logging to `game.py` (e.g., trajectories as JSON).
  - Fix Dockerfile (replace validate.sh with run_verification.py wrapper).
  - Populate PRD/DRD/TRD.md with specs from conversation.
  - Expand tests for real data and new features.
- **V&V Steps:**
  - Verify logs capture full interactions without side effects.
  - Docker build/run; assert passes all checks.
  - Review docs for traceability (lightweight matrix).

#### Sub-Phase F: Comprehensive Integration and Evaluation
Focus: End-to-end on public data; prepare for post-MVP.
- **TODO Items:**
  - Run simulations on recommended datasets (e.g., FRED-MD, Store Item Demand).
  - Tune configs for runtime constraints (<10 min, <4GB, 200 rounds).
  - Identify extensions (e.g., distributed stubs).
- **V&V Steps:**
  - Measure metrics on real data; confirm success threshold.
  - System-wide tests for no regressions.
  - Audit for risks (e.g., purity erosion). 
>>>>>>> be2a9a9 (3)

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

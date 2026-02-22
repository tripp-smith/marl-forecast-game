# TODO_002 — Implementation Backlog (Aligned to PRD/DRD/TRD)

## Scope
This backlog operationalizes the architecture and delivery gaps identified against `PRD.md`, `DRD.md`, and `TRD.md`.
It complements `planning/TODO_001.md` by converting broad phases into prioritized, testable implementation work.

## Priority and Phases

### P0 — Architecture and Protocol Foundation (COMPLETED)
**Objective:** Align core runtime and agent interactions with TRD architecture and component expectations.

**Implementation Tasks**
- [x] Add runtime abstraction for strategy backends (Python baseline + HaskellRLM adapter placeholder).
- [x] Introduce typed interaction protocol artifacts for round-level coordination.
- [x] Add Refactoring Agent and optional strategy refinement stage in game loop.
- [x] Add trajectory objects for observability and downstream refactoring.

**Validation Tasks**
- [x] Unit tests for backend selection and fallback behavior.
- [x] Unit tests for message ordering and deterministic replay with fixed seeds.
- [x] Regression tests ensuring max round caps still apply.

---

### P1 — Adversarial/Defense Modularity (COMPLETED)
**Objective:** Implement pluggable disturbances and defenses with measurable robustness deltas.

**Implementation Tasks**
- [x] Implement disturbance registry (noise, shift, evasion-like perturbation).
- [x] Implement defense registry (dampening, clipping).
- [x] Wire disturbance/defense selection from config into simulation rounds.
- [x] Extend metric reporting with robustness deltas and ratios.

**Validation Tasks**
- [x] Unit tests for registry operations and deterministic model outputs.
- [x] Verification checks for attacked-vs-clean comparatives.

---

### P2 — Data Adapters, Observability, and Delivery Hardening (COMPLETED)
**Objective:** Add DRD-aligned source adapters and richer output/reporting artifacts.

**Implementation Tasks**
- [x] Add common data source adapter interface and normalized schema.
- [x] Add exemplar adapters for a prediction market source and macro source.
- [x] Attach provenance metadata to records.
- [x] Extend game outputs with confidence intervals and convergence diagnostics.
- [x] Emit machine-readable verification report artifact.
- [x] Update README and implementation report.

**Validation Tasks**
- [x] Schema conformance tests for adapter outputs.
- [x] Confidence interval and report-shape tests.
- [x] End-to-end verification command execution.

---

## Sub-Phase A: Enhance Data Integration with Public Datasets (COMPLETED)
- [x] Add API wrappers for FRED, IMF, Polymarket.
- [x] Update data pipeline to handle diverse formats.
- [x] Implement validation checks (cross-source consistency, poisoning simulation).
- [x] Configure caching to manage rate limits/volatility.

## Sub-Phase B: Advance Agent and Adversarial Capabilities (COMPLETED)
- [x] Refactor agents for configurability; add Refactoring Agent.
- [x] Implement pluggable disturbance models (7 models in registry).
- [x] Add defense models (5 models + stacked combinator).
- [x] Add economic trade-offs (attack cost/reward).

## Sub-Phase C: Introduce LLM-Driven Refactoring Stubs (COMPLETED)
- [x] Add OllamaClient and DSPyLikeRepl integration.
- [x] Implement basic RLM loop for strategy generation.
- [x] Refactor agents.py to use LLM-generated policies (with mock fallback).
- [x] Add trajectory logging for refactoring inputs.

## Sub-Phase D: Haskell Migration for Core Logic (PARTIAL)
- [x] Port Types.hs and evolveState to Haskell.
- [ ] Update game.py to call Haskell for transitions via FFI.
- [x] Document migration rationale and mixed-language boundaries.

## Sub-Phase E: Enhance Observability, Testing, and Deployment (COMPLETED)
- [x] Add verbose logging to game.py (trajectories as JSON).
- [x] Fix Dockerfile (validate.sh + verification runner).
- [x] Populate PRD/DRD/TRD.md with specs.
- [x] Expand tests for real data and new features.

## Sub-Phase F: Comprehensive Integration and Evaluation (IN PROGRESS)
- [x] Run simulations on recommended datasets (FRED, IMF).
- [ ] Tune configs for runtime constraints.
- [ ] Identify extensions (distributed stubs).

## Traceability Matrix

| Backlog Task | Requirement References |
|---|---|
| Runtime abstraction and backend selection | TRD §3, TRD §6.3, PRD §2 |
| Refactoring Agent + iterative update loop | TRD §4.1, TRD §4.4, PRD §3.1, PRD §3.4 |
| Disturbance and defense registries | TRD §4.3, TRD §5.2, PRD §3.3 |
| Attack/clean comparative metrics | TRD §5.2, TRD §5.1, PRD §4.2 |
| Data source adapters + provenance | DRD §2.1, DRD §2.2, DRD §3 |
| Trajectories, logs, and report artifacts | TRD §5.3, TRD §6.1, PRD §4.3 |

## Out of Scope (for this backlog iteration)
- Full production Haskell runtime execution and distributed Cloud Haskell deployment.
- Live external API integration requiring credentials or paid access.
- Bayesian optimization and full MARL training algorithms beyond baseline placeholders.
- UI/dashboard development.

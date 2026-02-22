# Enhancement Specification: Ray Parallelism and Observability/Explainability UI  
**marl-forecast-game** Project  
Version: 0.1 (Declarative Draft)  
License: Apache-2.0 (consistent with repository)

## 1. Purpose and Scope

The system shall extend the existing MVP multi-agent Markov game forecasting framework to support scalable parallel execution and comprehensive observability/explainability through a unified user interface.

This specification covers two tightly integrated capability areas:
- Distributed parallel processing using Ray for simulations, data handling, and future agent training.
- Enhanced observability (system health, performance, resource usage) and explainability (agent decisions, simulation dynamics, adversarial impacts) via an interactive UI.

The enhancements preserve the core principles: immutable simulation state, pure-functional transitions, deterministic/reproducible behavior, chronological data handling, and adversarial evaluation via disturbance injection.

## 2. Functional Requirements

### 2.1 Parallel Execution (Ray Core)

The system shall:
- Execute multiple independent `ForecastGame` simulations concurrently across available CPU cores or distributed cluster nodes, parameterized by seed, disturbance flag, configuration variants, or horizon lengths.
- Support parallel runs for robustness evaluation (e.g., Monte Carlo statistics over seeds, clean vs. attacked scenario comparisons, multi-configuration sweeps).
- Enable future stateful agent representations as distributed actors (e.g., forecaster/adversary/defender instances maintaining internal memory or learned parameters).
- Allow parallel data preparation operations (ingestion, normalization, chronological splitting) when handling larger real-world OSINT datasets.
- Provide fault-tolerant execution: automatic task retries on transient failures and actor reconstruction on node loss.
- Integrate seamlessly with existing verification pipeline (`run_verification.py`) so parallel mode can replace or augment sequential execution.

### 2.2 Observability

The system shall expose real-time and historical visibility into:
- Ray cluster state: node health, resource utilization (CPU, memory, GPU if applicable), actor/task placement and lifecycle.
- Job and task progress: number of active/completed/failed simulation tasks, wall-clock time per scenario type.
- Custom application metrics: per-simulation and aggregated values including MAE, RMSE, MAPE, worst-case error, disturbance success rate, round counts, and deltas between clean/attacked outcomes.
- Logs and traces: structured logs from game rounds (state transitions, agent deltas, disturbance events) with distributed tracing across parallel executions.
- Alerts: configurable thresholds for anomalies (e.g., attacked MAE not exceeding clean MAE by expected margin, simulation exceeding max_rounds).

### 2.3 Explainability

The system shall provide interactive tools to understand and analyze agent behavior and simulation outcomes:
- Simulation replay: step-through visualization of game rounds showing immutable state evolution, forecasts vs. actual targets, trend/noise/exogenous/hidden components, and disturbance injections.
- Agent contribution breakdown: per-round and aggregate views of deltas applied by forecaster, adversary, and defender; influence heatmaps showing coordination patterns.
- Metric decomposition: attribution of error sources (e.g., portions due to trend extrapolation, noise, disturbances, agent miscoordination).
- What-if experimentation: interactive modification of parameters (disturbance probability, horizon, initial state, agent policies) with on-demand re-execution of affected simulations.
- Data lineage view: visualization of chronological train/valid/test splits, normalization applied, and (future) OSINT source ingestion flows.
- Future-proof placeholders for LLM-driven agent inspection: prompt history, generated code/refactoring diffs, and rationale logs when DSPy-REPL / HaskellRLM integration is active.

## 3. Non-Functional Requirements

- **Performance**: Parallel mode shall achieve near-linear speedup for embarrassingly parallel workloads (multiple independent simulations) up to available cores/nodes; no significant regression in single-run latency.
- **Observability Overhead**: Metric collection and logging shall add <5% runtime overhead in typical workloads.
- **Compatibility**: All enhancements use open-source components (Ray Core, Prometheus, Grafana, Streamlit or Dash); no proprietary dependencies.
- **Reproducibility**: Parallel runs preserve seeding and determinism where inputs are identical; UI replays reproduce exact simulation traces.
- **Scalability**: Design supports transition from laptop (multi-core) to cluster (Kubernetes/AWS/GCP via KubeRay or Anyscale) with minimal configuration changes.
- **Security**: UI access restricted to authenticated users when exposed beyond localhost; sensitive OSINT API keys never displayed.
- **Extensibility**: Architecture allows future integration of RLlib for MARL training and Ray Data for large-scale OSINT processing without breaking existing MVP flows.

## 4. Core Architectural Principles

- Ray serves as the unified distributed compute runtime for parallelism, task scheduling, actor management, and observability primitives.
- Observability builds on Ray Dashboard as primary entry point, extended with embedded Grafana panels for time-series metrics and custom application views.
- Explainability layer provided via lightweight Python web UI (Streamlit or Plotly Dash preferred) that queries Ray object store, serialized simulation results, and custom metrics/logs.
- All UI components read immutable artifacts (JSON-serialized states, forecast/target lists, metrics dicts) without mutating system state.
- Custom metrics exported via Ray's metrics API and scraped by Prometheus for Grafana visualization.
- Existing pure-Python framework modules remain unchanged where possible; extensions added via decorators, wrappers, or optional entry points.

## 5. Sequencing

1 (Foundation):
- Ray Core integration for parallel simulation execution.
- Basic custom metrics emission and Ray Dashboard + Prometheus/Grafana setup.

2 (Observability):
- Embedded Grafana panels in Ray Dashboard.
- Structured logging and distributed tracing.

3 (Explainability UI):
- Interactive simulation replay and metric decomposition views.
- What-if parameter tweaking interface.

4 (Future Alignment):
- Actor-based agent representations.
- RLlib environment wrappers for MARL experimentation.
- LLM inspection panels.

# PROPOSAL SPECIFICATION: Advanced MARL Defense and Aggregation Mechanisms

**Target Integration:** `marl-forecast-game` Core Framework
**Status:** Proposed Configuration Enhancement

## 1. Executive Summary

This specification outlines the technical integration of three advanced methodologies into the `marl-forecast-game` architecture. These enhancements aim to solve gradient starvation during Robust Adversarial RL (RARL) training, stress-test ensemble aggregators against coordinated attacks, and improve the calibration of probabilistic forecasts using predictive market dynamics.

| Feature | Primary Component Affected | Target Capability |
| --- | --- | --- |
| **Bounded Rationality Curriculum** | `framework/training.py` | MARL Training (RARL) |
| **Coordinated Ensemble Disruption** | `framework/agents.py`, `framework/disturbances.py` | Adversarial Robustness |
| **Kelly-Criterion BMA Weights** | `framework/aggregation.py` | Probabilistic Forecasting |

---

## 2. Feature Specification 1: Bounded Rationality Curriculum for RARL

### 2.1 Theoretical Foundation

Standard alternating RARL suffers from gradient starvation when a fully rational adversary overwhelmingly defeats a naive forecaster in early epochs. Implementing a bounded rationality curriculum resolves this by throttling adversarial competence early in training.

> **Citation:** *Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula* (arXiv:2311.01642).

### 2.2 Mathematical Formulation

The adversary's action selection is transitioned from an -greedy policy to a Quantal Response (Boltzmann) policy modulated by a temperature parameter .

The probability of selecting disturbance action  in state  is:


The temperature  decays across the alternating RARL epochs , transitioning from exploratory to exploitative:


### 2.3 Implementation Requirements

* **`SimulationConfig` Updates:** * Add `adversary_tau_init` (float, default: 5.0).
* Add `adversary_tau_final` (float, default: 0.1).
* Add `tau_decay_rate` (float, default: 0.05).


* **Agent Logic (`framework/agents.py`):** * Extend `AdversaryAgent` to accept a runtime `tau` parameter.
* Implement Boltzmann action sampling over the Q-table when `tau` is provided.


* **Training Loop (`framework/training.py`):** * In the `RARL` execution path, calculate  prior to freezing the adversary.
* Pass the calculated  to the adversary during the forecaster's training epoch.



---

## 3. Feature Specification 2: Coordinated Ensemble Disruption ("Wolfpack" Adversary)

### 3.1 Theoretical Foundation

Traditional multi-agent defenses fail when adversaries abandon random targeting in favor of correlated attacks aimed at breaking ensemble cooperation. A "Wolfpack" adversary specifically targets an agent and its highly correlated peers.

> **Citation:** *Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning* (arXiv:2502.02844).

### 3.2 Mathematical Formulation

Let  be the set of forecaster agents. The adversary computes the pairwise Pearson correlation coefficient  of historical prediction residuals.

Given a primary target  (e.g., the agent with the highest BMA weight), the adversary identifies a coalition target set  using a correlation threshold :


The adversary deploys a primary disturbance to  and secondary, correlated disturbances to all .

### 3.3 Implementation Requirements

* **Disturbance Factory (`framework/disturbances.py`):**
* Create `WolfpackDisturbance` capable of holding multi-agent targeting vectors.


* **Agent Logic (`framework/agents.py`):**
* Implement `WolfpackAdversary` subclass.
* Require read-only access to historical residuals to compute the correlation matrix .


* **Game Engine (`framework/game.py`):**
* Update `GameLoop.step()` to process multi-target disturbances, ensuring the pure-functional transition of `ForecastState` is maintained without state-leakage.



---

## 4. Feature Specification 3: Kelly-Criterion BMA Weights

### 4.1 Theoretical Foundation

Bayesian Model Averaging (BMA) can be modeled as a predictive game where forecasting is treated as a capital growth maximization problem. Agents wager a virtual bankroll, and weights are dynamically pruned based on strict proper scoring rules.

> **Citation:** *A Coding Theoretic Study of Homogeneous Markovian Predictive Games* (arXiv:2502.02433).

### 4.2 Mathematical Formulation

Each agent  maintains a virtual bankroll . In round , the agent receives a score  defined by the negative Continuous Ranked Probability Score (-CRPS) of its quantile fan against the realized target variable.

Bankroll scales via exponential growth:


The normalized BMA weight  for the subsequent round is:


### 4.3 Implementation Requirements

* **Aggregation Engine (`framework/aggregation.py`):**
* Refactor `BayesianAggregator` to initialize  for all registered forecasters.
* Replace standard Bayesian posterior updates with the exponential bankroll tracker.


* **Metrics Integration (`framework/metrics.py`):**
* Validate the `crps` function returns a log-likelihood equivalent suitable for .


* **Pruning Logic:**
* Introduce `bankruptcy_threshold` (e.g., 0.01) in `SimulationConfig`.
* If  `bankruptcy_threshold`, the agent is temporarily disabled in the `AgentRegistry` to optimize compute overhead.



---

## 5. Validation and Verification Strategy

To ensure strict compliance with the existing `marl-forecast-game` architecture, the following additions to the test suite are required:

1. **Hypothesis Property Tests (`tests/test_properties.py`):**
* Assert that  remains invariant under the Kelly-Criterion BMA logic.
* Assert that the Wolfpack target set  never exceeds the total number of registered forecasters.


2. **Scenario Validation (`scripts/run_validation_scenarios.py`):**
* Add Scenario 23: `RARL_Bounded_Rationality_Convergence` to measure the forecaster's early-epoch learning curve against  decay.
* Add Scenario 24: `Wolfpack_Ensemble_Stress_Test` to measure the robustness delta of the `stacked composition` defense under correlated attacks.
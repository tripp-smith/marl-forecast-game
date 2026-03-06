# Theory Notes

This note is intentionally implementation-oriented. It is not a formal paper, but it documents the mathematical assumptions that connect the code to standard forecasting and MARL language.

## 1. Markov Game View

At round \( t \), the environment is in state \( s_t \). The forecaster, adversary, and defender choose actions
\( a_t^F \), \( a_t^A \), and \( a_t^D \). The realized forecast is

\[
\hat{y}_t = v(s_t) + a_t^F + a_t^A + a_t^D + b_t,
\]

where \( v(s_t) \) is the current level encoded in the state and \( b_t \) is any accumulated refactor bias.

The next latent value follows the deterministic transition plus noise and disturbance used by `evolve_state(...)`:

\[
s_{t+1} = f(s_t, \epsilon_t, \delta_t),
\]

with \( \epsilon_t \) denoting base noise and \( \delta_t \) denoting disturbance mass.

## 2. Reward Design

The current runtime uses an accuracy-centered utility:

\[
r_t^F = -|\hat{y}_t - y_t|,
\]

with the adversary taking the sign-flipped reward and the defender sharing the forecaster's direction. This is deliberately simple and stable for tabular learning, but it can be generalized to include calibration penalties, attack cost, or social welfare terms.

## 3. Bayesian / Kelly-Style Aggregation

When multiple forecasters are active, the aggregator updates weights after observing forecast error. The implementation uses performance-sensitive reweighting analogous to repeated bankroll allocation:

\[
w_{i,t+1} \propto w_{i,t} \exp(\eta \cdot \text{score}_{i,t}),
\]

followed by normalization. In practice, this means strong experts accumulate probability mass while weak experts are gradually pruned or down-weighted.

## 4. Robustness As Explicit Opposition

Instead of treating corruption as exogenous random noise, the project models attack behavior as an agent policy. That matters because adversaries can adapt to the system's current weaknesses. The robust objective is therefore closer to

\[
\min_{\pi_F, \pi_D} \max_{\pi_A} \mathbb{E}\left[\sum_{t=1}^{T} r_t^F \right],
\]

than to ordinary risk minimization with IID perturbations.

## 5. MNPO Sketch

The codebase includes MNPO-oriented components to move from raw action values toward preference-informed policy updates. At a high level, the objective can be viewed as maximizing preference-consistent action probability while regularizing against unstable policy changes:

\[
\mathcal{L}_{\text{MNPO}}(\theta) =
- \mathbb{E}_{(s, a^+, a^-)} \left[ \log \sigma\left(\beta \left(Q_\theta(s, a^+) - Q_\theta(s, a^-)\right)\right)\right]
+ \lambda \, \Omega(\theta),
\]

where \( a^+ \) and \( a^- \) are preference-ranked actions and \( \Omega(\theta) \) is a stabilizing regularizer. The exact update path in code is tabular rather than deep, but the conceptual role is the same.

## 6. Regret Intuition

For the bandit-style learners in the repository, two standard regimes matter:

- In stochastic-gap settings, UCB-style methods aim for regret on the order of \( O(\log T) \).
- In adversarial settings, Tsallis-INF style guarantees are typically sublinear, commonly expressed as \( O(\sqrt{T}) \) up to constants and problem structure.

This matters for the benchmark interpretation: if a method is only stable under one regime, it should not be presented as generally robust.

## 7. Statistical Testing Guidance

The benchmark harness uses paired non-parametric testing because forecast error distributions are rarely Gaussian. For paired model comparisons on rolling windows, the Wilcoxon signed-rank test is a better default than a t-test unless normality is clearly justified.

## 8. Fairness And Bias Checks

The repository's fairness posture is preliminary. In synthetic subgroup experiments, the intended logic is:

\[
\Delta_{\text{group}} = |\text{MAE}_{g_1} - \text{MAE}_{g_2}|,
\]

and the system should flag cases where this gap exceeds a configured threshold. The current codebase has hooks for such checks, but richer demographic and domain-aware fairness analysis remains future work.

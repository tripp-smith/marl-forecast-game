# MNPO Integration

## Core Definitions

Preference Oracle P(x, y_i, {y_j}_{j≠i}) ∈ [0,1]
- Returns the probability that forecast y_i is preferred over the set of opponent forecasts {y_j}.
- In your framework: use negative CRPS (or PIT calibration score + Kelly BMA) as the underlying reward, then convert to probability via sigmoid or Bradley-Terry:\nP(y_i ≻ y_j) = 1 / (1 + exp(-(r_i - r_j)/β)), where r = -CRPS (lower CRPS = better).\nFor multiple opponents: average the win probabilities or use Plackett-Luce ranking.

Multiplayer Objective (for any policy π_i)\nJ(π_i, {π_j}{j≠i}) = E_x [ E{y_iπ_i, y_jπ_j} P(y_i ≻ {y_j}) ] - τ KL(π_i || π_ref)

Nash Equilibrium (target): A policy π* where no player can improve by unilaterally changing strategy.

## Loss

Squared-gap loss:

L(π) = E[(h(π, y_w, y_l) - target_gap)^2]

where

h(π, y_w, y_l) = log(π(y_w|x)/π(y_l|x)) − Σ_j λ_j log(π_j(y_w|x)/π_j(y_l|x))

target_gap = η/(2β)

## One-line switch from DPO-style to MNPO

```bash
python scripts/run_training.py --algorithm mnpo --mode TD --opponents 5 --eta 1.0
```

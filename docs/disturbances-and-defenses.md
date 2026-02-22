# Disturbances and Defenses

## Overview

The adversarial framework uses pluggable disturbance models to inject uncertainty into the simulation and defense models to counteract adversarial effects. Both are selected by name via factory functions and configured through `SimulationConfig`.

## Disturbance Models

All disturbance models implement the `DisturbanceModel` protocol:

```python
class DisturbanceModel(Protocol):
    def sample(self, state: ForecastState, rng: Random, config: SimulationConfig) -> float: ...
```

Each model uses `config.disturbance_prob` to decide whether to fire on a given round, and scales its output by `config.disturbance_scale` and `config.adversarial_intensity`.

### Model Catalog

| Name | Class | Trigger | Formula | Key Parameter |
|---|---|---|---|---|
| `gaussian` | `GaussianDisturbance` | `rng.random() <= prob` | `gauss(0, scale * intensity)` | -- |
| `shift` | `ShiftDisturbance` | `rng.random() <= prob` | `shift * intensity` | `shift=0.35` |
| `evasion` | `EvasionDisturbance` | `rng.random() <= prob` | `sign(value) * scale * factor * intensity` | `factor=0.2` |
| `volatility` | `VolatilityScaledDisturbance` | `rng.random() > prob` skips | `gauss(0, scale * dynamic_scale * intensity)` | `min_scale=0.05` |
| `regime_shift` | `RegimeShiftDisturbance` | `t % every_n == 0 && prob` | `+/- level_shift * intensity` | `level_shift=0.8`, `every_n_steps=12` |
| `volatility_burst` | `VolatilityBurstDisturbance` | `rng.random() > prob` skips | `gauss(0, scale * intensity * burst_scale)` | `burst_scale=2.5` |
| `drift` | `DriftDisturbance` | `rng.random() > prob` skips | `sign(exogenous) * step_drift * (t+1) * intensity` | `step_drift=0.03` |

### Model Details

**GaussianDisturbance** (default): Samples from a zero-mean Gaussian distribution. The standard deviation is `disturbance_scale * adversarial_intensity`. Produces symmetric, independent shocks.

**ShiftDisturbance**: Applies a constant-magnitude shift (`0.35 * intensity`) when triggered. Models sudden mean shifts like supply disruptions or policy changes.

**EvasionDisturbance**: Produces directional disturbances that depend on the sign of the current state value. If `value >= 0`, the disturbance is positive; otherwise negative. This models adversaries that try to amplify existing trends.

**VolatilityScaledDisturbance**: Dynamic standard deviation that scales with recent state volatility: `max(min_scale, |hidden_shift| + |exogenous|)`. Produces larger disturbances during already-volatile periods.

**RegimeShiftDisturbance**: Fires periodically (every `every_n_steps` rounds) with a random direction. Models structural breaks in the data generating process, such as recessions or policy regime changes.

**VolatilityBurstDisturbance**: Similar to Gaussian but with a larger scale multiplier (`burst_scale=2.5`). Models short-lived volatility spikes like flash crashes or natural disasters.

**DriftDisturbance**: Magnitude grows linearly with `t`, producing a systematic drift that becomes harder to forecast over longer horizons. Direction follows the sign of `exogenous`.

### Name Resolution

The `disturbance_from_name()` factory accepts these aliases:

| Input | Model |
|---|---|
| `gaussian`, `default` | `GaussianDisturbance` |
| `shift`, `regime_shift_basic` | `ShiftDisturbance` |
| `evasion`, `evasion_like` | `EvasionDisturbance` |
| `volatility`, `volatility_scaled`, `vol_scaled` | `VolatilityScaledDisturbance` |
| `regime_shift`, `regime` | `RegimeShiftDisturbance` |
| `volatility_burst`, `burst` | `VolatilityBurstDisturbance` |
| `drift`, `systematic_drift` | `DriftDisturbance` |

Unknown names log a warning and fall back to `GaussianDisturbance`.

## Defense Models

All defense models implement the `DefenseModel` protocol:

```python
class DefenseModel(Protocol):
    def defend(self, forecast_delta: float, adversary_delta: float) -> float: ...
```

The return value is a corrective delta added to the forecast.

### Model Catalog

| Name | Class | Strategy | Key Parameter |
|---|---|---|---|
| `dampening` | `DampeningDefense` | Proportional counter + forecast regularization | `dampening=0.6` |
| `clipping` | `ClippingDefense` | Full negation, clamped to range | `clip=0.2` |
| `bias_guard` | `BiasGuardDefense` | Full negation up to a maximum bias | `max_bias=0.12` |
| `ensemble` | `EnsembleDefense` | Average of dampening + clipping + bias_guard | -- |
| `stack:a,b` | `StackedDefense` | Sequential application of two models | -- |

### Model Details

**DampeningDefense** (default): Applies two corrections:
1. Negates `dampening` fraction (60%) of the adversary delta.
2. Subtracts a small regularization term (`0.1 * clamp(forecast_delta, -1, 1)`) to prevent forecast drift.

**ClippingDefense**: Fully negates the adversary delta but clamps the correction to `[-clip, clip]` (default `[-0.2, 0.2]`). Limits maximum correction magnitude.

**BiasGuardDefense**: Negates the adversary delta up to `max_bias` (default 0.12). If the adversary delta exceeds the threshold, the correction is capped. Prevents overcorrection from large attacks.

**EnsembleDefense**: Applies all three basic defenses (dampening, clipping, bias_guard) independently and averages their corrections:

```python
correction = (dampening.defend(...) + clipping.defend(...) + bias_guard.defend(...)) / 3.0
```

This provides a diversified defense that is robust to different attack patterns.

**StackedDefense**: Sequentially applies two defense models. The output of the first model modifies the inputs to the second:

```python
first_out = first.defend(forecast_delta, adversary_delta)
return second.defend(forecast_delta + first_out, adversary_delta + first_out)
```

### Stacked Composition

Stacked defenses are specified with the `stack:` prefix:

```python
defense_from_name("stack:dampening,clipping")
# Creates StackedDefense(DampeningDefense(), ClippingDefense())
```

This allows composing arbitrary defense strategies.

### Name Resolution

The `defense_from_name()` factory accepts these aliases:

| Input | Model |
|---|---|
| `dampening`, `default` | `DampeningDefense` |
| `clipping`, `clip` | `ClippingDefense` |
| `bias_guard`, `bias` | `BiasGuardDefense` |
| `ensemble`, `filter_ensemble` | `EnsembleDefense` |
| `stack:x,y` | `StackedDefense(x, y)` |

Unknown names log a warning and fall back to `DampeningDefense`.

## Configuration

Disturbance and defense models are selected through `SimulationConfig`:

```python
config = SimulationConfig(
    disturbance_model="regime_shift",
    defense_model="ensemble",
    disturbance_prob=0.2,
    disturbance_scale=1.2,
    adversarial_intensity=1.5,
)
```

See [configuration.md](configuration.md) for full parameter reference.

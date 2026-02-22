# Haskell Migration

## Overview

The project maintains a parallel Haskell scaffold under `haskell/` that mirrors the core Python types and transition logic. This scaffold serves as a foundation for migrating performance-critical pure functions to Haskell while keeping the Python orchestration layer. A subprocess bridge (`HaskellRLMRuntime`) enables the game engine to call Haskell functions at runtime.

## Project Structure

```
haskell/
├── src/
│   ├── Types.hs          # ForecastState algebraic data type
│   └── Game.hs           # Pure evolveState transition function
└── test/
    └── Main.hs           # QuickCheck property tests
```

## Types.hs

Defines the `ForecastState` ADT, mirroring the Python frozen dataclass:

```haskell
module Types where

data ForecastState = ForecastState
  { t :: Int
  , value :: Double
  , exogenous :: Double
  , hiddenShift :: Double
  } deriving (Eq, Show)
```

### Correspondence with Python

| Haskell Field | Python Field | Type |
|---|---|---|
| `t` | `t` | `Int` / `int` |
| `value` | `value` | `Double` / `float` |
| `exogenous` | `exogenous` | `Double` / `float` |
| `hiddenShift` | `hidden_shift` | `Double` / `float` |

The Haskell type currently omits `segment_id`, `segment_values`, and `macro_context` fields -- these would be added as the migration progresses to support hierarchical agents.

## Game.hs

Implements the pure state transition function:

```haskell
module Game where

import Types

evolveState :: ForecastState -> Double -> Double -> Double -> ForecastState
evolveState s baseTrend noise disturbance =
  ForecastState
    { t = t s + 1
    , value = value s + baseTrend + 0.4 * exogenous s + noise + disturbance
    , exogenous = 0.6 * exogenous s + 0.2 * disturbance
    , hiddenShift = disturbance
    }
```

### Formula Parity

The transition formula is identical to the Python `evolve_state` (without `macro_contribution`):

```
new_value     = value + baseTrend + 0.4 * exogenous + noise + disturbance
new_exogenous = 0.6 * exogenous + 0.2 * disturbance
new_shift     = disturbance
```

This ensures that given identical floating-point inputs, both implementations produce the same output (within IEEE 754 rounding).

## QuickCheck Property Tests

The `test/Main.hs` file would contain property-based tests using QuickCheck to verify:

1. **Determinism**: `evolveState s t n d == evolveState s t n d` for all inputs (1000 iterations).
2. **Time progression**: `t (evolveState s _ _ _) == t s + 1`.
3. **Disturbance propagation**: `hiddenShift (evolveState s _ _ d) == d`.

These properties mirror the Hypothesis tests on the Python side, providing cross-language parity verification.

## HaskellRLMRuntime (Subprocess Bridge)

The `HaskellRLMRuntime` class in `framework/strategy_runtime.py` invokes Haskell functions via a subprocess:

```python
@dataclass(frozen=True)
class HaskellRLMRuntime:
    fallback: PythonStrategyRuntime = PythonStrategyRuntime()
    haskell_dir: str = "haskell"
    timeout_s: float = 2.0
```

### How It Works

1. Serializes the current `ForecastState` to JSON:

```json
{
    "t": 5,
    "value": 12.3,
    "exogenous": 0.8,
    "hidden_shift": 0.1
}
```

2. Invokes `cabal run marl-forecast-game -- --delta` as a subprocess with JSON on stdin.
3. Reads a single float from stdout (the forecast delta).
4. Returns the float as the `forecast_delta`.

### Fallback Behavior

On any error (missing Cabal, compilation failure, timeout, parse error), the runtime silently falls back to `PythonStrategyRuntime.forecast_delta()`:

```python
return self.fallback.forecast_delta(state)  # 0.4 + 0.4 * exogenous
```

### Timeout

The subprocess has a 2-second timeout (`timeout_s=2.0`). This prevents Haskell compilation or startup delays from blocking the game loop.

### Selection

The Haskell runtime is selected via `SimulationConfig`:

```python
config = SimulationConfig(runtime_backend="haskell")
```

Or programmatically:

```python
from framework.strategy_runtime import runtime_from_name
runtime = runtime_from_name("haskell")
```

## Migration Roadmap

The current scaffold provides a minimal proof-of-concept. Full migration would involve:

1. **Extend Types.hs**: Add `segment_values` and `macro_context` fields (as `Map String Double`).
2. **Add macro_contribution**: Update `evolveState` to accept a coefficient map and compute the macro contribution term.
3. **Cabal project setup**: Create a proper `.cabal` file with executable targets and test suites.
4. **JSON I/O**: Use `aeson` for JSON serialization in the Haskell executable.
5. **QuickCheck tests**: Implement the full property test suite with cross-validation against Python outputs.
6. **Performance benchmarking**: Compare Haskell vs. Python transition function throughput for large-scale simulations.
7. **Bidirectional bridge**: Enable Python to call Haskell for both `evolveState` and agent policies, and Haskell to call Python for data loading.

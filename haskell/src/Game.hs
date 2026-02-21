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

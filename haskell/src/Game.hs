module Game (evolveState, forecastDelta, gameStep) where

import Types

evolveState :: ForecastState -> Double -> Double -> Double -> ForecastState
evolveState s baseTrend noise disturbance =
  ForecastState
    { t = t s + 1
    , value = value s + baseTrend + 0.4 * exogenous s + noise + disturbance
    , exogenous = 0.6 * exogenous s + 0.2 * disturbance
    , hiddenShift = disturbance
    }

forecastDelta :: ForecastState -> Double
forecastDelta s = 0.4 + 0.4 * exogenous s

gameStep :: ForecastState -> AgentAction -> AgentAction -> AgentAction -> Double -> (ForecastState, Double)
gameStep s fAction aAction dAction disturbance =
  let forecast  = value s + delta fAction + delta aAction + delta dAction
      nextState = evolveState s 0.4 0.0 disturbance
      target    = value nextState
      reward    = -(abs (target - forecast))
  in (nextState, reward)

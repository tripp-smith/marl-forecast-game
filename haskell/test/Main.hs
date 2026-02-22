module Main where

import Test.QuickCheck
import Game
import Types

prop_evolveDeterministic :: Double -> Double -> Bool
prop_evolveDeterministic x y =
  let s = ForecastState 0 x y 0
  in evolveState s 0.4 0.1 0.2 == evolveState s 0.4 0.1 0.2

prop_evolveTimestepIncrement :: Int -> Double -> Double -> Double -> Bool
prop_evolveTimestepIncrement tVal v e hs =
  let s = ForecastState tVal v e hs
      s' = evolveState s 0.4 0.0 0.0
  in t s' == tVal + 1

prop_evolveNoNaN :: Double -> Double -> Double -> Double -> Bool
prop_evolveNoNaN v e bt d =
  let s = ForecastState 0 v e 0
      s' = evolveState s bt 0.0 d
  in not (isNaN (value s')) && not (isInfinite (value s'))
     && not (isNaN (exogenous s')) && not (isInfinite (exogenous s'))

prop_forecastDeltaDeterministic :: Double -> Double -> Bool
prop_forecastDeltaDeterministic v e =
  let s = ForecastState 0 v e 0
  in forecastDelta s == forecastDelta s

prop_gameStepRewardSign :: Double -> Double -> Bool
prop_gameStepRewardSign v e =
  let s = ForecastState 0 v e 0
      fAct = AgentAction "forecaster" 0.1
      aAct = AgentAction "adversary" (-0.05)
      dAct = AgentAction "defender" 0.02
      (_, reward) = gameStep s fAct aAct dAct 0.0
  in reward <= 0.0

main :: IO ()
main = do
  putStrLn "=== QuickCheck property tests ==="
  quickCheckWith args prop_evolveDeterministic
  quickCheckWith args prop_evolveTimestepIncrement
  quickCheckWith args prop_evolveNoNaN
  quickCheckWith args prop_forecastDeltaDeterministic
  quickCheckWith args prop_gameStepRewardSign
  where
    args = stdArgs { maxSuccess = 1000 }

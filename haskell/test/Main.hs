module Main where

import Test.QuickCheck
import Game
import Types

prop_evolveDeterministic :: Double -> Double -> Bool
prop_evolveDeterministic x y =
  let s = ForecastState 0 x y 0
  in evolveState s 0.4 0.1 0.2 == evolveState s 0.4 0.1 0.2

main :: IO ()
main = quickCheckWith stdArgs {maxSuccess = 1000} prop_evolveDeterministic

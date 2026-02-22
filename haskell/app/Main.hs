module Main where

import System.Environment (getArgs)
import Game (evolveState, forecastDelta)
import Types (ForecastState(..))

parseState :: String -> Maybe ForecastState
parseState input =
  let tokens = words $ map (\c -> if c `elem` "{}\"," then ' ' else c) input
      pairs = go tokens
  in do
    tVal   <- lookup "t" pairs >>= safeReadInt
    vVal   <- lookup "value" pairs >>= safeReadDouble
    eVal   <- lookup "exogenous" pairs >>= safeReadDouble
    hsVal  <- lookup "hidden_shift" pairs >>= safeReadDouble
    Just ForecastState { t = tVal, value = vVal, exogenous = eVal, hiddenShift = hsVal }
  where
    go [] = []
    go (k : v : rest)
      | last k == ':' = (init k, v) : go rest
      | otherwise     = (k, v) : go rest
    go [_] = []
    safeReadDouble s = case reads s of
      [(x, "")] -> Just x
      _         -> Nothing
    safeReadInt s = case reads s of
      [(x, "")] -> Just x
      _         -> Nothing

main :: IO ()
main = do
  args <- getArgs
  input <- getContents
  case parseState input of
    Nothing -> putStrLn "error: invalid JSON input"
    Just st ->
      case args of
        ["--delta"]  -> print (forecastDelta st)
        ["--evolve"] -> print (evolveState st 0.4 0.0 0.0)
        _            -> print (forecastDelta st)

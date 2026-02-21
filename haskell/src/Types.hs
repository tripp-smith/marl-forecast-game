module Types where

data ForecastState = ForecastState
  { t :: Int
  , value :: Double
  , exogenous :: Double
  , hiddenShift :: Double
  } deriving (Eq, Show)

module Types where

data ForecastState = ForecastState
  { t :: Int
  , value :: Double
  , exogenous :: Double
  , hiddenShift :: Double
  } deriving (Eq, Show)

data AgentAction = AgentAction
  { actor :: String
  , delta :: Double
  } deriving (Eq, Show)

data ConfidenceInterval = ConfidenceInterval
  { lower :: Double
  , upper :: Double
  } deriving (Eq, Show)

data SimulationConfig = SimulationConfig
  { horizon :: Int
  , maxRounds :: Int
  , baseNoiseStd :: Double
  , disturbanceProb :: Double
  , disturbanceScale :: Double
  , adversarialIntensity :: Double
  , attackCost :: Double
  } deriving (Eq, Show)

defaultConfig :: SimulationConfig
defaultConfig = SimulationConfig
  { horizon = 100
  , maxRounds = 200
  , baseNoiseStd = 0.15
  , disturbanceProb = 0.1
  , disturbanceScale = 1.0
  , adversarialIntensity = 1.0
  , attackCost = 0.0
  }

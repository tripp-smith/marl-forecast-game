from .base import NormalizedRecord, SourceAdapter
from .bea import BEAAdapter
from .bis_policy_rate import BISPolicyRateAdapter
from .eurostat import EurostatAdapter
from .geopolitical_risk import GeopoliticalRiskAdapter
from .kaggle_demand import KaggleDemandAdapter
from .kalshi import KalshiAdapter
from .macro_fred import FredMacroAdapter
from .macro_imf import ImfMacroAdapter
from .oecd_cli import OECDCLIAdapter
from .prediction_polymarket import PolymarketAdapter
from .predictit import PredictItAdapter
from .world_bank import WorldBankAdapter

__all__ = [
    "NormalizedRecord",
    "SourceAdapter",
    "BEAAdapter",
    "BISPolicyRateAdapter",
    "EurostatAdapter",
    "FredMacroAdapter",
    "GeopoliticalRiskAdapter",
    "KaggleDemandAdapter",
    "KalshiAdapter",
    "ImfMacroAdapter",
    "OECDCLIAdapter",
    "PolymarketAdapter",
    "PredictItAdapter",
    "WorldBankAdapter",
]

from .base import NormalizedRecord, SourceAdapter
from .bis_policy_rate import BISPolicyRateAdapter
from .geopolitical_risk import GeopoliticalRiskAdapter
from .macro_fred import FredMacroAdapter
from .macro_imf import ImfMacroAdapter
from .oecd_cli import OECDCLIAdapter
from .prediction_polymarket import PolymarketAdapter

__all__ = [
    "NormalizedRecord",
    "SourceAdapter",
    "BISPolicyRateAdapter",
    "FredMacroAdapter",
    "GeopoliticalRiskAdapter",
    "ImfMacroAdapter",
    "OECDCLIAdapter",
    "PolymarketAdapter",
]

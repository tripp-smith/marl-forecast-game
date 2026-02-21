from .base import NormalizedRecord, SourceAdapter
from .macro_fred import FredMacroAdapter
from .macro_imf import ImfMacroAdapter
from .prediction_polymarket import PolymarketAdapter

__all__ = [
    "NormalizedRecord",
    "SourceAdapter",
    "FredMacroAdapter",
    "ImfMacroAdapter",
    "PolymarketAdapter",
]

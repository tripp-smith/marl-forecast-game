from .base import NormalizedRecord, SourceAdapter
from .macro_fred import FredMacroAdapter
from .prediction_polymarket import PolymarketAdapter

__all__ = [
    "NormalizedRecord",
    "SourceAdapter",
    "FredMacroAdapter",
    "PolymarketAdapter",
]

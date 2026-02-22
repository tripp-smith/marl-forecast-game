"""PredictIt adapter -- fetches prediction-market data from the public PredictIt JSON API."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen

from .base import NormalizedRecord

PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"


@dataclass(frozen=True)
class PredictItAdapter:
    name: str = "predictit"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = datetime(2024, 1, 1 + (i % 28))
            prob = 0.4 + 0.03 * (i % 10) - 0.01 * (i % 4)
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id=f"predictit_market_{i}",
                    target=prob,
                    promo=0.0,
                    macro_index=0.0,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        try:
            req = Request(PREDICTIT_URL, headers={"Accept": "application/json"})
            with urlopen(req, timeout=15) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

            markets = payload.get("markets", [])
            if not markets:
                return self._synthetic(periods)

            now = datetime.utcnow()
            rows: list[NormalizedRecord] = []
            for mkt in markets[:periods]:
                mkt_id = mkt.get("id", "unknown")
                contracts = mkt.get("contracts", [])
                for contract in contracts[:1]:
                    price = contract.get("lastTradePrice")
                    if price is None:
                        continue
                    ts_str = contract.get("dateEnd", "")
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None) if ts_str else now
                    except ValueError:
                        ts = now

                    rows.append(
                        NormalizedRecord(
                            timestamp=ts,
                            series_id=f"predictit_{mkt_id}",
                            target=float(price),
                            promo=0.0,
                            macro_index=0.0,
                            source=self.name,
                            fetched_at=now,
                        )
                    )

            rows.sort(key=lambda r: r.timestamp)
            return rows[-periods:] if rows else self._synthetic(periods)
        except Exception:
            logging.debug("PredictIt API fetch failed; using synthetic fallback", exc_info=True)
            return self._synthetic(periods)

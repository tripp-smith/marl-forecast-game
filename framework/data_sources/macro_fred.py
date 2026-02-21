from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from typing import Any

import json
from urllib.parse import urlencode
from urllib.request import urlopen

from .base import NormalizedRecord


@dataclass(frozen=True)
class FredMacroAdapter:
    name: str = "fred"
    series_id: str = "CPIAUCSL"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=i)
            macro = 100.0 + (0.08 * i)
            target = 50.0 + (0.12 * i) + 0.1 * macro
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="fred_gdp_proxy",
                    target=target,
                    promo=0.0,
                    macro_index=macro,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        key = os.getenv("FRED_API_KEY")
        if not key:
            return self._synthetic(periods)

        params = {
            "series_id": self.series_id,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": max(5, periods),
        }
        try:
            query = urlencode(params)
            with urlopen(f"https://api.stlouisfed.org/fred/series/observations?{query}", timeout=10) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            observations = payload.get("observations", [])
            parsed = [o for o in observations if o.get("value") not in {".", None}]
            parsed = list(reversed(parsed[-periods:]))
            now = datetime.utcnow()
            rows: list[NormalizedRecord] = []
            for idx, obs in enumerate(parsed):
                value = float(obs["value"])
                ts = datetime.fromisoformat(obs["date"])
                rows.append(
                    NormalizedRecord(
                        timestamp=ts,
                        series_id=f"fred_{self.series_id.lower()}",
                        target=value,
                        promo=0.0,
                        macro_index=100.0 + (0.03 * idx),
                        source=self.name,
                        fetched_at=now,
                    )
                )
            return rows if rows else self._synthetic(periods)
        except Exception:
            return self._synthetic(periods)

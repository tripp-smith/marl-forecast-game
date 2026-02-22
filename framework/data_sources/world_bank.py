"""World Bank adapter -- fetches indicator data from the public World Bank API (no key)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen

from .base import NormalizedRecord


@dataclass(frozen=True)
class WorldBankAdapter:
    name: str = "worldbank"
    indicator: str = "NY.GDP.MKTP.KD.ZG"
    country: str = "WLD"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        rows: list[NormalizedRecord] = []
        base_year = 2000
        for i in range(periods):
            year = base_year + i
            gdp_growth = 2.5 + 0.3 * (i % 5) - 0.2 * (i % 3)
            rows.append(
                NormalizedRecord(
                    timestamp=datetime(year, 7, 1),
                    series_id=f"worldbank_{self.indicator.lower().replace('.', '_')}",
                    target=gdp_growth,
                    promo=0.0,
                    macro_index=gdp_growth,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        url = (
            f"https://api.worldbank.org/v2/country/{self.country}/"
            f"indicator/{self.indicator}"
            f"?format=json&per_page={max(10, periods)}"
        )
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=15) as resp:
                payload: Any = json.loads(resp.read().decode("utf-8"))

            if not isinstance(payload, list) or len(payload) < 2:
                return self._synthetic(periods)

            data_array = payload[1]
            if not data_array:
                return self._synthetic(periods)

            now = datetime.utcnow()
            sid = f"worldbank_{self.indicator.lower().replace('.', '_')}"
            rows: list[NormalizedRecord] = []
            for entry in data_array:
                value = entry.get("value")
                date_str = entry.get("date")
                if value is None or date_str is None:
                    continue
                try:
                    ts = datetime(int(date_str), 7, 1)
                    val = float(value)
                except (ValueError, TypeError):
                    continue
                rows.append(
                    NormalizedRecord(
                        timestamp=ts,
                        series_id=sid,
                        target=val,
                        promo=0.0,
                        macro_index=val,
                        source=self.name,
                        fetched_at=now,
                    )
                )

            rows.sort(key=lambda r: r.timestamp)
            return rows[-periods:] if rows else self._synthetic(periods)
        except Exception:
            logging.debug("World Bank API fetch failed; using synthetic fallback", exc_info=True)
            return self._synthetic(periods)

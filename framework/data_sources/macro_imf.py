from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import json
from urllib.parse import urlencode
from urllib.request import urlopen

from .base import NormalizedRecord


@dataclass(frozen=True)
class ImfMacroAdapter:
    name: str = "imf"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=i)
            macro = 98.5 + (0.06 * i)
            target = 42.0 + (0.11 * i) + 0.12 * macro
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="imf_weo_proxy",
                    target=target,
                    promo=0.0,
                    macro_index=macro,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        url = "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH"
        try:
            with urlopen(url, timeout=10) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            values = payload.get("values", {}).get("NGDP_RPCH", {})
            country = values.get("WEO_WLD") or next(iter(values.values()), {})
            series = sorted(country.items(), key=lambda kv: kv[0])[-periods:]
            now = datetime.utcnow()
            rows: list[NormalizedRecord] = []
            for idx, (year, value) in enumerate(series):
                ts = datetime(int(year), 1, 1)
                v = float(value)
                rows.append(
                    NormalizedRecord(
                        timestamp=ts,
                        series_id="imf_ngdp_rpch",
                        target=v,
                        promo=0.0,
                        macro_index=100.0 + 0.1 * idx,
                        source=self.name,
                        fetched_at=now,
                    )
                )
            return rows if rows else self._synthetic(periods)
        except Exception:
            return self._synthetic(periods)

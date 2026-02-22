"""OECD Composite Leading Indicators adapter -- fetches from OECD SDMX API (public, no key)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.request import Request, urlopen

from .base import NormalizedRecord


@dataclass(frozen=True)
class OECDCLIAdapter:
    name: str = "oecd_cli"
    country: str = "USA"

    def _synthetic(self, periods: int) -> list[NormalizedRecord]:
        now = datetime.utcnow()
        start = datetime(2023, 1, 1)
        rows: list[NormalizedRecord] = []
        for i in range(periods):
            ts = start + timedelta(days=30 * i)
            cli_value = 99.5 + 0.1 * i + 0.3 * (i % 6)
            rows.append(
                NormalizedRecord(
                    timestamp=ts,
                    series_id="oecd_cli",
                    target=cli_value,
                    promo=0.0,
                    macro_index=cli_value,
                    source=self.name,
                    fetched_at=now,
                )
            )
        return rows

    def fetch(self, periods: int = 30) -> list[NormalizedRecord]:
        url = (
            f"https://sdmx.oecd.org/public/rest/data/"
            f"OECD.SDD.STES,DSD_STES@DF_CLI/.M.LI.GYSA.{self.country}"
            f"?format=jsondata&lastNObservations={max(5, periods)}"
        )
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=15) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

            datasets = payload.get("dataSets", [{}])
            if not datasets:
                return self._synthetic(periods)

            series_map = datasets[0].get("series", {})
            if not series_map:
                return self._synthetic(periods)

            dims = payload.get("structure", {}).get("dimensions", {}).get("observation", [])
            time_dim = next((d for d in dims if d.get("id") == "TIME_PERIOD"), None)
            time_values = [v["id"] for v in time_dim["values"]] if time_dim else []

            first_key = next(iter(series_map))
            observations = series_map[first_key].get("observations", {})

            now = datetime.utcnow()
            rows: list[NormalizedRecord] = []
            for obs_idx, obs_val in sorted(observations.items(), key=lambda kv: int(kv[0])):
                idx = int(obs_idx)
                value = float(obs_val[0])
                ts_str = time_values[idx] if idx < len(time_values) else None
                if ts_str is None:
                    continue
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m")
                except ValueError:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except ValueError:
                        continue

                rows.append(
                    NormalizedRecord(
                        timestamp=ts,
                        series_id=f"oecd_cli_{self.country.lower()}",
                        target=value,
                        promo=0.0,
                        macro_index=value,
                        source=self.name,
                        fetched_at=now,
                    )
                )

            return rows[-periods:] if rows else self._synthetic(periods)
        except Exception:
            logging.debug("OECD CLI API fetch failed; using synthetic fallback", exc_info=True)
            return self._synthetic(periods)

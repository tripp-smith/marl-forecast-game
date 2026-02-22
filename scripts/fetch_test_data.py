#!/usr/bin/env python3
"""Download historical qualitative test data and build a manifest CSV.

Usage:
    python scripts/fetch_test_data.py [--start 2010-01-01] [--end 2025-12-31]
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.data_sources.beige_book import BeigeBookAdapter
from framework.data_sources.earnings import EarningsAdapter
from framework.data_sources.pmi import PMIAdapter


_ADAPTERS = {
    "beige_book": BeigeBookAdapter,
    "pmi": PMIAdapter,
    "earnings": EarningsAdapter,
}

_SPLIT_BOUNDARIES = {
    "train": (datetime(2010, 1, 1), datetime(2020, 12, 31)),
    "valid": (datetime(2021, 1, 1), datetime(2023, 12, 31)),
    "test": (datetime(2024, 1, 1), datetime(2025, 12, 31)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch qualitative test data")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--output-dir", type=str, default="data/test_qualitative")
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    output_dir = Path(args.output_dir)

    manifest_rows: list[dict[str, str]] = []
    total_records = 0

    for adapter_name, adapter_cls in _ADAPTERS.items():
        adapter = adapter_cls(cache_dir=str(output_dir / adapter_name))
        records = adapter.fetch_releases(start_dt, end_dt)
        print(f"  {adapter_name}: {len(records)} records")

        for rec in records:
            ts_str = rec.timestamp.strftime("%Y-%m-%d")
            suffix = ""
            if adapter_name == "earnings":
                ticker = rec.metadata.get("ticker", "UNK")
                suffix = f"_{ticker}"
            local_path = output_dir / adapter_name / f"{ts_str}{suffix}.txt"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(rec.text, encoding="utf-8")

            checksum = sha256(rec.text.encode("utf-8")).hexdigest()
            url = rec.metadata.get("url", "")

            split = "unknown"
            for split_name, (s, e) in _SPLIT_BOUNDARIES.items():
                if s <= rec.timestamp <= e:
                    split = split_name
                    break

            manifest_rows.append({
                "source_id": rec.source_id,
                "timestamp": rec.timestamp.isoformat(),
                "url": url,
                "local_path": str(local_path),
                "checksum": checksum,
                "split": split,
            })
            total_records += 1

    manifest_path = output_dir / "test_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_id", "timestamp", "url", "local_path", "checksum", "split"],
        )
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: r["timestamp"]))

    print(f"\nTotal: {total_records} records written")
    print(f"Manifest: {manifest_path}")

    for split_name in ("train", "valid", "test"):
        count = sum(1 for r in manifest_rows if r["split"] == split_name)
        print(f"  {split_name}: {count}")


if __name__ == "__main__":
    main()

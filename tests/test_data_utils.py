from __future__ import annotations

from datetime import datetime, timedelta

from framework.data_utils import detect_isolation_forest_anomalies, detect_poisoned_rows


def test_isolation_forest_detects_synthetic_poison():
    base = datetime(2024, 1, 1)
    rows = [
        {
            "timestamp": base + timedelta(days=i),
            "series_id": "s",
            "target": 100.0 + (i % 3),
            "promo": 0.0,
            "macro_index": 1.0,
        }
        for i in range(40)
    ]
    for idx in range(3):
        rows[-(idx + 1)] = {
            **rows[-(idx + 1)],
            "target": 1000.0 + idx * 50.0,
        }
    result = detect_isolation_forest_anomalies(rows, contamination=0.1)
    assert len(result.suspects) >= 2


def test_detect_poisoned_rows_returns_detector_map():
    base = datetime(2024, 1, 1)
    rows = [
        {
            "timestamp": base + timedelta(days=i),
            "series_id": "s",
            "target": 10.0 if i < 15 else 150.0,
            "promo": 0.0,
            "macro_index": 1.0,
        }
        for i in range(20)
    ]
    results = detect_poisoned_rows(rows, poisoning_threshold=0.1)
    assert "isolation_forest" in results
    assert "autoencoder" in results

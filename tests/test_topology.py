from __future__ import annotations

import pytest

from framework.topology import CoalitionTopologyManager


def test_dynamic_coalitions_form_high_modularity_groups():
    nx = pytest.importorskip("networkx")
    _ = nx
    manager = CoalitionTopologyManager(reform_interval=1)
    utilities = {"a": 1.0, "b": 0.9, "c": -0.8, "d": -0.9}
    affinities = {
        ("a", "b"): 0.95,
        ("c", "d"): 0.95,
        ("a", "c"): 0.05,
        ("a", "d"): 0.05,
        ("b", "c"): 0.05,
        ("b", "d"): 0.05,
    }
    coalitions = manager.reform(utilities, affinities, round_idx=0, dynamic=True)
    assert any(set(group) == {"a", "b"} for group in coalitions)
    assert any(set(group) == {"c", "d"} for group in coalitions)
    assert manager.modularity() > 0.35

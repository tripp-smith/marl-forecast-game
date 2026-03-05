from __future__ import annotations

import pytest

from framework.training import QTableAgent, build_rl_agent, state_to_vector
from framework.types import ForecastState, SimulationConfig


def test_state_to_vector_flattens_high_dimensional_state():
    state = ForecastState(
        t=1,
        value=10.0,
        exogenous=0.5,
        hidden_shift=-0.2,
        segment_values={"a": 1.0, "b": 2.0},
        macro_context={"gdp": 0.1},
        qualitative_state=(1, -1),
        raw_qual_state=(0.2, 0.4),
        economic_regime=2,
    )
    vector = state_to_vector(state)
    assert vector.ndim == 1
    assert len(vector) >= 10


def test_build_rl_agent_uses_config_switches():
    tabular_cfg = SimulationConfig(rl_backend="tabular", epsilon_final=0.02)
    tabular = build_rl_agent(tabular_cfg)
    assert isinstance(tabular, QTableAgent)
    assert tabular.epsilon_min == pytest.approx(0.02)


def test_simulation_config_validates_new_rl_fields():
    with pytest.raises(ValueError):
        SimulationConfig(rl_backend="invalid")
    with pytest.raises(ValueError):
        SimulationConfig(rl_algorithm="invalid")
    with pytest.raises(ValueError):
        SimulationConfig(poisoning_threshold=2.0)


def test_deep_rl_backend_matches_tabular_policy_direction_when_torch_available():
    torch = pytest.importorskip("torch")
    _ = torch
    cfg = SimulationConfig(
        rl_backend="deep",
        rl_algorithm="dqn",
        replay_buffer_size=256,
        rl_batch_size=16,
        target_update_interval=10,
        epsilon_final=0.01,
    )
    state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
    next_state = ForecastState(t=1, value=11.0, exogenous=0.0, hidden_shift=0.0)

    tabular = QTableAgent(epsilon=0.0)
    action = tabular.action_space.delta_to_action(0.2)
    for _ in range(200):
        tabular.update(state, action, 1.0, next_state)

    deep = build_rl_agent(cfg, state_dim=len(state_to_vector(state)))
    for _ in range(300):
        deep.update(state, action, 1.0, next_state)

    tabular_q = tabular._get_q(  # type: ignore[attr-defined]
        __import__("framework.training", fromlist=["_state_hash"])._state_hash(state)
    )
    deep_q = deep.q_values(state)  # type: ignore[attr-defined]
    assert int(tabular_q.argmax()) == int(deep_q.argmax())
    assert float(deep_q[action]) == pytest.approx(float(deep_q.max()))
    assert all(abs(float(val)) < 1e6 for val in deep_q)

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from framework.agents import QLearnedAgent
from framework.training import QTableAgent, TrainingLoop
from framework.types import ForecastState


@given(
    t=st.integers(min_value=0, max_value=1_000),
    value=st.floats(min_value=-1_000, max_value=1_000, allow_nan=False, allow_infinity=False),
    exogenous=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    hidden_shift=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=1000)
def test_qlearned_agent_is_deterministic_for_same_state(tmp_path, t, value, exogenous, hidden_shift):
    model_path = tmp_path / "q_agent.pkl"
    q = QTableAgent(epsilon=0.0)
    TrainingLoop.save_q_table(q, model_path)

    agent = QLearnedAgent(name="qlearned", q_table_path=str(model_path), algorithm="q")
    state = ForecastState(t=t, value=value, exogenous=exogenous, hidden_shift=hidden_shift)

    first = agent.act(state)
    second = agent.act(state)
    assert first == second

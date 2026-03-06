from __future__ import annotations

import re
from unittest.mock import Mock, patch

import pytest

from framework.llm.client import (
    AnthropicProviderClient,
    GrokProviderClient,
    OllamaProviderClient,
    OpenAIProviderClient,
    bias_simulate,
    provider_client_from_config,
)
from framework.llm.refiner import RecursiveStrategyRefiner
from framework.types import AgentAction, ForecastState, TrajectoryEntry, frozen_mapping


def _mock_response(payload):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


@patch("framework.llm.client.requests.post")
def test_provider_clients_return_identical_strings_for_same_prompt(mock_post):
    prompt = "state -> forecast"
    mock_post.side_effect = [
        _mock_response({"response": "0.125"}),
        _mock_response({"choices": [{"message": {"content": "0.125"}}]}),
        _mock_response({"content": [{"text": "0.125"}]}),
        _mock_response({"choices": [{"message": {"content": "0.125"}}]}),
    ]
    outputs = [
        OllamaProviderClient(model="m").query(prompt),
        OpenAIProviderClient(model="m", api_key="k").query(prompt),
        AnthropicProviderClient(model="m", api_key="k").query(prompt),
        GrokProviderClient(model="m", api_key="k").query(prompt),
    ]
    assert outputs == ["0.125"] * 4


@patch("framework.llm.client.requests.post")
def test_provider_client_tracks_cost_and_retries(mock_post):
    mock_post.side_effect = [
        RuntimeError("transient"),
        _mock_response({"choices": [{"message": {"content": "final answer"}}]}),
    ]
    client = OpenAIProviderClient(model="mini", api_key="k", max_retries=2)
    text = client.query("hello world")
    assert text == "final answer"
    assert client.cost_tracker.total_cost_usd > 0.0


def test_provider_factory_falls_back_to_ollama_for_unknown_remote_failure():
    client = provider_client_from_config("ollama", model="local")
    assert isinstance(client, OllamaProviderClient)


def test_recursive_refiner_supports_query_clients():
    client = Mock()
    client.query.side_effect = [
        '{"bias_adjustment": -0.02, "rationale": "base"}',
        '{"bias_adjustment": -0.03, "rationale": "refined"}',
    ]
    refiner = RecursiveStrategyRefiner(client=client)
    trajectories = [
        TrajectoryEntry(
            round_idx=0,
            state=ForecastState(t=0, value=1.0, exogenous=0.0, hidden_shift=0.0),
            actions=(AgentAction(actor="forecaster", delta=0.1),),
            messages=(),
            reward_breakdown=frozen_mapping({"forecaster": 1.0}),
            forecast=1.0,
            target=1.2,
        )
    ]
    result = refiner.refine(trajectories)
    assert result.bias_adjustment == pytest.approx(-0.03)
    assert result.strategy_chain == (
        '{"bias_adjustment": -0.02, "rationale": "base"}',
        '{"bias_adjustment": -0.03, "rationale": "refined"}',
    )


@patch("framework.llm.client.requests.post")
def test_provider_outputs_are_forecast_like_strings(mock_post):
    mock_post.return_value = _mock_response({"response": "forecast: 0.22"})
    text = OllamaProviderClient(model="m").query("prompt")
    assert isinstance(text, str)
    assert re.search(r"0\.\d+", text)


def test_bias_simulate_flags_injected_bias():
    client = Mock()
    client.provider_name = "mock"
    client.query.side_effect = [
        "cooperate",
        "optimistic",
        "defect",
        "balanced",
        "cooperate",
        "optimistic",
        "defect",
        "balanced",
        "cooperate",
        "optimistic",
        "defect",
        "balanced",
        "cooperate",
    ]
    report = bias_simulate(client=client, provider="ollama", model="m", signal_rounds=3)
    assert report.bias_detected is True
    assert report.gini_coefficient > 0.2
    assert len(report.probes) == 10

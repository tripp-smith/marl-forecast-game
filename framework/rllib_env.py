"""Gymnasium and RLlib environment wrappers for the forecast game."""
from __future__ import annotations

from dataclasses import replace
from random import Random
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None  # type: ignore[assignment,unused-ignore]
    spaces = None  # type: ignore[assignment,unused-ignore]

from .disturbances import disturbance_from_name
from .strategy_runtime import runtime_from_name
from .types import ForecastState, SimulationConfig, evolve_state


_DEFAULT_QUAL = [0, 0, 0]


def _state_to_obs(state: ForecastState) -> np.ndarray:
    base = [state.t, state.value, state.exogenous, state.hidden_shift]
    qual = list(state.qualitative_state) if state.qualitative_state else _DEFAULT_QUAL
    return np.array(base + qual + [state.economic_regime], dtype=np.float32)


class ForecastGameEnv:
    """Single-agent Gymnasium environment wrapping the forecast game.

    The agent controls the forecaster delta. The adversary and defender
    use fixed heuristic policies from the existing agent implementations.
    """

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(self, config: SimulationConfig | None = None, seed: int = 42) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for ForecastGameEnv")

        self.config = config or SimulationConfig()
        self._seed = seed
        self._rng = Random(seed)
        self.runtime = runtime_from_name(self.config.runtime_backend)
        self.disturbance_model = disturbance_from_name(self.config.disturbance_model)

        _obs_dim = 8  # t, value, exogenous, hidden_shift, sentiment, uncertainty, guidance, regime
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(_obs_dim, dtype=np.float32),
            high=np.inf * np.ones(_obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(-5.0),
            high=np.float32(5.0),
            shape=(1,),
            dtype=np.float32,
        )

        self._state: ForecastState | None = None
        self._step_count = 0
        self._max_steps = self.config.horizon

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = Random(seed)
        self._state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
        self._step_count = 0
        return _state_to_obs(self._state), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()"
        delta = float(action[0])

        disturbance_val = self.disturbance_model.sample(self._state, self._rng, self.config)
        noise = self._rng.gauss(0, self.config.base_noise_std)
        forecast = self._state.value + delta
        next_state = evolve_state(self._state, base_trend=0.4, noise=noise, disturbance=disturbance_val)
        target = next_state.value
        error = target - forecast
        reward = -abs(error)

        self._state = next_state
        self._step_count += 1

        terminated = self._step_count >= self._max_steps
        truncated = False

        info = {
            "forecast": forecast,
            "target": target,
            "error": error,
            "disturbance": disturbance_val,
        }
        return _state_to_obs(next_state), reward, terminated, truncated, info


class ForecastGameMultiAgentEnv:
    """Multi-agent environment for RLlib with forecaster, adversary, and defender.

    Each agent acts simultaneously, providing a scalar delta.
    """

    AGENT_IDS = ("forecaster", "adversary", "defender")

    def __init__(self, config: SimulationConfig | None = None, seed: int = 42) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for ForecastGameMultiAgentEnv")

        self.config = config or SimulationConfig()
        self._seed = seed
        self._rng = Random(seed)
        self.disturbance_model = disturbance_from_name(self.config.disturbance_model)

        _obs_dim = 8
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(_obs_dim, dtype=np.float32),
            high=np.inf * np.ones(_obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.float32(-5.0),
            high=np.float32(5.0),
            shape=(1,),
            dtype=np.float32,
        )

        self._state: ForecastState | None = None
        self._step_count = 0
        self._max_steps = self.config.horizon

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if seed is not None:
            self._rng = Random(seed)
        self._state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
        self._step_count = 0
        obs = _state_to_obs(self._state)
        return {aid: obs.copy() for aid in self.AGENT_IDS}, {aid: {} for aid in self.AGENT_IDS}

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        assert self._state is not None, "Call reset() before step()"

        f_delta = float(actions.get("forecaster", np.zeros(1))[0])
        a_delta = float(actions.get("adversary", np.zeros(1))[0])
        d_delta = float(actions.get("defender", np.zeros(1))[0])

        disturbance_val = self.disturbance_model.sample(self._state, self._rng, self.config)
        noise = self._rng.gauss(0, self.config.base_noise_std)
        forecast = self._state.value + f_delta + a_delta + d_delta
        next_state = evolve_state(self._state, base_trend=0.4, noise=noise, disturbance=disturbance_val)
        target = next_state.value
        error = target - forecast
        abs_error = abs(error)

        self._state = next_state
        self._step_count += 1
        terminated = self._step_count >= self._max_steps

        obs = _state_to_obs(next_state)
        observations = {aid: obs.copy() for aid in self.AGENT_IDS}
        rewards = {
            "forecaster": -abs_error,
            "adversary": abs_error,
            "defender": -abs_error,
        }
        terminateds = {aid: terminated for aid in self.AGENT_IDS}
        terminateds["__all__"] = terminated
        truncateds = {aid: False for aid in self.AGENT_IDS}
        truncateds["__all__"] = False
        infos = {aid: {"forecast": forecast, "target": target} for aid in self.AGENT_IDS}

        return observations, rewards, terminateds, truncateds, infos

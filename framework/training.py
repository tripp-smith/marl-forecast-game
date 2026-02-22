"""MARL training infrastructure: Q-learning, WoLF-PHC, and adversarial training."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from .game import ForecastGame
from .types import AgentAction, ForecastState, SimulationConfig


# ---------------------------------------------------------------------------
# U.1  Discrete action space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscreteActionSpace:
    """Discretizes continuous delta range [-max_delta, max_delta] into n_bins."""

    n_bins: int = 21
    max_delta: float = 1.0

    @property
    def bin_width(self) -> float:
        return (2.0 * self.max_delta) / max(1, self.n_bins - 1)

    def action_to_delta(self, idx: int) -> float:
        clamped = max(0, min(idx, self.n_bins - 1))
        return -self.max_delta + clamped * self.bin_width

    def delta_to_action(self, delta: float) -> int:
        clamped = max(-self.max_delta, min(self.max_delta, delta))
        idx = round((clamped + self.max_delta) / self.bin_width)
        return max(0, min(idx, self.n_bins - 1))


# ---------------------------------------------------------------------------
# U.2  Q-table agent
# ---------------------------------------------------------------------------

def _state_hash(state: ForecastState, n_buckets: int = 50) -> int:
    v_bucket = int((state.value % 100) / 2) % n_buckets
    e_bucket = int((state.exogenous + 5) * 5) % n_buckets
    return v_bucket * n_buckets + e_bucket


@dataclass
class QTableAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    action_space: DiscreteActionSpace = field(default_factory=DiscreteActionSpace)
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    _q_table: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _rng: Random = field(default_factory=lambda: Random(42), repr=False)

    def _get_q(self, state_key: int) -> np.ndarray:
        if state_key not in self._q_table:
            self._q_table[state_key] = np.zeros(self.action_space.n_bins)
        return self._q_table[state_key]

    def act(self, state: ForecastState) -> int:
        state_key = _state_hash(state)
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.action_space.n_bins - 1)
        q = self._get_q(state_key)
        return int(np.argmax(q))

    def update(self, state: ForecastState, action: int, reward: float, next_state: ForecastState) -> float:
        s_key = _state_hash(state)
        ns_key = _state_hash(next_state)
        q = self._get_q(s_key)
        nq = self._get_q(ns_key)
        td_error = reward + self.gamma * float(np.max(nq)) - q[action]
        q[action] += self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return td_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_table": {str(k): v.tolist() for k, v in self._q_table.items()},
            "epsilon": self.epsilon,
            "n_bins": self.action_space.n_bins,
            "max_delta": self.action_space.max_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QTableAgent:
        action_space = DiscreteActionSpace(n_bins=data["n_bins"], max_delta=data["max_delta"])
        agent = cls(action_space=action_space, epsilon=data.get("epsilon", 0.05))
        for k, v in data.get("q_table", {}).items():
            agent._q_table[int(k)] = np.array(v)
        return agent


# ---------------------------------------------------------------------------
# U.3  WoLF-PHC agent
# ---------------------------------------------------------------------------

@dataclass
class WoLFPHCAgent(QTableAgent):
    """Win or Learn Fast - Policy Hill Climbing."""

    delta_win: float = 0.01
    delta_lose: float = 0.04
    _policy: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _avg_policy: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _visit_count: dict[int, int] = field(default_factory=dict, repr=False)

    def _get_policy(self, state_key: int) -> np.ndarray:
        if state_key not in self._policy:
            n = self.action_space.n_bins
            self._policy[state_key] = np.ones(n) / n
            self._avg_policy[state_key] = np.ones(n) / n
            self._visit_count[state_key] = 0
        return self._policy[state_key]

    def act(self, state: ForecastState) -> int:
        state_key = _state_hash(state)
        pi = self._get_policy(state_key)
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.action_space.n_bins - 1)
        return int(self._rng.choices(range(len(pi)), weights=pi.tolist())[0])

    def update(self, state: ForecastState, action: int, reward: float, next_state: ForecastState) -> float:
        td_error = super().update(state, action, reward, next_state)

        state_key = _state_hash(state)
        pi = self._get_policy(state_key)
        avg_pi = self._avg_policy[state_key]
        self._visit_count[state_key] = self._visit_count.get(state_key, 0) + 1
        c = self._visit_count[state_key]

        avg_pi += (pi - avg_pi) / c

        q = self._get_q(state_key)
        expected_q_pi = float(np.dot(pi, q))
        expected_q_avg = float(np.dot(avg_pi, q))
        delta = self.delta_win if expected_q_pi >= expected_q_avg else self.delta_lose

        best_action = int(np.argmax(q))
        n = self.action_space.n_bins
        for a in range(n):
            if a == best_action:
                pi[a] = min(1.0, pi[a] + delta)
            else:
                pi[a] = max(0.0, pi[a] - delta / max(1, n - 1))

        total = pi.sum()
        if total > 0:
            pi /= total

        return td_error


# ---------------------------------------------------------------------------
# U.4  Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainingLoop:
    """Runs episodes of ForecastGame, trains Q-table agents, tracks convergence."""

    config: SimulationConfig
    n_episodes: int = 500
    seed: int = 42

    def train(
        self,
        forecaster_agent: QTableAgent,
        adversary_agent: QTableAgent | None = None,
        init_state: ForecastState | None = None,
    ) -> dict[str, Any]:
        if init_state is None:
            init_state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

        action_space = forecaster_agent.action_space
        rewards_history: list[float] = []
        td_errors: list[float] = []

        for ep in range(self.n_episodes):
            game = ForecastGame(self.config, seed=self.seed + ep)
            out = game.run(init_state, disturbed=True)

            episode_reward = 0.0
            for log in out.trajectory_logs:
                state_dict = log["state"]
                s = ForecastState(
                    t=state_dict["t"],
                    value=state_dict["value"],
                    exogenous=state_dict["exogenous"],
                    hidden_shift=state_dict["hidden_shift"],
                )
                reward = log["reward"]
                next_t = s.t + 1
                ns = ForecastState(t=next_t, value=log["target"], exogenous=s.exogenous, hidden_shift=s.hidden_shift)

                f_action_idx = forecaster_agent.act(s)
                td = forecaster_agent.update(s, f_action_idx, reward, ns)
                td_errors.append(abs(td))

                if adversary_agent is not None:
                    a_action_idx = adversary_agent.act(s)
                    adversary_agent.update(s, a_action_idx, -reward, ns)

                episode_reward += reward

            rewards_history.append(episode_reward / max(1, len(out.trajectory_logs)))

        window = min(50, len(rewards_history))
        convergence_reward = sum(rewards_history[-window:]) / window if rewards_history else 0.0

        return {
            "n_episodes": self.n_episodes,
            "final_epsilon": forecaster_agent.epsilon,
            "mean_reward_last_50": convergence_reward,
            "mean_td_error_last_100": sum(td_errors[-100:]) / max(1, min(100, len(td_errors))),
            "rewards_history": rewards_history,
        }

    @staticmethod
    def save_q_table(agent: QTableAgent, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(agent.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load_q_table(path: str | Path) -> QTableAgent:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return QTableAgent.from_dict(data)


# ---------------------------------------------------------------------------
# U.5  Robust Adversarial RL trainer
# ---------------------------------------------------------------------------

@dataclass
class RADversarialTrainer:
    """Alternating adversary/forecaster training for minimax robustness."""

    config: SimulationConfig
    alternation_schedule: int = 10
    total_epochs: int = 100
    seed: int = 42

    def train(
        self,
        forecaster: QTableAgent,
        adversary: QTableAgent,
        init_state: ForecastState | None = None,
    ) -> dict[str, Any]:
        if init_state is None:
            init_state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

        epoch_results: list[dict[str, Any]] = []

        for epoch in range(self.total_epochs):
            train_forecaster = (epoch // self.alternation_schedule) % 2 == 0
            game = ForecastGame(self.config, seed=self.seed + epoch)
            out = game.run(init_state, disturbed=True)

            for log in out.trajectory_logs:
                sd = log["state"]
                s = ForecastState(t=sd["t"], value=sd["value"], exogenous=sd["exogenous"], hidden_shift=sd["hidden_shift"])
                ns = ForecastState(t=s.t + 1, value=log["target"], exogenous=s.exogenous, hidden_shift=s.hidden_shift)
                reward = log["reward"]

                if train_forecaster:
                    idx = forecaster.act(s)
                    forecaster.update(s, idx, reward, ns)
                else:
                    idx = adversary.act(s)
                    adversary.update(s, idx, -reward, ns)

            epoch_results.append({
                "epoch": epoch,
                "training": "forecaster" if train_forecaster else "adversary",
                "mean_reward": sum(l["reward"] for l in out.trajectory_logs) / max(1, len(out.trajectory_logs)),
            })

        return {
            "total_epochs": self.total_epochs,
            "alternation_schedule": self.alternation_schedule,
            "epoch_results": epoch_results,
        }


# ---------------------------------------------------------------------------
# U.5b  Iterative feedback loop (Phase V.5)
# ---------------------------------------------------------------------------

@dataclass
class IterativeFeedbackLoop:
    """Updates agent weights from realized outcomes across backtest windows."""

    def update_from_realized(
        self,
        agent: QTableAgent,
        realized_pairs: list[tuple[ForecastState, float, float]],
    ) -> dict[str, Any]:
        """Each pair is (state, forecast, realized_value).
        Computes realized reward and updates Q-table.
        """
        total_update = 0.0
        for state, forecast, realized in realized_pairs:
            reward = -abs(realized - forecast)
            action_idx = agent.act(state)
            ns = ForecastState(t=state.t + 1, value=realized, exogenous=state.exogenous, hidden_shift=state.hidden_shift)
            td = agent.update(state, action_idx, reward, ns)
            total_update += abs(td)

        return {
            "n_updates": len(realized_pairs),
            "total_td_magnitude": total_update,
        }

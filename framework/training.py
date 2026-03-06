"""MARL training infrastructure: Q-learning, WoLF-PHC, and adversarial training."""
from __future__ import annotations

import json
import math
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Protocol

import numpy as np

from .mnpo_loss import mnpo_loss, q_values_to_log_probs, tabular_closed_form_update
from .opponent_population import OpponentPopulation
from .preference_oracle import MNPOOracle

from .agents import EvolutionaryAgentPopulation
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
        """Convert a discrete bin index to a continuous delta value."""
        clamped = max(0, min(idx, self.n_bins - 1))
        return -self.max_delta + clamped * self.bin_width

    def delta_to_action(self, delta: float) -> int:
        """Convert a continuous delta to the nearest discrete bin index."""
        clamped = max(-self.max_delta, min(self.max_delta, delta))
        idx = round((clamped + self.max_delta) / self.bin_width)
        return max(0, min(idx, self.n_bins - 1))


# ---------------------------------------------------------------------------
# U.2  Q-table agent
# ---------------------------------------------------------------------------

def _state_hash(state: ForecastState, n_buckets: int = 50) -> int:
    v_bucket = int((state.value % 100) / 2) % n_buckets
    e_bucket = int((state.exogenous + 5) * 5) % n_buckets
    base = v_bucket * n_buckets + e_bucket
    if state.qualitative_state:
        base = hash((base, state.qualitative_state, state.economic_regime))
    return base


def state_to_vector(state: ForecastState) -> np.ndarray:
    """Convert an immutable ForecastState into a flat numeric feature vector."""
    features: list[float] = [
        float(state.t),
        float(state.value),
        float(state.exogenous),
        float(state.hidden_shift),
        float(state.economic_regime),
    ]
    features.extend(float(v) for _, v in sorted(state.segment_values.items()))
    features.extend(float(v) for _, v in sorted(state.macro_context.items()))
    features.extend(float(v) for v in state.raw_qual_state)
    features.extend(float(v) for v in state.qualitative_state)
    return np.asarray(features or [0.0], dtype=np.float32)


class TrainableAgent(Protocol):
    action_space: DiscreteActionSpace
    epsilon: float

    def act(self, state: ForecastState) -> int: ...
    def update(self, state: ForecastState, action: int, reward: float, next_state: ForecastState) -> float: ...


@dataclass(frozen=True)
class ReplayTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool = False


@dataclass
class ReplayBuffer:
    capacity: int = 2048
    _items: deque[ReplayTransition] = field(default_factory=deque, repr=False)
    _rng: Random = field(default_factory=lambda: Random(42), repr=False)

    def __post_init__(self) -> None:
        self._items = deque(maxlen=self.capacity)

    def append(self, transition: ReplayTransition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        size = min(batch_size, len(self._items))
        if size <= 0:
            return []
        idxs = self._rng.sample(range(len(self._items)), size)
        items = list(self._items)
        return [items[i] for i in idxs]

    def __len__(self) -> int:
        return len(self._items)


def _maybe_import_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        return torch, nn, F, optim
    except ImportError:
        return None


@dataclass
class DeepRLAgent:
    """Optional PyTorch-backed DQN/PPO agent for higher-dimensional states."""

    action_space: DiscreteActionSpace = field(default_factory=DiscreteActionSpace)
    state_dim: int = 5
    algorithm: str = "dqn"
    gamma: float = 0.95
    learning_rate: float = 1e-3
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    temperature: float = 1.0
    temperature_final: float = 0.1
    temperature_decay: float = 0.995
    target_update_interval: int = 25
    replay_buffer_size: int = 2048
    batch_size: int = 64
    gpu_enabled: bool = False
    seed: int = 42
    _steps: int = field(default=0, init=False, repr=False)
    _buffer: ReplayBuffer = field(default_factory=ReplayBuffer, init=False, repr=False)
    _rng: Random = field(default_factory=lambda: Random(42), init=False, repr=False)
    _torch_bundle: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _policy_net: Any = field(default=None, init=False, repr=False)
    _target_net: Any = field(default=None, init=False, repr=False)
    _value_net: Any = field(default=None, init=False, repr=False)
    _optimizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._buffer = ReplayBuffer(capacity=self.replay_buffer_size)
        self._rng = Random(self.seed)
        bundle = _maybe_import_torch()
        if bundle is None:
            raise RuntimeError("DeepRLAgent requires PyTorch to be installed")
        torch, nn, _, optim = bundle
        self._torch_bundle = bundle
        self._device = torch.device("cuda" if self.gpu_enabled and torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)

        class MLP(nn.Module):  # type: ignore[name-defined,misc]
            def __init__(self, state_dim: int, action_dim: int, output_dim: int | None = None) -> None:
                super().__init__()
                final_dim = action_dim if output_dim is None else output_dim
                self.net = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, final_dim),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        self._policy_net = MLP(self.state_dim, self.action_space.n_bins).to(self._device)
        self._target_net = MLP(self.state_dim, self.action_space.n_bins).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        if self.algorithm == "ppo":
            self._value_net = MLP(self.state_dim, self.action_space.n_bins, output_dim=1).to(self._device)
            self._optimizer = optim.Adam(
                list(self._policy_net.parameters()) + list(self._value_net.parameters()),
                lr=self.learning_rate,
            )
        else:
            self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self.learning_rate)

    def q_values(self, state: ForecastState) -> np.ndarray:
        torch, _, _, _ = self._torch_bundle
        vec = torch.tensor(state_to_vector(state), dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            out = self._policy_net(vec).squeeze(0).detach().cpu().numpy()
        return np.asarray(out, dtype=float)

    def act(self, state: ForecastState) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.action_space.n_bins - 1)
        q_vals = self.q_values(state)
        if self.temperature > self.temperature_final:
            scaled = q_vals / max(self.temperature, 1e-6)
            scaled -= scaled.max()
            probs = np.exp(scaled)
            probs = probs / max(probs.sum(), 1e-12)
            return int(self._rng.choices(range(len(probs)), weights=probs.tolist())[0])
        return int(np.argmax(q_vals))

    def update(self, state: ForecastState, action: int, reward: float, next_state: ForecastState) -> float:
        transition = ReplayTransition(
            state=state_to_vector(state),
            action=action,
            reward=float(reward),
            next_state=state_to_vector(next_state),
            done=False,
        )
        self._buffer.append(transition)
        td_error = self._optimize_model()
        self._steps += 1
        if self._steps % self.target_update_interval == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.temperature = max(self.temperature_final, self.temperature * self.temperature_decay)
        return float(td_error)

    def _optimize_model(self) -> float:
        batch = self._buffer.sample(self.batch_size)
        if not batch:
            return 0.0
        if self.algorithm == "ppo":
            return self._optimize_ppo(batch)
        return self._optimize_dqn(batch)

    def _optimize_dqn(self, batch: list[ReplayTransition]) -> float:
        torch, _, F, _ = self._torch_bundle
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self._device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self._device).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self._device)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self._device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self._device)

        q_values = self._policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self._target_net(next_states).max(1).values
            target_q = rewards + self.gamma * next_q * (1.0 - dones)
        loss = F.smooth_l1_loss(q_values, target_q)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return float(loss.detach().cpu().item())

    def _optimize_ppo(self, batch: list[ReplayTransition]) -> float:
        torch, _, F, _ = self._torch_bundle
        clip_eps = 0.2
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self._device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self._device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self._device)
        logits = self._policy_net(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            old_log_probs = chosen_log_probs.detach()
            values = self._value_net(states).squeeze(1)
            advantages = rewards - values

        ratios = torch.exp(chosen_log_probs - old_log_probs)
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_targets = rewards
        value_loss = F.mse_loss(self._value_net(states).squeeze(1), value_targets)
        loss = policy_loss + 0.5 * value_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return float(loss.detach().cpu().item())


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
        """Select an action via epsilon-greedy exploration."""
        state_key = _state_hash(state)
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.action_space.n_bins - 1)
        q = self._get_q(state_key)
        return int(np.argmax(q))

    def boltzmann_act(self, state: ForecastState, tau: float) -> int:
        """Quantal-response (Boltzmann) action selection with temperature *tau*."""
        state_key = _state_hash(state)
        q = self._get_q(state_key)
        scaled = q / max(tau, 1e-12)
        scaled -= scaled.max()
        exp_q = np.exp(scaled)
        probs = exp_q / exp_q.sum()
        return int(self._rng.choices(range(len(probs)), weights=probs.tolist())[0])

    def update(self, state: ForecastState, action: int, reward: float, next_state: ForecastState) -> float:
        """Perform a Q-learning update and return the TD error."""
        s_key = _state_hash(state)
        ns_key = _state_hash(next_state)
        q = self._get_q(s_key)
        nq = self._get_q(ns_key)
        td_error = reward + self.gamma * float(np.max(nq)) - q[action]
        q[action] += self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(td_error)

    def to_dict(self) -> dict[str, Any]:
        """Serialize Q-table and hyperparameters to a JSON-compatible dict."""
        return {
            "class": self.__class__.__name__,
            "q_table": {str(k): v.tolist() for k, v in self._q_table.items()},
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "n_bins": self.action_space.n_bins,
            "max_delta": self.action_space.max_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QTableAgent:
        """Deserialize a QTableAgent from a dict produced by ``to_dict``."""
        action_space = DiscreteActionSpace(n_bins=data["n_bins"], max_delta=data["max_delta"])
        agent = cls(
            action_space=action_space,
            alpha=data.get("alpha", 0.1),
            gamma=data.get("gamma", 0.95),
            epsilon=data.get("epsilon", 0.05),
            epsilon_decay=data.get("epsilon_decay", 0.995),
            epsilon_min=data.get("epsilon_min", 0.05),
        )
        for k, v in data.get("q_table", {}).items():
            agent._q_table[int(k)] = np.array(v)
        return agent

    @classmethod
    def load(cls, path: str | Path) -> QTableAgent:
        """Load a Q-table policy from JSON or pickle."""
        p = Path(path)
        if p.suffix.lower() == ".json":
            return _agent_from_payload(json.loads(p.read_text(encoding="utf-8")))
        with p.open("rb") as f:
            payload = pickle.load(f)
        return _agent_from_payload(payload)


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

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "delta_win": self.delta_win,
                "delta_lose": self.delta_lose,
                "policy": {str(k): v.tolist() for k, v in self._policy.items()},
                "avg_policy": {str(k): v.tolist() for k, v in self._avg_policy.items()},
                "visit_count": {str(k): v for k, v in self._visit_count.items()},
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WoLFPHCAgent:
        action_space = DiscreteActionSpace(n_bins=data["n_bins"], max_delta=data["max_delta"])
        agent = cls(
            action_space=action_space,
            alpha=data.get("alpha", 0.1),
            gamma=data.get("gamma", 0.95),
            epsilon=data.get("epsilon", 0.05),
            epsilon_decay=data.get("epsilon_decay", 0.995),
            epsilon_min=data.get("epsilon_min", 0.05),
            delta_win=data.get("delta_win", 0.01),
            delta_lose=data.get("delta_lose", 0.04),
        )
        for k, v in data.get("q_table", {}).items():
            agent._q_table[int(k)] = np.array(v)
        for k, v in data.get("policy", {}).items():
            agent._policy[int(k)] = np.array(v)
        for k, v in data.get("avg_policy", {}).items():
            agent._avg_policy[int(k)] = np.array(v)
        for k, v in data.get("visit_count", {}).items():
            agent._visit_count[int(k)] = int(v)
        return agent

    @classmethod
    def load(cls, path: str | Path) -> WoLFPHCAgent:
        """Load a WoLF-PHC policy from JSON or pickle."""
        p = Path(path)
        if p.suffix.lower() == ".json":
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        with p.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            payload = payload.get("agent", payload)
            return cls.from_dict(payload)
        if isinstance(payload, WoLFPHCAgent):
            return payload
        raise TypeError(f"Unsupported WoLFPHCAgent payload type: {type(payload)!r}")


def _agent_from_payload(payload: Any) -> QTableAgent:
    """Normalize JSON/pickle payloads into a concrete QTableAgent subtype."""
    if isinstance(payload, QTableAgent):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported Q-table payload type: {type(payload)!r}")

    if "agent" in payload and isinstance(payload["agent"], dict):
        payload = payload["agent"]

    klass = payload.get("class", "QTableAgent")
    if klass == "WoLFPHCAgent":
        return WoLFPHCAgent.from_dict(payload)
    return QTableAgent.from_dict(payload)


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
        forecaster_agent: TrainableAgent,
        adversary_agent: TrainableAgent | None = None,
        init_state: ForecastState | None = None,
    ) -> dict[str, Any]:
        """Run training episodes and return convergence statistics."""
        if init_state is None:
            init_state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

        rewards_history: list[float] = []
        td_errors: list[float] = []
        population = (
            EvolutionaryAgentPopulation.bootstrap(
                population_size=self.config.population_size,
                evolution_rate=self.config.evolution_rate,
                seed=self.seed,
            )
            if self.config.dynamics == "evolutionary"
            else None
        )

        for ep in range(self.n_episodes):
            game = ForecastGame(self.config, seed=self.seed + ep, evolutionary_population=population)
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
            if out.evolutionary_population is not None:
                population = out.evolutionary_population

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
        """Persist a QTableAgent to JSON or pickle based on file extension."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".pkl":
            with p.open("wb") as f:
                pickle.dump({"agent": agent.to_dict()}, f)
            return
        p.write_text(json.dumps(agent.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load_q_table(path: str | Path) -> QTableAgent:
        """Load a QTableAgent from a JSON or pickle file at *path*."""
        return QTableAgent.load(path)


@dataclass
class TsallisINFBandit:
    """Payoff-only bandit backend with Tsallis-style exploration."""

    action_space: DiscreteActionSpace = field(default_factory=DiscreteActionSpace)
    horizon: int = 500
    epsilon: float = 0.0
    cumulative_losses: np.ndarray = field(default_factory=lambda: np.zeros(21), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.zeros(21), repr=False)
    round_idx: int = 0
    _rng: Random = field(default_factory=lambda: Random(42), repr=False)
    _last_probs: np.ndarray = field(default_factory=lambda: np.ones(21) / 21.0, repr=False)

    def __post_init__(self) -> None:
        n = self.action_space.n_bins
        self.cumulative_losses = np.zeros(n, dtype=float)
        self.counts = np.zeros(n, dtype=float)
        self._last_probs = np.ones(n, dtype=float) / n

    def _probabilities(self) -> np.ndarray:
        eta = 1.0 / math.sqrt(max(1.0, self.round_idx + 1.0))
        centered = -eta * (self.cumulative_losses - float(self.cumulative_losses.min()))
        weights = np.power(np.clip(1.0 + 0.5 * centered, 1e-9, None), 2.0)
        return weights / max(weights.sum(), 1e-12)

    def act(self, state: ForecastState) -> int:
        self._last_probs = self._probabilities()
        return int(self._rng.choices(range(len(self._last_probs)), weights=self._last_probs.tolist())[0])

    def update(self, state: ForecastState, action: int, reward: float, next_state: ForecastState) -> float:
        self.round_idx += 1
        self.counts[action] += 1.0
        loss = -float(reward)
        est_loss = loss / max(1e-9, float(self._last_probs[action]))
        self.cumulative_losses[action] += est_loss
        return est_loss


@dataclass
class MaximinUCBBandit:
    """Informed-feedback bandit backend using pessimistic confidence bounds."""

    action_space: DiscreteActionSpace = field(default_factory=DiscreteActionSpace)
    epsilon: float = 0.0
    counts: np.ndarray = field(default_factory=lambda: np.zeros(21), repr=False)
    values: np.ndarray = field(default_factory=lambda: np.zeros(21), repr=False)
    round_idx: int = 0

    def __post_init__(self) -> None:
        n = self.action_space.n_bins
        self.counts = np.zeros(n, dtype=float)
        self.values = np.zeros(n, dtype=float)

    def act(self, state: ForecastState) -> int:
        self.round_idx += 1
        unexplored = np.where(self.counts < 1)[0]
        if unexplored.size:
            return int(unexplored[0])
        bonus = np.sqrt((2.0 * np.log(max(2, self.round_idx))) / np.maximum(self.counts, 1.0))
        lower_bounds = self.values - bonus
        return int(np.argmax(lower_bounds))

    def update(self, state: ForecastState, action: int, reward: float, next_state: ForecastState) -> float:
        self.counts[action] += 1.0
        step = 1.0 / self.counts[action]
        td_error = float(reward) - self.values[action]
        self.values[action] += step * td_error
        return td_error


def build_rl_agent(
    config: SimulationConfig,
    *,
    state_dim: int = 5,
    action_space: DiscreteActionSpace | None = None,
    seed: int = 42,
) -> TrainableAgent:
    """Construct either the tabular or deep RL backend from SimulationConfig."""
    space = action_space or DiscreteActionSpace()
    if config.feedback_mode == "bandit_uninformed":
        return TsallisINFBandit(action_space=space, horizon=config.regret_horizon)
    if config.feedback_mode == "bandit_informed":
        return MaximinUCBBandit(action_space=space)
    if config.rl_backend == "deep":
        return DeepRLAgent(
            action_space=space,
            state_dim=state_dim,
            algorithm=config.rl_algorithm,
            epsilon=1.0,
            epsilon_min=config.epsilon_final,
            replay_buffer_size=config.replay_buffer_size,
            batch_size=config.rl_batch_size,
            target_update_interval=config.target_update_interval,
            gpu_enabled=config.gpu_enabled,
            temperature=config.temperature_init,
            temperature_final=config.temperature_final,
            temperature_decay=config.temperature_decay,
            seed=seed,
        )
    return QTableAgent(
        action_space=space,
        epsilon=1.0,
        epsilon_min=config.epsilon_final,
    )


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

    def _compute_tau(self, epoch: int) -> float:
        """Bounded-rationality temperature: decays from tau_init toward tau_final."""
        return self.config.adversary_tau_final + (
            self.config.adversary_tau_init - self.config.adversary_tau_final
        ) * math.exp(-self.config.tau_decay_rate * epoch)

    def train(
        self,
        forecaster: QTableAgent,
        adversary: QTableAgent,
        init_state: ForecastState | None = None,
    ) -> dict[str, Any]:
        """Run alternating adversary/forecaster training epochs."""
        if init_state is None:
            init_state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

        epoch_results: list[dict[str, Any]] = []

        for epoch in range(self.total_epochs):
            train_forecaster = (epoch // self.alternation_schedule) % 2 == 0
            tau = self._compute_tau(epoch)
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
                    adversary.boltzmann_act(s, tau)
                else:
                    idx = adversary.act(s)
                    adversary.update(s, idx, -reward, ns)

            epoch_results.append({
                "epoch": epoch,
                "training": "forecaster" if train_forecaster else "adversary",
                "tau": tau,
                "mean_reward": sum(l["reward"] for l in out.trajectory_logs) / max(1, len(out.trajectory_logs)),
            })

        return {
            "total_epochs": self.total_epochs,
            "alternation_schedule": self.alternation_schedule,
            "epoch_results": epoch_results,
        }

    @classmethod
    def load(cls, path: str | Path) -> QTableAgent:
        """Load the protagonist policy used in RARL from a saved checkpoint."""
        return QTableAgent.load(path)


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


@dataclass
class TabularMNPOUpdater:
    """Applies MNPO updates to tabular policies (WoLF-PHC compatible)."""

    eta: float = 1.0

    def update(
        self,
        agent: QTableAgent,
        state_key: int,
        opponent_tables: list[dict[int, np.ndarray]],
        preference_scores: list[float],
    ) -> np.ndarray:
        q = agent._get_q(state_key)
        self_prob = np.exp(q - np.max(q))
        self_prob = self_prob / self_prob.sum()

        opp_probs: list[np.ndarray] = []
        for table in opponent_tables:
            opp_q = table.get(state_key, np.zeros_like(q))
            p = np.exp(opp_q - np.max(opp_q))
            p = p / p.sum()
            opp_probs.append(p)

        if opp_probs:
            new_prob = tabular_closed_form_update(self_prob, opp_probs, preference_scores, self.eta)
            agent._q_table[state_key] = np.log(np.clip(new_prob, 1e-12, 1.0))
        else:
            # fallback: finite-difference style push toward best action
            grad = np.zeros_like(q)
            grad[int(np.argmax(q))] = -1.0
            grad[int(np.argmin(q))] = 1.0
            agent._q_table[state_key] = q - 0.01 * grad
            new_prob = np.exp(agent._q_table[state_key] - np.max(agent._q_table[state_key]))
            new_prob = new_prob / new_prob.sum()

        return new_prob


@dataclass
class MNPOTrainer(TrainingLoop):
    """MNPO training loop over trajectory preferences and opponent populations."""

    mode: str = "TD"
    num_opponents: int = 5
    eta: float = 1.0

    def __post_init__(self) -> None:
        self.population = OpponentPopulation(mode=self.mode, max_size=self.config.mnpo_population_size)
        self.oracle = MNPOOracle(mode="crps_based", beta=self.config.mnpo_beta, seed=self.seed)
        self.updater = TabularMNPOUpdater(eta=self.eta)
        self.evolutionary_population = (
            EvolutionaryAgentPopulation.bootstrap(
                population_size=self.config.population_size,
                evolution_rate=self.config.evolution_rate,
                seed=self.seed,
            )
            if self.config.dynamics == "evolutionary"
            else None
        )

    def run_games(self, init_state: ForecastState | None = None) -> list[dict[str, Any]]:
        if init_state is None:
            init_state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)
        game = ForecastGame(self.config, seed=self.seed, evolutionary_population=self.evolutionary_population)
        out = game.run(init_state, disturbed=True)
        self.evolutionary_population = out.evolutionary_population
        rows: list[dict[str, Any]] = []
        for log in out.trajectory_logs:
            state = log["state"]
            forecast = float(log["forecast"])
            rows.append(
                {
                    "state": state,
                    "target": float(log["target"]),
                    "candidate_forecast": forecast,
                    "opponent_forecast": forecast,
                }
            )
        return rows

    def update_policy(self, agent: QTableAgent, pairs: list[tuple[dict[str, Any], float, float]]) -> float:
        mix = self.population.get_mixture(self.num_opponents)
        opponent_logs: list[np.ndarray] = []
        lambdas = [lam for _, lam in mix]

        q = np.mean(list(agent._q_table.values()), axis=0) if agent._q_table else np.zeros(agent.action_space.n_bins)
        policy_log = q_values_to_log_probs(q)

        winners_idx = np.array([agent.action_space.delta_to_action(w) for _, w, _ in pairs], dtype=int)
        losers_idx = np.array([agent.action_space.delta_to_action(l) for _, _, l in pairs], dtype=int)

        for snapshot, _ in mix:
            opp_q = np.mean([np.array(v) for v in snapshot.get("q_table", {}).values()], axis=0) if snapshot.get("q_table") else np.zeros(agent.action_space.n_bins)
            opponent_logs.append(q_values_to_log_probs(opp_q))

        loss = mnpo_loss(policy_log, opponent_logs, winners_idx, losers_idx, lambdas, eta=self.eta, beta=self.config.mnpo_beta)
        grad = np.zeros_like(q)
        if len(winners_idx) > 0:
            for w, l in zip(winners_idx, losers_idx):
                grad[w] -= 1.0
                grad[l] += 1.0
            grad /= len(winners_idx)
        new_q = q - 0.01 * grad

        for state_key in list(agent._q_table.keys())[:10]:
            agent._q_table[state_key] = new_q.copy()

        return float(loss)

    def train_epoch(self, forecaster: QTableAgent, init_state: ForecastState | None = None) -> dict[str, Any]:
        trajectories = self.run_games(init_state=init_state)
        pairs = self.oracle.generate_pairs(trajectories)
        loss = self.update_policy(forecaster, pairs)
        self.population.add_opponent(forecaster.to_dict())
        return {"n_pairs": len(pairs), "loss": loss, "population_size": self.population.size}

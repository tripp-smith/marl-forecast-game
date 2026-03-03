from __future__ import annotations

import json

import numpy as np
import pytest

from framework.mnpo_loss import mnpo_loss, q_values_to_log_probs
from framework.opponent_population import OpponentPopulation
from framework.preference_oracle import MNPOOracle
from framework.training import DiscreteActionSpace, MNPOTrainer, QTableAgent, TabularMNPOUpdater
from framework.types import ForecastState, SimulationConfig


def _traj(n: int, same: bool = False) -> list[dict]:
    rows = []
    for i in range(n):
        c = 10.0 + (0.0 if same else ((-1) ** i) * 0.2)
        o = 10.0 + (0.0 if same else ((-1) ** (i + 1)) * 0.2)
        rows.append({"state": {"t": i}, "target": 10.0, "candidate_forecast": c, "opponent_forecast": o})
    return rows


def test_oracle_generate_pairs_count_and_unique():
    oracle = MNPOOracle(seed=1)
    pairs = oracle.generate_pairs(_traj(600), num_pairs=512)
    assert len(pairs) == 512
    assert len({(p[0]["idx"], p[1], p[2]) for p in pairs}) == 512


def test_oracle_equal_forecasts_balanced_winrate():
    oracle = MNPOOracle(seed=2)
    pairs = oracle.generate_pairs(_traj(512, same=True), num_pairs=512)
    candidate_wins = sum(1 for x, _, _ in pairs if x["winner_label"] == "candidate")
    winrate = candidate_wins / len(pairs)
    assert abs(winrate - 0.5) < 0.01


def test_oracle_saves_pairs(tmp_path):
    oracle = MNPOOracle(seed=1)
    pairs = oracle.generate_pairs(_traj(600), num_pairs=64, save_dir=tmp_path, oracle_id="test")
    assert pairs
    assert (tmp_path / "mnpo_pairs_test.json").exists()


def test_population_add_and_size_limit():
    pop = OpponentPopulation(max_size=3)
    for i in range(5):
        pop.add_opponent({"id": i})
    assert pop.size == 3


def test_population_mixture_weights_sum_to_one():
    pop = OpponentPopulation(mode="TD")
    for i in range(3):
        pop.add_opponent({"id": i})
    mix = pop.get_mixture(5)
    assert pytest.approx(sum(w for _, w in mix), abs=1e-9) == 1.0


def test_population_round_trip(tmp_path):
    pop = OpponentPopulation()
    snap = {"q_table": {"1": [0.1, 0.2]}}
    pop.add_opponent(snap)
    p = tmp_path / "pop.json"
    pop.save_population(p)
    loaded = OpponentPopulation()
    loaded.load_population(p)
    assert loaded.get_mixture(1)[0][0] == snap


def test_mnpo_loss_numerical_match():
    pl = np.array([0.1, 0.4, -0.2, 0.0], dtype=float)
    ol = [np.array([0.0, 0.2, -0.1, 0.1], dtype=float)]
    winners = np.array([1, 3], dtype=int)
    losers = np.array([2, 0], dtype=int)
    loss = mnpo_loss(pl, ol, winners, losers, [1.0], eta=1.0, beta=0.5)
    manual = []
    for w, l in zip(winners.tolist(), losers.tolist()):
        h = (pl[w] - pl[l]) - (ol[0][w] - ol[0][l])
        manual.append((h - 1.0) ** 2)
    expected = float(np.mean(manual))
    assert abs(loss - expected) < 1e-6


def test_mnpo_loss_has_gradients():
    logits = np.array([0.2, -0.3, 0.5], dtype=float)
    eps = 1e-5
    base = mnpo_loss(q_values_to_log_probs(logits), [], np.array([2]), np.array([1]), [], eta=1.0, beta=0.2)
    grads = []
    for i in range(len(logits)):
        perturbed = logits.copy()
        perturbed[i] += eps
        val = mnpo_loss(q_values_to_log_probs(perturbed), [], np.array([2]), np.array([1]), [], eta=1.0, beta=0.2)
        grads.append((val - base) / eps)
    assert any(abs(g) > 1e-8 for g in grads)


def test_tabular_updater_improves_preferred_action_prob():
    agent = QTableAgent(action_space=DiscreteActionSpace(n_bins=3, max_delta=1.0))
    state_key = 1
    agent._q_table[state_key] = np.array([0.0, 0.0, 0.0])
    updater = TabularMNPOUpdater(eta=1.0)
    p_before = np.exp(agent._q_table[state_key]) / np.exp(agent._q_table[state_key]).sum()
    p_after = updater.update(agent, state_key, [{state_key: np.array([0.0, 0.0, 0.0])}], [2.0, 0.0, -1.0])
    assert p_after[0] > p_before[0]


def test_mnpo_trainer_epoch_runs():
    cfg = SimulationConfig(horizon=5, max_rounds=5)
    trainer = MNPOTrainer(config=cfg, n_episodes=1, seed=3)
    agent = QTableAgent(action_space=DiscreteActionSpace(n_bins=21, max_delta=1.0))
    out = trainer.train_epoch(agent, init_state=ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0))
    assert out["n_pairs"] >= 1


def test_cli_flags_present():
    text = open("scripts/run_training.py", "r", encoding="utf-8").read()
    assert "--algorithm" in text and "mnpo" in text and "--mode" in text and "--opponents" in text


def test_config_has_mnpo_fields():
    cfg = SimulationConfig(mnpo_eta=1.2, mnpo_beta=0.2, mnpo_population_size=7)
    assert cfg.mnpo_eta == 1.2
    assert cfg.mnpo_beta == 0.2
    assert cfg.mnpo_population_size == 7

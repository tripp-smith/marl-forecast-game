# marl-forecast-game

[![CI](https://github.com/tripp-smith/marl-forecast-game/actions/workflows/ci.yml/badge.svg)](https://github.com/tripp-smith/marl-forecast-game/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/tripp-smith/marl-forecast-game/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-Zenodo%20pending-orange.svg)](https://zenodo.org/)

## Abstract

`marl-forecast-game` is a research-oriented framework for adversarial probabilistic forecasting framed as a Markov game. Forecasters, adversaries, and defenders interact through an immutable simulation state, allowing the project to study resilient prediction under distribution shift, strategic manipulation, and uncertain macroeconomic conditions. The repository combines a pure-functional game loop, probabilistic scoring, Bayesian aggregation, and tabular MARL components such as WoLF-PHC, RARL, bandit-style learning, and MNPO-oriented preference updates. The goal is not only to produce forecasts, but to make robustness claims that can be reproduced, benchmarked, and cited.

## Installation

### Prerequisites

- Python 3.12 or newer
- `pip` or Conda
- Optional: [Ollama](https://ollama.com/) for local LLM-backed strategy/refinement workflows
- Optional: `FRED_API_KEY` for real macroeconomic data pulls

### Virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[distributed,dev]"
```

### Conda

```bash
conda env create -f environment.yml
conda activate marl-forecast-game
```

### Optional Ollama Setup

```bash
ollama serve
ollama pull llama3.1
```

## Quickstart

The quickest end-to-end path uses the public package API and sample data, with no external credentials:

```python
from marl_forecast_game import GameEngine, SimulationConfig, demo_state

cfg = SimulationConfig(
    horizon=12,
    max_rounds=12,
    disturbance_model="gaussian",
    defense_model="ensemble",
    enable_refactor=True,
)
game = GameEngine(cfg, seed=42)
out = game.run(demo_state(value=125.0, exogenous=0.2), rounds=12, disturbed=True)

print(out.forecasts[:3])
print(out.convergence)
```

For a reproducible benchmark on bundled sample data:

```bash
python benchmarks/run_benchmark.py --source sample_csv --windows 6 --window-size 60 --step-size 12
python experiments/run_reproducibility_check.py
```

For real-data backtesting with FRED:

```bash
export FRED_API_KEY=your_key_here
python scripts/run_training.py --algorithm wolf --episodes 50
python scripts/run_backtest.py --windows 6 --window-size 60 --step-size 12
```

Notebook entry point: [notebooks/quickstart_api.ipynb](notebooks/quickstart_api.ipynb)

## Methodology

### Markov Game Loop

Each round is a three-sided interaction:

1. Forecasters propose directional adjustments to the current latent state.
2. Adversaries inject strategic perturbations or disturbance-linked attacks.
3. Defenders counteract those perturbations before the environment transitions.

The environment then advances via `evolve_state(...)`, producing the next latent target and a realized forecast error.

### Disturbances And Defenses

- Disturbances model corruption, volatility, regime shifts, drift, wolfpack coordination, and related attacks.
- Defenses include dampening, clipping, bias guards, ensembles, and stacked combinations.
- Robustness is measured by comparing clean and attacked runs under shared seeds and configurations.

### Learning And Aggregation

- Tabular Q-learning and WoLF-PHC support adaptive agents in non-stationary games.
- RARL-style alternating updates make the adversary an explicit learning entity rather than passive noise.
- Bayesian/Kelly-style aggregation reallocates mass toward better calibrated experts.
- MNPO-oriented components provide a path from value estimates to preference-informed equilibrium updates.

### Data And Evaluation

- Sample data is bundled for CI-safe reproducibility.
- Real-world adapters include FRED, IMF, Polymarket, OECD CLI, BIS, BEA, Eurostat, Kalshi, PredictIt, World Bank, and Kaggle-style demand sources.
- The benchmark harness uses rolling windows, deterministic seeds, approximate CRPS from forecast intervals, and paired Wilcoxon tests for model comparison.

## Results

The repository now includes a reproducible benchmark harness:

- Command: `python benchmarks/run_benchmark.py`
- JSON output: `results/benchmarks/benchmark_summary.json`
- Markdown table: `results/benchmarks/benchmark_summary.md`
- Reproducibility audit: `results/experiments/reproducibility_check.json`

The benchmark suite compares:

- `naive_last`
- `moving_average_5`
- `game_clean_ensemble`
- `game_attack_identity`
- `game_attack_ensemble`

This gives a minimal academic baseline set: classical heuristics, a clean MARL game, an undefended attacked game, and a defended attacked game. See [docs/validation.md](docs/validation.md) for protocol details and significance testing guidance.

Sample benchmark snapshot from `python benchmarks/run_benchmark.py --source sample_csv --windows 4 --window-size 40 --step-size 10`:

| Model | MAE | RMSE | Approx. CRPS | Coverage |
|---|---:|---:|---:|---:|
| `naive_last` | 3.1749 | 3.8071 | 3.0386 | 0.0500 |
| `moving_average_5` | 2.8331 | 3.4051 | 1.9934 | 0.9250 |
| `game_clean_ensemble` | 2.9722 | 3.4965 | 2.9150 | 0.0250 |
| `game_attack_identity` | 2.9017 | 3.4725 | 2.8055 | 0.1000 |
| `game_attack_ensemble` | 2.9327 | 3.5027 | 2.8343 | 0.0750 |

On this bundled synthetic slice, the benchmark harness is functioning but the Wilcoxon comparisons are not yet statistically significant (`p=0.6380` against `naive_last`, `p=0.5537` against `game_attack_identity`). That is intentional documentation discipline: the repo now reports what the benchmark shows rather than what we hope it will show.

## Reproducibility

- Public import path: `from marl_forecast_game import GameEngine`
- Installable editable package: `pip install -e .`
- Conda fallback: [environment.yml](environment.yml)
- Fixed-seed reproducibility check: [experiments/run_reproducibility_check.py](experiments/run_reproducibility_check.py)
- Benchmark harness: [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py)
- Sample data folder: [data/sample](data/sample)
- Notebook demo: [notebooks/quickstart_api.ipynb](notebooks/quickstart_api.ipynb)

## Documentation

- [docs/architecture.md](docs/architecture.md): system structure
- [docs/validation.md](docs/validation.md): benchmark protocol, scenarios, tests, significance
- [docs/theory.md](docs/theory.md): equations and modeling assumptions
- [docs/literature.md](docs/literature.md): related work and BibTeX
- [docs/training.md](docs/training.md): MARL training modes
- [docs/data-pipeline.md](docs/data-pipeline.md): ingestion and caching
- [docs/deployment.md](docs/deployment.md): container and CI workflows
- [docs/mnpo_integration.md](docs/mnpo_integration.md): MNPO integration notes

## Citation

Repository citation metadata is in [CITATION.cff](CITATION.cff). Until a Zenodo DOI is minted, use the software citation below:

```bibtex
@software{smith2026marlforecastgame,
  author = {Smith, Tripp},
  title = {marl-forecast-game},
  year = {2026},
  url = {https://github.com/tripp-smith/marl-forecast-game},
  version = {0.1.0}
}
```

After the first archived release, add the DOI to `CITATION.cff` and replace the DOI badge.

## Limitations And Ethics

This project is intended for research and decision-support experimentation, not autonomous decision making.

- External data sources may encode sampling bias, reporting lag, survivorship effects, or market manipulation.
- LLM-backed components can import prompt bias, summarization artifacts, and provider instability.
- Approximate CRPS in the benchmark harness is derived from forecast intervals rather than a full predictive density export.
- Real-data claims should not be made from a single seed, single period, or single source.
- Computational cost rises quickly as adversaries, agents, and benchmark windows scale.

## Community

- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Community standards: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- License: [LICENSE](LICENSE)

## Roadmap

Near-term academic priorities:

1. Mint a Zenodo-backed release DOI and update the citation badge.
2. Add classical and deep-learning baselines as optional benchmark backends.
3. Expand fairness and subgroup robustness checks for synthetic and real data slices.

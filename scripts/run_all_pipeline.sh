#!/usr/bin/env bash
set -uo pipefail

RESULTS="${RESULTS_DIR:-/app/results}"
mkdir -p "$RESULTS"

echo "[pipeline] ============================================"
echo "[pipeline] MARL Forecast Game -- Full Pipeline"
echo "[pipeline] Results directory: $RESULTS"
echo "[pipeline] Started at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[pipeline] ============================================"

OVERALL_EXIT=0

echo ""
echo "[pipeline] === Phase 1: Unit + Property Tests ==="
python -m pytest -q --tb=short 2>&1 | tee "$RESULTS/pytest_output.txt"
PYTEST_EXIT=${PIPESTATUS[0]}
if [ "$PYTEST_EXIT" -ne 0 ]; then OVERALL_EXIT=1; fi
echo "[pipeline] Phase 1 exit code: $PYTEST_EXIT"

echo ""
echo "[pipeline] === Phase 2: Verification ==="
python scripts/run_verification.py --backend multiprocessing 2>&1 | tee "$RESULTS/verification_output.txt"
VER_EXIT=${PIPESTATUS[0]}
if [ "$VER_EXIT" -ne 0 ]; then OVERALL_EXIT=1; fi
cp planning/verification_report.json "$RESULTS/" 2>/dev/null || true
echo "[pipeline] Phase 2 exit code: $VER_EXIT"

echo ""
echo "[pipeline] === Phase 3: Training (smoke) ==="
python scripts/run_training.py --episodes 20 --horizon 30 --output-dir "$RESULTS/models" 2>&1 | tee "$RESULTS/training_output.txt"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "[pipeline] Phase 3 exit code: $TRAIN_EXIT"

python -c "
import json
from pathlib import Path
results_dir = '$RESULTS'
output = Path(results_dir) / 'training_output.txt'
info = {'source': 'pipeline', 'episodes': 20, 'horizon': 30, 'algorithm': 'wolf'}
if output.exists():
    lines = output.read_text().splitlines()
    for line in lines:
        if 'Mean reward' in line:
            try: info['mean_reward_last_50'] = float(line.split(':')[-1].strip())
            except ValueError: pass
        if 'Mean TD error' in line:
            try: info['mean_td_error_last_100'] = float(line.split(':')[-1].strip())
            except ValueError: pass
        if 'Final epsilon' in line:
            try: info['final_epsilon'] = float(line.split(':')[-1].strip())
            except ValueError: pass
with open(f'{results_dir}/training_results.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'[pipeline] Wrote {results_dir}/training_results.json')
" 2>&1

echo ""
echo "[pipeline] === Phase 4: Backtest ==="
python scripts/run_backtest.py --windows 5 --window-size 40 --step-size 15 --output-dir "$RESULTS" 2>&1 | tee "$RESULTS/backtest_output.txt"
echo "[pipeline] Phase 4 exit code: ${PIPESTATUS[0]}"

echo ""
echo "[pipeline] === Phase 5: Validation Scenarios ==="
python scripts/run_validation_scenarios.py --scenarios all --output-dir "$RESULTS" 2>&1 | tee "$RESULTS/validation_output.txt"
echo "[pipeline] Phase 5 exit code: ${PIPESTATUS[0]}"

echo ""
echo "[pipeline] === Phase 6: Stress Test ==="
python scripts/run_stress_test.py --games 20 --rounds 200 --workers 2 2>&1 | tee "$RESULTS/stress_output.txt"
echo "[pipeline] Phase 6 exit code: ${PIPESTATUS[0]}"

echo ""
echo "[pipeline] === Phase 7: Export Trajectory Artifacts ==="
python -c "
import json, time
from framework.game import ForecastGame
from framework.distributed import _game_outputs_to_dict
from framework.metrics import mae, rmse, mape, worst_case_abs_error
from framework.types import ForecastState, SimulationConfig

results_dir = '$RESULTS'
cfg = SimulationConfig(
    horizon=100, max_rounds=200,
    disturbance_prob=0.2, disturbance_scale=1.2,
    defense_model='ensemble',
)
state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

scenarios = [
    ('clean',        False, 42),
    ('attacked',     True,  42),
    ('attacked_s99', True,  99),
]
for label, disturbed, seed in scenarios:
    out = ForecastGame(cfg, seed=seed).run(state, disturbed=disturbed)
    d = _game_outputs_to_dict(out)
    d['seed'] = seed
    d['label'] = label
    d['disturbed'] = disturbed
    d['metrics'] = {
        'mae':  mae(out.targets, out.forecasts),
        'rmse': rmse(out.targets, out.forecasts),
        'mape': mape(out.targets, out.forecasts),
        'worst_case': worst_case_abs_error(out.targets, out.forecasts),
    }
    path = f'{results_dir}/simulation_{label}.json'
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
    print(f'[pipeline] Wrote {path}')

print('[pipeline] Trajectory export complete')
" 2>&1 | tee "$RESULTS/export_output.txt"
echo "[pipeline] Phase 7 exit code: ${PIPESTATUS[0]}"

echo ""
echo "[pipeline] ============================================"
echo "[pipeline] Pipeline Complete"
echo "[pipeline] Finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[pipeline] Overall exit code: $OVERALL_EXIT"
echo "[pipeline] ============================================"
echo "[pipeline] Results are available at: $RESULTS"
echo "[pipeline] Open Streamlit UI at http://localhost:8501"
echo "[pipeline] Open Grafana at http://localhost:3000"

echo "$OVERALL_EXIT" > "$RESULTS/.pipeline_exit_code"
exit "$OVERALL_EXIT"

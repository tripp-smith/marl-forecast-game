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
echo "[pipeline] === Phase 1: Programmatic Test Harness ==="
python run_project_tests.py --mode full --backend multiprocessing --report-dir "$RESULTS" 2>&1 | tee "$RESULTS/test_harness_output.txt"
HARNESS_EXIT=${PIPESTATUS[0]}
if [ "$HARNESS_EXIT" -ne 0 ]; then OVERALL_EXIT=1; fi
echo "[pipeline] Harness exit code: $HARNESS_EXIT"

echo ""
echo "[pipeline] === Phase 2: Export Trajectory Artifacts ==="
python -c "
import json
from framework.game import ForecastGame
from framework.distributed import _game_outputs_to_dict
from framework.metrics import mae, rmse, mape, worst_case_abs_error
from framework.types import ForecastState, SimulationConfig

results_dir = '$RESULTS'
cfg = SimulationConfig(
    horizon=100,
    max_rounds=200,
    disturbance_prob=0.2,
    disturbance_scale=1.2,
    defense_model='ensemble',
)
state = ForecastState(t=0, value=10.0, exogenous=0.0, hidden_shift=0.0)

scenarios = [
    ('clean', False, 42),
    ('attacked', True, 42),
    ('attacked_s99', True, 99),
]
for label, disturbed, seed in scenarios:
    out = ForecastGame(cfg, seed=seed).run(state, disturbed=disturbed)
    payload = _game_outputs_to_dict(out)
    payload['seed'] = seed
    payload['label'] = label
    payload['disturbed'] = disturbed
    payload['metrics'] = {
        'mae': mae(out.targets, out.forecasts),
        'rmse': rmse(out.targets, out.forecasts),
        'mape': mape(out.targets, out.forecasts),
        'worst_case': worst_case_abs_error(out.targets, out.forecasts),
    }
    path = f'{results_dir}/simulation_{label}.json'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f'[pipeline] Wrote {path}')
print('[pipeline] Trajectory export complete')
" 2>&1 | tee "$RESULTS/export_output.txt"
EXPORT_EXIT=${PIPESTATUS[0]}
if [ "$EXPORT_EXIT" -ne 0 ]; then OVERALL_EXIT=1; fi
echo "[pipeline] Export exit code: $EXPORT_EXIT"

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

#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-marl-forecast-game:test-harness}"
LOG_DIR="${LOG_DIR:-results}"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/container-test-harness-${TIMESTAMP}.log}"

mkdir -p "${LOG_DIR}"

echo "[harness] Writing log to ${LOG_FILE}"
{
  echo "[harness] Started at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[harness] Building Docker image ${IMAGE_TAG}"
  docker build -t "${IMAGE_TAG}" .

  echo "[harness] Running full containerized test suite"
  docker run --rm -v "$(pwd)/results:/app/results" "${IMAGE_TAG}" \
    sh -lc "python run_project_tests.py --mode full --backend multiprocessing --report-dir /app/results"

  echo "[harness] Finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
} 2>&1 | tee "${LOG_FILE}"

echo "[harness] Complete. Share this log file for debugging: ${LOG_FILE}"

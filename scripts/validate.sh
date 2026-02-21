#!/usr/bin/env sh
set -eu

pytest -q
python scripts/run_verification.py

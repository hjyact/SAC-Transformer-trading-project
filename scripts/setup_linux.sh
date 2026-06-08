#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-$USER}"

"$PYTHON_BIN" -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python scripts/gpu_smoke_test.py

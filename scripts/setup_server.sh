#!/usr/bin/env bash
# Run from the uploaded repository on the GPU server; does not start training.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "Use Python 3.11 or 3.12"'
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir \
  torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-server.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > environment.lock.txt
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python scripts/preflight.py --device cuda --out preflight.json
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python -m pytest -q tests/

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
export PYTHONPATH=.
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --log-level info

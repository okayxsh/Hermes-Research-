#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$ROOT/src" exec "$ROOT/.venv/bin/python" -m rq1.cli setup-stage candidate-models "$@"

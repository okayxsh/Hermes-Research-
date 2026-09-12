#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
PYTHONPATH="$ROOT/src" exec "$PYTHON" "$ROOT/scripts/runpod/validate_real_stack.py" "$@"

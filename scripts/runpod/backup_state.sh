#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${1:-}"
BACKUP_DIR="${2:-${RQ1_BACKUP_DIR:-/workspace/persistent/backups}}"
[[ -n "$RUN_ID" ]] || { echo "usage: backup_state.sh <experiment-id> [backup-directory]" >&2; exit 2; }
PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
PYTHONPATH="$ROOT/src" exec "$PYTHON" -m rq1.cli experiment backup --run-id "$RUN_ID" --backup-dir "$BACKUP_DIR"

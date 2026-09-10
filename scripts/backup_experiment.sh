#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <experiment-id> <backup-directory>" >&2
  exit 2
fi

python3 -m rq1.cli experiment backup --run-id "$1" --backup-dir "$2"

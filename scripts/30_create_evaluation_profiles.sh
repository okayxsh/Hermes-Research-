#!/usr/bin/env bash
set -euo pipefail
python -m rq1.cli evaluation profiles create "$@"

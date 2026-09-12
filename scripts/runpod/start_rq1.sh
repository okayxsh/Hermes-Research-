#!/usr/bin/env bash
set -Eeuo pipefail

# Launch gate for an already validated, manually approved machine. This script
# never provisions a Pod and never bypasses the repository's scientific gates.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERSISTENT_ROOT="${RQ1_RUNPOD_PERSISTENT_ROOT:-/workspace/persistent}"
BACKUP_DIR="${RQ1_BACKUP_DIR:-$PERSISTENT_ROOT/backups}"
ACTION="${1:-}"
shift || true
RUN_ID=""
PHASE="acquisition"
ACTIVATION=""
APPROVAL=""
FOREGROUND=0
APPROVE=0

usage() {
  cat <<'EOF'
Usage:
  start_rq1.sh start|resume --run-id ID --phase acquisition|evaluation [options]
  start_rq1.sh final --approval APPROVAL.json [options]
  start_rq1.sh status --run-id ID [--phase acquisition|evaluation|autopilot]
  start_rq1.sh logs --run-id ID

Launch options:
  --approve-launch         required explicit manual boundary after validation
  --activation-manifest P required for evaluation
  --approval P             required for final autopilot
  --backup-dir P           critical-file mirror (default /workspace/persistent/backups)
  --foreground             run attached instead of using tmux
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --activation-manifest) ACTIVATION="$2"; shift 2 ;;
    --approval) APPROVAL="$2"; shift 2 ;;
    --backup-dir) BACKUP_DIR="$2"; shift 2 ;;
    --approve-launch) APPROVE=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$ACTION" == "status" ]]; then
  [[ -n "$RUN_ID" ]] || { echo "--run-id is required" >&2; exit 2; }
  if [[ "$PHASE" == "autopilot" ]]; then
    exec "$PYTHON" -m rq1.cli autopilot status --run-id "$RUN_ID"
  fi
  exec "$PYTHON" -m rq1.cli experiment status --run-id "$RUN_ID"
fi

if [[ "$ACTION" == "logs" ]]; then
  [[ -n "$RUN_ID" ]] || { echo "--run-id is required" >&2; exit 2; }
  LOG_ROOT="$ROOT/results/final/$RUN_ID/logs"
  [[ -d "$LOG_ROOT" ]] || { echo "no log directory: $LOG_ROOT" >&2; exit 1; }
  LATEST="$(find "$LOG_ROOT" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {$1=""; sub(/^ /,""); print}')"
  if [[ -n "$LATEST" ]]; then tail -n 100 "$LATEST"; else find "$LOG_ROOT" -type f -print; fi
  exit 0
fi

[[ "$ACTION" == "start" || "$ACTION" == "resume" || "$ACTION" == "final" ]] || { echo "unknown action: ${ACTION:-<missing>}" >&2; usage >&2; exit 2; }
[[ "$APPROVE" == 1 ]] || { echo "Refusing launch: pass --approve-launch only after reviewing validation-report.json." >&2; exit 1; }
MARKER="$PERSISTENT_ROOT/rq1-validation-passed.json"
[[ -f "$MARKER" ]] || { echo "Refusing launch: validation marker is missing: $MARKER" >&2; exit 1; }
CURRENT_COMMIT="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null || true)"
MARKER_COMMIT="$(jq -r '.git_commit // empty' "$MARKER")"
[[ -n "$CURRENT_COMMIT" && "$CURRENT_COMMIT" == "$MARKER_COMMIT" ]] || { echo "Refusing launch: validation marker does not match current Git commit" >&2; exit 1; }
[[ -d "$BACKUP_DIR" && -w "$BACKUP_DIR" ]] || { echo "backup directory is not writable: $BACKUP_DIR" >&2; exit 1; }

COMMAND=()
ENV_PREFIX=()
if [[ "$ACTION" == "final" ]]; then
  [[ -n "$APPROVAL" && -f "$APPROVAL" ]] || { echo "--approval must point to an existing reviewed approval manifest" >&2; exit 2; }
  SESSION="rq1-final"
  COMMAND=(bash "$ROOT/scripts/rq1_autopilot.sh" final --approval "$APPROVAL" --yes)
elif [[ "$PHASE" == "acquisition" || "$PHASE" == "evaluation" ]]; then
  [[ -n "$RUN_ID" ]] || { echo "--run-id is required" >&2; exit 2; }
  SUBCOMMAND="$ACTION"
  if [[ "$PHASE" == "evaluation" ]]; then
    [[ -n "$ACTIVATION" && -f "$ACTIVATION" ]] || { echo "--activation-manifest must point to the immutable evaluation activation" >&2; exit 2; }
    ENV_PREFIX=(RQ1_RUN_FINAL_EVALUATION=1)
    COMMAND=("$PYTHON" -m rq1.cli evaluation "$SUBCOMMAND" --run-id "$RUN_ID" --activation-manifest "$ACTIVATION" --yes --backup-dir "$BACKUP_DIR" --require-backup)
  else
    COMMAND=("$PYTHON" -m rq1.cli acquisition "$SUBCOMMAND" --run-id "$RUN_ID" --yes --backup-dir "$BACKUP_DIR" --require-backup)
  fi
  SESSION="rq1-$PHASE-$RUN_ID"
else
  echo "--phase must be acquisition, evaluation, or use the final action" >&2
  exit 2
fi

LOG_DIR="$ROOT/results/final/${RUN_ID:-autopilot}/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/runpod-${ACTION}.log"
printf 'Launch gate approved at %s\nCommand:' "$(date -u +%FT%TZ)" > "$LOG_PATH"
printf ' %q' "${COMMAND[@]}" >> "$LOG_PATH"
printf '\n' >> "$LOG_PATH"

if [[ "$FOREGROUND" == 1 ]]; then
  cd "$ROOT"
  exec env "${ENV_PREFIX[@]}" "${COMMAND[@]}" 2>&1 | tee -a "$LOG_PATH"
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux is required for detached launch; pass --foreground if appropriate" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 1
fi
printf -v QROOT '%q' "$ROOT"
printf -v QLOG '%q' "$LOG_PATH"
CMD_TEXT=""
for item in "${COMMAND[@]}"; do printf -v QITEM '%q' "$item"; CMD_TEXT+="$QITEM "; done
ENV_TEXT=""
for item in "${ENV_PREFIX[@]}"; do ENV_TEXT+="$item "; done
tmux new-session -d -s "$SESSION" "cd $QROOT && exec env $ENV_TEXT $CMD_TEXT 2>&1 | tee -a $QLOG"
echo "started detached session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: $LOG_PATH"

#!/usr/bin/env bash
set -Eeuo pipefail

# Non-billable preparation happens on the local machine by reviewing this file;
# this script is intended to be run only after a Pod has been manually approved.
# It never calls runpodctl and never creates cloud resources.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERSISTENT_ROOT="${RQ1_RUNPOD_PERSISTENT_ROOT:-/workspace/persistent}"
MODE=""
VERBOSE=0

usage() {
  cat <<'EOF'
Usage: bootstrap.sh --dry-run [options]
       bootstrap.sh --apply [options]

Prepare an already-running Linux RunPod Pod. This script never provisions a Pod.
--dry-run              print the plan without installing or downloading anything
--apply                install the host/project prerequisites (target Pod only)
--repo-root PATH       repository checkout (default: this repository)
--persistent-root PATH persistent volume mount (default: /workspace/persistent)
--verbose              print commands as they run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--apply) [[ -z "$MODE" ]] || { echo "choose only one of --dry-run/--apply" >&2; exit 2; }; MODE="${1#--}"; shift ;;
    --repo-root) ROOT="$2"; [[ "$MODE" == "apply" && ! -d "$ROOT" ]] && { echo "repository does not exist: $ROOT" >&2; exit 1; }; [[ -d "$ROOT" ]] && ROOT="$(cd "$ROOT" && pwd)"; shift 2 ;;
    --persistent-root) PERSISTENT_ROOT="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$MODE" ]] || { echo "Refusing to run without --dry-run or --apply." >&2; usage >&2; exit 2; }

echo "Repository: $ROOT"
echo "Persistent root: $PERSISTENT_ROOT"
echo "Mode: $MODE"
echo "No RunPod resource provisioning is performed by this script."

if [[ "$MODE" == "dry-run" ]]; then
  cat <<EOF

Planned target-only actions:
  * verify Ubuntu/x86_64, NVIDIA visibility, RAM, and disk
  * install git curl build-essential zstd tmux jq ripgrep ffmpeg
  * run the existing resumable setup stages with the lockfile
  * keep ALFWorld data, Ollama models, final outputs, and backups below:
      $PERSISTENT_ROOT
  * run scripts/runpod/validate_real_stack.sh

Nothing was installed, downloaded, started, or provisioned.
EOF
  exit 0
fi

[[ "$(uname -s)" == "Linux" ]] || { echo "bootstrap requires Linux/Ubuntu; no changes made" >&2; exit 1; }
ARCH="$(uname -m)"
[[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]] || { echo "unsupported architecture: $ARCH" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is unavailable; refusing setup" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

FREE_GIB="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%.0f", $4/1024/1024}')"
RAM_GIB="$(awk '/MemTotal:/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)"
(( FREE_GIB >= 80 )) || { echo "need at least 80 GiB free on the repository/persistent filesystem; found ${FREE_GIB} GiB" >&2; exit 1; }
(( RAM_GIB >= 32 )) || { echo "need at least 32 GiB RAM; found ${RAM_GIB} GiB" >&2; exit 1; }
[[ -d "$PERSISTENT_ROOT" ]] || { echo "persistent mount is missing: $PERSISTENT_ROOT" >&2; exit 1; }
[[ -w "$PERSISTENT_ROOT" ]] || { echo "persistent mount is not writable: $PERSISTENT_ROOT" >&2; exit 1; }
git -C "$ROOT" rev-parse --verify HEAD >/dev/null 2>&1 || { echo "repository has no resolvable Git commit; commit the reviewed checkout before bootstrap" >&2; exit 1; }

if [[ "$VERBOSE" == 1 ]]; then set -x; fi

SUDO=()
if [[ "$(id -u)" != 0 ]]; then
  command -v sudo >/dev/null 2>&1 || { echo "root or sudo is required" >&2; exit 1; }
  SUDO=(sudo)
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git curl build-essential zstd tmux jq ripgrep ffmpeg ca-certificates xz-utils libffi-dev python3-dev

mkdir -p "$PERSISTENT_ROOT"/{models,alfworld_data,experiment_outputs,backups,logs,manifests}
export RQ1_ALFWORLD_DATA_DIR="$PERSISTENT_ROOT/alfworld_data"
export OLLAMA_MODELS="$PERSISTENT_ROOT/models/ollama"
export RQ1_RUNPOD_PERSISTENT_ROOT="$PERSISTENT_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_cli() {
  local python_bin="$ROOT/.venv/bin/python"
  if [[ ! -x "$python_bin" ]]; then python_bin="$(command -v python3)"; fi
  PYTHONPATH="$ROOT/src" "$python_bin" -m rq1.cli "$@"
}

run_stage() {
  local stage="$1"
  echo "== setup stage: $stage =="
  if run_cli setup-stage "$stage" --yes --resume --verbose; then
    return 0
  fi
  if [[ "$stage" == "base-profiles" ]]; then
    local state
    state="$(run_cli setup-status)"
    if [[ "$(jq -r '."base-profiles".status' <<<"$state")" == "blocked" ]]; then
      echo "base-profiles is capability-blocked; continuing to diagnostic verification"
      return 0
    fi
  fi
  echo "setup stage failed: $stage" >&2
  return 1
}

for stage in preflight system-packages python-environment ollama hermes alfworld-package alfworld-data candidate-models base-profiles installation-verification; do
  run_stage "$stage"
done

echo "== repository capability checks =="
run_cli validate-config
run_cli hermes-capabilities || true
run_cli alfworld capabilities || true

echo "== RunPod validation gate =="
"$ROOT/scripts/runpod/validate_real_stack.sh" --repo-root "$ROOT" --persistent-root "$PERSISTENT_ROOT" --backup-dir "$PERSISTENT_ROOT/backups"

echo "Bootstrap completed. The launch gate remains a separate manual approval step."

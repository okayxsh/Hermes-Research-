"""Observed acquisition environment identity for freezes and launch verification."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

from rq1.acquisition.protocol import ACQUISITION_MODEL
from rq1.acquisition.skill_creation import prompt_hashes
from rq1.experiment.persistence import bind_repository_configuration, runtime_manifest
from rq1.hermes.episode_driver import INFERENCE_SEED
from rq1.utils.hashing import sha256_file

HERMES_INSTALL_DIR = Path("/usr/local/lib/hermes-agent")
OLLAMA_URL = "http://127.0.0.1:11434"
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
DEFAULT_HF_HUB_CACHE = Path("/workspace/persistent/hf-cache/hub")
# Identities a scientific run or resume must reproduce.  Host name and GPU are
# recorded but not enforced, so a replacement Pod can resume the same run.
ENFORCED_AT_LAUNCH = (
    "repository_commit",
    "python_version",
    "dependency_lock_sha256",
    "alfworld_version",
    "hermes_version",
    "hermes_commit",
    "ollama_version",
    "model_tag",
    "model_digest",
    "inference_seed",
    "prompt_hashes",
    "config_hashes",
)


def _run(argv: Sequence[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(tuple(argv), cwd=cwd, text=True, capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout.strip()
    return output if completed.returncode == 0 and output else None


def _ollama(path: str) -> dict[str, Any]:
    try:
        with urlopen(OLLAMA_URL + path, timeout=15) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def model_digest(tag: str) -> str | None:
    for item in _ollama("/api/tags").get("models", []):
        if isinstance(item, dict) and item.get("name") in {tag, f"{tag}:latest"}:
            return item.get("digest")
    return None


def hermes_version() -> str | None:
    match = re.search(r"Hermes Agent v(\S+)", _run(("hermes", "--version")) or "")
    return match.group(1) if match else None


def sbert_snapshot_sha256(cache: Path) -> str | None:
    """SHA-256 of the compact sorted JSON map of snapshot file paths to file SHA-256."""
    snapshot = cache / "models--sentence-transformers--all-mpnet-base-v2" / "snapshots" / SBERT_REVISION
    if not snapshot.is_dir():
        return None
    files = {path.relative_to(snapshot).as_posix(): sha256_file(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def observed_environment(
    root: Path,
    *,
    task_queue_sha256: str | None,
    alfworld_data_identity: str | None,
    include_sbert: bool = True,
) -> dict[str, Any]:
    runtime = runtime_manifest(root)
    packages = runtime.get("packages", {})
    gpu = [item.strip() for item in (_run(("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")) or "").split(",")]
    cache = Path(os.environ.get("HF_HUB_CACHE") or DEFAULT_HF_HUB_CACHE)
    return {
        "repository_commit": runtime.get("git_commit"),
        "branch": _run(("git", "rev-parse", "--abbrev-ref", "HEAD"), cwd=root),
        "hostname": socket.gethostname(),
        "gpu": gpu[0] or None,
        "gpu_memory": gpu[1] if len(gpu) > 1 else None,
        "gpu_driver": gpu[2] if len(gpu) > 2 else None,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_environment": sys.prefix,
        "dependency_lock_sha256": sha256_file(root / "uv.lock") if (root / "uv.lock").is_file() else None,
        "packages": packages,
        "alfworld_version": packages.get("alfworld"),
        "alfworld_data_identity": alfworld_data_identity,
        "hermes_version": hermes_version(),
        "hermes_commit": _run(("git", "-C", str(HERMES_INSTALL_DIR), "rev-parse", "HEAD")),
        "ollama_version": _ollama("/api/version").get("version"),
        "model_tag": ACQUISITION_MODEL,
        "model_digest": model_digest(ACQUISITION_MODEL),
        "inference_seed": INFERENCE_SEED,
        "sbert_model": SBERT_MODEL,
        "sbert_revision": SBERT_REVISION,
        "sbert_snapshot_sha256": sbert_snapshot_sha256(cache) if include_sbert else None,
        "task_queue_sha256": task_queue_sha256,
        "prompt_hashes": prompt_hashes(root),
        "config_hashes": bind_repository_configuration(root, {})["repository_input_hashes"],
    }


def verify_launch_environment(root: Path, frozen: Mapping[str, Any]) -> list[str]:
    observed = observed_environment(
        root,
        task_queue_sha256=frozen.get("task_queue_sha256"),
        alfworld_data_identity=frozen.get("alfworld_data_identity"),
        include_sbert=False,
    )
    return [
        f"{key} differs from the approved acquisition environment freeze"
        for key in ENFORCED_AT_LAUNCH
        if observed.get(key) != frozen.get(key)
    ]

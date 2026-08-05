from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rq1.utils.time import utc_now


def _command_version(command: str) -> str | None:
    path = shutil.which(command)
    if not path:
        return None
    try:
        output = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=3, check=False)
        return (output.stdout or output.stderr).strip().splitlines()[0][:200] or path
    except (OSError, subprocess.TimeoutExpired):
        return f"{path} (version probe unavailable)"


def _nvidia() -> dict[str, str | None]:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"gpu": None, "vram_gb": None, "driver": None}
    try:
        output = subprocess.run(
            [command, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=False,
        ).stdout.strip().split(",")
        return {"gpu": output[0].strip(), "vram_gb": output[1].strip() if len(output) > 1 else None, "driver": output[2].strip() if len(output) > 2 else None}
    except OSError:
        return {"gpu": "unavailable", "vram_gb": None, "driver": None}


def machine_manifest(root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    is_wsl = "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ
    revision = _command_version("git")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=3, check=False).stdout.strip() or None
    except OSError:
        commit = None
    return {
        "generated_at": utc_now(), "os": platform.platform(), "kernel": platform.release(), "wsl": is_wsl,
        "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(), "ram_gb": None,
        "disk_free_gb": round(disk.free / 1024 ** 3, 2), "python": sys.version.split()[0],
        "git": revision, "git_commit": commit, "ollama": _command_version("ollama"),
        "hermes": _command_version("hermes"), "alfworld": _command_version("alfworld"), **_nvidia(),
    }

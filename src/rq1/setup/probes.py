from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rq1.setup.models import ProbeResult
from rq1.setup.runner import CommandRunner
from rq1.utils.time import utc_now


SUPPORTED_UBUNTU = {"22.04", "24.04"}
REQUIRED_HOSTS = (
    "https://astral.sh",
    "https://ollama.com",
    "https://registry.ollama.ai",
    "https://hermes-agent.nousresearch.com",
    "https://pypi.org",
    "https://files.pythonhosted.org",
    "https://github.com/alfworld/alfworld",
)


def read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def total_ram_gib(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 2)
    return None


def is_wsl() -> bool:
    return "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))


def wsl_generation() -> int | None:
    """Return 2 for WSL2, 1 for WSL1, and None for native Linux."""
    if not is_wsl():
        return None
    release = platform.release().lower()
    return 2 if "wsl2" in release or "microsoft-standard" in release else 1


def network_probe(url: str, timeout: int = 8) -> ProbeResult:
    try:
        request = Request(url, headers={"User-Agent": "rq1-installation-probe/1"})
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
        return ProbeResult(f"network:{url}", True, f"HTTP {status}")
    except HTTPError as exc:
        # An HTTPError still proves DNS, TLS, and HTTP-layer connectivity.
        return ProbeResult(f"network:{url}", True, f"HTTP {exc.code}")
    except (OSError, URLError) as exc:
        return ProbeResult(f"network:{url}", False, f"unreachable: {type(exc).__name__}")


def command_probe(runner: CommandRunner, command: str, *version_args: str) -> ProbeResult:
    path = runner.which(command)
    if not path:
        return ProbeResult(command, False, "command not found")
    args = version_args or ("--version",)
    result = runner.run((path, *args), timeout=15)
    output = (result.stdout or result.stderr).strip().splitlines()
    version = output[0][:300] if output else None
    return ProbeResult(command, result.ok, "command available", version, {"path": path})


def nvidia_probe(runner: CommandRunner) -> ProbeResult:
    path = runner.which("nvidia-smi")
    if not path:
        return ProbeResult("nvidia", False, "nvidia-smi not found; CPU fallback remains possible")
    result = runner.run(
        (
            path,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
        timeout=15,
    )
    if not result.ok:
        return ProbeResult("nvidia", False, "nvidia-smi probe failed")
    first = result.stdout.strip().splitlines()[0].split(",")
    metadata = {
        "model": first[0].strip() if first else None,
        "vram_mib": int(float(first[1])) if len(first) > 1 else None,
        "driver": first[2].strip() if len(first) > 2 else None,
    }
    return ProbeResult("nvidia", True, "NVIDIA GPU visible", metadata.get("driver"), metadata)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def python_executable(root: Path) -> Path:
    candidate = root / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    windows = root / ".venv" / "Scripts" / "python.exe"
    return windows if windows.exists() else candidate


def executable_in_venv(root: Path, name: str) -> Path:
    unix = root / ".venv" / "bin" / name
    if unix.exists():
        return unix
    suffix = ".exe" if os.name == "nt" else ""
    return root / ".venv" / "Scripts" / f"{name}{suffix}"


def data_inventory(path: Path) -> dict[str, Any]:
    import hashlib

    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "bytes": 0, "inventory_sha256": None}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    size = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        item_size = item.stat().st_size
        size += item_size
        digest.update(f"{relative}\0{item_size}\n".encode("utf-8"))
    return {
        "path": str(path),
        "exists": True,
        "file_count": len(files),
        "bytes": size,
        "inventory_sha256": digest.hexdigest(),
        "required_directories": {
            name: (path / name).is_dir() for name in ("json_2.1.1", "logic")
        },
    }


def write_machine_yaml(root: Path, runner: CommandRunner) -> Path:
    os_release = read_os_release()
    disk = shutil.disk_usage(root)
    gpu = nvidia_probe(runner)
    commit_result = runner.run(("git", "rev-parse", "HEAD"), cwd=root, timeout=15)
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "platform": {
            "os_id": os_release.get("ID", platform.system().lower()),
            "os_version": os_release.get("VERSION_ID"),
            "architecture": platform.machine(),
            "kernel": platform.release(),
            "wsl2": wsl_generation() == 2,
        },
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "ram_gib": total_ram_gib(),
            "disk_free_gib": round(disk.free / 1024**3, 2),
            "gpu": gpu.metadata if gpu.available else None,
            "cuda_compiler": command_probe(runner, "nvcc").version,
        },
        "repository": {"git_commit": commit_result.stdout.strip() if commit_result.ok else None},
        "privacy": {"hostname_recorded": False, "username_recorded": False, "network_ids_recorded": False},
    }
    path = root / "artifacts" / "manifests" / "machine_manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json_response(url: str, payload: bytes | None = None, timeout: int = 10) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = Request(url, data=payload, headers=headers, method="POST" if payload is not None else "GET")
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return value

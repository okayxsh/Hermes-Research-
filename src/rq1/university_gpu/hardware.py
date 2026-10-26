"""Read-only machine discovery and fail-closed GPU identity checks."""
from __future__ import annotations

import csv
import io
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rq1.university_gpu.models import Check, GPUDevice

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def parse_nvidia_smi_csv(text: str) -> list[GPUDevice]:
    devices: list[GPUDevice] = []
    for index, row in enumerate(csv.reader(io.StringIO(text), skipinitialspace=True)):
        if len(row) != 5:
            raise ValueError(f"expected five NVIDIA device fields on row {index + 1}")
        uuid, shown_index, name, total, free = (value.strip() for value in row)
        if not uuid.startswith("GPU-"):
            raise ValueError(f"GPU row {index + 1} lacks a physical GPU UUID")
        devices.append(GPUDevice(uuid, int(shown_index), name, int(total), int(free)))
    if not devices:
        raise ValueError("NVIDIA device query returned no GPUs")
    if len({device.uuid for device in devices}) != len(devices):
        raise ValueError("NVIDIA device query returned duplicate UUIDs")
    return devices


def discover_gpus(*, runner: CommandRunner = subprocess.run) -> tuple[list[GPUDevice], str | None]:
    try:
        result = runner(
            ["nvidia-smi", "--query-gpu=uuid,index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"nvidia-smi unavailable: {type(exc).__name__}"
    if result.returncode != 0:
        return [], (result.stderr or "nvidia-smi failed").strip()
    try:
        return parse_nvidia_smi_csv(result.stdout), None
    except (TypeError, ValueError) as exc:
        return [], str(exc)


def _memory_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            state = MEMORYSTATUSEX()
            state.dwLength = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return int(state.ullTotalPhys)
        except (AttributeError, OSError):
            pass
    return None


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def machine_report(
    *,
    output_dir: Path,
    gpu_count: int = 4,
    min_gpu_free_mib: int = 16_384,
    ollama_base_port: int = 11_534,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    gpus, gpu_error = discover_gpus(runner=runner)
    checks: list[Check] = []
    is_linux = sys.platform.startswith("linux")
    checks.append(Check("target_os", "PASS" if is_linux else "NOT_TESTED", f"Detected {platform.platform()}; university execution target is Linux/Ubuntu."))
    checks.append(Check("python", "PASS" if sys.version_info[:2] == (3, 11) else "FAIL", f"Detected Python {platform.python_version()}; require Python 3.11.x."))
    cpu_count = os.cpu_count()
    checks.append(Check("cpu", "PASS" if cpu_count and cpu_count >= 8 else "FAIL", f"Detected {cpu_count} logical CPUs; at least 8 required."))
    memory = _memory_bytes()
    checks.append(Check("ram", "PASS" if memory is not None and memory >= 64 * 1024**3 else ("BLOCKED" if memory is None else "FAIL"),
                        "Total physical RAM discovery.", {"bytes": memory, "minimum_bytes": 64 * 1024**3}))
    try:
        probe_path = output_dir
        while not probe_path.exists() and probe_path != probe_path.parent:
            probe_path = probe_path.parent
        disk = shutil.disk_usage(probe_path)
        writable = os.access(probe_path, os.W_OK)
        checks.append(Check("output_storage", "PASS" if writable and disk.free >= 20 * 10**9 else "FAIL",
                            "Output directory or nearest existing parent must be writable with at least 20 GB free.",
                            {"path": str(output_dir), "checked_path": str(probe_path), "free_bytes": disk.free, "writable": writable}))
    except OSError as exc:
        checks.append(Check("output_storage", "FAIL", f"Cannot inspect output directory: {exc}"))
    if gpu_error:
        checks.append(Check("nvidia_gpus", "BLOCKED", gpu_error))
    else:
        checks.append(Check("nvidia_gpus", "PASS" if len(gpus) == gpu_count else "FAIL",
                            f"Detected {len(gpus)} physical NVIDIA GPU(s); configured target is {gpu_count}.",
                            {"devices": [gpu.to_dict() for gpu in gpus]}))
        enough_memory = all(gpu.memory_free_mib >= min_gpu_free_mib for gpu in gpus)
        checks.append(Check("gpu_free_memory", "PASS" if enough_memory else "FAIL",
                            f"Each GPU must have at least {min_gpu_free_mib} MiB free.",
                            {gpu.uuid: gpu.memory_free_mib for gpu in gpus}))
    port_checks = {str(port): port_is_free("127.0.0.1", port) for port in range(ollama_base_port, ollama_base_port + gpu_count)}
    checks.append(Check("worker_ports", "PASS" if all(port_checks.values()) else "FAIL",
                        "Dedicated Ollama endpoint ports must be available.", port_checks))
    return {
        "schema": "rq1-university-gpu-machine-report-v1",
        "status": "PASS" if all(check.status == "PASS" for check in checks) else "BLOCKED",
        "scientific_evidence": False,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "checks": [check.to_dict() for check in checks],
        "devices": [gpu.to_dict() for gpu in gpus],
    }


def parse_compute_processes(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row or len(row) < 2:
            continue
        rows.append({"pid": int(row[0].strip()), "gpu_uuid": row[1].strip(),
                     "used_memory_mib": int(row[2].strip()) if len(row) > 2 and row[2].strip().isdigit() else None})
    return rows


def linux_descendants(pid: int, *, proc_root: Path = Path("/proc")) -> set[int]:
    """Return a PID and observed Linux descendants; unreadable entries are ignored."""
    found = {pid}
    pending = [pid]
    while pending:
        parent = pending.pop()
        children_path = proc_root / str(parent) / "task" / str(parent) / "children"
        try:
            children = [int(value) for value in children_path.read_text(encoding="ascii").split()]
        except (OSError, ValueError):
            continue
        for child in children:
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def verify_worker_placements(
    worker_pids: Mapping[str, int], process_rows: Sequence[Mapping[str, Any]], *,
    descendant_provider: Callable[[int], set[int]] = linux_descendants,
) -> dict[str, Any]:
    assignments: dict[str, str] = {}
    errors: list[str] = []
    for worker_id, pid in worker_pids.items():
        tree = descendant_provider(int(pid))
        seen = {str(row.get("gpu_uuid")) for row in process_rows if int(row.get("pid", -1)) in tree}
        if len(seen) != 1:
            errors.append(f"{worker_id} has no unique observed GPU placement: {sorted(seen)}")
        else:
            assignments[worker_id] = next(iter(seen))
    if len(set(assignments.values())) != len(assignments):
        errors.append("multiple workers are observed on the same physical GPU")
    return {"passed": not errors and len(assignments) == len(worker_pids), "assignments": assignments, "errors": errors}

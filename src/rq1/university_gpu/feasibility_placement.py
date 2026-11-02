"""Engineering-only WSL placement evidence; the scientific launcher stays PID-only."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from rq1.university_gpu.ops import capture_placement


def _is_wsl() -> bool:
    try:
        return os.name == "posix" and "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _owns_log(server: Any) -> bool:
    """Check the still-owned server's environment and its stdout log descriptor."""
    try:
        proc = Path('/proc') / str(server.process.pid)
        env = dict(item.split(b'=', 1) for item in (proc / 'environ').read_bytes().split(b'\0') if b'=' in item)
        st = (proc / 'fd' / '1').stat()
        return (server.process.poll() is None and (st.st_dev, st.st_ino) == server.log_identity
                and env.get(b'CUDA_VISIBLE_DEVICES') == server.spec.gpu_uuid.encode()
                and env.get(b'OLLAMA_HOST') == f'127.0.0.1:{server.spec.port}'.encode())
    except (OSError, ValueError, AttributeError):
        return False


def _pci(value: str) -> str:
    match = re.fullmatch(r'(?:(?:[0-9a-fA-F]{4,8}):)?([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value.strip())
    if not match:
        raise ValueError('malformed PCI address')
    # Preserve domain: dropping it could conflate distinct physical devices.
    parts = value.strip().split(':')
    domain = int(parts[0], 16) if len(parts) == 3 else 0
    return f'{domain:08x}:{match.group(1).lower()}:{match.group(2).lower()}.{match.group(3)}'


def capture_feasibility_placement(servers: list[Any], *, runner: Any = subprocess.run,
                                  pid_capture: Any = capture_placement,
                                  wsl_check: Any = _is_wsl, owner_check: Any = _owns_log) -> dict[str, Any]:
    """Prefer PID evidence; use fresh, process-bound CUDA logs only when WSL omits PIDs."""
    expected = {s.spec.worker_id: s.spec.gpu_uuid for s in servers}
    pids = {s.spec.worker_id: int(s.process.pid) for s in servers if s.process is not None}
    if (len(servers) != 4 or len(expected) != 4 or len(pids) != 4
            or len(set(expected.values())) != 4 or len(set(pids.values())) != 4
            or any(s.process.poll() is not None for s in servers)):
        return {'passed': False, 'method': None, 'assignments': {}, 'errors': ['four distinct live worker processes and GPUs required']}
    pid = pid_capture(pids)
    observed = pid.get('assignments', {})
    if not isinstance(observed, dict) or any(k not in expected or expected[k] != v for k, v in observed.items()):
        return {'passed': False, 'method': None, 'assignments': {}, 'errors': ['PID placement conflicts with expected GPUs'], 'pid_evidence': pid}
    if pid.get('passed') is True:
        if observed == expected and all(server.process.poll() is None for server in servers):
            return {**pid, 'method': 'pid'}
        return {'passed': False, 'method': None, 'assignments': {}, 'errors': ['PID placement incomplete'], 'pid_evidence': pid}
    errors = pid.get('errors')
    missing = set(expected) - set(observed)
    allowed_errors = {f'{worker} has no unique observed GPU placement: []' for worker in missing}
    if not isinstance(errors, list) or set(errors) != allowed_errors or not missing:
        return {'passed': False, 'method': None, 'assignments': {},
                'errors': ['PID evidence is ambiguous or query failed'], 'pid_evidence': pid}
    result: dict[str, Any] = {'passed': False, 'method': None, 'assignments': {}, 'pid_evidence': pid, 'errors': []}
    try:
        if not wsl_check():
            raise ValueError('WSL-only fallback unavailable')
        query = runner(['nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=False, timeout=20)
        if query.returncode != 0:
            raise ValueError('GPU PCI inventory query failed')
        inventory: dict[str, str] = {}
        for row in csv.reader(io.StringIO(query.stdout)):
            if len(row) != 2 or not row[0].strip():
                raise ValueError('malformed GPU PCI inventory')
            gpu, bus = row[0].strip(), _pci(row[1])
            if gpu in inventory or bus in inventory.values():
                raise ValueError('ambiguous GPU PCI inventory')
            inventory[gpu] = bus
        if set(inventory) != set(expected.values()):
            raise ValueError('live PCI inventory differs from four expected GPUs')
        evidence = {}
        for server in servers:
            if not owner_check(server):
                raise ValueError(f'{server.spec.worker_id}: live process ownership not verified')
            path = server.spec.worker_dir / 'ollama-server.log'
            if path.is_symlink() or server.log_start_offset is None or server.log_identity is None:
                raise ValueError('worker log identity unavailable')
            with path.open('rb') as f:
                st = os.fstat(f.fileno())
                if (st.st_dev, st.st_ino) != server.log_identity or st.st_size <= server.log_start_offset:
                    raise ValueError('worker log replaced, empty, or stale')
                f.seek(server.log_start_offset)
                raw = f.read()
            lines = raw.decode('utf-8', errors='strict').splitlines()
            gpu_lines = [line for line in lines if 'msg="inference compute"' in line and re.search(r'\blibrary=CUDA\b', line)]
            devices = []
            for line in gpu_lines:
                buses = re.findall(r'\bpci_id="?([0-9a-fA-F:.]+)"?', line)
                names = re.findall(r'\bname="?(CUDA\d+)"?', line)
                if len(buses) != 1 or len(names) != 1:
                    raise ValueError('ambiguous CUDA discovery line')
                devices.append((_pci(buses[0]), names[0]))
            if not devices or set(devices) != {(inventory[server.spec.gpu_uuid], 'CUDA0')}:
                raise ValueError(f'{server.spec.worker_id}: CUDA discovery conflicts with expected GPU')
            loaded = [line for line in lines if re.search(r'load_tensors:.*\bCUDA0 model buffer size\s*=\s*[1-9][0-9.]*\s*MiB\b', line)
                      or re.search(r'msg="model weights".*\bdevice=CUDA0\b.*\bsize="?[1-9][0-9.]*\s*(?:MiB|GiB)', line)]
            if not loaded:
                raise ValueError(f'{server.spec.worker_id}: CUDA model loading not evidenced')
            if not owner_check(server):
                raise ValueError('worker stopped or ownership changed during verification')
            evidence[server.spec.worker_id] = {'pid': server.process.pid, 'gpu_uuid': server.spec.gpu_uuid,
                'pci_bus_id': inventory[server.spec.gpu_uuid], 'log_path': str(path), 'start_offset': server.log_start_offset,
                'end_offset': server.log_start_offset + len(raw), 'fresh_log_sha256': hashlib.sha256(raw).hexdigest(),
                'cuda_discovery_lines': gpu_lines, 'cuda_model_load_lines': loaded}
        result.update(passed=True, method='wsl_log_pci', assignments=expected, errors=[],
                      gpu_pci_inventory=inventory, workers=evidence)
    except (OSError, ValueError, UnicodeError, AttributeError, subprocess.TimeoutExpired) as exc:
        result['errors'].append(str(exc))
    return result

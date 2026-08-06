"""Non-mutating local supervisor health probes."""
from __future__ import annotations
import os, shutil
from pathlib import Path
from typing import Any
from rq1.utils.time import utc_now

def sample(root: Path) -> dict[str, Any]:
    disk=shutil.disk_usage(root)
    return {"schema_version":1,"timestamp":utc_now(),"pid":os.getpid(),"disk_free_bytes":disk.free,"disk_total_bytes":disk.total,"systemd_user_available":shutil.which("systemd-run") is not None,"tmux_available":shutil.which("tmux") is not None,"gpu_available":shutil.which("nvidia-smi") is not None}

"""Fail-closed guard for process operations that require POSIX semantics."""
from __future__ import annotations

import os


class UnsupportedProcessManagementError(RuntimeError):
    """Raised when process ownership cannot be checked safely on this platform."""


def require_posix_process_management(operation: str) -> None:
    """Reject native Windows before any POSIX-style PID probing or worker signals.

    Python's ``os.kill(pid, 0)`` does not perform a liveness probe on Windows:
    Windows treats signal values other than its two console-event values as a
    request to terminate the target process. Linux, including Ubuntu under
    WSL2, retains the expected POSIX existence-check behavior.
    """
    if os.name == "nt":
        raise UnsupportedProcessManagementError(
            f"{operation} requires POSIX process management; run this command under Linux/WSL2, "
            "not native Windows Python"
        )

"""Runtime protocol shared by deterministic fake and capability-gated real modes."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rq1.pilot.models import PilotTestSpec, RuntimeExecution


class PilotRuntime(Protocol):
    simulated: bool

    def execute(self, spec: PilotTestSpec, *, run_id: str, attempt_id: str, output_dir: Path) -> RuntimeExecution: ...

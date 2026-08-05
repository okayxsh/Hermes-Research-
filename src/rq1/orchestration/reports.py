from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rq1.utils.time import utc_now


@dataclass
class StageReport:
    stage: str
    attempt_id: str
    status: str
    started_at: str
    completed_at: str
    dry_run: bool
    outputs: list[str]
    warnings: list[str]
    error: str | None = None
    next_command: str | None = None
    metadata: dict[str, Any] | None = None

    def write(self, path: Path, overwrite: bool = False) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite report: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed_report(stage: str, attempt_id: str, started_at: str, **kwargs: Any) -> StageReport:
    return StageReport(stage=stage, attempt_id=attempt_id, status="passed", started_at=started_at, completed_at=utc_now(), **kwargs)

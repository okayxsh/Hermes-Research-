"""Append-only recovery JSONL evidence."""
from __future__ import annotations
import json
from pathlib import Path
from rq1.recovery.models import RecoveryEvent, to_dict

def append_event(path: Path, event: RecoveryEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(to_dict(event), sort_keys=True) + "\n")

def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

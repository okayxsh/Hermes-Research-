"""Summarize four observed pilot attempts; no inference is performed here."""
from datetime import datetime
from pathlib import Path

from rq1.university_gpu.models import load_json
from rq1.university_gpu.scheduler import RunStore, SchedulerError


def report(run_root: Path, manifest_path: Path) -> dict:
    manifest = load_json(manifest_path)
    store = RunStore(run_root, manifest)
    events = store.event_rows()
    starts = [row for row in events if row["event"] == "attempt_started"][:4]
    if len(starts) != 4 or len({row["worker_id"] for row in starts}) != 4:
        raise SchedulerError("benchmark requires the first four attempts on four distinct workers")
    finishes = {row["attempt_id"]: row for row in events if row["event"] == "unit_completed"}
    failed = {row["attempt_id"] for row in events if row["event"] == "attempt_failed"}
    if any(row["attempt_id"] not in finishes or row["attempt_id"] in failed for row in starts):
        raise SchedulerError("all four benchmark attempts must complete without infrastructure failure")
    rows = []
    for start in starts:
        end = finishes[start["attempt_id"]]
        a = datetime.fromisoformat(start["timestamp"])
        b = datetime.fromisoformat(end["timestamp"])
        rows.append({"worker_id": start["worker_id"], "unit_key": start["unit_key"],
                     "seconds": (b - a).total_seconds(), "started_at": start["timestamp"], "completed_at": end["timestamp"]})
    seconds = (max(datetime.fromisoformat(finishes[row["attempt_id"]]["timestamp"]) for row in starts)
               - min(datetime.fromisoformat(row["timestamp"]) for row in starts)).total_seconds()
    if seconds <= 0 or any(row["seconds"] <= 0 for row in rows):
        raise SchedulerError("invalid benchmark timestamps")
    return {"status": "PASS", "benchmark_attempts": rows, "elapsed_seconds": seconds,
            "episodes_per_hour": 4 * 3600 / seconds,
            "projected_hours_36": 36 * seconds / 4 / 3600,
            "projected_hours_540": 540 * seconds / 4 / 3600,
            "projected_hours_576": 576 * seconds / 4 / 3600,
            "limitations": "Four pilot episodes provide a rough rate only; differing tasks, retries, model load, backups, contention and transfers can change total time.",
            "scientific_evidence": False}

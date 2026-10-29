"""Descriptive reporting for incomplete or complete verified runs."""
from collections import Counter
from pathlib import Path

from rq1.university_gpu.manifest import CONDITIONS
from rq1.university_gpu.models import load_json
from rq1.university_gpu.scheduler import Coordinator, RunStore, SchedulerError


def report(run_root: Path, manifest_path: Path) -> dict:
    manifest = load_json(manifest_path)
    store = RunStore(run_root, manifest)
    validation = Coordinator(store, [type("OfflineWorker", (), {"worker_id": "report"})()]).validate()
    if validation["problems"]:
        raise SchedulerError("invalid run: " + "; ".join(validation["problems"]))
    events = store.event_rows()
    completed = store.completed()
    counts = Counter()
    successes = Counter()
    for envelope in completed.values():
        row = envelope["result"]
        if not isinstance(row.get("recovery_success"), bool):
            raise SchedulerError("completed result lacks a binary recovery outcome")
        condition = row["condition"]
        counts[condition] += 1
        successes[condition] += int(row["recovery_success"])
    planned = Counter(unit["condition"] for unit in manifest["units"])
    return {"status": "COMPLETE" if validation["passed"] else "INCOMPLETE", "planned": validation["expected"],
            "completed": validation["completed"], "missing_unit_keys": validation["missing"],
            "failed_attempts": sum(row["event"] == "attempt_failed" for row in events),
            "conditions": {condition: {"planned": planned[condition], "completed": counts[condition],
                                       "successes": successes[condition],
                                       "recovery_rate": successes[condition] / counts[condition] if counts[condition] else None}
                           for condition in CONDITIONS},
            "interpretation": "Counts from an incomplete run are preliminary and may be unbalanced; do not use them for confirmatory comparisons."}

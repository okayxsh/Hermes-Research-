"""Resumable single-process work queue for five-arm and pilot-diagnostic manifests.

Layout under ``<output_root>/<run_id>/``::

    manifest.json        the exact manifest executed (written once; must match on resume)
    journal.jsonl        durable events: attempt_started / attempt_finished / protocol_violation / interrupted_attempt
    records.jsonl        one measurement record per attempt (fsync'd append)
    attempts/<unit>/a<k> fresh directory per attempt (never reused)

Rules (protocol SD-11): units run in the manifest's global order; only MISSING_INFRASTRUCTURE attempts are retried, up
to ``max_infrastructure_retries`` times, with the same unit identity; a unit with a final non-infrastructure record is
never re-run (resume cannot duplicate a scientific unit); an attempt that started but never finished (process killed)
is closed on resume as an ``infrastructure_interruption`` record; a protocol violation halts the whole queue.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from rq1.experiment.persistence import ExperimentLock, atomic_write_json, durable_append_jsonl
from rq1.five_arm import readiness
from rq1.five_arm.executor import ProtocolViolation
from rq1.five_arm.measurements import validate_diagnostic_record, validate_record

KINDS = {"five_arm": validate_record, "diagnostic": validate_diagnostic_record}


class QueueError(RuntimeError):
    pass


class QueueHalted(QueueError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class RunQueue:
    def __init__(self, *, output_root: Path, manifest: Mapping[str, Any], executor: Callable[..., dict[str, Any]],
                 kind: str, max_infrastructure_retries: int, synthetic: bool,
                 interruption_record: Callable[[Mapping[str, Any], int, str], dict[str, Any]]) -> None:
        if kind not in KINDS:
            raise QueueError(f"kind must be one of {sorted(KINDS)}")
        if bool(manifest.get("synthetic")) != synthetic:
            raise QueueError("manifest synthetic flag differs from the queue's execution mode (real and synthetic never mix)")
        if not synthetic and (manifest.get("status") != "frozen"
                              or not readiness.valid_approval(manifest.get("approval"), manifest.get("manifest_sha256"))):
            raise QueueError("real execution needs a frozen manifest with a human approval bound to its hash")
        if max_infrastructure_retries < 0:
            raise QueueError("retry limit must be non-negative")
        self.manifest, self.executor, self.kind, self.synthetic = dict(manifest), executor, kind, synthetic
        self.max_retries = int(max_infrastructure_retries)
        self.interruption_record = interruption_record
        self.root = Path(output_root) / str(manifest["run_id"])
        self.units = sorted(manifest["units"], key=lambda u: int(u["global_order"]))
        keys = [u["unit_key"] for u in self.units]
        if len(set(keys)) != len(keys):
            raise QueueError("duplicate unit identities in the manifest")

    # -- state ------------------------------------------------------------------------------------------------------
    def _prepare(self) -> None:
        path = self.root / "manifest.json"
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("manifest_sha256") != self.manifest.get("manifest_sha256"):
                raise QueueError("this run directory belongs to a different manifest; refusing to mix runs")
        else:
            atomic_write_json(path, self.manifest)

    def records(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.root / "records.jsonl")

    def _attempts(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for record in self.records():
            out.setdefault(str(record["unit_key"]), []).append(record)
        for rows in out.values():
            rows.sort(key=lambda r: int(r["attempt"]))
        return out

    def _close_stale_attempts(self) -> None:
        journal = _read_jsonl(self.root / "journal.jsonl")
        started = {(e["unit_key"], int(e["attempt"])) for e in journal if e["event"] == "attempt_started"}
        finished = {(r["unit_key"], int(r["attempt"])) for r in self.records()}
        by_key = {u["unit_key"]: u for u in self.units}
        for unit_key, attempt in sorted(started - finished):
            record = self.interruption_record(by_key[unit_key], attempt,
                                              "attempt started but never finished (process interrupted); closed on resume")
            self._append(record)
            durable_append_jsonl(self.root / "journal.jsonl", {"event": "interrupted_attempt", "unit_key": unit_key, "attempt": attempt})

    def _append(self, record: Mapping[str, Any]) -> None:
        problems = KINDS[self.kind](record)
        if problems:
            raise QueueError(f"executor produced an invalid record: {'; '.join(problems)}")
        if bool(record.get("synthetic")) != self.synthetic:
            raise QueueError("record synthetic flag differs from the queue's execution mode")
        durable_append_jsonl(self.root / "records.jsonl", record)

    def next_attempt(self, unit: Mapping[str, Any], attempts: Mapping[str, list[dict[str, Any]]]) -> int | None:
        done = attempts.get(unit["unit_key"], [])
        if not done:
            return 1
        last = done[-1]
        if last["outcome_status"] != "MISSING_INFRASTRUCTURE":
            return None
        if len(done) >= self.max_retries + 1:
            return None
        return int(last["attempt"]) + 1

    # -- execution ----------------------------------------------------------------------------------------------------
    def run(self, *, max_attempts: int | None = None, stop_requested: Callable[[], bool] = lambda: False) -> dict[str, Any]:
        executed = 0
        with ExperimentLock(self.root / ".queue.lock"):
            self._prepare()
            self._close_stale_attempts()
            for unit in self.units:
                while True:
                    if stop_requested() or (max_attempts is not None and executed >= max_attempts):
                        return self.summary(stopped=True)
                    attempt = self.next_attempt(unit, self._attempts())
                    if attempt is None:
                        break
                    attempt_dir = self.root / "attempts" / str(unit["unit_key"])[:24] / f"a{attempt}"
                    if attempt_dir.exists():
                        raise QueueError(f"attempt directory already exists (never reused): {attempt_dir}")
                    durable_append_jsonl(self.root / "journal.jsonl", {"event": "attempt_started", "unit_key": unit["unit_key"],
                                                                      "attempt": attempt})
                    try:
                        record = self.executor({**unit, "run_id": self.manifest["run_id"]}, attempt=attempt, attempt_dir=attempt_dir)
                    except ProtocolViolation as exc:
                        durable_append_jsonl(self.root / "journal.jsonl", {"event": "protocol_violation", "unit_key": unit["unit_key"],
                                                                          "attempt": attempt, "detail": str(exc)[:1000]})
                        raise QueueHalted(f"protocol violation in {unit['unit_key']} attempt {attempt}: {exc}") from exc
                    if record["unit_key"] != unit["unit_key"] or int(record["attempt"]) != attempt \
                            or record["condition"] != unit["condition"] or int(record["seed"]) != int(unit["seed"]):
                        raise QueueError("executor record does not match the scheduled unit")
                    self._append(record)
                    durable_append_jsonl(self.root / "journal.jsonl", {"event": "attempt_finished", "unit_key": unit["unit_key"],
                                                                      "attempt": attempt, "outcome_status": record["outcome_status"]})
                    executed += 1
        return self.summary(stopped=False)

    def summary(self, *, stopped: bool) -> dict[str, Any]:
        attempts = self._attempts()
        final = {k: v[-1] for k, v in attempts.items()}
        pending = [u["unit_key"] for u in self.units if self.next_attempt(u, attempts) is not None]
        statuses: dict[str, int] = {}
        for record in final.values():
            statuses[record["outcome_status"]] = statuses.get(record["outcome_status"], 0) + 1
        return {"run_id": self.manifest["run_id"], "kind": self.kind, "synthetic": self.synthetic, "stopped_early": stopped,
                "units": len(self.units), "units_with_records": len(final), "pending_units": len(pending),
                "attempts": sum(len(v) for v in attempts.values()), "final_status_counts": dict(sorted(statuses.items()))}

    def validate(self) -> dict[str, Any]:
        """Post-run validation: record schema, unit identity, attempt numbering, retry rule, completeness."""
        problems: list[str] = []
        planned = {u["unit_key"]: u for u in self.units}
        attempts = self._attempts()
        for unit_key, rows in attempts.items():
            if unit_key not in planned:
                problems.append(f"record for an unplanned unit {unit_key}")
                continue
            if [int(r["attempt"]) for r in rows] != list(range(1, len(rows) + 1)):
                problems.append(f"{unit_key}: attempts not numbered 1..n")
            if len(rows) > self.max_retries + 1:
                problems.append(f"{unit_key}: more attempts than the retry limit")
            if any(r["outcome_status"] != "MISSING_INFRASTRUCTURE" for r in rows[:-1]):
                problems.append(f"{unit_key}: a non-infrastructure outcome was retried")
            if sum(r["outcome_status"] == "VALID" for r in rows) > 1:
                problems.append(f"{unit_key}: duplicate scientific outcome")
            for r in rows:
                problems += [f"{unit_key}/a{r['attempt']}: {p}" for p in KINDS[self.kind](r)]
        missing = sorted(set(planned) - set(attempts))
        journal = _read_jsonl(self.root / "journal.jsonl")
        violations = [e for e in journal if e["event"] == "protocol_violation"]
        return {"ok": not problems and not violations, "problems": problems, "units_without_records": len(missing),
                "protocol_violations": len(violations),
                "complete": not missing and all(self.next_attempt(u, attempts) is None for u in self.units)}

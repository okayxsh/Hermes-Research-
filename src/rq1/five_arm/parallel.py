"""Isolated multi-lane (four-GPU) execution for the seven-condition protocol (DRAFT; never run locally on a model).

Architecture:

* ONE authoritative ``Coordinator`` per run directory (process-exclusive lock).  It owns the durable journal and the
  records; workers never write records themselves.
* Frozen, balanced ``lane assignment`` bound to the manifest hash: every unit of a task-seed cell (all conditions of
  that task and seed) is assigned to the SAME lane, so within-cell comparisons share one GPU and one provider endpoint.
* Durable attempt CLAIMS: a claim is journalled (fsync) with a random claim token before a worker may execute; a result
  is accepted only with the active token of that exact unit and attempt.
* Safe restart: on start the coordinator verifies the record hash chain (corruption halts) and REVOKES every claim left
  open by a previous coordinator process, recording an ``infrastructure_interruption`` attempt; results later submitted
  with revoked or expired tokens are rejected as LATE.  A second result for a completed claim is rejected as DUPLICATE.
* Leases: a claim without a heartbeat for ``lease_seconds`` expires (worker loss) and is closed the same way.
* Retry policy unchanged: only MISSING_INFRASTRUCTURE attempts are retried, up to ``max_infrastructure_retries``.
* Mandatory halts (persistent HALTED.json; no further claims until a human clears it with a recorded reason): CPU
  fallback or partial GPU offload, runtime drift (model digest / runtime identity), provider token discrepancy,
  provenance inconsistent with the lane, schema-invalid results, protocol violations, evidence corruption.
  Results that trigger a halt are QUARANTINED (kept verbatim for review), except token-discrepancy results, which are
  valid INVALID_MEASUREMENT evidence and are recorded before halting.
* Worker isolation: each lane worker is a separate process with its own provider endpoint (``CUDA_VISIBLE_DEVICES``
  pinned to one GPU), its own episode driver / bridge, and per-attempt directories (fresh Hermes workspace per session).
  Workers talk to the coordinator through an atomic file transport (``inbox/`` -> ``outbox/``).
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rq1.experiment.persistence import ExperimentLock, atomic_write_json, durable_append_jsonl
from rq1.five_arm.executor import ProtocolViolation

ASSIGNMENT_SCHEMA = "rq1-seven-condition-lane-assignment-v1"
PROVENANCE_REQUIRED = ("mode", "lane_id", "worker_id", "endpoint", "gpu_uuid", "cuda_visible_devices", "provider_version",
                       "model_digest", "processor_before", "processor_after", "host", "pid", "hermes_workspace",
                       "claim_token_sha256")
TOKEN_HALT_PREFIXES = ("provider_token_discrepancy", "provider_context_limit_reached", "provider_prompt_eval_count_missing")


class CoordinatorHalted(RuntimeError):
    pass


class ClaimRejected(RuntimeError):
    pass


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- frozen lane assignment -------------------------------------------------------------------------------------------
def assign_lanes(manifest: Mapping[str, Any], lanes: Sequence[Mapping[str, Any]], *, salt: str) -> dict[str, Any]:
    """Balanced, deterministic, outcome-blind assignment of task-seed cells to lanes (all units of a cell together)."""
    if len(salt) < 16:
        raise ValueError("the lane-assignment salt must have at least 16 characters")
    ids = [str(lane["lane_id"]) for lane in lanes]
    if not ids or len(set(ids)) != len(ids) or any(not lane.get("endpoint") or not lane.get("gpu_uuid") for lane in lanes):
        raise ValueError("lanes need unique lane_id, an endpoint and a gpu_uuid")
    if len({lane["endpoint"] for lane in lanes}) != len(lanes) or len({lane["gpu_uuid"] for lane in lanes}) != len(lanes):
        raise ValueError("every lane needs its own provider endpoint and its own GPU")
    cells: dict[str, list[str]] = defaultdict(list)
    for unit in manifest["units"]:
        cells[f"{unit['task_id']}|{unit['seed']}"].append(unit["unit_key"])
    order = sorted(cells, key=lambda c: hashlib.sha256(f"{salt}|{c}".encode()).hexdigest())
    load = {lane_id: 0 for lane_id in ids}
    cell_lane: dict[str, str] = {}
    for cell in order:
        lane_id = min(ids, key=lambda l: (load[l], ids.index(l)))
        cell_lane[cell] = lane_id
        load[lane_id] += len(cells[cell])
    body = {"schema": ASSIGNMENT_SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
            "lanes": [dict(lane) for lane in lanes], "salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
            "cells": dict(sorted(cell_lane.items())),
            "unit_lane": {key: cell_lane[cell] for cell, keys in cells.items() for key in keys},
            "lane_unit_counts": load, "rule": "hash-ordered task-seed cells, each to the currently least-loaded lane"}
    body["assignment_sha256"] = _sha(body)
    return body


def validate_assignment(assignment: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    problems = []
    if assignment.get("schema") != ASSIGNMENT_SCHEMA:
        problems.append("not a lane assignment")
    if assignment.get("manifest_sha256") != manifest.get("manifest_sha256"):
        problems.append("assignment is bound to a different manifest")
    body = {k: v for k, v in assignment.items() if k != "assignment_sha256"}
    if assignment.get("assignment_sha256") != _sha(body):
        problems.append("assignment hash does not match its contents")
    unit_lane = assignment.get("unit_lane", {})
    for unit in manifest.get("units", []):
        if unit["unit_key"] not in unit_lane:
            problems.append(f"unit {unit['unit_key'][:12]} has no lane")
        elif unit_lane[unit["unit_key"]] != assignment["cells"].get(f"{unit['task_id']}|{unit['seed']}"):
            problems.append(f"unit {unit['unit_key'][:12]} is not on its task-seed cell's lane")
    return problems


# --- coordinator --------------------------------------------------------------------------------------------------------
class Coordinator:
    def __init__(self, root: Path, *, manifest: Mapping[str, Any], assignment: Mapping[str, Any], kind_validator: Callable,
                 interruption_record: Callable[[Mapping[str, Any], int, str], dict[str, Any]], max_infrastructure_retries: int,
                 expected_runtime_identity_sha256: str, expected_model_digest: str, synthetic: bool,
                 lease_seconds: float = 900.0, clock: Callable[[], float] = time.time) -> None:
        problems = validate_assignment(assignment, manifest)
        if problems:
            raise ValueError("; ".join(problems))
        if bool(manifest.get("synthetic")) != synthetic:
            raise ValueError("manifest synthetic flag differs from the coordinator mode (real and synthetic never mix)")
        self.root, self.manifest, self.assignment = Path(root), dict(manifest), dict(assignment)
        self.validator, self.interruption_record = kind_validator, interruption_record
        self.max_retries, self.lease, self.clock, self.synthetic = int(max_infrastructure_retries), float(lease_seconds), clock, synthetic
        self.expected_runtime, self.expected_digest = expected_runtime_identity_sha256, expected_model_digest
        self.units = {u["unit_key"]: u for u in manifest["units"]}
        self.lanes = {str(l["lane_id"]): dict(l) for l in assignment["lanes"]}
        self.lane_units: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for unit in sorted(manifest["units"], key=lambda u: int(u["global_order"])):
            self.lane_units[assignment["unit_lane"][unit["unit_key"]]].append(unit)
        self._lock = ExperimentLock(self.root / ".coordinator.lock")
        self.claims: dict[str, dict[str, Any]] = {}
        self.chain = "0" * 64

    # -- lifecycle ----------------------------------------------------------------------------------------------------
    def start(self) -> "Coordinator":
        self._lock.__enter__()
        try:
            self._bind_once("manifest.json", self.manifest, "manifest_sha256")
            self._bind_once("assignment.json", self.assignment, "assignment_sha256")
            self._replay()
            self._verify_evidence()
            for token, claim in list(self.claims.items()):
                if claim["status"] == "active":
                    self._close(token, "revoked_on_restart",
                                "claim left open by a previous coordinator process; revoked on restart")
        except BaseException:
            self._lock.__exit__(None, None, None)
            raise
        return self

    def stop(self) -> None:
        self._lock.__exit__(None, None, None)

    def _bind_once(self, name: str, value: Mapping[str, Any], key: str) -> None:
        path = self.root / name
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8")).get(key) != value.get(key):
                raise ValueError(f"{name} in this run directory differs; refusing to mix runs")
        else:
            atomic_write_json(path, value)

    def _journal(self, event: str, **payload: Any) -> None:
        durable_append_jsonl(self.root / "journal.jsonl", {"event": event, "at": self.clock(), **payload})

    def _replay(self) -> None:
        for e in _read_jsonl(self.root / "journal.jsonl"):
            if e["event"] == "claimed":
                self.claims[e["token"]] = {"unit_key": e["unit_key"], "attempt": e["attempt"], "lane_id": e["lane_id"],
                                           "worker_id": e["worker_id"], "status": "active", "last_seen": e["at"]}
            elif e["event"] == "heartbeat" and e["token"] in self.claims:
                self.claims[e["token"]]["last_seen"] = e["at"]
            elif e["event"] in ("accepted", "revoked_on_restart", "lease_expired", "quarantined", "violation"):
                if e.get("token") in self.claims:
                    self.claims[e["token"]]["status"] = {"accepted": "completed"}.get(e["event"], e["event"])
                if e["event"] in ("accepted", "revoked_on_restart", "lease_expired"):
                    self.chain = e["chain"]

    def _verify_evidence(self) -> None:
        accepted = [e for e in _read_jsonl(self.root / "journal.jsonl") if e["event"] in ("accepted", "revoked_on_restart", "lease_expired")]
        records = _read_jsonl(self.root / "records.jsonl")
        chain = "0" * 64
        corruption = None
        if len(records) != len(accepted):
            corruption = "evidence corruption: records.jsonl and the journal disagree on the number of records"
        for event, record in zip(accepted, records):
            digest = _sha(record)
            chain = hashlib.sha256((chain + digest).encode()).hexdigest()
            if corruption is None and (digest != event["record_sha256"] or chain != event["chain"]):
                corruption = f"evidence corruption: record for {event['unit_key'][:12]} a{event['attempt']} fails its hash chain"
        if corruption:
            self._halt(corruption)
            raise CoordinatorHalted(corruption)

    def _append_record(self, record: Mapping[str, Any]) -> tuple[str, str]:
        durable_append_jsonl(self.root / "records.jsonl", record)
        digest = _sha(record)
        self.chain = hashlib.sha256((self.chain + digest).encode()).hexdigest()
        return digest, self.chain

    # -- halts ----------------------------------------------------------------------------------------------------------
    def halted(self) -> dict[str, Any] | None:
        path = self.root / "HALTED.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _halt(self, reason: str, **detail: Any) -> None:
        if not self.halted():
            atomic_write_json(self.root / "HALTED.json", {"reason": reason, "at": self.clock(), **detail})
            self._journal("halted", reason=reason)

    def clear_halt(self, *, reviewed_by: str, note: str) -> None:
        """Operational acknowledgement after human review (recorded; not a scientific approval)."""
        if not reviewed_by.strip() or not note.strip():
            raise ValueError("clearing a halt needs a reviewer and a note")
        state = self.halted()
        if state:
            self._journal("halt_cleared", reason=state["reason"], reviewed_by=reviewed_by, note=note)
            (self.root / "HALTED.json").replace(self.root / f"HALTED-cleared-{uuid.uuid4().hex[:8]}.json")

    def _quarantine(self, token: str, record: Mapping[str, Any], reason: str) -> None:
        atomic_write_json(self.root / "quarantine" / f"{hashlib.sha256(token.encode()).hexdigest()[:16]}.json",
                          {"reason": reason, "record": dict(record)})
        if token in self.claims:
            self.claims[token]["status"] = "quarantined"
        self._journal("quarantined", token=token, reason=reason)

    # -- attempts -------------------------------------------------------------------------------------------------------
    def _attempts(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in _read_jsonl(self.root / "records.jsonl"):
            out[record["unit_key"]].append(record)
        return out

    def _next_attempt(self, unit_key: str, attempts: Mapping[str, list[dict[str, Any]]]) -> int | None:
        done = sorted(attempts.get(unit_key, []), key=lambda r: int(r["attempt"]))
        if not done:
            return 1
        if done[-1]["outcome_status"] != "MISSING_INFRASTRUCTURE" or len(done) >= self.max_retries + 1:
            return None
        return int(done[-1]["attempt"]) + 1

    def _close(self, token: str, event: str, detail: str) -> None:
        claim = self.claims[token]
        record = self.interruption_record(self.units[claim["unit_key"]], int(claim["attempt"]), detail)
        digest, chain = self._append_record(record)
        claim["status"] = event
        self._journal(event, token=token, unit_key=claim["unit_key"], attempt=claim["attempt"], record_sha256=digest, chain=chain)

    def expire_leases(self) -> int:
        now, expired = self.clock(), 0
        for token, claim in list(self.claims.items()):
            if claim["status"] == "active" and now - claim["last_seen"] > self.lease:
                self._close(token, "lease_expired", f"no heartbeat for more than {self.lease:.0f}s (worker loss); lease expired")
                expired += 1
        return expired

    def claim(self, lane_id: str, worker_id: str) -> dict[str, Any] | None:
        if self.halted():
            raise CoordinatorHalted(self.halted()["reason"])
        if lane_id not in self.lanes:
            raise ClaimRejected(f"unknown lane {lane_id!r}")
        self.expire_leases()
        active = {c["unit_key"] for c in self.claims.values() if c["status"] == "active"}
        attempts = self._attempts()
        for unit in self.lane_units[lane_id]:
            if unit["unit_key"] in active:
                continue
            attempt = self._next_attempt(unit["unit_key"], attempts)
            if attempt is None:
                continue
            token = secrets.token_hex(16)
            self.claims[token] = {"unit_key": unit["unit_key"], "attempt": attempt, "lane_id": lane_id, "worker_id": worker_id,
                                  "status": "active", "last_seen": self.clock()}
            self._journal("claimed", token=token, unit_key=unit["unit_key"], attempt=attempt, lane_id=lane_id, worker_id=worker_id)
            return {"token": token, "unit": {**unit, "run_id": self.manifest["run_id"]}, "attempt": attempt,
                    "lane": self.lanes[lane_id]}
        return None

    def heartbeat(self, token: str) -> None:
        claim = self.claims.get(token)
        if claim is None or claim["status"] != "active":
            raise ClaimRejected("heartbeat for an unknown or closed claim")
        claim["last_seen"] = self.clock()
        self._journal("heartbeat", token=token)

    def report_violation(self, token: str, detail: str) -> None:
        if token in self.claims:
            self.claims[token]["status"] = "violation"
        self._journal("violation", token=token, detail=detail[:1000])
        self._halt(f"protocol violation reported by a worker: {detail[:300]}")

    def submit(self, token: str, record: Mapping[str, Any]) -> dict[str, Any]:
        claim = self.claims.get(token)
        if claim is None:
            self._journal("rejected", token_sha256=hashlib.sha256(token.encode()).hexdigest(), reason="unknown claim token")
            raise ClaimRejected("unknown claim token")
        if claim["status"] == "completed":
            self._journal("rejected", token=token, reason="duplicate result")
            raise ClaimRejected("duplicate result for an already accepted claim")
        if claim["status"] != "active":
            self._journal("rejected", token=token, reason=f"late result ({claim['status']})")
            raise ClaimRejected(f"late result: the claim was {claim['status']}")
        unit = self.units[claim["unit_key"]]
        if (record.get("unit_key"), int(record.get("attempt", -1)), record.get("condition"), int(record.get("seed", -1))) != \
                (unit["unit_key"], int(claim["attempt"]), unit["condition"], int(unit["seed"])):
            self._quarantine(token, record, "result does not match the claimed unit")
            self._halt("integrity: a result did not match its claimed unit")
            raise CoordinatorHalted("integrity: result does not match the claim")
        problems = list(self.validator(record))
        problems += self._provenance_problems(token, claim, record)
        if problems:
            self._quarantine(token, record, "; ".join(problems))
            self._halt("result failed validation or lane provenance", problems=problems[:10])
            raise CoordinatorHalted("; ".join(problems[:3]))
        if bool(record.get("synthetic")) != self.synthetic:
            self._quarantine(token, record, "synthetic flag mismatch")
            self._halt("real and synthetic records mixed")
            raise CoordinatorHalted("real and synthetic records mixed")
        digest, chain = self._append_record(record)
        claim["status"] = "completed"
        self._journal("accepted", token=token, unit_key=claim["unit_key"], attempt=claim["attempt"], record_sha256=digest, chain=chain)
        detail = str(record.get("failure_detail") or "")
        if detail.startswith(TOKEN_HALT_PREFIXES):
            self._halt(f"provider token discrepancy: {detail[:200]}")
        return {"accepted": True, "record_sha256": digest}

    def _provenance_problems(self, token: str, claim: Mapping[str, Any], record: Mapping[str, Any]) -> list[str]:
        prov = record.get("execution_provenance") or {}
        lane = self.lanes[claim["lane_id"]]
        problems = [f"provenance lacks {k}" for k in PROVENANCE_REQUIRED if prov.get(k) in (None, "")]
        if problems:
            return problems
        if prov["mode"] != "parallel" or prov["lane_id"] != claim["lane_id"] or prov["worker_id"] != claim["worker_id"]:
            problems.append("provenance lane/worker differs from the claim")
        if prov["endpoint"] != lane["endpoint"] or prov["gpu_uuid"] != lane["gpu_uuid"] or prov["cuda_visible_devices"] != lane["gpu_uuid"]:
            problems.append("provenance endpoint/GPU differs from the frozen lane")
        if prov["claim_token_sha256"] != hashlib.sha256(token.encode()).hexdigest():
            problems.append("provenance claim token differs")
        if prov["processor_before"] != "gpu_full" or prov["processor_after"] != "gpu_full":
            problems.append("CPU fallback or partial GPU offload detected")
        if prov["model_digest"] != self.expected_digest or record.get("runtime_identity_sha256") != self.expected_runtime:
            problems.append("runtime drift: model digest or runtime identity differs from the approved runtime")
        return problems

    # -- reporting ------------------------------------------------------------------------------------------------------
    def records(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.root / "records.jsonl")

    def summary(self) -> dict[str, Any]:
        attempts = self._attempts()
        pending = [k for k in self.units if self._next_attempt(k, attempts) is not None]
        final = Counter(sorted(v, key=lambda r: int(r["attempt"]))[-1]["outcome_status"] for v in attempts.values())
        return {"units": len(self.units), "pending_units": len(pending), "halted": self.halted(),
                "final_status_counts": dict(sorted(final.items())),
                "active_claims": sum(c["status"] == "active" for c in self.claims.values())}

    def validate(self) -> dict[str, Any]:
        problems = []
        attempts = self._attempts()
        for unit_key, rows in attempts.items():
            rows = sorted(rows, key=lambda r: int(r["attempt"]))
            if unit_key not in self.units:
                problems.append(f"record for an unplanned unit {unit_key[:12]}")
            if [int(r["attempt"]) for r in rows] != list(range(1, len(rows) + 1)):
                problems.append(f"{unit_key[:12]}: attempts not numbered 1..n")
            if any(r["outcome_status"] != "MISSING_INFRASTRUCTURE" for r in rows[:-1]):
                problems.append(f"{unit_key[:12]}: a non-infrastructure outcome was retried")
            if sum(r["outcome_status"] == "VALID" for r in rows) > 1:
                problems.append(f"{unit_key[:12]}: duplicate scientific outcome")
        return {"ok": not problems and not self.halted(), "problems": problems,
                "complete": all(self._next_attempt(k, attempts) is None for k in self.units)}



# --- worker side --------------------------------------------------------------------------------------------------------
def ollama_lane_provenance(request: Callable[[str, str], Mapping[str, Any]], endpoint: str, model_tag: str) -> dict[str, Any]:
    """Metadata-only snapshot of one provider endpoint (/api/version, /api/ps).  ``processor`` is ``gpu_full`` only when
    the loaded model is entirely in VRAM; anything else is CPU fallback or partial offload."""
    version = request(endpoint, "/api/version")
    loaded = next((m for m in request(endpoint, "/api/ps").get("models", []) if m.get("name") == model_tag), None)
    if loaded is None:
        processor = "not_loaded"
    else:
        size, vram = int(loaded.get("size") or 0), int(loaded.get("size_vram") or 0)
        processor = "gpu_full" if size > 0 and vram >= size else ("cpu" if vram == 0 else "gpu_partial")
    return {"provider_version": version.get("version"), "model_digest": (loaded or {}).get("digest"), "processor": processor}


class LaneWorker:
    """One lane: claim -> execute (own driver / endpoint / attempt directories) -> attach provenance -> submit."""

    def __init__(self, client: Any, *, lane_id: str, worker_id: str, executor: Callable[..., dict[str, Any]], root: Path,
                 snapshot: Callable[[], Mapping[str, Any]], heartbeat_seconds: float = 60.0) -> None:
        self.client, self.lane_id, self.worker_id, self.executor = client, lane_id, worker_id, executor
        self.root, self.snapshot, self.heartbeat_seconds = Path(root), snapshot, heartbeat_seconds

    def _heartbeats(self, token: str, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_seconds):
            try:
                self.client.heartbeat(token)
            except (ClaimRejected, CoordinatorHalted, TimeoutError):
                return

    def run_once(self) -> bool:
        claim = self.client.claim(self.lane_id, self.worker_id)
        if claim is None:
            return False
        token, unit, attempt, lane = claim["token"], claim["unit"], int(claim["attempt"]), claim["lane"]
        attempt_dir = self.root / "lanes" / self.lane_id / "attempts" / str(unit["unit_key"])[:24] / f"a{attempt}-{token[:8]}"
        before = self.snapshot()
        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeats, args=(token, stop), daemon=True)
        beat.start()
        try:
            record = self.executor(unit, attempt=attempt, attempt_dir=attempt_dir)
        except ProtocolViolation as exc:
            self.client.report_violation(token, str(exc))
            raise
        finally:
            stop.set()
            beat.join(timeout=5)
        after = self.snapshot()
        record = dict(record)
        record["execution_provenance"] = {
            **dict(record.get("execution_provenance") or {}), "mode": "parallel", "lane_id": self.lane_id,
            "worker_id": self.worker_id, "endpoint": lane["endpoint"], "gpu_uuid": lane["gpu_uuid"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", before.get("cuda_visible_devices")),
            "provider_version": before.get("provider_version"), "model_digest": before.get("model_digest"),
            "processor_before": before.get("processor"), "processor_after": after.get("processor"),
            "model_digest_after": after.get("model_digest"), "host": socket.gethostname(), "pid": os.getpid(),
            "hermes_workspace": str(attempt_dir / "episode"), "claim_token_sha256": hashlib.sha256(token.encode()).hexdigest()}
        if after.get("model_digest") != before.get("model_digest"):
            record["execution_provenance"]["model_digest"] = f"changed:{before.get('model_digest')}->{after.get('model_digest')}"
        self.client.submit(token, record)
        return True


# --- file transport (separate processes) ---------------------------------------------------------------------------------
OPERATIONS = ("claim", "heartbeat", "submit", "report_violation")


def serve_once(coordinator: Coordinator, root: Path) -> int:
    """Process every pending request in ``root/inbox`` (oldest first); write one response each to ``root/outbox``."""
    inbox, outbox = Path(root) / "inbox", Path(root) / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    handled = 0
    for path in sorted(inbox.glob("*.json")) if inbox.is_dir() else []:
        request = json.loads(path.read_text(encoding="utf-8"))
        try:
            if request["op"] not in OPERATIONS:
                raise ClaimRejected(f"unknown operation {request['op']!r}")
            result = getattr(coordinator, request["op"])(*request.get("args", []))
            response = {"ok": True, "result": result}
        except (ClaimRejected, CoordinatorHalted) as exc:
            response = {"ok": False, "error": type(exc).__name__, "detail": str(exc)}
        atomic_write_json(outbox / path.name, response)
        path.unlink()
        handled += 1
    return handled


class FileCoordinatorClient:
    """Worker-side client.  ``pump`` lets tests drive ``serve_once`` in-process; in production the coordinator process
    serves the inbox in a loop."""

    def __init__(self, root: Path, *, timeout: float = 600.0, poll: float = 0.5, pump: Callable[[], Any] | None = None) -> None:
        self.root, self.timeout, self.poll, self.pump = Path(root), timeout, poll, pump

    def _call(self, op: str, *args: Any) -> Any:
        name = f"{time.time_ns():020d}-{uuid.uuid4().hex}.json"
        inbox = self.root / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        tmp = inbox / (name + ".tmp")
        tmp.write_text(json.dumps({"op": op, "args": list(args)}, default=str), encoding="utf-8")
        tmp.replace(inbox / name)
        deadline = time.monotonic() + self.timeout
        out = self.root / "outbox" / name
        while not out.is_file():
            if self.pump is not None:
                self.pump()
            elif time.monotonic() > deadline:
                raise TimeoutError(f"coordinator did not answer {op} within {self.timeout}s")
            else:
                time.sleep(self.poll)
        response = json.loads(out.read_text(encoding="utf-8"))
        out.unlink()
        if not response["ok"]:
            raise {"ClaimRejected": ClaimRejected, "CoordinatorHalted": CoordinatorHalted}[response["error"]](response["detail"])
        return response["result"]

    def claim(self, lane_id: str, worker_id: str):
        return self._call("claim", lane_id, worker_id)

    def heartbeat(self, token: str):
        return self._call("heartbeat", token)

    def submit(self, token: str, record: Mapping[str, Any]):
        return self._call("submit", token, record)

    def report_violation(self, token: str, detail: str):
        return self._call("report_violation", token, detail)

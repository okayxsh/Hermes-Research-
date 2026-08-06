"""Real ALFWorld pilot operations through Fix 1; no random task or fake fallback."""
from __future__ import annotations
import json
from pathlib import Path
from rq1.bridge.environment import RealALFWorldAdapter, real_adapter_capability
from rq1.bridge.episode_manager import EpisodeManager
from rq1.bridge.models import EpisodeStartRequest
from rq1.pilot.gates import validate_task_manifest
from rq1.pilot.models import EvidenceLevel
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked, failed, passed

def _manifest(context: RealExecutionContext, split: str = "valid_seen") -> tuple[dict, dict] | None:
    path = context.root / "data" / "task_lists" / "pilot_seen.json"
    try: payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError): return None
    errors = validate_task_manifest(payload, allowed_splits={split})
    tasks = payload.get("tasks", [])
    return (payload, {"path": str(path.relative_to(context.root)), "errors": errors, "tasks": tasks})

def standalone(context: RealExecutionContext):
    capability = real_adapter_capability()
    if not capability.real_adapter_ready:
        return blocked("alfworld_adapter_unavailable", capability.details, "Run `rq1 alfworld capabilities` and resolve package/data/index prerequisites.", {"handler": "alfworld", "capabilities": capability.to_dict()})
    manifest = _manifest(context)
    if not manifest or manifest[1]["errors"] or not manifest[1]["tasks"]:
        return blocked("pilot_task_manifest_unavailable", "Approved valid_seen pilot tasks are missing or invalid.", "Provide an approved valid_seen task manifest; never use valid_unseen.", {"handler": "alfworld"})
    task = manifest[1]["tasks"][0]
    manager = EpisodeManager(RealALFWorldAdapter, context.output_dir / "bridge")
    try:
        start = manager.start(EpisodeStartRequest(task["task_id"], "valid_seen", 1, 12))
        if not start.admissible_actions: return failed("no_admissible_actions", "Real ALFWorld start exposed no admissible action.", {"handler": "alfworld", "start": start.to_dict()})
        valid = manager.step(start.episode_id, start.admissible_actions[0])
        invalid = manager.step(start.episode_id, "rq1 intentionally invalid action") if not valid.done else valid
        status = manager.status(start.episode_id)
        reset = manager.reset(start.episode_id) if not status.done else None
        aborted = manager.abort(start.episode_id, "pilot standalone cleanup") if reset else None
        return passed(EvidenceLevel.REAL_COMPONENT, {"handler": "alfworld", "operation_executed": True, "task_id": task["task_id"], "start": start.to_dict(), "valid_step": valid.to_dict(), "invalid_step": invalid.to_dict(), "status": status.to_dict(), "reset": reset.to_dict() if reset else None, "abort": aborted.to_dict() if aborted else None})
    except Exception as exc:
        return failed("alfworld_lifecycle_failed", str(exc), {"handler": "alfworld", "task_id": task.get("task_id")})

def trajectory(context: RealExecutionContext, *, complete: bool = False):
    manifest = _manifest(context)
    if not manifest or manifest[1]["errors"] or not manifest[1]["tasks"]:
        return blocked("pilot_task_manifest_unavailable", "Approved valid_seen task manifest is unavailable.", "Supply validated task records with explicit pilot_actions.", {"handler": "alfworld_trajectory"})
    records = manifest[1]["tasks"]
    if any(not isinstance(item.get("pilot_actions"), list) or not item["pilot_actions"] for item in records):
        return blocked("approved_trajectory_missing", "Complete/multi-step pilot actions are absent from the approved task manifest.", "Add explicit validated pilot_actions; the runtime will not invent trajectories.", {"handler": "alfworld_trajectory"})
    if complete and len({item.get("task_family") for item in records}) < 6:
        return blocked("six_family_coverage_missing", "Approved task manifest does not cover all six ALFWorld task families.", "Approve one valid_seen trajectory per indexed family.", {"handler": "alfworld_complete"})
    return blocked("real_trajectory_executor_requires_observed_hermes_dispatch", "Manifest trajectories are present but real Hermes dispatch has not been capability-observed.", "Complete pilot_07 with an installed-version dispatch adapter before integrated execution.", {"handler": "alfworld_complete" if complete else "alfworld_trajectory"})

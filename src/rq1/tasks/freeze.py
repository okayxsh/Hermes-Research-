from __future__ import annotations
import json
from pathlib import Path
from rq1.freeze.validation import git_state, validate_final_gates
from rq1.tasks.models import ManifestState, TaskManifest
from rq1.tasks.validation import manifest_hash, validate_manifest
from rq1.utils.time import utc_now

class TaskFreezeError(RuntimeError): pass

def gate_kind(root: Path, kind: str) -> None:
    if kind == "evaluation":
        gates = validate_final_gates(root)
        if not gates.valid: raise TaskFreezeError("evaluation task manifest is blocked: " + "; ".join(gates.reasons))

def freeze_manifest(root: Path, proposed: TaskManifest, approval: dict, destination: Path) -> TaskManifest:
    if proposed.status != ManifestState.PROPOSED.value: raise TaskFreezeError("only proposed manifests can be frozen")
    if not approval.get("approved_by") or not approval.get("approved_at"): raise TaskFreezeError("approval metadata is required")
    gate_kind(root, proposed.manifest_type)
    commit, clean, error = git_state(root)
    if error or not clean: raise TaskFreezeError("freezing requires a clean committed repository")
    if destination.exists(): raise FileExistsError("refusing to overwrite frozen manifest")
    value = proposed.to_dict(); value.update({"status": ManifestState.FROZEN.value, "repository_commit": commit, "approved_at": str(approval["approved_at"]), "approval_reference": str(approval.get("reference", approval["approved_by"])), "generated_at": utc_now(), "manifest_sha256": ""})
    value["manifest_sha256"] = manifest_hash(value); frozen = TaskManifest(**{**value, "tasks": tuple(proposed.tasks), "exclusions": tuple(proposed.exclusions), "duplicate_resolution": tuple(proposed.duplicate_resolution)})
    errors = validate_manifest(frozen, require_frozen=True)
    if errors: raise TaskFreezeError("cannot freeze invalid manifest: " + "; ".join(errors))
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(frozen.to_dict(), indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return frozen

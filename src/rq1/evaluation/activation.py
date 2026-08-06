"""Immutable, fail-closed authority for final valid_unseen evaluation."""
from __future__ import annotations
import hashlib, json, os
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4
from rq1.freeze.validation import git_state, validate_final_gates
from rq1.tasks.models import TaskManifest, TaskRecord
from rq1.tasks.validation import validate_manifest
from rq1.utils.time import utc_now

REQUIRED_EVIDENCE = ("pilot_report", "evaluation_task_manifest", "acquisition_validation", "snapshot_validation", "profile_validation", "checkpoint_replay", "perturbation", "solvability", "recovery_context", "relevance_rules")

@dataclass(frozen=True)
class EvidenceReference:
    name: str; path: str; sha256: str
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class ActivationManifest:
    schema_version: int; activation_id: str; status: str; activated_at: str; repository_commit: str
    environment_freeze_sha256: str; protocol_freeze_sha256: str; pilot_run_id: str; pilot_report_sha256: str
    model_digest: str; alfworld_data_sha256: str; hermes_version: str; hermes_capability_sha256: str
    evaluation_task_manifest_sha256: str; acquisition_validation_sha256: str; snapshot_set_sha256: str
    recovery_profile_validation_hashes: tuple[str, ...]; checkpoint_set_sha256: str; perturbation_set_sha256: str
    recovery_context_sha256: str; prompt_hashes: dict[str, str]; relevance_rule_sha256: str
    repetition_count: int; action_budget: int; timeout_seconds: int; queue_policy_version: str
    approval_reference: str; approval_file_sha256: str; evidence: tuple[EvidenceReference, ...]; content_sha256: str
    def to_dict(self):
        value=asdict(self); value["evidence"]=[item.to_dict() for item in self.evidence]; return value

class ActivationError(RuntimeError): pass

def _sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()
def _sha(value: object) -> str: return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
def _read(path: Path) -> dict:
    data=json.loads(path.read_text(encoding="utf-8"));
    if not isinstance(data, dict): raise ActivationError(f"evidence must be an object: {path}")
    return data

def _task_manifest(path: Path) -> TaskManifest:
    value=_read(path); value["tasks"]=tuple(TaskRecord(**item) for item in value.get("tasks", [])); value["exclusions"]=tuple(value.get("exclusions", [])); value["duplicate_resolution"]=tuple(value.get("duplicate_resolution", [])); return TaskManifest(**value)

def prerequisite_report(root: Path, evidence_paths: dict[str, str]) -> tuple[list[EvidenceReference], list[str]]:
    reasons=[]; refs=[]; gates=validate_final_gates(root)
    if not gates.valid: reasons.extend(gates.reasons)
    for name in REQUIRED_EVIDENCE:
        raw=evidence_paths.get(name)
        if not isinstance(raw, str): reasons.append(f"missing evidence: {name}"); continue
        path=Path(raw)
        try: payload=_read(path)
        except (OSError, ValueError, ActivationError) as exc: reasons.append(f"invalid evidence {name}: {type(exc).__name__}"); continue
        refs.append(EvidenceReference(name, str(path), _sha_bytes(path.read_bytes())))
        if name == "pilot_report" and (payload.get("mode") != "real" or payload.get("experimental_ready") is not True or payload.get("go_no_go", {}).get("decision") != "go"): reasons.append("pilot report is not a real experimental-ready go")
        if name == "evaluation_task_manifest":
            try: reasons.extend("evaluation task manifest: " + item for item in validate_manifest(_task_manifest(path), require_frozen=True))
            except (TypeError, ValueError) as exc: reasons.append(f"invalid frozen evaluation task manifest: {type(exc).__name__}")
        if name not in {"pilot_report", "evaluation_task_manifest"} and payload.get("valid") is not True and payload.get("status") not in {"passed", "validated"}: reasons.append(f"evidence is not validated: {name}")
    commit, clean, error=git_state(root)
    if error or not clean: reasons.append("repository working tree is not clean")
    return refs, reasons

def build_activation(root: Path, approval: dict, evidence_paths: dict[str, str]) -> ActivationManifest:
    refs, reasons=prerequisite_report(root, evidence_paths)
    if reasons: raise ActivationError("activation blocked: " + "; ".join(reasons))
    required={"approved_by", "approved_at", "reference", "repetition_count", "action_budget", "timeout_seconds", "queue_policy_version"}
    if not required.issubset(approval): raise ActivationError("approval file lacks required activation metadata")
    env=root/"artifacts"/"freezes"/"environment-freeze.json"; protocol=root/"artifacts"/"freezes"/"protocol-freeze.json"
    pilot=_read(Path(evidence_paths["pilot_report"])); env_payload=_read(env); protocol_payload=_read(protocol)
    inputs=env_payload["inputs"]
    if int(approval["repetition_count"]) != int(inputs["repetition_count"]): raise ActivationError("approved repetition count differs from environment freeze")
    values={"schema_version":1,"activation_id":str(uuid4()),"status":"active","activated_at":utc_now(),"repository_commit":env_payload["repository_commit"],"environment_freeze_sha256":_sha_bytes(env.read_bytes()),"protocol_freeze_sha256":_sha_bytes(protocol.read_bytes()),"pilot_run_id":pilot["pilot_run_id"],"pilot_report_sha256":_sha_bytes(Path(evidence_paths["pilot_report"]).read_bytes()),"model_digest":str(inputs["model_digest"]),"alfworld_data_sha256":str(inputs["alfworld_data_sha256"]),"hermes_version":str(inputs["hermes_version"]),"hermes_capability_sha256":_sha(inputs.get("hermes_capabilities", {})),"evaluation_task_manifest_sha256":_sha_bytes(Path(evidence_paths["evaluation_task_manifest"]).read_bytes()),"acquisition_validation_sha256":_sha_bytes(Path(evidence_paths["acquisition_validation"]).read_bytes()),"snapshot_set_sha256":_sha_bytes(Path(evidence_paths["snapshot_validation"]).read_bytes()),"recovery_profile_validation_hashes":(_sha_bytes(Path(evidence_paths["profile_validation"]).read_bytes()),),"checkpoint_set_sha256":_sha_bytes(Path(evidence_paths["checkpoint_replay"]).read_bytes()),"perturbation_set_sha256":_sha_bytes(Path(evidence_paths["perturbation"]).read_bytes()),"recovery_context_sha256":_sha_bytes(Path(evidence_paths["recovery_context"]).read_bytes()),"prompt_hashes":dict(inputs["prompt_hashes"]),"relevance_rule_sha256":_sha_bytes(Path(evidence_paths["relevance_rules"]).read_bytes()),"repetition_count":int(approval["repetition_count"]),"action_budget":int(approval["action_budget"]),"timeout_seconds":int(approval["timeout_seconds"]),"queue_policy_version":str(approval["queue_policy_version"]),"approval_reference":str(approval["reference"]),"approval_file_sha256":_sha(approval),"evidence":tuple(refs),"content_sha256":""}
    hash_payload=dict(values); hash_payload.pop("content_sha256"); hash_payload["evidence"]=[item.to_dict() for item in values["evidence"]]
    values["content_sha256"]=_sha(hash_payload); return ActivationManifest(**values)

def validate_activation(root: Path, manifest: ActivationManifest) -> list[str]:
    values=manifest.to_dict(); expected=values.pop("content_sha256")
    if expected != _sha(values): return ["activation content hash mismatch"]
    if manifest.status != "active": return ["activation is not active"]
    paths={item.name:item.path for item in manifest.evidence}; refs, reasons=prerequisite_report(root, paths)
    if {item.name:item.sha256 for item in refs} != {item.name:item.sha256 for item in manifest.evidence}: reasons.append("referenced evidence hash drift")
    commit, _clean, _error=git_state(root)
    if commit != manifest.repository_commit: reasons.append("repository commit drift")
    return reasons

def invalidation_records(manifest_path: Path) -> tuple[Path, ...]:
    return tuple(sorted((manifest_path.parent / "invalidations").glob("*.json")))

def write_activation(root: Path, manifest: ActivationManifest) -> Path:
    path=root/"artifacts"/"evaluation_activations"/manifest.activation_id/"activation.json"
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True); path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True)+"\n", encoding="utf-8"); return path

def invalidate(root: Path, manifest_path: Path, reason: str) -> Path:
    manifest=_read(manifest_path); destination=manifest_path.parent/"invalidations"/(utc_now().replace(":","-")+".json")
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps({"schema_version":1,"activation_id":manifest.get("activation_id"),"reason":reason,"invalidated_at":utc_now(),"activation_sha256":_sha_bytes(manifest_path.read_bytes())}, indent=2, sort_keys=True)+"\n", encoding="utf-8"); return destination

def log_task_access(manifest_path: Path, task_id: str) -> None:
    with (manifest_path.parent/"valid-unseen-access.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"timestamp":utc_now(),"task_id":task_id,"activation_manifest":manifest_path.name}, sort_keys=True)+"\n")

def require_runtime_opt_in(root: Path, manifest_path: Path) -> ActivationManifest:
    if os.environ.get("RQ1_RUN_FINAL_EVALUATION") != "1": raise ActivationError("final evaluation requires RQ1_RUN_FINAL_EVALUATION=1")
    payload=_read(manifest_path); payload["evidence"]=tuple(EvidenceReference(**item) for item in payload["evidence"]); manifest=ActivationManifest(**payload)
    errors=validate_activation(root, manifest)
    if invalidation_records(manifest_path): errors.append("activation has an invalidation record")
    if errors: raise ActivationError("activation is invalid: " + "; ".join(errors))
    return manifest

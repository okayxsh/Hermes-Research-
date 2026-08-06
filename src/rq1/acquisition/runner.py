"""Queue/validation primitives; live acquisition deliberately remains capability gated."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from rq1.acquisition.models import AcquisitionAttempt, AcquisitionPlan, SkillOperation
from rq1.freeze.validation import validate_final_gates
from rq1.logging.run_registry import Run, RunRegistry

class AcquisitionError(RuntimeError): pass

def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

class AcquisitionRunner:
    def __init__(self, root: Path) -> None:
        self.root = root; self.registry = RunRegistry(root / "state" / "run_registry.sqlite")

    def plan(self, tasks: list[dict[str, object]], run_id: str | None = None) -> AcquisitionPlan:
        if any(item.get("split") != "train" or not isinstance(item.get("task_id"), str) for item in tasks):
            raise AcquisitionError("final acquisition accepts frozen train task records only")
        ids = tuple(sorted(str(item["task_id"]) for item in tasks))
        if len(ids) != len(set(ids)): raise AcquisitionError("frozen acquisition queue contains duplicate task IDs")
        return AcquisitionPlan(run_id or f"acquisition-{uuid4()}", ids)

    def install_plan(self, plan: AcquisitionPlan) -> None:
        for task_id in plan.task_ids:
            self.registry.plan(Run(f"{plan.run_id}:{task_id}", task_id, "train", "acquisition", "rq1-acquisition", 1, "planned"))

    def run(self, plan: AcquisitionPlan) -> dict[str, object]:
        gates = validate_final_gates(self.root)
        if not gates.valid: raise AcquisitionError("final gate blocked: " + "; ".join(gates.reasons))
        # A real Hermes execution adapter must be observed before this method can drive it.
        raise AcquisitionError("real acquisition execution is blocked until a version-specific Hermes session/skill-write adapter is observed")

def validate_history(attempts: list[AcquisitionAttempt], operations: list[SkillOperation]) -> list[str]:
    errors: list[str] = []
    successful = {item.attempt_id: item for item in attempts if item.status == "completed" and item.episode_log}
    seen_tasks: set[str] = set(); seen_skills: set[str] = set(); expected_index = 1
    for item in attempts:
        if item.task_id in seen_tasks and item.status == "completed": errors.append("duplicate successful task execution: " + item.task_id)
        if item.status == "completed": seen_tasks.add(item.task_id)
        if "valid_" in item.task_id: errors.append("evaluation task leakage in acquisition")
    for operation in sorted(operations, key=lambda x: x.operation_index):
        if operation.operation_index != expected_index: errors.append("skill chronology is incomplete")
        expected_index += 1
        source = successful.get(operation.source_attempt_id)
        if source is None or source.task_id != operation.source_task_id: errors.append("skill operation lacks successful source episode")
        if operation.operation not in {"create", "patch"}: errors.append("unapproved skill operation")
        if operation.skill_id in seen_skills: errors.append("duplicate skill operation")
        seen_skills.add(operation.skill_id)
    return errors

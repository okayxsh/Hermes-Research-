"""Chronological train-only acquisition through the durable experiment boundary."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4
from rq1.acquisition.gates import queue_identity_sha256, validate_acquisition_gates
from rq1.acquisition.models import AcquisitionAttempt, AcquisitionPlan, SkillOperation
from rq1.experiment.models import ExperimentUnit, RunExecutor
from rq1.experiment.persistence import ExperimentStore, bind_repository_configuration
from rq1.experiment.runner import DurableExperimentRunner, RunnerOptions
from rq1.logging.run_registry import Run, RunRegistry
from rq1.tasks.models import TaskManifest

PhaseHook = Callable[[Sequence[ExperimentUnit], Mapping[str, Mapping[str, Any]]], Any]

class AcquisitionError(RuntimeError): pass

class AcquisitionRunner:
    def __init__(self, root: Path) -> None:
        self.root = root; self.registry = RunRegistry(root / "state" / "run_registry.sqlite")

    def plan(self, tasks: list[dict[str, object]], run_id: str | None = None) -> AcquisitionPlan:
        if any(item.get("split") != "train" or not isinstance(item.get("task_id"), str) for item in tasks):
            raise AcquisitionError("final acquisition accepts frozen train task records only")
        ids = tuple(sorted(str(item["task_id"]) for item in tasks))
        if len(ids) != len(set(ids)): raise AcquisitionError("frozen acquisition queue contains duplicate task IDs")
        return AcquisitionPlan(run_id or f"acquisition-{uuid4()}", ids)

    def plan_from_manifest(self, manifest: TaskManifest, run_id: str) -> AcquisitionPlan:
        """Preserve the frozen queue order and family of every task.

        An ``acquisition-extension`` queue also carries its parent run and logical
        position offset from the manifest lineage.
        """
        if manifest.manifest_type not in {"acquisition", "acquisition-extension"} or manifest.split != "train":
            raise AcquisitionError("acquisition requires a TRAIN acquisition task manifest")
        extension = manifest.manifest_type == "acquisition-extension"
        lineage = manifest.lineage or {}
        if extension and (not lineage.get("parent_run_id") or not isinstance(lineage.get("logical_index_offset"), int)):
            raise AcquisitionError("an acquisition extension queue requires parent lineage")
        if not extension and manifest.lineage is not None:
            raise AcquisitionError("only an acquisition extension queue may carry parent lineage")
        tasks = sorted(manifest.tasks, key=lambda item: item.order_index)
        if any(task.split != "train" or not task.task_id.startswith("train:") for task in tasks):
            raise AcquisitionError("acquisition queue contains a non-TRAIN task")
        ids = tuple(task.task_id for task in tasks)
        if len(ids) != len(set(ids)): raise AcquisitionError("frozen acquisition queue contains duplicate task IDs")
        return AcquisitionPlan(
            run_id, ids, task_families=tuple(task.family for task in tasks), queue_sha256=queue_identity_sha256(manifest),
            parent_run_id=str(lineage["parent_run_id"]) if extension else None,
            logical_index_offset=int(lineage["logical_index_offset"]) if extension else 0,
        )

    def install_plan(self, plan: AcquisitionPlan) -> None:
        for task_id in plan.task_ids:
            self.registry.plan(Run(f"{plan.run_id}:{task_id}", task_id, "train", "acquisition", "rq1-acquisition", 1, "planned"))

    def run_resumable(
        self,
        plan: AcquisitionPlan,
        executor: RunExecutor,
        *,
        configuration: dict[str, object],
        options: RunnerOptions | None = None,
        output_base: Path | None = None,
        initial_library_hash: str,
        initial_library_size: int = 0,
        task_manifest_path: Path | None = None,
        scientific: bool = True,
        store: ExperimentStore | None = None,
        preflight: PhaseHook | None = None,
        checkpoint_extension: PhaseHook | None = None,
        progress: Callable[[str], None] | None = print,
        gate: Callable[..., Any] | None = None,
    ) -> dict[str, object]:
        """Run the queue through the durable boundary.

        Scientific runs require the approved acquisition freezes, the frozen
        queue, and ``results/final``.  Non-scientific checks are confined to
        ``artifacts/prelaunch`` and must be labelled ``scientific_evidence=false``.
        ``gate`` replaces the initial-acquisition gate for an approved
        continuation (the acquisition extension gate).
        """
        if scientific:
            gates = (gate or validate_acquisition_gates)(self.root, task_manifest_path=task_manifest_path)
            if not gates.valid:
                raise AcquisitionError("acquisition gate blocked: " + "; ".join(gates.reasons))
            if gates.task_manifest is None or plan.queue_sha256 != queue_identity_sha256(gates.task_manifest):
                raise AcquisitionError("scientific acquisition must execute the approved frozen queue")
            if configuration.get("scientific_evidence") is not True:
                raise AcquisitionError("scientific acquisition configuration must set scientific_evidence=true")
            required_base = (self.root / "results" / "final").resolve()
        else:
            if configuration.get("scientific_evidence") is not False:
                raise AcquisitionError("non-scientific acquisition checks must set scientific_evidence=false")
            required_base = (self.root / "artifacts" / "prelaunch").resolve()
        store = store or ExperimentStore(self.root, plan.run_id, base=output_base)
        directory = store.directory.resolve()
        if store.experiment_id != plan.run_id or not directory.is_relative_to(required_base) or (scientific and directory.parent != required_base):
            raise AcquisitionError(f"acquisition outputs must be written under {required_base}")
        units = acquisition_units(
            plan, initial_library_hash=initial_library_hash,
            initial_library_size=initial_library_size,
        )
        return DurableExperimentRunner(
            store, progress=progress, preflight=preflight, checkpoint_extension=checkpoint_extension,
        ).run(
            "acquisition", units,
            bind_repository_configuration(self.root, configuration),
            executor, options,
        )


def acquisition_units(
    plan: AcquisitionPlan,
    *,
    initial_library_hash: str | None = None,
    initial_library_size: int = 0,
) -> list[ExperimentUnit]:
    if plan.task_families and len(plan.task_families) != len(plan.task_ids):
        raise ValueError("acquisition plan task families must align with task IDs")

    def lineage(index: int) -> dict[str, object]:
        if plan.parent_run_id is None:
            return {}
        return {"parent_run_id": plan.parent_run_id, "logical_acquisition_index": plan.logical_index_offset + index}

    return [
        ExperimentUnit(
            phase="acquisition",
            task_id=task_id,
            task_index=index,
            condition="acquisition",
            seed=None,
            identity={
                "phase": "acquisition",
                "task_id": task_id,
                "task_index": index,
                "profile": plan.profile,
                **lineage(index),
            },
            payload={
                "split": plan.split,
                "profile": plan.profile,
                **({"task_family": plan.task_families[index - 1]} if plan.task_families else {}),
                **lineage(index),
            },
            library_name=plan.profile,
            library_size=initial_library_size,
            library_hash=initial_library_hash,
        )
        for index, task_id in enumerate(plan.task_ids, 1)
    ]

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

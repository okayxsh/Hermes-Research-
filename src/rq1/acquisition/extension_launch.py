"""Operational commands of the balanced acquisition extension (Decision 011).

- ``propose``: the 60-task continuation queue proposal and the immutable
  starting-pool snapshot (TRAIN metadata only).
- ``check``/``check-report``: NON-SCIENTIFIC extension checks under
  ``artifacts/prelaunch/acquisition-extension-check``.  They start from the real
  parent pool read-only and never use a task of a scientific queue.
- ``prepare-approvals``: UNAPPROVED extension freeze requests; it never approves.
- ``freeze-tasks``: freezes the queue from a human-approved request.
- ``run``/``resume``/``retry-failed``/``validate``: the approved scientific extension
  under ``results/final/<run-id>``; the parent run directory is never written.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rq1.acquisition.environment import _run, observed_environment, verify_launch_environment
from rq1.acquisition.extension import (
    continuation_problems,
    ensure_starting_pool,
    extension_run_configuration,
    load_parent,
    parent_directory,
    propose_extension_manifest,
    starting_pool_path,
    starting_pool_problems,
    validate_extension_gates,
    validate_extension_queue_manifest,
)
from rq1.acquisition.extension_protocol import (
    EXTENSION_APPROVAL_DIR,
    EXTENSION_CHECK_BASE,
    EXTENSION_CHECK_PREFIX,
    EXTENSION_CHECK_REPORT,
    EXTENSION_DECISION_RECORD,
    EXTENSION_DECISION_RECORDS,
    EXTENSION_FROZEN_DIR,
    EXTENSION_MANIFEST_TYPE,
    EXTENSION_PROPOSAL_ARCHIVE_DIR,
    EXTENSION_PROPOSAL_DIR,
    EXTENSION_TASK_COUNT,
    EXTENSION_TASKS_PER_FAMILY,
    FROZEN_MODEL_DIGEST,
    PARENT,
    ParentReference,
    extension_protocol_definition,
    extension_protocol_sha256,
)
from rq1.acquisition.gates import load_task_manifest, queue_identity_sha256
from rq1.acquisition.launch import (
    CHECK_PREFIX,
    RUNNING_STATES,
    _execute,
    _real_episode_evidence,
    _scientific_queue_task_ids,
    _train_task_family,
    attempt_lineage,
    hard_cap_status,
)
from rq1.acquisition.models import AcquisitionPlan
from rq1.acquisition.protocol import (
    ACQUISITION_ACTION_BUDGET,
    ACQUISITION_MODEL,
    ACQUISITION_TEMPERATURE,
    protocol_sha256,
)
from rq1.acquisition.reporting import write_report
from rq1.acquisition.runner import AcquisitionRunner
from rq1.acquisition.skill_creation import prompt_hashes
from rq1.acquisition.skill_pool import SkillPoolError, pool_hash, rebuild_pool, verify_snapshot
from rq1.bridge.adapters.capabilities import default_data_dir, probe_alfworld_capabilities
from rq1.experiment.models import canonical_hash
from rq1.experiment.persistence import ExperimentStateError, ExperimentStore, atomic_write_json, durable_append_jsonl
from rq1.experiment.runner import RunnerOptions
from rq1.freeze.validation import ACQUISITION_EXTENSION_EVIDENCE_MODE, git_state
from rq1.hermes.episode_driver import (
    ACTION_SELECTION_PROTOCOL,
    INFERENCE_SEED,
    MAX_SELECTION_ATTEMPTS,
    MODEL_CONTEXT_LENGTH,
    MODEL_QUANTIZATION,
    OUTPUT_TOKEN_CAP,
)
from rq1.skills.leakage import find_leakage
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.discovery import discover_tasks
from rq1.tasks.freeze import TaskFreezeError, freeze_manifest
from rq1.tasks.reporting import write_immutable
from rq1.utils.hashing import sha256_file
from rq1.utils.time import utc_now

CHECK_LABEL = "NON-SCIENTIFIC PRELAUNCH ACQUISITION EXTENSION CHECK"
EXTENSION_APPROVAL_REQUEST_NAMES = ("extension-task-freeze", "acquisition-extension-environment", "acquisition-extension-protocol")


def _blocked(**details: Any) -> dict[str, Any]:
    return {"ok": False, "status": "blocked", **details}


def _next_step(status: str, run_id: str) -> str:
    command = "python -m rq1.cli acquisition-extension"
    return {
        "completed": f"{command} validate --run-id {run_id}",
        "paused": f"{command} resume --run-id {run_id} --yes",
        "incomplete": f"{command} resume --run-id {run_id} --yes",
        "interrupted": f"{command} resume --run-id {run_id} --yes",
        "failed": f"fix the infrastructure error in errors.jsonl, then {command} retry-failed --run-id {run_id} --yes",
        "blocked": "manual review of checkpoint.json blocking_error is required; do not edit scientific results",
    }.get(status, "inspect checkpoint.json")


def _per_family(skills: Any) -> dict[str, int]:
    return {family: sum(skill.task_family == family for skill in skills) for family in TASK_FAMILIES}


def propose_extension(root: Path, args: argparse.Namespace, parent: ParentReference = PARENT) -> dict[str, Any]:
    commit, clean, error = git_state(root)
    if error or not clean or not commit:
        return _blocked(reason="an extension proposal requires a clean committed repository")
    state = load_parent(root, parent)
    if state.problems or state.manifest is None:
        return _blocked(reason="the completed parent run is not verified", parent_problems=state.problems)
    pool_path = ensure_starting_pool(root, state, parent)
    data_dir = default_data_dir()
    discovery = discover_tasks(data_dir, "train")
    try:
        manifest = propose_extension_manifest(
            discovery, state.manifest, parent,
            alfworld_version=probe_alfworld_capabilities(data_dir).version, repository_commit=commit,
        )
    except ValueError as exc:
        return _blocked(reason=str(exc))
    errors = [
        *validate_extension_queue_manifest(manifest, state.manifest, parent, require_frozen=False),
        *continuation_problems(discovery, state.manifest, manifest),
    ]
    if errors:
        return _blocked(reasons=errors)
    path = root / EXTENSION_PROPOSAL_DIR / f"{EXTENSION_MANIFEST_TYPE}-{manifest.manifest_sha256[:16]}.json"
    write_immutable(path, manifest.to_dict())
    lineage = manifest.lineage or {}
    return {
        "ok": True,
        "status": "proposed",
        "proposal": str(path),
        "proposal_sha256": sha256_file(path),
        "manifest_sha256": manifest.manifest_sha256,
        "task_queue_sha256": queue_identity_sha256(manifest),
        "actual_count": manifest.actual_count,
        "family_counts": dict(manifest.family_counts),
        "logical_positions": [lineage["units"][0]["logical_acquisition_index"], lineage["units"][-1]["logical_acquisition_index"]],
        "overlap_with_parent_queue": lineage["proof"]["overlap_with_parent_queue"],
        "internal_duplicate_task_ids": lineage["proof"]["internal_duplicate_task_ids"],
        "parent_queue_sha256": parent.queue_sha256,
        "starting_pool": {
            "path": str(pool_path), "sha256": sha256_file(pool_path), "size": len(state.pool),
            "hash": pool_hash(state.pool), "per_family": _per_family(state.pool),
        },
    }


def extension_plan(root: Path, args: argparse.Namespace, parent: ParentReference = PARENT) -> dict[str, Any]:
    manifest_path = Path(args.task_manifest) if getattr(args, "task_manifest", None) else None
    gate = validate_extension_gates(root, task_manifest_path=manifest_path, parent=parent)
    cap = hard_cap_status(root, getattr(args, "run_id", None), [task.family for task in gate.task_manifest.tasks] if gate.task_manifest else [])
    return {
        "ok": True,
        "dry_run": True,
        "launch_permitted": gate.valid and cap["permitted"],
        "gate": gate.to_dict(),
        "hard_cap": cap,
        "parent": parent.to_dict(),
        "extension_protocol_sha256": extension_protocol_sha256(parent),
        "extension_protocol": extension_protocol_definition(parent),
    }


def _os_release() -> str:
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return platform.platform()


def extension_observed_environment(
    root: Path, *, task_queue_sha256: str | None, alfworld_data_identity: str | None, parent: ParentReference = PARENT,
) -> dict[str, Any]:
    """The acquisition environment identity plus the OS, CUDA, and torch stack and parent run."""
    value = observed_environment(root, task_queue_sha256=task_queue_sha256, alfworld_data_identity=alfworld_data_identity)
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", _run(("nvidia-smi",)) or "")
    value.update({
        "os": _os_release(),
        "kernel": platform.release(),
        "cuda_version": cuda.group(1) if cuda else None,
        "torch_version": (value.get("packages") or {}).get("torch"),
        "torch_cuda_version": _run((sys.executable, "-c", "import torch; print(torch.version.cuda)")),
        "parent_run_id": parent.run_id,
        "parent_repository_commit": parent.repository_commit,
    })
    return value


def extension_check(root: Path, args: argparse.Namespace, parent: ParentReference = PARENT) -> dict[str, Any]:
    run_id = str(args.run_id)
    if not run_id.startswith(EXTENSION_CHECK_PREFIX):
        return _blocked(reason=f"non-scientific extension check run IDs must start with {EXTENSION_CHECK_PREFIX}")
    store = ExperimentStore(root, run_id, base=root / EXTENSION_CHECK_BASE)
    plan_path = store.directory / "check-plan.json"
    commit, clean, _error = git_state(root)
    state = load_parent(root, parent)
    if state.problems:
        return _blocked(reason="the completed parent run is not verified", parent_problems=state.problems)
    requested = list(getattr(args, "task_id", None) or [])
    if args.resume:
        if not plan_path.is_file():
            return _blocked(reason="cannot resume an unknown non-scientific extension check")
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if requested and requested != saved["task_ids"]:
            return _blocked(reason="check queue differs from the saved check plan")
        if saved.get("parent_run_id") != parent.run_id or saved.get("starting_pool_hash") != pool_hash(state.pool):
            return _blocked(reason="check parent or starting pool differs from the saved check plan")
        task_ids, families = list(saved["task_ids"]), list(saved["task_families"])
    else:
        if plan_path.exists():
            return _blocked(reason="check already exists; use --resume")
        if not requested:
            return _blocked(reason="at least one --task-id is required")
        overlap = sorted(set(requested) & (_scientific_queue_task_ids(root) | state.queue_task_ids))
        if overlap:
            return _blocked(reason="check tasks overlap a scientific acquisition or extension queue", overlap=overlap)
        families = [_train_task_family(default_data_dir(), task_id) for task_id in requested]
        task_ids = requested
        atomic_write_json(plan_path, {
            "schema_version": 1,
            "label": CHECK_LABEL,
            "scientific_evidence": False,
            "task_ids": task_ids,
            "task_families": families,
            "model_name": ACQUISITION_MODEL,
            "repository_commit": commit,
            "parent_run_id": parent.run_id,
            "starting_pool_size": len(state.pool),
            "starting_pool_hash": pool_hash(state.pool),
            "parent_results_sha256": sha256_file(parent_directory(root, parent) / "results.jsonl"),
            "created_at": utc_now(),
        })
    queue_sha = canonical_hash({"task_ids": task_ids, "task_families": families, "parent_run_id": parent.run_id})
    plan = AcquisitionPlan(
        run_id, tuple(task_ids), task_families=tuple(families), queue_sha256=queue_sha,
        parent_run_id=parent.run_id, logical_index_offset=parent.completed_units,
    )
    durable_append_jsonl(store.directory / "invocations.jsonl", {
        "mode": "resume" if args.resume else "run",
        "repository_commit": commit,
        "clean": clean,
        "model_name": ACQUISITION_MODEL,
        "max_runs": args.max_runs,
        "timestamp": utc_now(),
    })
    result = _execute(
        root, plan, store,
        configuration=extension_run_configuration(root, queue_sha256=queue_sha, scientific=False, parent=parent),
        options=RunnerOptions(resume=bool(args.resume), max_runs=args.max_runs, fail_fast=True),
        scientific=False,
        parent_pool=state.pool,
        parent_run_id=parent.run_id,
    )
    return {"ok": result["status"] in RUNNING_STATES, "label": CHECK_LABEL, "scientific_evidence": False, **result}


def _attempt_directories(store: ExperimentStore, run_key: str) -> int:
    directory = store.logs / "acquisition" / run_key
    return sum(1 for item in directory.iterdir() if item.is_dir()) if directory.is_dir() else 0


def extension_check_report(root: Path, run_id: str, parent: ParentReference = PARENT) -> dict[str, Any]:
    store = ExperimentStore(root, run_id, base=root / EXTENSION_CHECK_BASE)
    plan_path = store.directory / "check-plan.json"
    if not run_id.startswith(EXTENSION_CHECK_PREFIX) or not store.manifest_path.is_file() or not plan_path.is_file():
        return _blocked(reason="unknown non-scientific acquisition extension check")
    check_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = [row for row in store.read_results(repair_tail=False) if row.get("phase") == "acquisition"]
    lineage = attempt_lineage(rows)
    try:
        results = store.terminal_results(phase="acquisition", repair_tail=False)
    except ExperimentStateError:
        results = {}
    records = sorted(results.values(), key=lambda item: int(item.get("task_index", 0)))
    completed = [item for item in records if item.get("status") == "completed"]
    configuration = json.loads((store.manifests / "acquisition.json").read_text(encoding="utf-8")).get("configuration", {})
    runtime = configuration.get("runtime_settings", {})
    bound = configuration.get("extension") or {}
    invocations_path = store.directory / "invocations.jsonl"
    invocations = [json.loads(line) for line in invocations_path.read_text(encoding="utf-8").splitlines() if line.strip()] if invocations_path.is_file() else []
    checkpoint, _source = store.load_checkpoint()
    phase_state = (checkpoint or {}).get("phase_state") or {}
    commit, clean, _error = git_state(root)
    state = load_parent(root, parent)
    try:
        pool = rebuild_pool(records, base=state.pool)
        verify_snapshot(store.directory, pool)
        pool_consistent = not state.problems
    except SkillPoolError:
        pool, pool_consistent = (), False
    appended = pool[len(state.pool):]
    first = records[0] if records else {}
    attempts = Counter(str(row.get("run_key")) for row in rows)
    parent_dir = parent_directory(root, parent).resolve()
    checks = {
        "non_scientific_configuration": configuration.get("scientific_evidence") is False and all(item.get("scientific_evidence") is False for item in completed),
        "parent_run_verified_read_only": not state.problems,
        "parent_results_unchanged": not state.problems and check_plan.get("parent_results_sha256") == sha256_file(parent_dir / "results.jsonl"),
        "check_tasks_outside_scientific_queues": not (set(check_plan.get("task_ids", [])) & (_scientific_queue_task_ids(root) | state.queue_task_ids)),
        "starts_from_exact_parent_pool": bool(records) and first.get("skill_pool_size_before") == parent.pool_size and first.get("skill_pool_hash_before") == parent.pool_hash,
        "first_unit_is_first_extension_position": bool(records) and first.get("task_index") == 1 and first.get("logical_acquisition_index") == parent.first_logical_index,
        "skill_pool_consistent_with_results": pool_consistent,
        "parent_skills_preserved_unchanged": pool_consistent and len(state.pool) == parent.pool_size
            and [skill.identity() for skill in pool[: len(state.pool)]] == [skill.identity() for skill in state.pool],
        "appended_skills_have_extension_provenance": pool_consistent and all(
            skill.pool_index > parent.pool_size
            and skill.provenance.get("parent_run_id") == parent.run_id
            and skill.provenance.get("logical_acquisition_index") == parent.completed_units + skill.source_task_index
            and skill.provenance.get("model") == ACQUISITION_MODEL
            and not find_leakage(skill.text)
            for skill in appended
        ),
        "checkpoint_records_continuation_pool": ((phase_state.get("skill_pool") or {}).get("hash") == pool_hash(pool)
            and (phase_state.get("skill_pool") or {}).get("size") == len(pool)
            and (phase_state.get("continuation") or {}).get("parent_run_id") == parent.run_id
            and (phase_state.get("continuation") or {}).get("starting_pool_hash") == parent.pool_hash),
        "parent_lineage_on_every_result": bool(completed) and all(
            item.get("parent_run_id") == parent.run_id
            and item.get("logical_acquisition_index") == parent.completed_units + int(item.get("task_index", 0))
            for item in completed
        ),
        "results_persisted": bool(completed) and all(item.get("skill_library_hash_after") and item.get("library_size_after") is not None for item in completed),
        "real_hermes_episodes_executed": bool(completed) and all(_real_episode_evidence(store.directory, item) for item in completed),
        "resume_invocation_observed": len(invocations) >= 2 and any(item.get("mode") == "resume" for item in invocations),
        "completed_units_not_replayed": bool(completed) and all(_attempt_directories(store, str(item.get("run_key"))) == attempts[str(item.get("run_key"))] for item in completed),
        "frozen_action_budget_50": runtime.get("acquisition_action_budget") == ACQUISITION_ACTION_BUDGET and all(int(item.get("actions") or 0) <= ACQUISITION_ACTION_BUDGET for item in completed),
        "output_token_cap_2048": runtime.get("output_token_cap") == OUTPUT_TOKEN_CAP,
        "model_context_32768": runtime.get("model_context_length") == MODEL_CONTEXT_LENGTH,
        "temperature_0_seed_42": runtime.get("temperature") == ACQUISITION_TEMPERATURE and runtime.get("seed") == INFERENCE_SEED,
        "action_interface_v3_three_attempts": runtime.get("action_selection_protocol") == ACTION_SELECTION_PROTOCOL and runtime.get("max_selection_attempts") == MAX_SELECTION_ATTEMPTS,
        "exact_model_identity": configuration.get("model_name") == ACQUISITION_MODEL and all(
            (item.get("skill_candidate") or {}).get("model") in (None, ACQUISITION_MODEL) for item in completed
        ),
        "no_scientific_retrieval": bool(completed) and all(item.get("scientific_retrieval_count") == 0 and not item.get("retrieved_skill_ids") for item in completed),
        "authorized_retry_lineage": lineage["authorized"] and bool(results),
        "no_duplicate_completed_units": lineage["duplicate_completed_units"] == 0,
        "no_unresolved_infrastructure_failures": bool(records) and all(item.get("status") == "completed" for item in records),
        "outputs_separate_from_parent": store.directory.resolve() != parent_dir and not store.directory.resolve().is_relative_to(parent_dir),
        "extension_configuration_bound": bound.get("parent_run_id") == parent.run_id
            and bound.get("protocol_sha256") == extension_protocol_sha256(parent)
            and bound.get("starting_pool_hash") == parent.pool_hash
            and bound.get("logical_index_offset") == parent.completed_units,
        "single_clean_commit": bool(clean) and bool(invocations) and all(item.get("repository_commit") == commit and item.get("clean") is True for item in invocations),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "mode": ACQUISITION_EXTENSION_EVIDENCE_MODE,
        "label": CHECK_LABEL,
        "scientific_evidence": False,
        "run_id": run_id,
        "repository_commit": commit,
        "generated_at": utc_now(),
        "passed": passed,
        "checks": checks,
        "observations": {
            "successful_episode_observed": any(item.get("success") is True for item in completed),
            "extension_skill_appended": bool(appended),
            "skill_candidate_statuses": dict(Counter(str((item.get("skill_candidate") or {}).get("status")) for item in completed)),
        },
        "parent": {"run_id": parent.run_id, "starting_pool_size": len(state.pool), "starting_pool_hash": pool_hash(state.pool), "problems": state.problems},
        "units": [
            {
                "task_index": item.get("task_index"),
                "logical_acquisition_index": item.get("logical_acquisition_index"),
                "task_id": item.get("task_id"),
                "task_family": item.get("task_family"),
                "status": item.get("status"),
                "success": item.get("success"),
                "termination_reason": item.get("termination_reason"),
                "actions": item.get("actions"),
                "skill_candidate_status": (item.get("skill_candidate") or {}).get("status"),
                "rejection_reasons": (item.get("skill_candidate") or {}).get("rejection_reasons"),
                "skill_pool_size_before": item.get("skill_pool_size_before"),
                "skill_pool_size_after": item.get("library_size_after"),
            }
            for item in records
        ],
        "skill_pool": {
            "size": len(pool),
            "hash": pool_hash(pool),
            "appended": [
                {"pool_index": skill.pool_index, "skill_id": skill.skill_id, "task_family": skill.task_family, "source_task_id": skill.source_task_id, "text": skill.text}
                for skill in appended
            ],
        },
        "attempt_lineage": lineage,
        "output_directory": str(store.directory),
    }
    path = store.directory / EXTENSION_CHECK_REPORT
    atomic_write_json(path, report)
    return {"ok": passed, "report": str(path), "report_sha256": sha256_file(path), **report}


def prepare_extension_approvals(root: Path, proposal_path: Path, evidence_path: Path, parent: ParentReference = PARENT) -> dict[str, Any]:
    commit, clean, error = git_state(root)
    if error or not clean or not commit:
        return _blocked(reason="approval requests require a clean committed repository")
    state = load_parent(root, parent)
    proposal = load_task_manifest(proposal_path)
    problems = [
        *state.problems,
        *validate_extension_queue_manifest(proposal, state.manifest, parent, require_frozen=False),
        *starting_pool_problems(root, state, parent),
    ]
    if proposal.status != "proposed":
        problems.append("task manifest is not a proposal")
    if proposal.repository_commit != commit:
        problems.append("extension queue proposal was generated at a different commit")
    discovery = discover_tasks(default_data_dir(), "train")
    problems.extend(continuation_problems(discovery, state.manifest, proposal))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("mode") != ACQUISITION_EXTENSION_EVIDENCE_MODE
        or evidence.get("passed") is not True
        or evidence.get("scientific_evidence") is not False
        or evidence.get("repository_commit") != commit
    ):
        problems.append("evidence report is not a passed non-scientific acquisition extension check at this commit")
    if problems:
        return _blocked(reasons=problems)
    directory = root / EXTENSION_APPROVAL_DIR / commit[:12]
    paths = {name: directory / f"{name}.approval.json" for name in EXTENSION_APPROVAL_REQUEST_NAMES}
    if any(path.exists() for path in paths.values()):
        return _blocked(reason=f"approval requests already exist and are immutable: {directory}")
    queue_sha = queue_identity_sha256(proposal)
    lineage = proposal.lineage or {}
    pool_path = starting_pool_path(root, parent)
    closeout = {"path": parent.closeout_manifest, "sha256": parent.closeout_manifest_sha256}
    unapproved = {"status": "UNAPPROVED", "approved_by": None, "approved_at": None, "reference": None}
    how = "A human reviewer sets status to APPROVED and fills approved_by, approved_at (UTC ISO-8601), and reference, then runs the command."
    evidence_reference = {"path": str(evidence_path), "sha256": sha256_file(evidence_path), "run_id": evidence.get("run_id")}
    first, last = parent.first_logical_index, parent.last_logical_index
    documents = {
        "extension-task-freeze": {
            "schema_version": 1,
            "approval_kind": "acquisition-extension-task-freeze",
            **unapproved,
            "subject": {
                "proposal_path": str(proposal_path),
                "proposal_sha256": sha256_file(proposal_path),
                "manifest_sha256": proposal.manifest_sha256,
                "task_queue_sha256": queue_sha,
                "repository_commit": commit,
                "split": proposal.split,
                "actual_count": proposal.actual_count,
                "family_counts": dict(proposal.family_counts),
                "selection_policy": dict(proposal.selection_policy),
                "data_root_identity": proposal.data_root_identity,
                "queue_order": lineage.get("units"),
                "parent_run_id": parent.run_id,
                "parent_queue_sha256": parent.queue_sha256,
                "parent_task_manifest_sha256": parent.task_manifest_sha256,
                "parent_closeout_manifest": closeout,
                "overlap_proof": {
                    **dict(lineage.get("proof") or {}),
                    "recomputed_selection_prefix_equals_parent_queue": True,
                    "extension_is_recomputed_selection_tail": True,
                },
            },
            "attestation": [
                f"The queue is exactly {EXTENSION_TASK_COUNT} ALFWorld 0.4.2 TRAIN tasks, {EXTENSION_TASKS_PER_FAMILY} per RQ1 family, in the recorded order; they are logical acquisition positions {first}-{last}.",
                f"Selection recomputed the frozen task-selection-v1 policy (seed 1, round-robin family balancing) at {last} using TRAIN metadata only; positions 1-{parent.completed_units} reproduce the parent frozen queue exactly and positions {first}-{last} are this queue. No task was chosen by apparent difficulty or outcome; no valid_seen, valid_unseen, or evaluation information was used.",
                f"The queue has no overlap with the {parent.completed_units} parent tasks and no duplicate task.",
                "This activates the frozen later hard cap of the acquisition protocol (240 tasks, 40 per family; Decision 011); acquisition never exceeds 240.",
            ],
            "how_to_approve": how,
            "command": f"python -m rq1.cli acquisition-extension freeze-tasks --proposal {proposal_path} --approval-file {paths['extension-task-freeze']} --yes",
        },
        "acquisition-extension-environment": {
            "schema_version": 1,
            "approval_kind": "acquisition-extension-environment",
            "approval": dict(unapproved),
            "inputs": extension_observed_environment(root, task_queue_sha256=queue_sha, alfworld_data_identity=proposal.data_root_identity, parent=parent),
            "evidence_report": evidence_reference,
            "attestation": [
                f"The recorded extension commit, branch, OS and host, GPU, driver and CUDA, Python, torch, ALFWorld, Hermes, Ollama, {ACQUISITION_MODEL} ({MODEL_QUANTIZATION}, digest {FROZEN_MODEL_DIGEST}), and provider settings (num_predict {OUTPUT_TOKEN_CAP}, num_ctx {MODEL_CONTEXT_LENGTH}, temperature 0, seed {INFERENCE_SEED}, think false) are the approved extension environment.",
                f"Every enforced scientific identity equals the approved environment of the parent run {parent.run_id}; the repository commit and configuration hashes are those of the extension-preparation commit.",
                "Seed and temperature are fixed, but provider/model inference is not claimed to be deterministic (Decision 010).",
                "The referenced NON-SCIENTIFIC extension check passed at this commit and is not scientific data.",
            ],
            "how_to_approve": how,
            "command": f"python -m rq1.cli freeze acquisition-extension-environment --approval-file {paths['acquisition-extension-environment']} --pilot-report {evidence_path} --yes",
        },
        "acquisition-extension-protocol": {
            "schema_version": 1,
            "approval_kind": "acquisition-extension-protocol",
            "approval": dict(unapproved),
            "inputs": {
                "repository_commit": commit,
                "protocol": extension_protocol_definition(parent),
                "protocol_sha256": extension_protocol_sha256(parent),
                "inherited_protocol_sha256": protocol_sha256(),
                "task_queue_sha256": queue_sha,
                "acquisition_action_budget": ACQUISITION_ACTION_BUDGET,
                "inference_seed": INFERENCE_SEED,
                "prompt_hashes": prompt_hashes(root),
                "decision_record_sha256": sha256_file(root / EXTENSION_DECISION_RECORD),
                "decision_records_sha256": {record: sha256_file(root / record) for record in EXTENSION_DECISION_RECORDS},
                "parent_run_id": parent.run_id,
                "parent_pool_size": parent.pool_size,
                "parent_pool_hash": parent.pool_hash,
                "parent_queue_sha256": parent.queue_sha256,
                "parent_closeout_manifest_sha256": parent.closeout_manifest_sha256,
                "starting_pool_path": str(pool_path),
                "starting_pool_sha256": sha256_file(pool_path),
            },
            "evidence_report": evidence_reference,
            "attestation": [
                f"Decision 011 activates the previously approved later hard cap (180 to 240) of the frozen acquisition protocol after the parent run {parent.run_id} completed normally; no final RQ1 evaluation has started and no evaluation outcomes exist; no acquisition result is discarded or rerun.",
                f"{EXTENSION_TASK_COUNT} additional TRAIN episodes, {EXTENSION_TASKS_PER_FAMILY} per family, logical positions {first}-{last}, in a separate run whose starting pool is the exact final {parent.pool_size}-skill parent pool (hash {parent.pool_hash}).",
                f"The inherited scientific protocol (sha256 {protocol_sha256()}) is unchanged: fresh session per task; {ACQUISITION_ACTION_BUDGET}-action cap; full observable within-episode history; verbatim initial observation at every decision; inventory only through the inventory action; exactly one ACTION_INDEX line; three action-selection attempts; {OUTPUT_TOKEN_CAP}-token output cap; the same {ACQUISITION_MODEL} agent writes at most one create-only candidate after success; exact-normalized duplicate rejection only; zero acquisition retrieval.",
                "Persistence is append-only: parent results, checkpoint, and skills are never modified; completed units are never rerun; invalid or capped model output is a scientific outcome, and only genuine execution failures are infrastructure failures, which halt for chronological retry-failed.",
            ],
            "how_to_approve": how,
            "command": f"python -m rq1.cli freeze acquisition-extension-protocol --approval-file {paths['acquisition-extension-protocol']} --pilot-report {evidence_path} --yes",
        },
    }
    for name, document in documents.items():
        write_report(paths[name], document)
    return {
        "ok": True,
        "status": "UNAPPROVED",
        "approval_requests": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "task_queue_sha256": queue_sha,
        "proposal_manifest_sha256": proposal.manifest_sha256,
        "extension_protocol_sha256": extension_protocol_sha256(parent),
        "inherited_protocol_sha256": protocol_sha256(),
    }


def freeze_extension_tasks(root: Path, args: argparse.Namespace, parent: ParentReference = PARENT) -> dict[str, Any]:
    if not getattr(args, "yes", False):
        return _blocked(reason="freezing the extension queue requires --yes")
    proposal_path = Path(args.proposal)
    try:
        proposal = load_task_manifest(proposal_path)
    except (OSError, TypeError, ValueError) as exc:
        return _blocked(reason=f"extension queue proposal is unreadable: {type(exc).__name__}")
    state = load_parent(root, parent)
    problems = [*state.problems, *validate_extension_queue_manifest(proposal, state.manifest, parent, require_frozen=False)]
    if proposal.status != "proposed":
        problems.append("task manifest is not a proposal")
    problems.extend(continuation_problems(discover_tasks(default_data_dir(), "train"), state.manifest, proposal))
    if problems:
        return _blocked(reasons=problems)
    approval = json.loads(Path(args.approval_file).read_text(encoding="utf-8"))
    frozen_path = root / EXTENSION_FROZEN_DIR / f"{EXTENSION_MANIFEST_TYPE}-{proposal.manifest_sha256[:16]}.json"
    try:
        frozen = freeze_manifest(root, proposal, approval, frozen_path)
    except (TaskFreezeError, FileExistsError) as exc:
        return _blocked(reason=str(exc))
    archive = root / EXTENSION_PROPOSAL_ARCHIVE_DIR / proposal_path.name
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(proposal_path, archive)
    return {
        "ok": True,
        "frozen": str(frozen_path),
        "frozen_sha256": sha256_file(frozen_path),
        "archived_proposal": str(archive),
        "manifest_sha256": frozen.manifest_sha256,
        "task_queue_sha256": queue_identity_sha256(frozen),
    }


def extension_run(root: Path, args: argparse.Namespace, *, resume: bool, retry_failed: bool, parent: ParentReference = PARENT) -> dict[str, Any]:
    if not getattr(args, "yes", False):
        return _blocked(reason="the scientific acquisition extension requires --yes")
    run_id = str(args.run_id)
    if run_id == parent.run_id or run_id.startswith((CHECK_PREFIX, EXTENSION_CHECK_PREFIX)):
        return _blocked(reason="the extension run ID must differ from the parent run and from non-scientific check prefixes")
    manifest_path = Path(args.task_manifest) if getattr(args, "task_manifest", None) else None
    gate = validate_extension_gates(root, task_manifest_path=manifest_path, parent=parent)
    if not gate.valid or gate.task_manifest is None or gate.environment is None or gate.protocol is None:
        return _blocked(gate=gate.to_dict())
    cap = hard_cap_status(root, run_id, [task.family for task in gate.task_manifest.tasks])
    if not cap["permitted"]:
        return _blocked(hard_cap=cap)
    drift = verify_launch_environment(root, gate.environment.inputs)
    state = load_parent(root, parent)
    drift.extend(state.problems)
    drift.extend(continuation_problems(discover_tasks(default_data_dir(), "train"), state.manifest, gate.task_manifest))
    if drift:
        return _blocked(environment_drift=drift)
    plan = AcquisitionRunner(root).plan_from_manifest(gate.task_manifest, run_id)
    store = ExperimentStore(root, run_id)
    if plan.parent_run_id != parent.run_id or plan.logical_index_offset != parent.completed_units:
        return _blocked(reason="frozen extension queue lineage differs from the parent reference")
    if store.directory.resolve() == parent_directory(root, parent).resolve():
        return _blocked(reason="extension outputs must be separate from the parent run")
    configuration = extension_run_configuration(
        root,
        queue_sha256=str(plan.queue_sha256),
        scientific=True,
        parent=parent,
        freezes={
            "task_manifest_sha256": gate.task_manifest.manifest_sha256,
            "acquisition-extension-environment": gate.environment.input_fingerprint,
            "acquisition-extension-protocol": gate.protocol.input_fingerprint,
        },
    )
    backup = Path(args.backup_dir) if getattr(args, "backup_dir", None) else None
    options = RunnerOptions(
        resume=resume,
        retry_failed=retry_failed,
        max_runs=getattr(args, "max_runs", None),
        fail_fast=True,
        backup_dir=backup,
        require_backup=bool(getattr(args, "require_backup", False)),
    )

    def extension_gate(gate_root: Path, task_manifest_path: Path | None = None) -> Any:
        return validate_extension_gates(gate_root, task_manifest_path=task_manifest_path, parent=parent)

    result = _execute(
        root, plan, store,
        configuration=configuration, options=options, scientific=True,
        task_manifest_path=gate.task_manifest_path,
        parent_pool=state.pool, parent_run_id=parent.run_id, gate=extension_gate,
    )
    return {"ok": result["status"] in RUNNING_STATES, "scientific_evidence": True, **result, "next": _next_step(result["status"], run_id)}


def validate_extension_run(root: Path, run_id: str, parent: ParentReference = PARENT) -> dict[str, Any]:
    store = ExperimentStore(root, run_id)
    if run_id == parent.run_id or not store.manifest_path.is_file():
        return _blocked(reason="unknown scientific acquisition extension run")
    state = load_parent(root, parent)
    if state.problems:
        return {"ok": False, "status": "invalid", "reason": "the completed parent run is not verified", "parent_problems": state.problems}
    rows = [row for row in store.read_results(repair_tail=False) if row.get("phase") == "acquisition"]
    records = sorted(store.terminal_results(phase="acquisition", repair_tail=False).values(), key=lambda item: int(item.get("task_index", 0)))
    try:
        pool = rebuild_pool(records, base=state.pool)
        verify_snapshot(store.directory, pool)
    except SkillPoolError as exc:
        return {"ok": False, "status": "invalid", "reason": str(exc)}
    completed = [item for item in records if item.get("status") == "completed"]
    appended = pool[len(state.pool):]
    lineage = attempt_lineage(rows)
    problems = []
    if any(not str(item.get("task_id", "")).startswith("train:") for item in records):
        problems.append("non-TRAIN task in extension results")
    if set(item.get("task_id") for item in records) & state.queue_task_ids:
        problems.append("extension results contain a parent queue task")
    if any(item.get("scientific_retrieval_count") != 0 or item.get("retrieved_skill_ids") for item in completed):
        problems.append("scientific retrieval occurred during acquisition")
    if any(item.get("scientific_evidence") is not True for item in completed):
        problems.append("completed result is not marked as scientific evidence")
    if any(item.get("parent_run_id") != parent.run_id or item.get("logical_acquisition_index") != parent.completed_units + int(item.get("task_index", 0)) for item in completed):
        problems.append("completed result lacks the parent lineage")
    if any(skill.provenance.get("parent_run_id") != parent.run_id for skill in appended):
        problems.append("appended skill lacks the parent lineage")
    if lineage["duplicate_completed_units"] or not lineage["authorized"]:
        problems.append("extension results contain a duplicate completed unit or an unauthorized retry")
    return {
        "ok": not problems,
        "status": "valid" if not problems else "invalid",
        "problems": problems,
        "parent_run_id": parent.run_id,
        "terminal": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "successful": sum(item.get("success") is True for item in completed),
        "starting_pool_size": len(state.pool),
        "starting_pool_hash": pool_hash(state.pool),
        "skill_pool_size": len(pool),
        "skill_pool_hash": pool_hash(pool),
        "appended_skills": len(appended),
        "appended_skills_per_family": _per_family(appended),
        "combined_skills_per_family": _per_family(pool),
        "combined_completed_units": parent.completed_units + len(completed),
        "retried_units": len(lineage["retried_units"]),
        "duplicate_completed_units": lineage["duplicate_completed_units"],
    }

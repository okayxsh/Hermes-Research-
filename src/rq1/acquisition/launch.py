"""Operational acquisition commands.

- ``run``/``resume``/``retry-failed``: SCIENTIFIC acquisition under
  ``results/final/<run-id>``, only after approved task, environment, and protocol
  freezes and a matching live environment.
- ``check``/``check-report``: NON-SCIENTIFIC prelaunch validation under
  ``artifacts/prelaunch/acquisition-check``.
- ``prepare-approvals``: writes UNAPPROVED approval requests; it never approves.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rq1.acquisition.environment import observed_environment, verify_launch_environment
from rq1.acquisition.executor import RealAcquisitionExecutor
from rq1.acquisition.gates import (
    load_task_manifest,
    queue_identity_sha256,
    validate_acquisition_gates,
    validate_queue_manifest,
)
from rq1.acquisition.models import AcquisitionPlan
from rq1.acquisition.protocol import (
    ACQUISITION_ACTION_BUDGET,
    ACQUISITION_ENVIRONMENT_SEED,
    ACQUISITION_MODEL,
    ACQUISITION_POLICY_VERSION,
    ACQUISITION_TEMPERATURE,
    DECISION_RECORD,
    SKILL_GENERATION_ATTEMPTS,
    SKILL_GENERATION_PROTOCOL,
    protocol_definition,
    protocol_sha256,
)
from rq1.acquisition.reporting import write_report
from rq1.acquisition.runner import AcquisitionError, AcquisitionRunner
from rq1.acquisition.skill_creation import prompt_hashes
from rq1.acquisition.skill_pool import EMPTY_POOL_HASH, SkillPoolError, pool_hash, rebuild_pool, verify_snapshot
from rq1.bridge.adapters.capabilities import default_data_dir
from rq1.bridge.adapters.task_index import _resolve_task_family
from rq1.experiment.models import canonical_hash
from rq1.experiment.persistence import ExperimentStore, atomic_write_json, durable_append_jsonl
from rq1.experiment.runner import RunnerOptions
from rq1.freeze.validation import ACQUISITION_EVIDENCE_MODE, git_state
from rq1.hermes.episode_driver import ACTION_SELECTION_PROTOCOL, INFERENCE_SEED, MAX_SELECTION_ATTEMPTS, RealEpisodeDriver
from rq1.skills.leakage import find_leakage
from rq1.tasks.discovery import CANONICAL_FAMILIES, discover_tasks
from rq1.utils.hashing import sha256_file
from rq1.utils.time import utc_now

CHECK_BASE = Path("artifacts") / "prelaunch" / "acquisition-check"
CHECK_PREFIX = "prelaunch-acquisition-check-"
CHECK_REPORT = "acquisition-check-report.json"
APPROVAL_DIR = Path("artifacts") / "approvals" / "acquisition"
RUNNING_STATES = {"completed", "paused", "incomplete"}


def run_configuration(
    root: Path,
    *,
    queue_sha256: str,
    scientific: bool,
    freezes: Mapping[str, str] | None = None,
    model_name: str = ACQUISITION_MODEL,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": ACQUISITION_POLICY_VERSION,
        "model_name": model_name,
        "runtime_settings": {
            "temperature": ACQUISITION_TEMPERATURE,
            "seed": INFERENCE_SEED,
            "action_selection_protocol": ACTION_SELECTION_PROTOCOL,
            "max_selection_attempts": MAX_SELECTION_ATTEMPTS,
            "acquisition_action_budget": ACQUISITION_ACTION_BUDGET,
            "environment_seed": ACQUISITION_ENVIRONMENT_SEED,
            "skill_generation_protocol": SKILL_GENERATION_PROTOCOL,
            "skill_generation_attempts": SKILL_GENERATION_ATTEMPTS,
        },
        "protocol_sha256": protocol_sha256(),
        "queue_sha256": queue_sha256,
        "prompt_hashes": prompt_hashes(root),
        "scientific_evidence": scientific,
    }
    if freezes:
        value["freeze_fingerprints"] = dict(freezes)
    return value


def _execute(
    root: Path,
    plan: AcquisitionPlan,
    store: ExperimentStore,
    *,
    configuration: Mapping[str, Any],
    options: RunnerOptions,
    scientific: bool,
    task_manifest_path: Path | None = None,
    model_name: str = ACQUISITION_MODEL,
) -> dict[str, Any]:
    with RealEpisodeDriver(
        root, data_dir=default_data_dir(), model_name=model_name, bridge_log_root=store.directory / "logs" / "bridge",
    ) as driver:
        executor = RealAcquisitionExecutor(root, store, driver, scientific=scientific, queue_sha256=plan.queue_sha256)
        return AcquisitionRunner(root).run_resumable(
            plan,
            executor,
            configuration=dict(configuration),
            options=options,
            output_base=store.directory.parent,
            initial_library_hash=EMPTY_POOL_HASH,
            initial_library_size=0,
            task_manifest_path=task_manifest_path,
            scientific=scientific,
            store=store,
            preflight=executor.preflight,
            checkpoint_extension=executor.checkpoint_state,
        )


def _next_step(status: str, run_id: str) -> str:
    command = "python -m rq1.cli acquisition"
    return {
        "completed": f"{command} validate --run-id {run_id}",
        "paused": f"{command} resume --run-id {run_id} --yes",
        "incomplete": f"{command} resume --run-id {run_id} --yes",
        "interrupted": f"{command} resume --run-id {run_id} --yes",
        "failed": f"fix the infrastructure error in errors.jsonl, then {command} retry-failed --run-id {run_id} --yes",
        "blocked": "manual review of checkpoint.json blocking_error is required; do not edit scientific results",
    }.get(status, "inspect checkpoint.json")


def acquisition_plan(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.task_manifest) if getattr(args, "task_manifest", None) else None
    gate = validate_acquisition_gates(root, task_manifest_path=manifest_path)
    return {
        "ok": True,
        "dry_run": True,
        "launch_permitted": gate.valid,
        "gate": gate.to_dict(),
        "protocol_sha256": protocol_sha256(),
        "protocol": protocol_definition(),
    }


def scientific_run(root: Path, args: argparse.Namespace, *, resume: bool, retry_failed: bool) -> dict[str, Any]:
    if not getattr(args, "yes", False):
        return {"ok": False, "status": "blocked", "reason": "scientific acquisition requires --yes"}
    if str(args.run_id).startswith(CHECK_PREFIX):
        return {"ok": False, "status": "blocked", "reason": f"run IDs starting with {CHECK_PREFIX} are reserved for non-scientific checks"}
    manifest_path = Path(args.task_manifest) if getattr(args, "task_manifest", None) else None
    gate = validate_acquisition_gates(root, task_manifest_path=manifest_path)
    if not gate.valid or gate.task_manifest is None or gate.environment is None or gate.protocol is None:
        return {"ok": False, "status": "blocked", "gate": gate.to_dict()}
    drift = verify_launch_environment(root, gate.environment.inputs)
    if discover_tasks(default_data_dir(), "train").data_root_identity != gate.task_manifest.data_root_identity:
        drift.append("ALFWorld TRAIN data identity differs from the approved frozen queue")
    if drift:
        return {"ok": False, "status": "blocked", "environment_drift": drift}
    plan = AcquisitionRunner(root).plan_from_manifest(gate.task_manifest, str(args.run_id))
    store = ExperimentStore(root, str(args.run_id))
    configuration = run_configuration(
        root,
        queue_sha256=str(plan.queue_sha256),
        scientific=True,
        freezes={
            "task_manifest_sha256": gate.task_manifest.manifest_sha256,
            "acquisition-environment": gate.environment.input_fingerprint,
            "acquisition-protocol": gate.protocol.input_fingerprint,
        },
    )
    backup = Path(args.backup_dir) if getattr(args, "backup_dir", None) else None
    options = RunnerOptions(
        resume=resume,
        retry_failed=retry_failed,
        max_runs=getattr(args, "max_runs", None),
        # Acquisition halts at an infrastructure failure so it can be retried
        # before any later chronological result exists.
        fail_fast=True,
        backup_dir=backup,
        require_backup=bool(getattr(args, "require_backup", False)),
    )
    result = _execute(root, plan, store, configuration=configuration, options=options, scientific=True, task_manifest_path=gate.task_manifest_path)
    return {"ok": result["status"] in RUNNING_STATES, "scientific_evidence": True, **result, "next": _next_step(result["status"], str(args.run_id))}


def _scientific_queue_task_ids(root: Path) -> set[str]:
    identifiers: set[str] = set()
    for folder in ("proposals", "proposal_archive", "frozen"):
        for path in (root / "artifacts" / "task_manifests" / folder).glob("acquisition-*.json"):
            identifiers.update(task.task_id for task in load_task_manifest(path).tasks)
    return identifiers


def _train_task_family(data_dir: Path, task_id: str) -> str:
    if not task_id.startswith("train:"):
        raise AcquisitionError(f"acquisition checks accept TRAIN tasks only: {task_id}")
    source = data_dir / "json_2.1.1" / "train" / task_id.split(":", 1)[1] / "traj_data.json"
    if not source.is_file() or not source.with_name("game.tw-pddl").is_file():
        raise AcquisitionError(f"unknown playable TRAIN task: {task_id}")
    native = _resolve_task_family(json.loads(source.read_text(encoding="utf-8")).get("task_type"))
    if native not in CANONICAL_FAMILIES:
        raise AcquisitionError(f"TRAIN task is outside the six RQ1 families: {task_id}")
    return CANONICAL_FAMILIES[native]


def prelaunch_check(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_id = str(args.run_id)
    if not run_id.startswith(CHECK_PREFIX):
        return {"ok": False, "status": "blocked", "reason": f"non-scientific check run IDs must start with {CHECK_PREFIX}"}
    store = ExperimentStore(root, run_id, base=root / CHECK_BASE)
    plan_path = store.directory / "check-plan.json"
    commit, clean, _error = git_state(root)
    requested = list(getattr(args, "task_id", None) or [])
    requested_model = getattr(args, "model", None)
    if args.resume:
        if not plan_path.is_file():
            return {"ok": False, "status": "blocked", "reason": "cannot resume an unknown non-scientific check"}
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if requested and requested != saved["task_ids"]:
            return {"ok": False, "status": "blocked", "reason": "check queue differs from the saved check plan"}
        model_name = str(saved.get("model_name", ACQUISITION_MODEL))
        if requested_model and requested_model != model_name:
            return {"ok": False, "status": "blocked", "reason": "check model differs from the saved check plan"}
        task_ids, families = list(saved["task_ids"]), list(saved["task_families"])
    else:
        if plan_path.exists():
            return {"ok": False, "status": "blocked", "reason": "check already exists; use --resume"}
        if not requested:
            return {"ok": False, "status": "blocked", "reason": "at least one --task-id is required"}
        overlap = sorted(set(requested) & _scientific_queue_task_ids(root))
        if overlap:
            return {"ok": False, "status": "blocked", "reason": "check tasks overlap the scientific acquisition queue", "overlap": overlap}
        families = [_train_task_family(default_data_dir(), task_id) for task_id in requested]
        task_ids = requested
        model_name = str(requested_model or ACQUISITION_MODEL)
        atomic_write_json(plan_path, {
            "schema_version": 1,
            "label": "NON-SCIENTIFIC PRELAUNCH ACQUISITION CHECK",
            "scientific_evidence": False,
            "task_ids": task_ids,
            "task_families": families,
            "model_name": model_name,
            "repository_commit": commit,
            "created_at": utc_now(),
        })
    queue_sha = canonical_hash({"task_ids": task_ids, "task_families": families})
    plan = AcquisitionPlan(run_id, tuple(task_ids), task_families=tuple(families), queue_sha256=queue_sha)
    durable_append_jsonl(store.directory / "invocations.jsonl", {
        "mode": "resume" if args.resume else "run",
        "repository_commit": commit,
        "clean": clean,
        "model_name": model_name,
        "max_runs": args.max_runs,
        "timestamp": utc_now(),
    })
    result = _execute(
        root, plan, store,
        configuration=run_configuration(root, queue_sha256=queue_sha, scientific=False, model_name=model_name),
        options=RunnerOptions(resume=bool(args.resume), max_runs=args.max_runs, fail_fast=True),
        scientific=False,
        model_name=model_name,
    )
    return {"ok": result["status"] in RUNNING_STATES, "label": "NON-SCIENTIFIC PRELAUNCH ACQUISITION CHECK", "scientific_evidence": False, **result}


def _real_episode_evidence(directory: Path, record: Mapping[str, Any]) -> bool:
    for relative in record.get("log_paths") or []:
        path = directory / relative
        if path.name != "episode-events.jsonl" or not path.is_file():
            continue
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        tools = {event.get("payload", {}).get("tool") for event in events if event.get("event") == "tool_result"}
        plugin = path.with_name("plugin-events.jsonl")
        plugin_calls = plugin.is_file() and any(
            json.loads(line).get("event") == "plugin_pre_tool_call"
            for line in plugin.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        return (
            {"alfworld_start", "alfworld_step"} <= tools
            and any(event.get("event") == "model_selection" for event in events)
            and any(event.get("event") == "task_goal_frozen" for event in events)
            and plugin_calls
        )
    return False


def check_report(root: Path, run_id: str) -> dict[str, Any]:
    store = ExperimentStore(root, run_id, base=root / CHECK_BASE)
    if not run_id.startswith(CHECK_PREFIX) or not store.manifest_path.is_file():
        return {"ok": False, "status": "blocked", "reason": "unknown non-scientific acquisition check"}
    results = store.terminal_results(phase="acquisition", repair_tail=False)
    records = sorted(results.values(), key=lambda item: int(item.get("task_index", 0)))
    completed = [item for item in records if item.get("status") == "completed"]
    configuration = json.loads((store.manifests / "acquisition.json").read_text(encoding="utf-8")).get("configuration", {})
    invocations_path = store.directory / "invocations.jsonl"
    invocations = [json.loads(line) for line in invocations_path.read_text(encoding="utf-8").splitlines() if line.strip()] if invocations_path.is_file() else []
    checkpoint, _source = store.load_checkpoint()
    commit, clean, _error = git_state(root)
    try:
        pool = rebuild_pool(records)
        verify_snapshot(store.directory, pool)
        pool_consistent = True
    except SkillPoolError:
        pool, pool_consistent = (), False
    checkpoint_pool = ((checkpoint or {}).get("phase_state") or {}).get("skill_pool") or {}
    runtime = configuration.get("runtime_settings", {})
    checks = {
        "non_scientific_configuration": configuration.get("scientific_evidence") is False and all(item.get("scientific_evidence") is False for item in completed),
        "frozen_action_budget_50": runtime.get("acquisition_action_budget") == ACQUISITION_ACTION_BUDGET,
        "inference_seed_42": runtime.get("seed") == INFERENCE_SEED,
        "train_only": bool(records) and all(str(item.get("task_id", "")).startswith("train:") for item in records),
        "real_hermes_episodes_executed": bool(completed) and all(_real_episode_evidence(store.directory, item) for item in completed),
        "results_persisted": bool(completed) and all(item.get("skill_library_hash_after") and item.get("library_size_after") is not None for item in completed),
        "skill_pool_consistent_with_results": pool_consistent,
        "successful_episode_observed": any(item.get("success") is True for item in completed),
        "decision_003_skill_created": bool(pool) and all(not find_leakage(skill.text) for skill in pool),
        "skill_pool_size_hash_updated": bool(pool) and pool_hash(pool) != EMPTY_POOL_HASH and checkpoint_pool.get("hash") == pool_hash(pool) and checkpoint_pool.get("size") == len(pool),
        "later_unit_observed_restored_pool": any(int(item.get("skill_pool_size_before") or 0) >= 1 for item in completed),
        "resume_invocation_observed": len(invocations) >= 2 and any(item.get("mode") == "resume" for item in invocations),
        "no_completed_unit_rerun": len(store.read_results(repair_tail=False)) == len(results),
        "no_scientific_retrieval": bool(completed) and all(item.get("scientific_retrieval_count") == 0 and not item.get("retrieved_skill_ids") for item in completed),
        "single_clean_commit": bool(clean) and bool(invocations) and all(item.get("repository_commit") == commit and item.get("clean") is True for item in invocations),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "mode": ACQUISITION_EVIDENCE_MODE,
        "label": "NON-SCIENTIFIC PRELAUNCH ACQUISITION CHECK",
        "scientific_evidence": False,
        "run_id": run_id,
        "repository_commit": commit,
        "generated_at": utc_now(),
        "passed": passed,
        "checks": checks,
        "units": [
            {
                "task_index": item.get("task_index"),
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
            "skills": [{"skill_id": skill.skill_id, "task_family": skill.task_family, "source_task_id": skill.source_task_id, "text": skill.text} for skill in pool],
        },
        "output_directory": str(store.directory),
    }
    path = store.directory / CHECK_REPORT
    atomic_write_json(path, report)
    return {"ok": passed, "report": str(path), "report_sha256": sha256_file(path), **report}


def validate_run(root: Path, run_id: str) -> dict[str, Any]:
    store = ExperimentStore(root, run_id)
    if not store.manifest_path.is_file():
        return {"ok": False, "status": "blocked", "reason": "unknown scientific acquisition run"}
    results = store.terminal_results(phase="acquisition", repair_tail=False)
    records = sorted(results.values(), key=lambda item: int(item.get("task_index", 0)))
    try:
        pool = rebuild_pool(records)
        verify_snapshot(store.directory, pool)
    except SkillPoolError as exc:
        return {"ok": False, "status": "invalid", "reason": str(exc)}
    completed = [item for item in records if item.get("status") == "completed"]
    problems = []
    if any(not str(item.get("task_id", "")).startswith("train:") for item in records):
        problems.append("non-TRAIN task in acquisition results")
    if any(item.get("scientific_retrieval_count") != 0 or item.get("retrieved_skill_ids") for item in completed):
        problems.append("scientific retrieval occurred during acquisition")
    if any(item.get("scientific_evidence") is not True for item in completed):
        problems.append("completed result is not marked as scientific evidence")
    return {
        "ok": not problems,
        "status": "valid" if not problems else "invalid",
        "problems": problems,
        "terminal": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "successful": sum(item.get("success") is True for item in completed),
        "skill_pool_size": len(pool),
        "skill_pool_hash": pool_hash(pool),
        "skills_per_family": dict(Counter(skill.task_family for skill in pool)),
    }


def prepare_approvals(root: Path, proposal_path: Path, evidence_path: Path) -> dict[str, Any]:
    commit, clean, error = git_state(root)
    if error or not clean or not commit:
        return {"ok": False, "status": "blocked", "reason": "approval requests require a clean committed repository"}
    proposal = load_task_manifest(proposal_path)
    problems = validate_queue_manifest(proposal, require_frozen=False)
    if proposal.status != "proposed":
        problems.append("task manifest is not a proposal")
    if proposal.repository_commit != commit:
        problems.append("task proposal was generated at a different commit")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("mode") != ACQUISITION_EVIDENCE_MODE or evidence.get("passed") is not True or evidence.get("repository_commit") != commit:
        problems.append("evidence report is not a passed non-scientific acquisition check at this commit")
    if problems:
        return {"ok": False, "status": "blocked", "reasons": problems}
    directory = root / APPROVAL_DIR / commit[:12]
    paths = {name: directory / f"{name}.approval.json" for name in ("task-freeze", "acquisition-environment", "acquisition-protocol")}
    if any(path.exists() for path in paths.values()):
        return {"ok": False, "status": "blocked", "reason": f"approval requests already exist and are immutable: {directory}"}
    queue_sha = queue_identity_sha256(proposal)
    unapproved = {"status": "UNAPPROVED", "approved_by": None, "approved_at": None, "reference": None}
    how = "A human reviewer sets status to APPROVED and fills approved_by, approved_at (UTC ISO-8601), and reference, then runs the command."
    evidence_reference = {"path": str(evidence_path), "sha256": sha256_file(evidence_path), "run_id": evidence.get("run_id")}
    documents = {
        "task-freeze": {
            "schema_version": 1,
            "approval_kind": "acquisition-task-freeze",
            **unapproved,
            "subject": {
                "proposal_path": str(proposal_path),
                "manifest_sha256": proposal.manifest_sha256,
                "task_queue_sha256": queue_sha,
                "repository_commit": commit,
                "split": proposal.split,
                "actual_count": proposal.actual_count,
                "family_counts": dict(proposal.family_counts),
                "selection_policy": dict(proposal.selection_policy),
                "data_root_identity": proposal.data_root_identity,
            },
            "attestation": [
                "The queue is exactly 180 ALFWorld 0.4.2 TRAIN tasks, 30 per RQ1 family, in the recorded order.",
                "Selection used only TRAIN metadata (task-selection-v1, seed 1, round-robin family balancing); no valid_seen or valid_unseen information.",
                "This queue is the initial scientific acquisition queue; it is not automatically extended to 240.",
            ],
            "how_to_approve": how,
            "command": f"python -m rq1.cli tasks freeze --kind acquisition --proposal {proposal_path} --approval-file {paths['task-freeze']} --yes",
        },
        "acquisition-environment": {
            "schema_version": 1,
            "approval_kind": "acquisition-environment",
            "approval": dict(unapproved),
            "inputs": observed_environment(root, task_queue_sha256=queue_sha, alfworld_data_identity=proposal.data_root_identity),
            "evidence_report": evidence_reference,
            "attestation": [
                "The recorded commit, Python environment, ALFWorld data, Hermes, Ollama, and hermes3:8b digest are the approved acquisition environment.",
                "The referenced NON-SCIENTIFIC acquisition check passed at this commit and is not scientific data.",
                "A resume on replacement hardware must reproduce every enforced identity; host name and GPU are recorded only.",
            ],
            "how_to_approve": how,
            "command": f"python -m rq1.cli freeze acquisition-environment --approval-file {paths['acquisition-environment']} --pilot-report {evidence_path} --yes",
        },
        "acquisition-protocol": {
            "schema_version": 1,
            "approval_kind": "acquisition-protocol",
            "approval": dict(unapproved),
            "inputs": {
                "repository_commit": commit,
                "protocol": protocol_definition(),
                "protocol_sha256": protocol_sha256(),
                "task_queue_sha256": queue_sha,
                "acquisition_action_budget": ACQUISITION_ACTION_BUDGET,
                "inference_seed": INFERENCE_SEED,
                "prompt_hashes": prompt_hashes(root),
                "decision_record_sha256": sha256_file(root / DECISION_RECORD),
            },
            "evidence_report": evidence_reference,
            "attestation": [
                "Decision 007 was made on 2026-09-13 before any scientific acquisition data existed.",
                "The acquisition protocol is: 50-action budget; fresh session per task; no retrieval; the same hermes3:8b agent writes at most one create-only candidate after success; exact-normalized duplicate rejection only; no retrospective deduplication; results-authoritative skill pool; fail-closed resume.",
            ],
            "how_to_approve": how,
            "command": f"python -m rq1.cli freeze acquisition-protocol --approval-file {paths['acquisition-protocol']} --pilot-report {evidence_path} --yes",
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
        "protocol_sha256": protocol_sha256(),
    }

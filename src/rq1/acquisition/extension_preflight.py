"""Technical preflight of the scientific acquisition extension; it starts no episode.

Before human approval the queue proposal and the UNAPPROVED requests stand in for
the frozen manifest and freezes they become, so every identity of
``acquisition-extension run`` is verified against the live repository, the
completed parent run, and the live environment.  Human approval must be the only
remaining blocker.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from rq1.acquisition.environment import model_digest, verify_launch_environment
from rq1.acquisition.extension import (
    EXTENSION_APPROVAL_PENDING_REASONS,
    continuation_problems,
    extension_run_configuration,
    load_parent,
    parent_directory,
    starting_pool_path,
    starting_pool_problems,
    validate_extension_gates,
    validate_extension_queue_manifest,
)
from rq1.acquisition.extension_launch import EXTENSION_APPROVAL_REQUEST_NAMES
from rq1.acquisition.extension_protocol import (
    EXTENSION_APPROVAL_DIR,
    EXTENSION_CHECK_PREFIX,
    EXTENSION_PREFLIGHT_BASE,
    EXTENSION_RUN_ID,
    EXTENSION_TASK_COUNT,
    EXTENSION_TASKS_PER_FAMILY,
    FROZEN_MODEL_DIGEST,
    PARENT,
    ParentReference,
    extension_protocol_definition,
    extension_protocol_sha256,
)
from rq1.acquisition.gates import ENVIRONMENT_FREEZE, PROTOCOL_FREEZE, load_task_manifest, queue_identity_sha256
from rq1.acquisition.launch import CHECK_PREFIX, HERMES_PYTHON, PRODUCTION_BACKUP_DIR, _read_request, _writable, hard_cap_status
from rq1.acquisition.protocol import ACQUISITION_ACTION_BUDGET, ACQUISITION_MODEL, ACQUISITION_TEMPERATURE, protocol_sha256
from rq1.acquisition.runner import AcquisitionError, AcquisitionRunner
from rq1.acquisition.skill_creation import prompt_hashes
from rq1.acquisition.skill_pool import pool_hash
from rq1.bridge.adapters.capabilities import default_data_dir, probe_alfworld_capabilities
from rq1.experiment.persistence import atomic_write_json
from rq1.freeze.validation import ACQUISITION_EXTENSION_ENVIRONMENT_REQUIRED, ACQUISITION_EXTENSION_EVIDENCE_MODE, git_state, read_freeze
from rq1.hermes.episode_driver import (
    ACTION_SELECTION_PROTOCOL,
    INFERENCE_SEED,
    MAX_SELECTION_ATTEMPTS,
    MODEL_CONTEXT_LENGTH,
    MODEL_QUANTIZATION,
    OUTPUT_TOKEN_CAP,
    provider_settings,
)
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.discovery import discover_tasks
from rq1.utils.hashing import sha256_file
from rq1.utils.time import utc_now

# Scientific identities the extension environment must share with the approved
# parent environment freeze.  Commit and configuration hashes necessarily differ;
# host name and GPU are recorded only, as for the parent run.
PARENT_ENVIRONMENT_IDENTITIES = (
    "python_version", "python_executable", "python_environment", "dependency_lock_sha256", "packages",
    "alfworld_version", "alfworld_data_identity", "hermes_version", "hermes_commit", "ollama_version",
    "model_tag", "model_digest", "model_quantization", "provider_settings", "inference_seed",
    "sbert_model", "sbert_revision", "sbert_snapshot_sha256", "prompt_hashes",
)
RUNNER_COMMANDS = {"run", "resume", "retry-failed", "check"}


def extension_commands(run_id: str = EXTENSION_RUN_ID, backup_dir: str = PRODUCTION_BACKUP_DIR) -> dict[str, str]:
    durable = f"--run-id {run_id} --yes --backup-dir {backup_dir} --require-backup"
    return {
        "run": f"python -m rq1.cli acquisition-extension run {durable}",
        "resume": f"python -m rq1.cli acquisition-extension resume {durable}",
        "retry_failed": f"python -m rq1.cli acquisition-extension retry-failed {durable}",
        "status": f"python -m rq1.cli experiment status --run-id {run_id}",
        "validate": f"python -m rq1.cli acquisition-extension validate --run-id {run_id}",
    }


def active_acquisition_processes() -> list[str]:
    """Command lines of live acquisition or extension runners and checks (Linux /proc)."""
    found: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            words = [part.decode("utf-8", "replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
        except OSError:
            continue
        if "rq1.cli" in words and {"acquisition", "acquisition-extension"} & set(words) and RUNNER_COMMANDS & set(words):
            found.append(" ".join(words))
    return found


def extension_preflight(root: Path, args: argparse.Namespace, parent: ParentReference = PARENT) -> dict[str, Any]:
    run_id = str(getattr(args, "run_id", None) or EXTENSION_RUN_ID)
    backup_dir = Path(str(getattr(args, "backup_dir", None) or PRODUCTION_BACKUP_DIR))
    commit, clean, error = git_state(root)
    checks: dict[str, bool] = {"clean_committed_repository": bool(commit) and clean and not error}
    details: dict[str, Any] = {"repository_commit": commit, "parent": parent.to_dict()}

    state = load_parent(root, parent)
    checks.update(state.checks)
    details["parent_problems"] = state.problems
    pool_path = starting_pool_path(root, parent)
    checks["starting_pool_snapshot_matches_parent"] = not starting_pool_problems(root, state, parent)
    details["starting_pool"] = {
        "path": str(pool_path),
        "sha256": sha256_file(pool_path) if pool_path.is_file() else None,
        "size": len(state.pool),
        "hash": pool_hash(state.pool),
        "per_family": {family: sum(skill.task_family == family for skill in state.pool) for family in TASK_FAMILIES},
    }

    approval_dir = Path(args.approval_dir) if getattr(args, "approval_dir", None) else root / EXTENSION_APPROVAL_DIR / str(commit or "")[:12]
    requests = {name: _read_request(approval_dir / f"{name}.approval.json") for name in EXTENSION_APPROVAL_REQUEST_NAMES}
    checks["approval_requests_present"] = all(value is not None for value in requests.values())
    task_request = requests["extension-task-freeze"] or {}
    environment_request = requests["acquisition-extension-environment"] or {}
    protocol_request = requests["acquisition-extension-protocol"] or {}
    approval_status = {
        "extension-task-freeze": task_request.get("status"),
        "acquisition-extension-environment": (environment_request.get("approval") or {}).get("status"),
        "acquisition-extension-protocol": (protocol_request.get("approval") or {}).get("status"),
    }

    subject = task_request.get("subject") or {}
    proposal_path = Path(args.proposal) if getattr(args, "proposal", None) else Path(str(subject.get("proposal_path") or ""))
    try:
        proposal = load_task_manifest(proposal_path)
    except (OSError, TypeError, ValueError):
        proposal = None
    queue_sha = queue_identity_sha256(proposal) if proposal is not None else None
    identifiers = [task.task_id for task in proposal.tasks] if proposal is not None else []
    balanced = {family: EXTENSION_TASKS_PER_FAMILY for family in TASK_FAMILIES}
    queue_errors = (
        validate_extension_queue_manifest(proposal, state.manifest, parent, require_frozen=False)
        if proposal is not None else ["extension queue proposal is unreadable"]
    )
    discovery = discover_tasks(default_data_dir(), "train")
    continuation = continuation_problems(discovery, state.manifest, proposal) if proposal is not None else ["extension queue proposal is unreadable"]
    checks["extension_queue_proposal_valid"] = proposal is not None and proposal.status == "proposed" and not queue_errors
    checks["extension_queue_exactly_60"] = proposal is not None and proposal.actual_count == len(identifiers) == EXTENSION_TASK_COUNT
    checks["extension_queue_10_per_family"] = proposal is not None and dict(proposal.family_counts) == balanced and dict(Counter(task.family for task in proposal.tasks)) == balanced
    checks["extension_no_overlap_with_parent_queue"] = proposal is not None and state.manifest is not None and not set(identifiers) & state.queue_task_ids
    checks["extension_no_internal_duplicates"] = proposal is not None and len(set(identifiers)) == len(identifiers)
    checks["extension_queue_hash_matches_request"] = (
        proposal is not None and subject.get("manifest_sha256") == proposal.manifest_sha256 and subject.get("task_queue_sha256") == queue_sha
    )
    checks["queue_proposal_at_commit"] = proposal is not None and proposal.repository_commit == commit
    checks["extension_continues_frozen_selection"] = not continuation
    units = (proposal.lineage or {}).get("units") or [{}] if proposal is not None else [{}]
    details["queue"] = {
        "proposal_path": str(proposal_path),
        "proposal_sha256": sha256_file(proposal_path) if proposal is not None else None,
        "manifest_sha256": proposal.manifest_sha256 if proposal is not None else None,
        "task_queue_sha256": queue_sha,
        "count": proposal.actual_count if proposal is not None else None,
        "family_counts": dict(proposal.family_counts) if proposal is not None else None,
        "logical_positions": [units[0].get("logical_acquisition_index"), units[-1].get("logical_acquisition_index")],
        "errors": queue_errors,
        "continuation_problems": continuation,
    }

    environment_inputs = environment_request.get("inputs") or {}
    protocol_inputs = protocol_request.get("inputs") or {}
    checks["environment_request_complete"] = bool(environment_inputs) and not (ACQUISITION_EXTENSION_ENVIRONMENT_REQUIRED - set(environment_inputs))
    checks["environment_request_at_commit_and_queue"] = (
        environment_inputs.get("repository_commit") == commit
        and environment_inputs.get("task_queue_sha256") == queue_sha
        and environment_inputs.get("parent_run_id") == parent.run_id
    )
    parent_environment, _errors = read_freeze(root / ENVIRONMENT_FREEZE, "acquisition-environment")
    parent_protocol, _errors = read_freeze(root / PROTOCOL_FREEZE, "acquisition-protocol")
    differing = [
        key for key in PARENT_ENVIRONMENT_IDENTITIES
        if parent_environment is None or environment_inputs.get(key) != parent_environment.inputs.get(key)
    ]
    checks["environment_identical_to_parent_run"] = not differing
    details["environment_differences_from_parent"] = differing
    details["environment"] = {key: environment_inputs.get(key) for key in (
        "repository_commit", "branch", "os", "kernel", "hostname", "gpu", "gpu_driver", "cuda_version", "python_version",
        "torch_version", "torch_cuda_version", "alfworld_version", "hermes_version", "hermes_commit", "ollama_version",
        "model_tag", "model_digest", "model_quantization",
    )}
    pool_sha = sha256_file(pool_path) if pool_path.is_file() else None
    checks["protocol_request_matches_repository"] = (
        protocol_inputs.get("repository_commit") == commit
        and protocol_inputs.get("protocol") == extension_protocol_definition(parent)
        and protocol_inputs.get("protocol_sha256") == extension_protocol_sha256(parent)
        and protocol_inputs.get("inherited_protocol_sha256") == protocol_sha256()
        and protocol_inputs.get("task_queue_sha256") == queue_sha
        and protocol_inputs.get("acquisition_action_budget") == ACQUISITION_ACTION_BUDGET
        and protocol_inputs.get("inference_seed") == INFERENCE_SEED
        and protocol_inputs.get("parent_run_id") == parent.run_id
        and protocol_inputs.get("parent_pool_size") == parent.pool_size
        and protocol_inputs.get("parent_pool_hash") == parent.pool_hash
        and protocol_inputs.get("parent_queue_sha256") == parent.queue_sha256
        and protocol_inputs.get("parent_closeout_manifest_sha256") == parent.closeout_manifest_sha256
        and pool_sha is not None
        and protocol_inputs.get("starting_pool_sha256") == pool_sha
    )
    definition = extension_protocol_definition(parent)
    checks["inherited_protocol_identical_to_parent_freeze"] = (
        parent_protocol is not None
        and parent_protocol.inputs.get("protocol_sha256") == protocol_sha256()
        and parent_protocol.inputs.get("protocol") == definition["inherited_acquisition_protocol"]
    )
    checks["prompt_hashes_consistent"] = environment_inputs.get("prompt_hashes") == protocol_inputs.get("prompt_hashes") == prompt_hashes(root)
    checks["exact_model_identity"] = (
        environment_inputs.get("model_tag") == ACQUISITION_MODEL
        and environment_inputs.get("model_quantization") == MODEL_QUANTIZATION
        and environment_inputs.get("model_digest") == FROZEN_MODEL_DIGEST
    )
    settings = environment_inputs.get("provider_settings") or {}
    checks["exact_provider_settings"] = (
        settings == provider_settings()
        and settings.get("think") is False
        and settings.get("options") == {"temperature": ACQUISITION_TEMPERATURE, "seed": INFERENCE_SEED, "num_predict": OUTPUT_TOKEN_CAP, "num_ctx": MODEL_CONTEXT_LENGTH}
    )
    checks["zero_acquisition_retrieval"] = (
        definition["scientific_retrieval_during_acquisition"] is False
        and definition["inherited_acquisition_protocol"]["scientific_retrieval_during_acquisition"] is False
    )
    drift = verify_launch_environment(root, environment_inputs) if environment_inputs else ["extension environment request is missing"]
    checks["live_environment_matches_request"] = not drift
    details["environment_drift"] = drift
    checks["alfworld_train_data_identity"] = (
        proposal is not None and state.manifest is not None
        and discovery.data_root_identity == proposal.data_root_identity == environment_inputs.get("alfworld_data_identity") == state.manifest.data_root_identity
    )
    checks["real_alfworld_adapter_ready"] = probe_alfworld_capabilities(default_data_dir()).real_adapter_ready
    checks["hermes_runtime_present"] = HERMES_PYTHON.is_file() and bool(environment_inputs.get("hermes_version")) and bool(environment_inputs.get("hermes_commit"))
    checks["ollama_serves_frozen_digest"] = model_digest(ACQUISITION_MODEL) == FROZEN_MODEL_DIGEST == environment_inputs.get("model_digest")

    plan = None
    if proposal is not None:
        try:
            plan = AcquisitionRunner(root).plan_from_manifest(proposal, run_id)
        except AcquisitionError as exc:
            details["plan_error"] = str(exc)
    checks["production_plan_is_extension_queue"] = (
        plan is not None
        and len(plan.task_ids) == EXTENSION_TASK_COUNT
        and plan.queue_sha256 == queue_sha
        and plan.parent_run_id == parent.run_id
        and plan.logical_index_offset == parent.completed_units
    )
    configuration = extension_run_configuration(root, queue_sha256=str(queue_sha), scientific=True, parent=parent)
    runtime = configuration["runtime_settings"]
    details["production_configuration"] = {key: configuration[key] for key in ("model_name", "runtime_settings", "protocol_sha256", "extension")}
    checks["production_configuration_frozen_controller"] = (
        configuration["model_name"] == ACQUISITION_MODEL
        and configuration["scientific_evidence"] is True
        and runtime.get("output_token_cap") == OUTPUT_TOKEN_CAP
        and runtime.get("model_context_length") == MODEL_CONTEXT_LENGTH
        and runtime.get("acquisition_action_budget") == ACQUISITION_ACTION_BUDGET
        and runtime.get("temperature") == ACQUISITION_TEMPERATURE
        and runtime.get("seed") == INFERENCE_SEED
        and runtime.get("action_selection_protocol") == ACTION_SELECTION_PROTOCOL
        and runtime.get("max_selection_attempts") == MAX_SELECTION_ATTEMPTS
    )
    evidence_reference = environment_request.get("evidence_report") or {}
    evidence_path = Path(str(evidence_reference.get("path") or ""))
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        evidence = {}
    checks["extension_check_evidence_passed_at_commit"] = (
        evidence.get("mode") == ACQUISITION_EXTENSION_EVIDENCE_MODE
        and evidence.get("passed") is True
        and evidence.get("scientific_evidence") is False
        and evidence.get("repository_commit") == commit
        and sha256_file(evidence_path) == evidence_reference.get("sha256") == (protocol_request.get("evidence_report") or {}).get("sha256")
    )
    details["evidence_report"] = {"path": str(evidence_path), "sha256": evidence_reference.get("sha256"), "run_id": evidence.get("run_id")}

    run_directory = root / "results" / "final" / run_id
    parent_dir = parent_directory(root, parent).resolve()
    checks["extension_run_id_unused"] = (
        run_id != parent.run_id and not run_id.startswith((CHECK_PREFIX, EXTENSION_CHECK_PREFIX)) and not run_directory.exists()
    )
    cap = hard_cap_status(root, run_id, [task.family for task in proposal.tasks] if proposal is not None else [])
    checks["acquisition_hard_cap_respected"] = proposal is not None and cap["permitted"]
    details["hard_cap"] = cap
    checks["extension_outputs_separate_from_parent"] = run_directory.resolve() != parent_dir and not run_directory.resolve().is_relative_to(parent_dir)
    checks["extension_checkpoint_path_available"] = not (run_directory / "checkpoint.json").exists() and not (run_directory / "results.jsonl").exists()
    checks["results_final_writable"] = _writable(root / "results" / "final")
    checks["extension_backup_path_available"] = _writable(backup_dir) and not (backup_dir / run_id).exists()
    checks["concurrency_lock_free"] = not (run_directory / ".experiment.lock").exists()
    active = active_acquisition_processes()
    checks["no_active_acquisition_runner"] = not active
    details["active_acquisition_processes"] = active
    details["paths"] = {
        "results": str(run_directory),
        "checkpoint": str(run_directory / "checkpoint.json"),
        "backup": str(backup_dir / run_id),
        "lock": str(run_directory / ".experiment.lock"),
    }
    free = shutil.disk_usage(root).free
    details["results_filesystem_free_gb"] = round(free / 1e9, 1)
    checks["results_filesystem_free_space_20gb"] = free >= 20e9

    gate = validate_extension_gates(root, parent=parent)
    technical_reasons = [reason for reason in gate.reasons if reason not in EXTENSION_APPROVAL_PENDING_REASONS]
    checks["extension_gate_blocked_only_by_pending_approval"] = not technical_reasons
    details["extension_gate_reasons"] = list(gate.reasons)

    technical_pass = all(checks.values())
    approval_pending = bool(gate.reasons) or any(status != "APPROVED" for status in approval_status.values())
    generated = utc_now()
    report = {
        "schema_version": 1,
        "label": "NON-SCIENTIFIC ACQUISITION EXTENSION PREFLIGHT",
        "scientific_evidence": False,
        "generated_at": generated,
        "run_id": run_id,
        "parent_run_id": parent.run_id,
        "technical_pass": technical_pass,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "human_approval_pending": approval_pending,
        "only_blocker": ("HUMAN APPROVAL" if approval_pending else None) if technical_pass else "TECHNICAL CHECKS FAILED",
        "approval_status": approval_status,
        "approval_directory": str(approval_dir),
        "checks": checks,
        "details": details,
        "commands": extension_commands(run_id, str(backup_dir)),
    }
    stamp = generated.replace(":", "").replace("-", "")
    path = root / EXTENSION_PREFLIGHT_BASE / f"preflight-{stamp}.json"
    atomic_write_json(path, report)
    return {"ok": technical_pass, "report": str(path), "report_sha256": sha256_file(path), **report}

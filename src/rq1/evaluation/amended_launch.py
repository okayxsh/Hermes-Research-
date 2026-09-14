"""Operational commands of the amended controlled-recovery evaluation (Decision 012).

- ``check``/``check-report``: NON-SCIENTIFIC end-to-end check on valid_seen.
- ``prepare-approvals``: UNAPPROVED amendment/task/environment/protocol requests; never approves.
- ``freeze-tasks``: freezes the human-approved task set with its controlled failures.
- ``build-libraries``: deterministic Rule A libraries from the completed human core review.
- ``plan``/``preflight``: the launch gate and every technical check; no episode starts.
- ``run``/``resume``/``retry-failed --shard k``: approved scientific units only.
- ``validate``/``merge``/``rater-export``/``analyze``: post-run evidence handling.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq1.acquisition.environment import (
    DEFAULT_HF_HUB_CACHE,
    SBERT_MODEL,
    SBERT_REVISION,
    _run,
    model_digest,
    observed_environment,
    sbert_snapshot_sha256,
    verify_launch_environment,
)
from rq1.acquisition.extension_launch import _os_release
from rq1.acquisition.gates import _approved, load_task_manifest, queue_identity_sha256
from rq1.acquisition.launch import HERMES_PYTHON, PRODUCTION_BACKUP_DIR, _read_request, _writable, attempt_lineage, scientific_acquisition_allocation
from rq1.acquisition.reporting import write_report
from rq1.acquisition.skill_creation import prompt_hashes
from rq1.bridge.adapters.capabilities import default_data_dir, probe_alfworld_capabilities
from rq1.bridge.adapters.task_index import _resolve_task_family
from rq1.bridge.adapters.unseen_access import SCIENTIFIC_EVALUATION, UNSEEN_ACCESS_ENV, require_unseen_access
from rq1.evaluation.amended_analysis import analyze, read_labels, retrieval_quality
from rq1.evaluation.amended_executor import RESULT_SCHEMA, AmendedEvaluationExecutor
from rq1.evaluation.amended_libraries import (
    AmendedLibrary,
    build_amended_libraries,
    library_freeze_payload,
    load_raw_pool,
    nesting_problems,
    parse_review_csv,
    raw_feasibility,
    review_rows,
    select_core,
)
from rq1.evaluation.amended_matrix import (
    EvaluationTask,
    MatrixError,
    build_matrix,
    coverage_problems,
    evaluation_tasks,
    experiment_units,
    matrix_sha256,
    merge_shards,
    shard_manifest,
    shard_run_id,
    unit_identity,
)
from rq1.evaluation.amended_protocol import (
    AMENDMENT_DECISION_RECORD,
    AMENDMENT_FREEZE,
    AMENDMENT_VERSION,
    APPROVAL_DIR,
    CHECK_BASE,
    CHECK_PREFIX,
    CHECK_REPORT,
    CHECKPOINT_POLICY,
    COMBINED_CLOSEOUT_MANIFEST,
    CONDITIONS,
    CORE_REVIEW_FILE,
    EMBEDDING_DIMENSION,
    ENVIRONMENT_FREEZE,
    EVALUATION_POLICY_VERSION,
    EVALUATION_RUN_PREFIX,
    EVALUATION_SEEDS,
    EVALUATION_SPLIT,
    EVALUATION_TASK_COUNT,
    FROZEN_TASK_DIR,
    LIBRARY_DIR,
    LIBRARY_SIZES,
    MANIFEST_TYPE,
    MODEL_DIGEST,
    PREFLIGHT_BASE,
    PREPARATION_DIR,
    PROPOSAL_ARCHIVE_DIR,
    PROTOCOL_DOCUMENT,
    PROTOCOL_FREEZE,
    RAW_POOL_HASH,
    RAW_POOL_SNAPSHOT,
    RELEVANCE_RUBRIC_DOCUMENT,
    REPORT_DIR,
    RETRIEVAL_TOP_K,
    RUBRIC_DOCUMENT,
    SBERT_SNAPSHOT_SHA256,
    SHARD_COUNT,
    TASK_SELECTION,
    TASKS_PER_FAMILY,
    TOTAL_EPISODE_ACTION_BUDGET,
    evaluation_protocol_definition,
    evaluation_protocol_sha256,
)
from rq1.evaluation.task_preparation import validate_controlled_failure
from rq1.experiment.models import ExperimentUnit, canonical_hash
from rq1.experiment.persistence import ExperimentStore, atomic_write_json, bind_repository_configuration, durable_append_jsonl
from rq1.experiment.runner import DurableExperimentRunner, RunnerOptions
from rq1.freeze.validation import EVALUATION_ENVIRONMENT_REQUIRED, EVALUATION_EVIDENCE_MODE, git_state, read_freeze
from rq1.hermes.episode_driver import (
    ACTION_SELECTION_PROTOCOL,
    EXPERIMENT_MODEL,
    MAX_SELECTION_ATTEMPTS,
    MODEL_CONTEXT_LENGTH,
    MODEL_QUANTIZATION,
    MODEL_TIMEOUT_SECONDS,
    OUTPUT_TOKEN_CAP,
    RealEpisodeDriver,
    provider_settings,
)
from rq1.recovery.reference_route import derive_handcoded_reference
from rq1.retrieval.query import CANONICAL_FAILURE_MESSAGE, INVENTORY_NOT_OBSERVED_MARKER, QUERY_TEMPLATE_VERSION, query_template_hash
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.discovery import CANONICAL_FAMILIES, discover_tasks
from rq1.tasks.freeze import TaskFreezeError, freeze_manifest
from rq1.tasks.models import TaskManifest
from rq1.tasks.reporting import write_immutable
from rq1.tasks.validation import validate_manifest
from rq1.utils.hashing import sha256_file
from rq1.utils.time import utc_now

CHECK_TASK_ID = "valid_seen:look_at_obj_in_light-AlarmClock-None-DeskLamp-323/trial_T20190909_044715_250790"
CHECK_CONDITIONS = ("NoLib", "Accum-18")
CHECK_SEED = EVALUATION_SEEDS[0]
CHECK_LABEL = "NON-SCIENTIFIC PRELAUNCH EVALUATION CHECK"
MATRIX_DIR = Path("artifacts") / "evaluation-matrix"
REQUEST_NAMES = ("evaluation-amendment", "evaluation-task-freeze", "evaluation-environment", "evaluation-protocol")
DECISION_RECORD_FILES = tuple(f"docs/decisions/{name}" for name in (
    "003-skill-creation-policy.md", "005-relevance-labelling.md", "007-acquisition-execution-policy.md", "008-action-selection-episode-history.md",
    "009-observation-interface-corrections.md", "010-gemma-backbone-and-model-output-failures.md", "011-acquisition-extension-181-240.md",
    "012-pre-evaluation-library-amendment.md"))
# Files that determine the prepared checkpoints, detours, and oracle evidence.  They must be
# byte-identical between the preparation commit and the frozen commit.
PREPARATION_CODE_FILES = (
    "src/rq1/evaluation/task_preparation.py", "src/rq1/pilot/real_runtime/harnesses.py", "src/rq1/recovery/reference_route.py",
    "src/rq1/recovery/controlled_failure.py", "src/rq1/bridge/adapters/task_index.py", "src/rq1/bridge/adapters/alfworld_v042.py",
    "src/rq1/bridge/adapters/unseen_access.py", "src/rq1/hermes/episode_driver.py", "src/rq1/tasks/discovery.py", "src/rq1/bridge/app.py",
    "src/rq1/bridge/episode_manager.py",
)
APPROVAL_PENDING_REASONS = frozenset({
    "invalid evaluation-amendment freeze: FileNotFoundError",
    "invalid evaluation-environment freeze: FileNotFoundError",
    "invalid evaluation-protocol freeze: FileNotFoundError",
    "exactly one frozen evaluation task manifest is required (found 0)",
    "exactly one frozen controlled-failure set is required (found 0)",
})
CORE_PENDING_PREFIXES = ("exactly one evaluation library freeze is required (found 0)",)
RESULT_KEYS = ("result_schema", "task_id", "task_family", "seed", "condition", "library_hash", "model_digest", "checkpoint_id", "checkpoint_digest",
               "perturbation", "perturbation_digest", "failure_context", "retrieval", "retrieval_events", "recovery_memory_sha256", "post_failure_actions",
               "recovery_success", "task_completed", "termination_reason", "recovery_latency_actions", "recovery_latency_seconds",
               "invalid_action_selections", "retries", "selection_exhausted", "recovery_actions", "skill_writes")


def _blocked(**details: Any) -> dict[str, Any]:
    return {"ok": False, "status": "blocked", **details}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


# ============================================================== preparation artifacts


def preparation_directory(root: Path, value: str | None = None) -> Path:
    if value:
        return Path(value)
    candidates = sorted(path.parent for path in (root / PREPARATION_DIR).glob("*/controlled-failures.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"exactly one evaluation preparation is required (found {len(candidates)})")
    return candidates[0]


def load_preparation(directory: Path) -> dict[str, Any]:
    proposals = sorted(directory.glob(f"{MANIFEST_TYPE}-*.json"))
    if len(proposals) != 1:
        raise FileNotFoundError(f"exactly one evaluation task proposal is required in {directory}")
    return {
        "directory": directory,
        "proposal_path": proposals[0],
        "proposal": load_task_manifest(proposals[0]),
        "failures_path": directory / "controlled-failures.json",
        "failures": json.loads((directory / "controlled-failures.json").read_text(encoding="utf-8")),
        "routes_path": directory / "route-lengths.json",
        "access_log": directory / "valid-unseen-access.jsonl",
    }


def task_manifest_problems(manifest: TaskManifest, *, require_frozen: bool) -> list[str]:
    errors = list(validate_manifest(manifest, require_frozen=require_frozen))
    if manifest.manifest_type != MANIFEST_TYPE:
        errors.append("task manifest is not an amended evaluation manifest")
    if manifest.split != EVALUATION_SPLIT or any(task.split != EVALUATION_SPLIT or not task.task_id.startswith("valid_unseen:") for task in manifest.tasks):
        errors.append("evaluation tasks must be valid_unseen only")
    if manifest.actual_count != EVALUATION_TASK_COUNT or dict(manifest.family_counts) != {family: TASKS_PER_FAMILY for family in TASK_FAMILIES}:
        errors.append(f"evaluation needs exactly {EVALUATION_TASK_COUNT} tasks, {TASKS_PER_FAMILY} per family")
    if dict(manifest.selection_policy) != TASK_SELECTION:
        errors.append("evaluation task selection policy differs from the protocol")
    return errors


def failures_problems(failures: Mapping[str, Any], manifest: TaskManifest) -> list[str]:
    problems = []
    if failures.get("policy") != CHECKPOINT_POLICY or failures.get("canonical_failure_message") != CANONICAL_FAILURE_MESSAGE:
        problems.append("controlled-failure policy or canonical failure message differs from the protocol")
    try:
        tasks = evaluation_tasks(manifest.tasks, failures)
    except (MatrixError, KeyError, TypeError, ValueError) as exc:
        return [*problems, f"controlled failures do not match the task set: {exc}"]
    for task in tasks:
        definition = next(item for item in failures["tasks"] if item["task_id"] == task.task_id)
        prefix = len(task.prefix_actions)
        if (task.reference_actions[:prefix] != task.prefix_actions or task.reference_actions[prefix] != task.expected_next_action
                or not task.detour_action.startswith("go to ") or task.detour_action == task.expected_next_action
                or task.recovery_action_budget != TOTAL_EPISODE_ACTION_BUDGET - prefix - 1
                or task.recovery_action_budget < len(task.reference_actions) - prefix
                or definition.get("split") != EVALUATION_SPLIT or not re.fullmatch(r"[0-9a-f]{64}", task.checkpoint_digest)
                or not re.fullmatch(r"[0-9a-f]{64}", task.post_detour_digest)):
            problems.append(f"controlled failure is not protocol-valid: {task.task_id}")
    if failures.get("model_calls") not in (0, None):
        problems.append("task preparation called a model")
    return problems


def preparation_code_problems(root: Path, preparation_commit: str) -> list[str]:
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", preparation_commit, "HEAD"], cwd=root, capture_output=True)
    if ancestor.returncode != 0:
        return ["the preparation commit is not an ancestor of the repository head"]
    diff = subprocess.run(["git", "diff", "--quiet", preparation_commit, "HEAD", "--", *PREPARATION_CODE_FILES], cwd=root, capture_output=True)
    return [] if diff.returncode == 0 else ["code that determines the prepared checkpoints changed after the preparation"]


# ============================================================== libraries


def check_libraries(root: Path) -> dict[str, AmendedLibrary]:
    """NON-SCIENTIFIC plumbing libraries: rank-1 placeholder core; this is not a core selection."""
    pool = load_raw_pool(root / RAW_POOL_SNAPSHOT)
    return build_amended_libraries(pool, {family: next(entry.skill_id for entry in pool if entry.task_family == family and entry.family_rank == 1)
                                          for family in TASK_FAMILIES})


def build_libraries(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not getattr(args, "yes", False):
        return _blocked(reason="building the evaluation libraries requires --yes")
    commit, _clean, _error = git_state(root)
    amendment, errors = read_freeze(root / AMENDMENT_FREEZE, "evaluation-amendment")
    if amendment is None or errors or not _approved(amendment.approval) or amendment.repository_commit != commit:
        return _blocked(reason="the evaluation amendment must be human-approved and frozen at this commit first", errors=errors)
    review_path = Path(args.review_file) if getattr(args, "review_file", None) else root / CORE_REVIEW_FILE
    pool = load_raw_pool(root / RAW_POOL_SNAPSHOT)
    selection = select_core(pool, parse_review_csv(review_path.read_bytes()))
    if not selection.complete:
        return _blocked(reason="the human core validation is incomplete or has a family without a PASS", selection=selection.to_dict())
    libraries = build_amended_libraries(pool, selection.core)
    payload = library_freeze_payload(libraries, selection=selection, review_sha256=sha256_file(review_path),
                                     pool_snapshot_sha256=sha256_file(root / RAW_POOL_SNAPSHOT))
    payload.update({"repository_commit": commit, "review_file": str(review_path), "amendment_freeze_fingerprint": amendment.input_fingerprint,
                    "generated_at": utc_now()})
    path = root / LIBRARY_DIR / f"libraries-{canonical_hash(payload['library_hashes'])[:16]}.json"
    if list((root / LIBRARY_DIR).glob("libraries-*.json")):
        return _blocked(reason=f"an evaluation library freeze already exists and is immutable: {root / LIBRARY_DIR}")
    write_immutable(path, payload)
    return {"ok": True, "status": "libraries_built", "path": str(path), "sha256": sha256_file(path), "core": selection.core,
            "library_hashes": payload["library_hashes"], "sizes": {condition: libraries[condition].size for condition in CONDITIONS}}


def load_library_freeze(root: Path) -> tuple[dict[str, Any] | None, dict[str, AmendedLibrary], list[str]]:
    paths = sorted((root / LIBRARY_DIR).glob("libraries-*.json"))
    if len(paths) != 1:
        return None, {}, [f"exactly one evaluation library freeze is required (found {len(paths)})"]
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    review_path = Path(payload.get("review_file") or root / CORE_REVIEW_FILE)
    problems = []
    try:
        pool = load_raw_pool(root / RAW_POOL_SNAPSHOT)
        selection = select_core(pool, parse_review_csv(review_path.read_bytes()))
        rebuilt = build_amended_libraries(pool, selection.core) if selection.complete else {}
    except (OSError, ValueError, KeyError) as exc:
        return payload, {}, [f"library freeze cannot be recomputed: {type(exc).__name__}: {exc}"]
    if not selection.complete:
        problems.append("core validation is incomplete or has a family without a PASS")
    if sha256_file(review_path) != payload.get("core_review_sha256"):
        problems.append("core review changed after the library freeze")
    if rebuilt and {condition: rebuilt[condition].content_sha256 for condition in CONDITIONS} != payload.get("library_hashes"):
        problems.append("library freeze differs from the Rule A reconstruction")
    libraries = {condition: AmendedLibrary(condition, tuple(payload["libraries"][condition]["skills"])) for condition in CONDITIONS}
    problems.extend(nesting_problems(libraries))
    if {condition: libraries[condition].content_sha256 for condition in CONDITIONS} != payload.get("library_hashes"):
        problems.append("library freeze content hash mismatch")
    return payload, libraries, problems


# ============================================================== configuration and environment


def evaluation_run_configuration(root: Path, *, scientific: bool, libraries: Mapping[str, AmendedLibrary], matrix_sha: str,
                                 shard: int | None = None, freezes: Mapping[str, str] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": EVALUATION_POLICY_VERSION,
        "model_name": EXPERIMENT_MODEL,
        "runtime_settings": {
            "temperature": 0,
            "inference_seed": "unit_replicate_seed",
            "action_selection_protocol": ACTION_SELECTION_PROTOCOL,
            "max_selection_attempts": MAX_SELECTION_ATTEMPTS,
            "output_token_cap": OUTPUT_TOKEN_CAP,
            "model_context_length": MODEL_CONTEXT_LENGTH,
            "model_timeout_seconds": MODEL_TIMEOUT_SECONDS,
            "total_action_budget": TOTAL_EPISODE_ACTION_BUDGET,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "embedding_model": SBERT_MODEL,
            "embedding_model_revision": SBERT_REVISION,
        },
        "protocol_sha256": evaluation_protocol_sha256(),
        "matrix_sha256": matrix_sha,
        "shard": shard,
        "library_hashes": {condition: libraries[condition].content_sha256 for condition in CONDITIONS},
        "library_sizes": {condition: libraries[condition].size for condition in CONDITIONS},
        "prompt_hashes": prompt_hashes(root),
        "scientific_evidence": scientific,
    }
    if freezes:
        value["freeze_fingerprints"] = dict(freezes)
    return value


def evaluation_observed_environment(root: Path, *, task_queue_sha256: str | None, data_identity: str | None, matrix_sha: str | None) -> dict[str, Any]:
    value = observed_environment(root, task_queue_sha256=task_queue_sha256, alfworld_data_identity=data_identity)
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", _run(("nvidia-smi",)) or "")
    value.update({
        "os": _os_release(),
        "kernel": os.uname().release if hasattr(os, "uname") else None,
        "cuda_version": cuda.group(1) if cuda else None,
        "torch_version": (value.get("packages") or {}).get("torch"),
        "torch_cuda_version": _run((sys.executable, "-c", "import torch; print(torch.version.cuda)")),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "evaluation_matrix_sha256": matrix_sha,
        "valid_unseen_access": SCIENTIFIC_EVALUATION,
    })
    return value


def _embedder() -> Any:
    from rq1.retrieval.embedder import SentenceBERTEmbedder

    embedder = SentenceBERTEmbedder(SBERT_MODEL, cache_folder=DEFAULT_HF_HUB_CACHE, revision=SBERT_REVISION, local_files_only=True)
    dimension = len(embedder.encode(["embedding dimension probe"])[0])
    if dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(f"Sentence-BERT embedding dimension {dimension} differs from the frozen {EMBEDDING_DIMENSION}")
    return embedder


def _task_family(data_dir: Path, task_id: str) -> str:
    split, relative = task_id.split(":", 1)
    payload = json.loads((data_dir / "json_2.1.1" / split / relative / "traj_data.json").read_text(encoding="utf-8"))
    return CANONICAL_FAMILIES[_resolve_task_family(payload.get("task_type"))]


# ============================================================== non-scientific check (valid_seen)


def evaluation_check(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_id = str(args.run_id)
    if not run_id.startswith(CHECK_PREFIX):
        return _blocked(reason=f"non-scientific evaluation check run IDs must start with {CHECK_PREFIX}")
    if os.environ.get(UNSEEN_ACCESS_ENV):
        return _blocked(reason=f"unset {UNSEEN_ACCESS_ENV}: the evaluation check uses valid_seen only")
    store = ExperimentStore(root, run_id, base=root / CHECK_BASE)
    plan_path = store.directory / "check-plan.json"
    commit, clean, _error = git_state(root)
    libraries = check_libraries(root)
    if args.resume and not plan_path.is_file():
        return _blocked(reason="cannot resume an unknown evaluation check")
    if not args.resume and plan_path.exists():
        return _blocked(reason="check already exists; use --resume")
    embedder = _embedder()
    data_dir = default_data_dir()
    family = _task_family(data_dir, CHECK_TASK_ID)
    store.directory.mkdir(parents=True, exist_ok=True)
    with RealEpisodeDriver(root, data_dir=data_dir, bridge_log_root=store.directory / "logs" / "bridge") as driver:
        if not args.resume:
            route = derive_handcoded_reference(data_dir, CHECK_TASK_ID, "valid_seen")
            definition, attempts = validate_controlled_failure(driver, output_dir=store.directory / "preparation", task_id=CHECK_TASK_ID,
                                                               task_family=family, split="valid_seen", reference_actions=route.actions,
                                                               run_id=run_id + "-preparation")
            if definition is None:
                return _blocked(reason="the valid_seen check task admits no validated controlled failure", attempts=attempts)
            atomic_write_json(plan_path, {"schema_version": 1, "label": CHECK_LABEL, "scientific_evidence": False, "task": definition,
                                          "checkpoint_attempts": attempts, "conditions": list(CHECK_CONDITIONS), "seed": CHECK_SEED,
                                          "library_note": "rank-1 placeholder core for plumbing only; not a core selection",
                                          "library_hashes": {condition: libraries[condition].content_sha256 for condition in CONDITIONS},
                                          "embedding_dimension": EMBEDDING_DIMENSION, "repository_commit": commit, "created_at": utc_now()})
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        definition = plan["task"]
        task = EvaluationTask(CHECK_TASK_ID, family, 1, definition["checkpoint_id"], definition["checkpoint_digest"], definition["post_detour_digest"],
                              definition["detour_action"], definition["expected_next_action"], tuple(definition["prefix_actions"]),
                              tuple(definition["reference_actions"]), int(definition["recovery_action_budget"]))
        units = []
        for index, condition in enumerate(CHECK_CONDITIONS, 1):
            identity = {**unit_identity(task, CHECK_SEED, condition), "non_scientific_check": run_id}
            units.append(ExperimentUnit(
                phase="evaluation", task_id=task.task_id, task_index=index, condition=condition, seed=CHECK_SEED, identity=identity,
                payload={"global_order": index, "unit_key": canonical_hash(identity), "cell_index": 0, "position_in_cell": index - 1, "shard": 0,
                         "task_family": family, "checkpoint_id": task.checkpoint_id, "recovery_action_budget": task.recovery_action_budget},
                library_name=condition, library_size=libraries[condition].size, library_hash=libraries[condition].content_sha256,
            ))
        matrix_sha = canonical_hash([unit.run_key for unit in units])
        durable_append_jsonl(store.directory / "invocations.jsonl", {"mode": "resume" if args.resume else "run", "repository_commit": commit,
                                                                     "clean": clean, "max_runs": args.max_runs, "timestamp": utc_now()})
        executor = AmendedEvaluationExecutor(driver, tasks={task.task_id: task}, libraries=libraries, embedder=embedder, scientific=False,
                                             split="valid_seen", provenance={"evaluation_protocol_sha256": evaluation_protocol_sha256(),
                                                                             "evaluation_matrix_sha256": matrix_sha, "check_plan_sha256": sha256_file(plan_path)})
        configuration = evaluation_run_configuration(root, scientific=False, libraries=libraries, matrix_sha=matrix_sha)
        result = DurableExperimentRunner(store, progress=print).run(
            "evaluation", units, bind_repository_configuration(root, configuration), executor,
            RunnerOptions(resume=bool(args.resume), max_runs=args.max_runs, fail_fast=True),
        )
    return {"ok": result["status"] in {"completed", "paused", "incomplete"}, "label": CHECK_LABEL, "scientific_evidence": False, **result}


def evaluation_check_report(root: Path, run_id: str) -> dict[str, Any]:
    store = ExperimentStore(root, run_id, base=root / CHECK_BASE)
    plan_path = store.directory / "check-plan.json"
    if not run_id.startswith(CHECK_PREFIX) or not plan_path.is_file() or not store.manifest_path.is_file():
        return _blocked(reason="unknown non-scientific evaluation check")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = [row for row in store.read_results(repair_tail=False) if row.get("phase") == "evaluation"]
    lineage = attempt_lineage(rows)
    records = sorted(store.terminal_results(phase="evaluation", repair_tail=False).values(), key=lambda item: int(item["task_index"]))
    completed = [record for record in records if record.get("status") == "completed"]
    configuration = json.loads((store.manifests / "evaluation.json").read_text(encoding="utf-8"))["configuration"]
    runtime = configuration.get("runtime_settings", {})
    invocations = _jsonl(store.directory / "invocations.jsonl")
    commit, clean, _error = git_state(root)
    events: dict[str, list[dict[str, Any]]] = {}
    retrieval_lines: dict[str, int] = {}
    for record in completed:
        paths = [store.directory / path for path in record.get("log_paths") or []]
        events[record["condition"]] = next((_jsonl(path) for path in paths if path.name == "episode-events.jsonl"), [])
        retrieval_lines[record["condition"]] = sum(len(_jsonl(path)) for path in paths if path.name == "retrieval.jsonl")
    by_condition = {record["condition"]: record for record in completed}

    def prompts(condition: str) -> list[str]:
        return [event["payload"]["prompt"] for event in events.get(condition, []) if event.get("event") == "model_selection"]

    def memory_blocks(condition: str) -> list[str]:
        return [prompt.split("RECOVERY MEMORY:")[1].split("ADMISSIBLE ACTIONS:")[0] for prompt in prompts(condition) if "RECOVERY MEMORY:" in prompt]

    library, nolib = by_condition.get("Accum-18", {}), by_condition.get("NoLib", {})
    retrieval_events = [json.loads(line) for record in completed for path in record.get("log_paths") or [] if path.endswith("retrieval.jsonl")
                        for line in (store.directory / path).read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = {
        "non_scientific_configuration": configuration.get("scientific_evidence") is False and all(record.get("scientific_evidence") is False for record in completed),
        "both_conditions_completed": set(by_condition) == set(CHECK_CONDITIONS) and len(records) == len(CHECK_CONDITIONS),
        "real_episodes_executed": bool(completed) and all(
            {"alfworld_start", "alfworld_step"} <= {(event.get("payload") or {}).get("tool") for event in events.get(condition, []) if event.get("event") == "tool_result"}
            and any(event.get("event") == "task_goal_frozen" for event in events.get(condition, [])) and prompts(condition)
            for condition in by_condition),
        "controlled_failure_oracle_validated_before_run": (plan["task"].get("oracle") or {}).get("validated") is True,
        "frozen_checkpoint_and_detour_reproduced": bool(completed) and all(
            record.get("checkpoint_digest") == plan["task"]["checkpoint_digest"] and record.get("perturbation_digest") == plan["task"]["post_detour_digest"]
            for record in completed),
        "library_retrieval_exactly_once_top3": library.get("retrieval_events") == 1 and len((library.get("retrieval") or {}).get("top") or []) == 3
                                               and retrieval_lines.get("Accum-18") == 1,
        "nolib_no_retrieval": nolib.get("retrieval_events") == 0 and nolib.get("no_retrieval") is True and not (nolib.get("retrieval") or {}).get("top"),
        "retrieval_query_frozen": bool(retrieval_events) and all(event.get("query_version") == QUERY_TEMPLATE_VERSION
                                                                 and event.get("query_template_hash") == query_template_hash() for event in retrieval_events),
        "recovery_memory_injected_once_per_episode": all(sum(event.get("event") == "recovery_memory_injected" for event in events.get(condition, [])) == 1
                                                         for condition in CHECK_CONDITIONS),
        "memory_without_scores_and_nolib_marker": bool(memory_blocks("Accum-18")) and all("score" not in block for block in memory_blocks("Accum-18"))
                                                  and all('"no_retrieved_skills_available": true' in block for block in memory_blocks("NoLib")),
        "no_condition_label_in_prompts": all(label not in prompt for condition in CHECK_CONDITIONS for prompt in prompts(condition) for label in CONDITIONS),
        "inference_seed_is_replicate_seed": all(event["payload"].get("inference_seed") == CHECK_SEED for condition in CHECK_CONDITIONS
                                                for event in events.get(condition, []) if event.get("event") == "model_selection"),
        "frozen_controller_settings": configuration.get("model_name") == EXPERIMENT_MODEL and runtime.get("output_token_cap") == OUTPUT_TOKEN_CAP
                                      and runtime.get("model_context_length") == MODEL_CONTEXT_LENGTH and runtime.get("temperature") == 0
                                      and runtime.get("max_selection_attempts") == MAX_SELECTION_ATTEMPTS
                                      and runtime.get("action_selection_protocol") == ACTION_SELECTION_PROTOCOL and runtime.get("retrieval_top_k") == 3,
        "episode_action_budget_respected": bool(completed) and all(
            record.get("post_failure_actions", 0) <= record.get("recovery_action_budget", -1)
            and sum((event.get("payload") or {}).get("tool") == "alfworld_step" for event in events.get(record["condition"], [])
                    if event.get("event") == "tool_result") <= TOTAL_EPISODE_ACTION_BUDGET for record in completed),
        "result_schema_complete": bool(completed) and all(all(key in record for key in RESULT_KEYS) and record.get("result_schema") == RESULT_SCHEMA
                                                          for record in completed),
        "recovery_latency_logged": bool(completed) and all((record.get("recovery_latency_actions") is not None) == bool(record.get("recovery_success"))
                                                           for record in completed),
        "no_skill_writes": all(record.get("skill_writes") is False for record in completed)
                           and not any(event.get("event") == "post_success_learning" for condition in CHECK_CONDITIONS for event in events.get(condition, [])),
        "resume_invocation_observed": len(invocations) >= 2 and any(item.get("mode") == "resume" for item in invocations),
        "no_unresolved_infrastructure_failures": bool(records) and all(record.get("status") == "completed" for record in records),
        "authorized_retry_lineage": lineage["authorized"] and lineage["duplicate_completed_units"] == 0,
        "embedding_dimension_768": plan.get("embedding_dimension") == EMBEDDING_DIMENSION,
        "single_clean_commit": bool(clean) and bool(invocations) and all(item.get("repository_commit") == commit and item.get("clean") is True for item in invocations),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1, "mode": EVALUATION_EVIDENCE_MODE, "label": CHECK_LABEL, "scientific_evidence": False, "run_id": run_id,
        "repository_commit": commit, "generated_at": utc_now(), "passed": passed, "checks": checks,
        "units": [{key: record.get(key) for key in ("condition", "status", "recovery_success", "termination_reason", "post_failure_actions",
                                                     "recovery_latency_actions", "recovery_latency_seconds", "invalid_action_selections", "retrieval_events")}
                  for record in records],
        "controlled_failure": {key: plan["task"].get(key) for key in ("task_id", "checkpoint_index", "prefix_actions", "expected_next_action", "detour_action",
                                                                      "recovery_action_budget", "checkpoint_digest", "post_detour_digest")},
        "retrieved_skill_ids": [item.get("skill_id") for item in (library.get("retrieval") or {}).get("top") or []],
        "output_directory": str(store.directory),
    }
    path = store.directory / CHECK_REPORT
    atomic_write_json(path, report)
    return {"ok": passed, "report": str(path), "report_sha256": sha256_file(path), **report}


# ============================================================== approval requests and task freeze


def prepare_approvals(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    commit, clean, error = git_state(root)
    if error or not clean or not commit:
        return _blocked(reason="approval requests require a clean committed repository")
    preparation = load_preparation(preparation_directory(root, getattr(args, "preparation", None)))
    proposal, failures = preparation["proposal"], preparation["failures"]
    evidence_path = Path(args.evidence_report)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    problems = [*task_manifest_problems(proposal, require_frozen=False), *failures_problems(failures, proposal),
                *preparation_code_problems(root, str(proposal.repository_commit))]
    if proposal.status != "proposed" or failures.get("task_manifest_sha256") != proposal.manifest_sha256:
        problems.append("preparation proposal and controlled failures do not belong together")
    if evidence.get("mode") != EVALUATION_EVIDENCE_MODE or evidence.get("passed") is not True or evidence.get("repository_commit") != commit:
        problems.append("evidence report is not a passed non-scientific evaluation check at this commit")
    pool = load_raw_pool(root / RAW_POOL_SNAPSHOT)
    if not raw_feasibility(pool)["feasible"]:
        problems.append("the raw pool cannot supply 3 skills per family")
    template = root / CORE_REVIEW_FILE
    if not template.is_file():
        problems.append("the fast core-validation package is missing")
    if problems:
        return _blocked(reasons=problems)
    tasks = evaluation_tasks(proposal.tasks, failures)
    matrix = build_matrix(tasks)
    matrix_sha = matrix_sha256(matrix)
    matrix_dir = root / MATRIX_DIR / commit[:12]
    if matrix_dir.exists():
        return _blocked(reason=f"matrix manifests already exist and are immutable: {matrix_dir}")
    write_immutable(matrix_dir / "matrix.json", {"schema_version": 1, "kind": "rq1-evaluation-matrix", "matrix_sha256": matrix_sha,
                                                 "schedule": evaluation_protocol_definition()["matrix"]["schedule"], "units": matrix})
    shard_hashes = {}
    for shard in range(1, SHARD_COUNT + 1):
        write_immutable(matrix_dir / f"shard-{shard}-of-{SHARD_COUNT}.json", shard_manifest(matrix, shard))
        shard_hashes[str(shard)] = sha256_file(matrix_dir / f"shard-{shard}-of-{SHARD_COUNT}.json")
    directory = root / APPROVAL_DIR / commit[:12]
    paths = {name: directory / f"{name}.approval.json" for name in REQUEST_NAMES}
    if any(path.exists() for path in paths.values()):
        return _blocked(reason=f"approval requests already exist and are immutable: {directory}")
    queue_sha = queue_identity_sha256(proposal)
    unapproved = {"status": "UNAPPROVED", "approved_by": None, "approved_at": None, "reference": None}
    how = "A human reviewer sets status to APPROVED and fills approved_by, approved_at (UTC ISO-8601), and reference, then runs the command."
    evidence_reference = {"path": str(evidence_path), "sha256": sha256_file(evidence_path), "run_id": evidence.get("run_id")}
    definition = evaluation_protocol_definition()
    documents: dict[str, dict[str, Any]] = {
        "evaluation-amendment": {
            "schema_version": 1, "approval_kind": "evaluation-amendment", "approval": dict(unapproved),
            "inputs": {"repository_commit": commit, "amendment_version": AMENDMENT_VERSION, "protocol": definition, "protocol_sha256": evaluation_protocol_sha256(),
                       "raw_pool_hash": RAW_POOL_HASH, "raw_pool_snapshot_sha256": sha256_file(root / RAW_POOL_SNAPSHOT),
                       "combined_closeout_manifest_sha256": sha256_file(root / COMBINED_CLOSEOUT_MANIFEST),
                       "skill_quality_rubric_sha256": sha256_file(root / RUBRIC_DOCUMENT), "core_review_template_sha256": sha256_file(template),
                       "decision_record_sha256": sha256_file(root / AMENDMENT_DECISION_RECORD)},
            "evidence_report": evidence_reference,
            "attestation": [
                "Acquisition ended at the pre-approved hard cap of 240 before evaluation; the raw 50-skill pool (11/8/4/9/15/3) cannot build Clean-24, Accum-60, or Accum-96.",
                "No evaluation episode has run and no evaluation outcome exists; the amendment is driven by acquisition feasibility only.",
                "Replacement conditions NoLib 0, Core-6 6, Accum-12 12, Accum-18 18 under Rule A: the core is the earliest human PASS per family; extras are the earliest remaining acquired skills; balanced, nested, identical core.",
                "Four conditions, 30 valid_unseen tasks (5 per family), seeds 11/29/47, and 360 episodes are retained.",
                "No skill is fabricated, no acquisition episode is discarded, no quality rule is loosened, and no semantic deduplication is introduced; skill-quality-rubric-v1 is derived only from Decisions 003/007 and the existing validation prompts.",
            ],
            "how_to_approve": how,
            "command": f"python -m rq1.cli freeze evaluation-amendment --approval-file {paths['evaluation-amendment']} --pilot-report {evidence_path} --yes",
        },
        "evaluation-task-freeze": {
            "schema_version": 1, "approval_kind": "evaluation-task-freeze", **unapproved,
            "subject": {"proposal_path": str(preparation["proposal_path"]), "proposal_sha256": sha256_file(preparation["proposal_path"]),
                        "manifest_sha256": proposal.manifest_sha256, "task_queue_sha256": queue_sha, "repository_commit": proposal.repository_commit,
                        "split": proposal.split, "actual_count": proposal.actual_count, "family_counts": dict(proposal.family_counts),
                        "selection_policy": dict(proposal.selection_policy), "data_root_identity": proposal.data_root_identity,
                        "tasks": [{"task_id": task.task_id, "family": task.task_family, "reference_route_length": len(task.reference_actions),
                                   "checkpoint_prefix_length": len(task.prefix_actions), "expected_next_action": task.expected_next_action,
                                   "detour_action": task.detour_action, "recovery_action_budget": task.recovery_action_budget} for task in tasks],
                        "controlled_failures": {"path": str(preparation["failures_path"]), "sha256": sha256_file(preparation["failures_path"])},
                        "route_lengths": {"path": str(preparation["routes_path"]), "sha256": sha256_file(preparation["routes_path"])},
                        "valid_unseen_access_log": {"path": str(preparation["access_log"]), "sha256": sha256_file(preparation["access_log"])},
                        "replaced_candidates": {family: [item["task_id"] for item in items] for family, items in failures.get("replaced_candidates", {}).items()}},
            "attestation": [
                "Exactly 30 valid_unseen tasks, 5 per family: the longest ALFWorld hand-coded expert routes per family, ties by task ID; replacements only for tasks without an oracle-validated controlled failure before freeze.",
                "Selection used valid_unseen metadata and hand-coded expert routes only under the logged task-preparation authorization; no model was called and no agent or acquisition outcome was used.",
                "Every task has a midpoint-near navigation checkpoint and a deterministic reversible go-to detour proven solvable by the real-bridge oracle within the 50-action budget; oracle information is never shown to the model.",
            ],
            "how_to_approve": how,
            "command": (f"python -m rq1.cli evaluation-amended freeze-tasks --proposal {preparation['proposal_path']} "
                        f"--controlled-failures {preparation['failures_path']} --approval-file {paths['evaluation-task-freeze']} --yes"),
        },
        "evaluation-environment": {
            "schema_version": 1, "approval_kind": "evaluation-environment", "approval": dict(unapproved),
            "inputs": evaluation_observed_environment(root, task_queue_sha256=queue_sha, data_identity=proposal.data_root_identity, matrix_sha=matrix_sha),
            "evidence_report": evidence_reference,
            "attestation": [
                f"The recorded commit, OS, GPU, driver/CUDA, Python, torch, ALFWorld, Hermes, Ollama, {EXPERIMENT_MODEL} ({MODEL_QUANTIZATION}, digest {MODEL_DIGEST}), provider settings (num_predict {OUTPUT_TOKEN_CAP}, num_ctx {MODEL_CONTEXT_LENGTH}, temperature 0, think false), and Sentence-BERT {SBERT_MODEL} @ {SBERT_REVISION} (snapshot {SBERT_SNAPSHOT_SHA256}, {EMBEDDING_DIMENSION} dimensions) are the approved evaluation environment.",
                "Each evaluation worker (shard) must reproduce every enforced identity at launch; host name and GPU are recorded only.",
                "Seed and temperature are fixed, but provider inference is not claimed deterministic (Decision 010).",
            ],
            "how_to_approve": how,
            "command": f"python -m rq1.cli freeze evaluation-environment --approval-file {paths['evaluation-environment']} --pilot-report {evidence_path} --yes",
        },
        "evaluation-protocol": {
            "schema_version": 1, "approval_kind": "evaluation-protocol", "approval": dict(unapproved),
            "inputs": {"repository_commit": commit, "protocol": definition, "protocol_sha256": evaluation_protocol_sha256(),
                       "task_manifest_sha256": proposal.manifest_sha256, "task_queue_sha256": queue_sha,
                       "controlled_failures_sha256": sha256_file(preparation["failures_path"]), "evaluation_matrix_sha256": matrix_sha,
                       "shard_manifest_sha256": shard_hashes, "prompt_hashes": prompt_hashes(root),
                       "decision_records_sha256": {path: sha256_file(root / path) for path in DECISION_RECORD_FILES},
                       "inference_seeds": list(EVALUATION_SEEDS), "skill_quality_rubric_sha256": sha256_file(root / RUBRIC_DOCUMENT),
                       "relevance_rubric_sha256": sha256_file(root / RELEVANCE_RUBRIC_DOCUMENT), "protocol_document_sha256": sha256_file(root / PROTOCOL_DOCUMENT),
                       "preparation_commit": proposal.repository_commit, "preparation_code_unchanged": True,
                       "total_action_budget": TOTAL_EPISODE_ACTION_BUDGET},
            "evidence_report": evidence_reference,
            "attestation": [
                "Conditions NoLib/Core-6/Accum-12/Accum-18; seeds 11/29/47 (also the inference seed of each unit); 360 units in 6 balanced shards; no outcome-dependent ordering.",
                "Controlled reversible action perturbation after a frozen midpoint-near checkpoint; canonical failure message; each unit must reproduce the frozen digests.",
                "Sentence-BERT top-3 exactly once immediately after the failure (none for NoLib); query = task goal, observation, inventory, canonical failure message; no action history; scores logged but never shown; one structured recovery-memory block.",
                "Primary metric: conditional recovery rate; secondary: task completion, recovery latency (actions, seconds), invalid selections, retries, infrastructure failures; Precision@3 and retrieval noise from two blinded binary raters with Cohen's kappa and adjudication; bootstrap CIs; descriptive Spearman.",
                "Evaluation never writes skills; libraries are built deterministically from the completed human core review before launch.",
            ],
            "how_to_approve": how,
            "command": f"python -m rq1.cli freeze evaluation-protocol --approval-file {paths['evaluation-protocol']} --pilot-report {evidence_path} --yes",
        },
    }
    for name, document in documents.items():
        write_report(paths[name], document)
    return {"ok": True, "status": "UNAPPROVED", "approval_requests": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
            "task_queue_sha256": queue_sha, "matrix_sha256": matrix_sha, "matrix_directory": str(matrix_dir), "shard_manifest_sha256": shard_hashes,
            "protocol_sha256": evaluation_protocol_sha256()}


def freeze_tasks(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not getattr(args, "yes", False):
        return _blocked(reason="freezing the evaluation tasks requires --yes")
    proposal_path, failures_path = Path(args.proposal), Path(args.controlled_failures)
    proposal = load_task_manifest(proposal_path)
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    approval = json.loads(Path(args.approval_file).read_text(encoding="utf-8"))
    subject = approval.get("subject") or {}
    problems = [*task_manifest_problems(proposal, require_frozen=False), *failures_problems(failures, proposal)]
    if (subject.get("controlled_failures") or {}).get("sha256") != sha256_file(failures_path):
        problems.append("approval does not reference these controlled failures")
    if problems:
        return _blocked(reasons=problems)
    destination = root / FROZEN_TASK_DIR / f"{MANIFEST_TYPE}-{proposal.manifest_sha256[:16]}.json"
    try:
        frozen = freeze_manifest(root, proposal, approval, destination)
    except (TaskFreezeError, FileExistsError) as exc:
        return _blocked(reason=str(exc))
    failures_copy = root / FROZEN_TASK_DIR / f"controlled-failures-{sha256_file(failures_path)[:16]}.json"
    if not failures_copy.exists():
        failures_copy.write_bytes(failures_path.read_bytes())
    archive = root / PROPOSAL_ARCHIVE_DIR / proposal_path.name
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(proposal_path, archive)
    return {"ok": True, "frozen": str(destination), "frozen_sha256": sha256_file(destination), "controlled_failures": str(failures_copy),
            "task_queue_sha256": queue_identity_sha256(frozen), "manifest_sha256": frozen.manifest_sha256}


# ============================================================== launch gate


@dataclass
class EvaluationGate:
    valid: bool
    reasons: list[str]
    manifest: TaskManifest | None = None
    failures: dict[str, Any] | None = None
    failures_path: Path | None = None
    tasks: tuple[EvaluationTask, ...] = ()
    matrix: list[dict[str, Any]] = field(default_factory=list)
    libraries: dict[str, AmendedLibrary] = field(default_factory=dict)
    library_payload: dict[str, Any] | None = None
    freezes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "reasons": list(self.reasons), "task_queue_sha256": queue_identity_sha256(self.manifest) if self.manifest else None,
                "matrix_sha256": matrix_sha256(self.matrix) if self.matrix else None,
                "library_hashes": {condition: library.content_sha256 for condition, library in self.libraries.items()} or None,
                "freeze_fingerprints": {kind: freeze.input_fingerprint for kind, freeze in self.freezes.items()}}


def validate_evaluation_gates(root: Path) -> EvaluationGate:
    reasons: list[str] = []
    commit, clean, error = git_state(root)
    if error:
        reasons.append(error)
    elif not clean:
        reasons.append("repository working tree is not clean")
    freezes = {}
    for kind, path in (("evaluation-amendment", AMENDMENT_FREEZE), ("evaluation-environment", ENVIRONMENT_FREEZE), ("evaluation-protocol", PROTOCOL_FREEZE)):
        freeze, errors = read_freeze(root / path, kind)
        reasons.extend(errors)
        if freeze is None:
            continue
        freezes[kind] = freeze
        if freeze.repository_commit != commit or freeze.inputs.get("repository_commit") != commit:
            reasons.append(f"repository commit changed since {kind} freeze")
        if not _approved(freeze.approval):
            reasons.append(f"{kind} freeze lacks human approval")
    gate = EvaluationGate(False, reasons, freezes=freezes)
    manifests = sorted((root / FROZEN_TASK_DIR).glob(f"{MANIFEST_TYPE}-*.json"))
    failure_files = sorted((root / FROZEN_TASK_DIR).glob("controlled-failures-*.json"))
    if len(manifests) != 1:
        reasons.append(f"exactly one frozen evaluation task manifest is required (found {len(manifests)})")
    if len(failure_files) != 1:
        reasons.append(f"exactly one frozen controlled-failure set is required (found {len(failure_files)})")
    if len(manifests) == 1 and len(failure_files) == 1:
        gate.manifest = load_task_manifest(manifests[0])
        gate.failures_path = failure_files[0]
        gate.failures = json.loads(failure_files[0].read_text(encoding="utf-8"))
        reasons.extend(task_manifest_problems(gate.manifest, require_frozen=True))
        if gate.manifest.repository_commit != commit:
            reasons.append("frozen evaluation task manifest was frozen at a different commit")
        if not gate.manifest.approved_at or not gate.manifest.approval_reference:
            reasons.append("frozen evaluation task manifest lacks approval metadata")
        failure_problems = failures_problems(gate.failures, gate.manifest) if gate.failures.get("task_manifest_sha256") else ["controlled failures lack their task manifest reference"]
        reasons.extend(failure_problems)
        if not failure_problems:
            gate.tasks = evaluation_tasks(gate.manifest.tasks, gate.failures)
            gate.matrix = build_matrix(gate.tasks)
    amendment = freezes.get("evaluation-amendment")
    if amendment is not None and (amendment.inputs.get("protocol_sha256") != evaluation_protocol_sha256() or amendment.inputs.get("raw_pool_hash") != RAW_POOL_HASH):
        reasons.append("evaluation amendment freeze differs from the repository protocol or raw pool")
    protocol = freezes.get("evaluation-protocol")
    if protocol is not None:
        inputs = protocol.inputs
        if inputs.get("protocol") != evaluation_protocol_definition() or inputs.get("protocol_sha256") != evaluation_protocol_sha256():
            reasons.append("evaluation protocol freeze differs from the repository protocol")
        if inputs.get("inference_seeds") != list(EVALUATION_SEEDS) or inputs.get("prompt_hashes") != prompt_hashes(root):
            reasons.append("evaluation protocol freeze seeds or prompts differ")
        if gate.manifest is not None and inputs.get("task_queue_sha256") != queue_identity_sha256(gate.manifest):
            reasons.append("evaluation protocol freeze references a different task set")
        if gate.failures_path is not None and inputs.get("controlled_failures_sha256") != sha256_file(gate.failures_path):
            reasons.append("evaluation protocol freeze references different controlled failures")
        if gate.matrix and inputs.get("evaluation_matrix_sha256") != matrix_sha256(gate.matrix):
            reasons.append("evaluation protocol freeze references a different matrix")
    environment = freezes.get("evaluation-environment")
    if environment is not None:
        inputs = environment.inputs
        if (inputs.get("model_tag"), inputs.get("model_digest"), inputs.get("model_quantization"), inputs.get("provider_settings")) != (
                EXPERIMENT_MODEL, MODEL_DIGEST, MODEL_QUANTIZATION, provider_settings()):
            reasons.append("evaluation environment freeze model identity differs from the frozen model")
        if (inputs.get("sbert_model"), inputs.get("sbert_revision"), inputs.get("sbert_snapshot_sha256"), inputs.get("embedding_dimension")) != (
                SBERT_MODEL, SBERT_REVISION, SBERT_SNAPSHOT_SHA256, EMBEDDING_DIMENSION):
            reasons.append("evaluation environment freeze Sentence-BERT identity differs from the protocol")
        if gate.matrix and inputs.get("evaluation_matrix_sha256") != matrix_sha256(gate.matrix):
            reasons.append("evaluation environment freeze references a different matrix")
        if gate.manifest is not None and inputs.get("alfworld_data_identity") != gate.manifest.data_root_identity:
            reasons.append("evaluation environment freeze valid_unseen data identity differs from the frozen tasks")
    if environment is not None and protocol is not None and environment.inputs.get("prompt_hashes") != protocol.inputs.get("prompt_hashes"):
        reasons.append("environment/protocol freeze prompt hashes differ")
    gate.library_payload, gate.libraries, library_problems = load_library_freeze(root)
    reasons.extend(library_problems)
    allocation = scientific_acquisition_allocation(root)
    if sum(item["units"] for item in allocation["runs"].values()) != 240 or allocation["problems"]:
        reasons.append("the scientific acquisition ledger is not exactly the closed 240 units")
    gate.valid = not reasons
    return gate


def pending_only(reasons: Sequence[str]) -> tuple[list[str], list[str]]:
    human = [reason for reason in reasons if reason in APPROVAL_PENDING_REASONS or reason.startswith(CORE_PENDING_PREFIXES)]
    return human, [reason for reason in reasons if reason not in human]


# ============================================================== scientific shards


def evaluation_commands(backup_dir: str = PRODUCTION_BACKUP_DIR) -> dict[str, str]:
    env = f"{UNSEEN_ACCESS_ENV}={SCIENTIFIC_EVALUATION}"
    durable = f"--yes --backup-dir {backup_dir} --require-backup"
    commands = {f"shard_{shard}": f"{env} python -m rq1.cli evaluation-amended run --shard {shard} {durable}" for shard in range(1, SHARD_COUNT + 1)}
    commands.update({
        "single_worker": f"for k in $(seq 1 {SHARD_COUNT}); do {env} python -m rq1.cli evaluation-amended run --shard $k {durable} || break; done",
        "resume_shard_k": f"{env} python -m rq1.cli evaluation-amended resume --shard <k> {durable}",
        "retry_failed_shard_k": f"{env} python -m rq1.cli evaluation-amended retry-failed --shard <k> {durable}",
        "validate": "python -m rq1.cli evaluation-amended validate",
        "merge": "python -m rq1.cli evaluation-amended merge --yes",
        "rater_export": "python -m rq1.cli evaluation-amended rater-export --yes",
        "analyze": "python -m rq1.cli evaluation-amended analyze",
    })
    return commands


def evaluation_plan(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate = validate_evaluation_gates(root)
    human, technical = pending_only(gate.reasons)
    shards = {str(shard): (root / "results" / "final" / shard_run_id(shard)).exists() for shard in range(1, SHARD_COUNT + 1)}
    return {"ok": True, "dry_run": True, "launch_permitted": gate.valid, "gate": gate.to_dict(), "human_blockers": human, "technical_blockers": technical,
            "shard_directories_exist": shards, "protocol_sha256": evaluation_protocol_sha256(), "commands": evaluation_commands()}


def evaluation_run(root: Path, args: argparse.Namespace, *, resume: bool, retry_failed: bool) -> dict[str, Any]:
    if not getattr(args, "yes", False):
        return _blocked(reason="scientific evaluation requires --yes")
    shard = int(args.shard)
    require_unseen_access(SCIENTIFIC_EVALUATION)
    gate = validate_evaluation_gates(root)
    if not gate.valid or gate.manifest is None:
        return _blocked(gate=gate.to_dict())
    environment = gate.freezes["evaluation-environment"]
    drift = verify_launch_environment(root, environment.inputs)
    cache = Path(os.environ.get("HF_HUB_CACHE") or DEFAULT_HF_HUB_CACHE)
    if sbert_snapshot_sha256(cache) != SBERT_SNAPSHOT_SHA256:
        drift.append("Sentence-BERT snapshot differs from the frozen snapshot")
    if discover_tasks(default_data_dir(), EVALUATION_SPLIT, allow_unseen_metadata=True).data_root_identity != gate.manifest.data_root_identity:
        drift.append("valid_unseen data identity differs from the frozen tasks")
    if drift:
        return _blocked(environment_drift=drift)
    run_id = shard_run_id(shard)
    store = ExperimentStore(root, run_id)
    sizes = {condition: gate.libraries[condition].size for condition in CONDITIONS}
    hashes = {condition: gate.libraries[condition].content_sha256 for condition in CONDITIONS}
    units = experiment_units(gate.matrix, shard, sizes, hashes)
    freezes = {kind: freeze.input_fingerprint for kind, freeze in gate.freezes.items()}
    freezes["task_manifest_sha256"] = gate.manifest.manifest_sha256
    configuration = evaluation_run_configuration(root, scientific=True, libraries=gate.libraries, matrix_sha=matrix_sha256(gate.matrix), shard=shard, freezes=freezes)
    provenance = {"evaluation_protocol_sha256": evaluation_protocol_sha256(), "evaluation_matrix_sha256": matrix_sha256(gate.matrix),
                  "task_queue_sha256": queue_identity_sha256(gate.manifest), "task_manifest_sha256": gate.manifest.manifest_sha256,
                  "controlled_failures_sha256": sha256_file(gate.failures_path), "library_freeze_sha256": canonical_hash(gate.library_payload),
                  "freeze_fingerprints": freezes, "shard": shard}
    backup = Path(args.backup_dir) if getattr(args, "backup_dir", None) else None
    options = RunnerOptions(resume=resume, retry_failed=retry_failed, max_runs=getattr(args, "max_runs", None), fail_fast=True,
                            backup_dir=backup, require_backup=bool(getattr(args, "require_backup", False)))
    embedder = _embedder()
    with RealEpisodeDriver(root, data_dir=default_data_dir(), bridge_log_root=store.directory / "logs" / "bridge") as driver:
        executor = AmendedEvaluationExecutor(driver, tasks={task.task_id: task for task in gate.tasks}, libraries=gate.libraries, embedder=embedder,
                                             scientific=True, split=EVALUATION_SPLIT, provenance=provenance)
        result = DurableExperimentRunner(store, progress=print).run("evaluation", units, bind_repository_configuration(root, configuration), executor, options)
    return {"ok": result["status"] in {"completed", "paused", "incomplete"}, "scientific_evidence": True, "shard": shard, **result}


def shard_records(root: Path, shard: int) -> list[dict[str, Any]]:
    store = ExperimentStore(root, shard_run_id(shard))
    if not store.manifest_path.is_file():
        return []
    return list(store.terminal_results(phase="evaluation", repair_tail=False).values())


def validate_runs(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate = validate_evaluation_gates(root)
    if not gate.matrix:
        return _blocked(reason="the frozen matrix is unavailable", gate=gate.to_dict())
    shards = [int(args.shard)] if getattr(args, "shard", None) else list(range(1, SHARD_COUNT + 1))
    report = {}
    problems = []
    for shard in shards:
        records = shard_records(root, shard)
        keys = {unit["unit_key"] for unit in gate.matrix if unit["shard"] == shard}
        completed = [record for record in records if record.get("status") == "completed"]
        bad = [record["run_key"] for record in completed if record.get("skill_writes") is not False
               or (record.get("condition") == "NoLib" and record.get("retrieval_events") != 0)
               or (record.get("condition") != "NoLib" and (record.get("retrieval_events") != 1 or len((record.get("retrieval") or {}).get("top") or []) != 3))
               or record.get("scientific_evidence") is not True]
        outside = [record["run_key"] for record in records if record["run_key"] not in keys]
        report[str(shard)] = {"terminal": len(records), "completed": len(completed), "failed": len(records) - len(completed), "planned": len(keys),
                              "protocol_violations": bad, "outside_shard": outside}
        if bad or outside:
            problems.append(f"shard {shard} has protocol violations or foreign units")
    return {"ok": not problems, "problems": problems, "shards": report}


def merge_results(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not getattr(args, "yes", False):
        return _blocked(reason="merging requires --yes")
    gate = validate_evaluation_gates(root)
    if not gate.matrix:
        return _blocked(reason="the frozen matrix is unavailable", gate=gate.to_dict())
    records = {shard: shard_records(root, shard) for shard in range(1, SHARD_COUNT + 1)}
    merged, problems = merge_shards(gate.matrix, records)
    validation = validate_runs(root, argparse.Namespace(shard=None))
    problems.extend(validation["problems"])
    if problems:
        return _blocked(reasons=problems)
    output = root / REPORT_DIR / EVALUATION_RUN_PREFIX
    if (output / "merged-results.jsonl").exists():
        return _blocked(reason=f"merged results already exist and are immutable: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "merged-results.jsonl").open("x", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = {"schema_version": 1, "kind": "rq1-amended-evaluation-report", "generated_at": utc_now(), "units": len(merged),
              "matrix_sha256": matrix_sha256(gate.matrix), "protocol_sha256": evaluation_protocol_sha256(), "library_hashes": gate.to_dict()["library_hashes"],
              "completed": sum(record.get("status") == "completed" for record in merged), "failed": sum(record.get("status") == "failed" for record in merged),
              "shard_results_sha256": {str(shard): sha256_file(ExperimentStore(root, shard_run_id(shard)).results_path) for shard in range(1, SHARD_COUNT + 1)},
              "merged_results_sha256": sha256_file(output / "merged-results.jsonl")}
    write_immutable(output / "evaluation-report.json", report)
    return {"ok": True, **report, "output": str(output)}


def rater_export(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    import csv
    import io

    if not getattr(args, "yes", False):
        return _blocked(reason="rater export requires --yes")
    output = root / REPORT_DIR / EVALUATION_RUN_PREFIX
    merged = _jsonl(output / "merged-results.jsonl")
    if not merged:
        return _blocked(reason="merged evaluation results are required")
    directory = output / "relevance"
    if directory.exists():
        return _blocked(reason=f"rater export already exists and is immutable: {directory}")
    _payload, libraries, problems = load_library_freeze(root)
    if problems:
        return _blocked(reasons=problems)
    texts = {skill["skill_id"]: skill["text"] for library in libraries.values() for skill in library.skills}
    items, keys = [], []
    for record in merged:
        top = (record.get("retrieval") or {}).get("top") or []
        if record.get("status") != "completed" or record.get("condition") == "NoLib" or not top:
            continue
        item_id = canonical_hash({"unit": record["run_key"], "event": record["retrieval"].get("event_id")})[:16]
        context = record["failure_context"]
        for entry in top:
            items.append({"item_id": item_id, "rank": entry["rank"], "task_goal": context["task_instruction"], "failure_observation": context["observation"],
                          "inventory_field": ", ".join(context.get("inventory") or []) or INVENTORY_NOT_OBSERVED_MARKER,
                          "canonical_failure_message": context["failure_message"], "skill_id": entry["skill_id"], "skill_text": texts[entry["skill_id"]],
                          "rater_label": "", "rater_notes": ""})
        keys.append({"item_id": item_id, "unit_key": record["run_key"], "condition": record["condition"], "task_id": record["task_id"], "seed": record["seed"],
                     "event_id": record["retrieval"].get("event_id"), "top_count": len(top)})
    items.sort(key=lambda item: (canonical_hash({"shuffle": 20260914, "item": item["item_id"]}), item["rank"]))
    directory.mkdir(parents=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(items[0].keys()), lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(items)
    for name in ("rater-A.csv", "rater-B.csv"):
        (directory / name).write_bytes(("﻿" + buffer.getvalue()).encode("utf-8"))
    write_immutable(directory / "KEY-DO-NOT-SHARE-WITH-RATERS.json", {"schema_version": 1, "items": keys})
    (directory / "instructions.md").write_text((root / RELEVANCE_RUBRIC_DOCUMENT).read_text(encoding="utf-8"), encoding="utf-8")
    return {"ok": True, "items": len(keys), "rows": len(items), "directory": str(directory)}


def analyze_results(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    output = root / REPORT_DIR / EVALUATION_RUN_PREFIX
    merged = _jsonl(output / "merged-results.jsonl")
    if not merged:
        return _blocked(reason="merged evaluation results are required; analysis never runs before scientific data exists")
    quality = None
    if getattr(args, "rater_a", None) and getattr(args, "rater_b", None) and getattr(args, "adjudicated", None):
        from rq1.evaluation.amended_analysis import unit_rows

        keys = json.loads((output / "relevance" / "KEY-DO-NOT-SHARE-WITH-RATERS.json").read_text(encoding="utf-8"))["items"]
        quality = retrieval_quality(keys, unit_rows(merged), rater_a=read_labels(Path(args.rater_a).read_bytes(), "rater_label"),
                                    rater_b=read_labels(Path(args.rater_b).read_bytes(), "rater_label"),
                                    adjudicated=read_labels(Path(args.adjudicated).read_bytes(), "adjudicated_label"))
    metrics = analyze(merged, quality=quality)
    path = output / "analysis" / f"metrics-{utc_now().replace(':', '').replace('-', '')}.json"
    atomic_write_json(path, {**metrics, "merged_results_sha256": sha256_file(output / "merged-results.jsonl")})
    return {"ok": True, "path": str(path), "sha256": sha256_file(path), "by_condition": metrics["by_condition"]}


# ============================================================== technical preflight


def active_evaluation_processes() -> list[str]:
    found = []
    proc = Path("/proc")
    for entry in proc.iterdir() if proc.is_dir() else ():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            words = [part.decode("utf-8", "replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
        except OSError:
            continue
        if "rq1.cli" in words and ({"evaluation-amended", "acquisition", "acquisition-extension"} & set(words)) and ({"run", "resume", "retry-failed", "check"} & set(words)):
            found.append(" ".join(words))
    return found


def evaluation_preflight(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    backup_dir = Path(str(getattr(args, "backup_dir", None) or PRODUCTION_BACKUP_DIR))
    commit, clean, error = git_state(root)
    checks: dict[str, bool] = {"clean_committed_repository": bool(commit) and clean and not error}
    details: dict[str, Any] = {"repository_commit": commit}
    allocation = scientific_acquisition_allocation(root)
    checks["acquisition_hard_cap_reached_240"] = sum(item["units"] for item in allocation["runs"].values()) == 240 and not allocation["problems"]
    exports = Path("/workspace/persistent/exports")
    validation = exports / "rq1-acquisition-240-final-20260914.validation.json"
    archive = exports / "rq1-acquisition-240-final-20260914.tar.gz"
    sha_file = exports / "rq1-acquisition-240-final-20260914.tar.gz.sha256"
    checks["acquisition_archive_valid"] = (validation.is_file() and json.loads(validation.read_text(encoding="utf-8")).get("passed") is True
                                           and archive.is_file() and sha_file.is_file() and sha256_file(archive) == sha_file.read_text().split()[0])
    pool = load_raw_pool(root / RAW_POOL_SNAPSHOT)
    feasibility = raw_feasibility(pool)
    checks["raw_pool_supports_3_per_family"] = feasibility["feasible"]
    probe = build_amended_libraries(pool, {family: next(entry.skill_id for entry in pool if entry.task_family == family and entry.family_rank == 1) for family in TASK_FAMILIES})
    checks["library_constructor_nesting_self_test"] = not nesting_problems(probe) and {c: probe[c].size for c in CONDITIONS} == LIBRARY_SIZES
    review_path = root / CORE_REVIEW_FILE
    selection = select_core(pool, parse_review_csv(review_path.read_bytes())) if review_path.is_file() else None
    checks["core_review_package_intact"] = selection is not None and not [problem for problem in selection.problems if "immutable column" in problem or "exactly the" in problem]
    core_status = "COMPLETE" if selection is not None and selection.complete else ("NO_PASS_IN_" + ",".join(selection.failed_families) if selection and selection.failed_families else "PENDING")
    details["core_validation"] = selection.to_dict() if selection is not None else None
    try:
        preparation = load_preparation(preparation_directory(root, getattr(args, "preparation", None)))
    except (OSError, ValueError) as exc:
        preparation = None
        details["preparation_error"] = str(exc)
    matrix: list[dict[str, Any]] = []
    if preparation is not None:
        proposal, failures = preparation["proposal"], preparation["failures"]
        checks["evaluation_task_proposal_valid"] = proposal.status == "proposed" and not task_manifest_problems(proposal, require_frozen=False)
        checks["controlled_failures_oracle_validated"] = not failures_problems(failures, proposal) and failures.get("task_manifest_sha256") == proposal.manifest_sha256
        checks["preparation_code_unchanged"] = not preparation_code_problems(root, str(proposal.repository_commit))
        try:
            matrix = build_matrix(evaluation_tasks(proposal.tasks, failures))
        except (MatrixError, KeyError, ValueError) as exc:
            details["matrix_error"] = str(exc)
        details["tasks"] = {family: [task.task_id for task in proposal.tasks if task.family == family] for family in TASK_FAMILIES}
    else:
        checks["evaluation_task_proposal_valid"] = checks["controlled_failures_oracle_validated"] = checks["preparation_code_unchanged"] = False
    checks["matrix_exactly_360_unique"] = bool(matrix) and not coverage_problems(matrix)
    checks["shards_cover_360_exactly_once"] = bool(matrix) and sorted(unit["unit_key"] for shard in range(1, SHARD_COUNT + 1) for unit in matrix if unit["shard"] == shard) == sorted(unit["unit_key"] for unit in matrix)
    matrix_sha = matrix_sha256(matrix) if matrix else None
    details["matrix_sha256"] = matrix_sha
    requests = {name: _read_request(root / APPROVAL_DIR / str(commit or "")[:12] / f"{name}.approval.json") for name in REQUEST_NAMES}
    checks["approval_requests_present"] = all(value is not None for value in requests.values())
    approval_status = {name: (value or {}).get("status") or ((value or {}).get("approval") or {}).get("status") for name, value in requests.items()}
    protocol_inputs = ((requests.get("evaluation-protocol") or {}).get("inputs") or {})
    environment_inputs = ((requests.get("evaluation-environment") or {}).get("inputs") or {})
    checks["protocol_request_matches_repository_and_matrix"] = (protocol_inputs.get("protocol") == evaluation_protocol_definition()
                                                               and protocol_inputs.get("protocol_sha256") == evaluation_protocol_sha256()
                                                               and protocol_inputs.get("evaluation_matrix_sha256") == matrix_sha and matrix_sha is not None
                                                               and protocol_inputs.get("repository_commit") == commit)
    checks["environment_request_complete"] = bool(environment_inputs) and not (EVALUATION_ENVIRONMENT_REQUIRED - set(environment_inputs))
    checks["model_identity_exact"] = (environment_inputs.get("model_tag"), environment_inputs.get("model_digest"), environment_inputs.get("model_quantization")) == (
        EXPERIMENT_MODEL, MODEL_DIGEST, MODEL_QUANTIZATION) and model_digest(EXPERIMENT_MODEL) == MODEL_DIGEST
    checks["provider_settings_exact"] = environment_inputs.get("provider_settings") == provider_settings()
    cache = Path(os.environ.get("HF_HUB_CACHE") or DEFAULT_HF_HUB_CACHE)
    checks["sbert_identity_exact"] = (environment_inputs.get("sbert_revision") == SBERT_REVISION and sbert_snapshot_sha256(cache) == SBERT_SNAPSHOT_SHA256)
    try:
        _embedder()
        checks["sbert_embedding_dimension_768"] = True
    except Exception as exc:  # recorded as a failed technical check
        checks["sbert_embedding_dimension_768"] = False
        details["embedder_error"] = f"{type(exc).__name__}: {exc}"
    definition = evaluation_protocol_definition()
    checks["retrieval_top3_once_post_failure_nolib_none"] = (definition["retrieval"]["top_k"] == 3 and definition["retrieval"]["repeated_or_per_step_retrieval"] is False
                                                             and definition["retrieval"]["pre_failure_retrieval"] is False and LIBRARY_SIZES["NoLib"] == 0)
    drift = verify_launch_environment(root, environment_inputs) if environment_inputs else ["evaluation environment request is missing"]
    checks["live_environment_matches_request"] = not drift
    details["environment_drift"] = drift
    evidence = ((requests.get("evaluation-protocol") or {}).get("evidence_report") or {})
    evidence_path = Path(str(evidence.get("path") or ""))
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    checks["evaluation_check_evidence_passed_at_commit"] = (evidence_payload.get("mode") == EVALUATION_EVIDENCE_MODE and evidence_payload.get("passed") is True
                                                            and evidence_payload.get("repository_commit") == commit and sha256_file(evidence_path) == evidence.get("sha256"))
    checks["real_alfworld_adapter_ready"] = probe_alfworld_capabilities(default_data_dir()).real_adapter_ready
    checks["hermes_runtime_present"] = HERMES_PYTHON.is_file()
    run_dirs = [root / "results" / "final" / shard_run_id(shard) for shard in range(1, SHARD_COUNT + 1)]
    checks["shard_result_paths_unused"] = not any(path.exists() for path in run_dirs)
    checks["results_final_writable"] = _writable(root / "results" / "final")
    checks["backup_writable_and_unused"] = _writable(backup_dir) and not any((backup_dir / shard_run_id(shard)).exists() for shard in range(1, SHARD_COUNT + 1))
    checks["concurrency_locks_free"] = not any((path / ".experiment.lock").exists() for path in run_dirs)
    active = active_evaluation_processes()
    checks["no_active_runner"] = not active
    details["active_processes"] = active
    free = shutil.disk_usage(root).free
    details["results_filesystem_free_gb"] = round(free / 1e9, 1)
    checks["disk_free_20gb"] = free >= 20e9
    checks["valid_unseen_locked_without_authorization"] = os.environ.get(UNSEEN_ACCESS_ENV) is None
    gate = validate_evaluation_gates(root)
    human, technical = pending_only(gate.reasons)
    checks["gate_blocked_only_by_human_steps"] = not technical
    details["gate_reasons"] = gate.reasons
    technical_pass = all(checks.values())
    blockers = []
    if core_status != "COMPLETE":
        blockers.append("HUMAN CORE VALIDATION")
    if any(status != "APPROVED" for status in approval_status.values()) or human:
        blockers.append("HUMAN APPROVAL")
    report = {"schema_version": 1, "label": "NON-SCIENTIFIC EVALUATION PREFLIGHT", "scientific_evidence": False, "generated_at": utc_now(),
              "technical_pass": technical_pass, "failed_checks": [name for name, passed in checks.items() if not passed],
              "remaining_blockers": blockers if technical_pass else ["TECHNICAL CHECKS FAILED", *blockers], "core_validation_status": core_status,
              "approval_status": approval_status, "checks": checks, "details": details, "commands": evaluation_commands(str(backup_dir))}
    path = root / PREFLIGHT_BASE / f"preflight-{report['generated_at'].replace(':', '').replace('-', '')}.json"
    atomic_write_json(path, report)
    return {"ok": technical_pass, "report": str(path), "report_sha256": sha256_file(path), **report}


def dispatch(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    command = args.evaluation_amended_command
    if command == "check":
        return evaluation_check(root, args)
    if command == "check-report":
        return evaluation_check_report(root, args.run_id)
    if command == "prepare-approvals":
        return prepare_approvals(root, args)
    if command == "freeze-tasks":
        return freeze_tasks(root, args)
    if command == "build-libraries":
        return build_libraries(root, args)
    if command == "plan":
        return evaluation_plan(root, args)
    if command == "preflight":
        return evaluation_preflight(root, args)
    if command in {"run", "resume", "retry-failed"}:
        return evaluation_run(root, args, resume=command != "run", retry_failed=command == "retry-failed")
    if command == "validate":
        return validate_runs(root, args)
    if command == "merge":
        return merge_results(root, args)
    if command == "rater-export":
        return rater_export(root, args)
    if command == "analyze":
        return analyze_results(root, args)
    return _blocked(reason=f"unsupported command: {command}")

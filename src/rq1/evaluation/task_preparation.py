"""Evaluation task preparation (Decision 012): valid_unseen selection and controlled-failure oracle validation.

Requires ``RQ1_VALID_UNSEEN_ACCESS=evaluation-task-preparation``.  It reads valid_unseen
task metadata and derives ALFWorld hand-coded expert routes, then proves one controlled
failure per task in the real bridge.  It never calls a model; no agent outcome exists.
Every output is immutable and precedes the task freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq1.bridge.adapters.capabilities import default_data_dir, probe_alfworld_capabilities
from rq1.bridge.adapters.unseen_access import TASK_PREPARATION, log_unseen_access, require_unseen_access
from rq1.evaluation.amended_protocol import (
    CHECKPOINT_POLICY,
    EVALUATION_POLICY_VERSION,
    EVALUATION_SPLIT,
    EVALUATION_TASK_COUNT,
    MANIFEST_TYPE,
    PREPARATION_DIR,
    TASK_SELECTION,
    TASKS_PER_FAMILY,
    TOTAL_EPISODE_ACTION_BUDGET,
)
from rq1.experiment.models import canonical_hash
from rq1.freeze.validation import git_state
from rq1.hermes.episode_driver import EpisodeDriverError
from rq1.pilot.real_runtime.harnesses import RealRecoveryHarness, bridge_state_digest
from rq1.recovery.controlled_failure import CANONICAL_FAILURE_MESSAGE, ControlledFailureError
from rq1.recovery.reference_route import ReferenceRoute, derive_handcoded_reference, midpoint_candidates
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.discovery import discover_tasks
from rq1.tasks.models import ManifestState, TaskManifest, TaskRecord
from rq1.tasks.reporting import write_immutable
from rq1.tasks.validation import manifest_hash
from rq1.utils.hashing import sha256_file
from rq1.utils.time import utc_now

PROTOCOL_REPLAY_ERROR = "Reference/oracle action was invalid or terminal before completion"


def _safe_name(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]


def _route_worker(arguments: tuple[str, str, str]) -> tuple[str, dict[str, Any]]:
    data_dir, task_id, split = arguments
    try:
        route = derive_handcoded_reference(Path(data_dir), task_id, split)
    except Exception as exc:  # recorded per task; a task without an expert route cannot be ranked
        return task_id, {"error": f"{type(exc).__name__}: {exc}"}
    return task_id, {"length": len(route.actions), "actions": list(route.actions)}


def derive_routes(data_dir: Path, task_ids: Sequence[str], split: str, workers: int) -> dict[str, dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        return dict(pool.map(_route_worker, [(str(data_dir), task_id, split) for task_id in task_ids], chunksize=1))


def ranked_candidates(records: Sequence[TaskRecord], routes: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Longest hand-coded expert route first, ties by task ID, per family."""
    ranking: dict[str, list[dict[str, Any]]] = {family: [] for family in TASK_FAMILIES}
    for record in records:
        route = routes.get(record.task_id) or {}
        if "length" in route and record.family in ranking:
            ranking[record.family].append({"task_id": record.task_id, "length": int(route["length"])})
    for family in ranking:
        ranking[family].sort(key=lambda item: (-item["length"], item["task_id"]))
    return ranking


def validate_controlled_failure(
    driver: Any,
    *,
    output_dir: Path,
    task_id: str,
    task_family: str,
    split: str,
    reference_actions: Sequence[str],
    seed: int = 0,
    run_id: str = "evaluation-preparation",
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """First midpoint-near checkpoint whose deterministic detour passes the real oracle."""
    route = ReferenceRoute(task_id, split, tuple(reference_actions))
    attempts: list[dict[str, Any]] = []
    for index, prefix, continuation in midpoint_candidates(route):
        budget = TOTAL_EPISODE_ACTION_BUDGET - len(prefix) - 1
        if not continuation[0].startswith("go to "):
            attempts.append({"checkpoint_index": index, "result": "rejected", "code": "checkpoint_not_navigation_eligible"})
            continue
        if budget < len(continuation):
            attempts.append({"checkpoint_index": index, "result": "rejected", "code": "reference_rejoin_exceeds_recovery_budget"})
            continue
        harness = RealRecoveryHarness(
            driver, output_dir=output_dir / f"checkpoint-{index}", run_id=run_id, attempt_id=f"checkpoint-{index}",
            reference_actions=reference_actions, total_action_limit=TOTAL_EPISODE_ACTION_BUDGET, allowed_split=split,
            profile_prefix="rq1-evaluation-preparation",
        )
        try:
            harness.start_and_replay(task_id, split, seed, prefix)
            assert harness.session is not None and harness.session.state is not None
            checkpoint = dict(harness.session.state)
            perturbation = harness.apply_controlled_action_perturbation(f"{EVALUATION_POLICY_VERSION}:{task_id}:checkpoint-{index}")
            post = dict(harness.post_detour_state or {})
        except ControlledFailureError as exc:
            attempts.append({"checkpoint_index": index, "result": "rejected", "code": exc.code, "error": str(exc)})
            continue
        except EpisodeDriverError as exc:
            if str(exc).startswith(PROTOCOL_REPLAY_ERROR):
                attempts.append({"checkpoint_index": index, "result": "rejected", "code": "prefix_replay_invalid", "error": str(exc)})
                continue
            raise  # infrastructure: never let an execution fault change task selection
        finally:
            harness.close()
        attempts.append({"checkpoint_index": index, "result": "accepted"})
        definition = {
            "task_id": task_id,
            "task_family": task_family,
            "split": split,
            "reference_route_length": len(reference_actions),
            "reference_actions": list(reference_actions),
            "reference_route_sha256": canonical_hash(list(reference_actions)),
            "checkpoint_id": f"{EVALUATION_POLICY_VERSION}:{task_id}:checkpoint-{index}",
            "checkpoint_index": index,
            "prefix_actions": list(prefix),
            "expected_next_action": continuation[0],
            "continuation_length": len(continuation),
            "checkpoint_observation": checkpoint.get("observation"),
            "checkpoint_admissible_actions": checkpoint.get("admissible_actions"),
            "checkpoint_step_number": checkpoint.get("step_number"),
            "checkpoint_digest": bridge_state_digest(checkpoint),
            "detour_action": perturbation.action,
            "post_detour_observation": post.get("observation"),
            "post_detour_admissible_actions": post.get("admissible_actions"),
            "post_detour_digest": perturbation.post_state_digest,
            "canonical_failure_message": CANONICAL_FAILURE_MESSAGE,
            "oracle": {"validated": True, "solvable": perturbation.solvable, "selection_rule": perturbation.selection_rule,
                       "method": CHECKPOINT_POLICY["oracle_validation"], "evidence_directory": str(output_dir / f"checkpoint-{index}")},
            "total_action_budget": TOTAL_EPISODE_ACTION_BUDGET,
            "recovery_action_budget": budget,
        }
        return definition, attempts
    return None, attempts


def _family_worker(arguments: tuple[str, str, str, list[dict[str, Any]], str]) -> dict[str, Any]:
    from rq1.hermes.episode_driver import RealEpisodeDriver

    root, data_dir, family, candidates, output = arguments
    access_log = Path(output) / "valid-unseen-access.jsonl"
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with RealEpisodeDriver(Path(root), data_dir=Path(data_dir), bridge_log_root=Path(output) / "bridge" / family) as driver:
        for candidate in candidates:
            if len(accepted) == TASKS_PER_FAMILY:
                break
            log_unseen_access(access_log, operation="controlled_failure_oracle_validation", task_id=candidate["task_id"])
            definition, attempts = validate_controlled_failure(
                driver, output_dir=Path(output) / "oracle" / family / _safe_name(candidate["task_id"]), task_id=candidate["task_id"],
                task_family=family, split=EVALUATION_SPLIT, reference_actions=candidate["actions"],
            )
            if definition is None:
                rejected.append({"task_id": candidate["task_id"], "length": candidate["length"], "attempts": attempts})
            else:
                accepted.append({**definition, "selection_rank": candidate["rank"], "checkpoint_attempts": attempts})
    return {"family": family, "accepted": accepted, "rejected": rejected}


def prepare_evaluation_tasks(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    require_unseen_access(TASK_PREPARATION)
    commit, clean, error = git_state(root)
    if error or not clean or not commit:
        return {"ok": False, "status": "blocked", "reason": "evaluation task preparation requires a clean committed repository"}
    if not getattr(args, "yes", False):
        return {"ok": False, "status": "blocked", "reason": "evaluation task preparation requires --yes"}
    data_dir = default_data_dir()
    output = root / PREPARATION_DIR / commit[:12]
    if output.exists():
        return {"ok": False, "status": "blocked", "reason": f"evaluation preparation already exists and is immutable: {output}"}
    output.mkdir(parents=True)
    access_log = output / "valid-unseen-access.jsonl"
    log_unseen_access(access_log, operation="preparation_started", detail=f"repository_commit={commit}")
    discovery = discover_tasks(data_dir, EVALUATION_SPLIT, allow_unseen_metadata=True)
    log_unseen_access(access_log, operation="task_metadata_discovery", detail=f"{len(discovery.records)} valid_unseen task records")
    workers = int(getattr(args, "workers", None) or min(32, os.cpu_count() or 4))
    routes = derive_routes(data_dir, [record.task_id for record in discovery.records], EVALUATION_SPLIT, workers)
    for task_id in sorted(routes):
        log_unseen_access(access_log, operation="handcoded_expert_route", task_id=task_id)
    ranking = ranked_candidates(discovery.records, routes)
    write_immutable(output / "route-lengths.json", {
        "schema_version": 1,
        "kind": "rq1-evaluation-route-lengths",
        "selection": TASK_SELECTION,
        "data_root_identity": discovery.data_root_identity,
        "tasks": {task_id: {"family": record.family, **({"length": routes[task_id]["length"]} if "length" in routes[task_id] else {"error": routes[task_id]["error"]})}
                  for record in discovery.records for task_id in [record.task_id]},
        "ranking": ranking,
    })
    jobs = []
    for family in TASK_FAMILIES:
        candidates = [{**item, "rank": rank, "actions": routes[item["task_id"]]["actions"]} for rank, item in enumerate(ranking[family], 1)]
        jobs.append((str(root), str(data_dir), family, candidates, str(output)))
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("fork")) as pool:
        results = {item["family"]: item for item in pool.map(_family_worker, jobs)}
    problems = [f"{family}: only {len(results[family]['accepted'])} validated tasks" for family in TASK_FAMILIES if len(results[family]["accepted"]) != TASKS_PER_FAMILY]
    records = {record.task_id: record for record in discovery.records}
    definitions = [definition for family in TASK_FAMILIES for definition in results[family]["accepted"]]
    tasks = tuple(TaskRecord(**{**records[item["task_id"]].to_dict(), "order_index": index}) for index, item in enumerate(definitions, 1))
    chosen = {task.task_id for task in tasks}
    exclusions = [
        *discovery.exclusions,
        *({"task_id": item["task_id"], "reason": "replaced_before_freeze_no_oracle_validated_controlled_failure",
           "codes": ",".join(sorted({attempt.get("code", "") for attempt in item["attempts"] if attempt.get("code")}))}
          for family in TASK_FAMILIES for item in results[family]["rejected"]),
        *({"task_id": record.task_id, "reason": "not_selected"} for record in discovery.records
          if record.task_id not in chosen and all(record.task_id != item["task_id"] for family in TASK_FAMILIES for item in results[family]["rejected"])),
    ]
    value: dict[str, Any] = {
        "schema_version": 1, "manifest_type": MANIFEST_TYPE, "status": ManifestState.PROPOSED.value, "split": EVALUATION_SPLIT,
        "alfworld_version": probe_alfworld_capabilities(data_dir).version, "data_root_identity": discovery.data_root_identity,
        "repository_commit": commit, "selection_policy": dict(TASK_SELECTION), "requested_count": EVALUATION_TASK_COUNT,
        "actual_count": len(tasks), "family_counts": dict(sorted(Counter(task.family for task in tasks).items())),
        "tasks": [task.to_dict() for task in tasks], "exclusions": exclusions, "duplicate_resolution": [],
        "generated_at": utc_now(), "approved_at": None, "approval_reference": None, "manifest_sha256": "",
    }
    value["manifest_sha256"] = manifest_hash(value)
    proposal = output / f"{MANIFEST_TYPE}-{value['manifest_sha256'][:16]}.json"
    write_immutable(proposal, value)
    failures = output / "controlled-failures.json"
    write_immutable(failures, {
        "schema_version": 1,
        "kind": "rq1-evaluation-controlled-failures",
        "policy": CHECKPOINT_POLICY,
        "policy_version": EVALUATION_POLICY_VERSION,
        "canonical_failure_message": CANONICAL_FAILURE_MESSAGE,
        "repository_commit": commit,
        "task_manifest_sha256": value["manifest_sha256"],
        "task_count": len(definitions),
        "tasks": definitions,
        "replaced_candidates": {family: results[family]["rejected"] for family in TASK_FAMILIES},
        "generated_at": utc_now(),
        "model_calls": 0,
    })
    log_unseen_access(access_log, operation="preparation_finished", detail=f"{len(tasks)} tasks; problems={len(problems)}")
    return {
        "ok": not problems and len(tasks) == EVALUATION_TASK_COUNT,
        "status": "proposed" if not problems else "blocked",
        "problems": problems,
        "proposal": str(proposal),
        "proposal_sha256": sha256_file(proposal),
        "manifest_sha256": value["manifest_sha256"],
        "controlled_failures": str(failures),
        "controlled_failures_sha256": sha256_file(failures),
        "family_counts": value["family_counts"],
        "selected": {family: [(item["task_id"], item["reference_route_length"], item["checkpoint_index"], item["recovery_action_budget"])
                              for item in results[family]["accepted"]] for family in TASK_FAMILIES},
        "replaced": {family: [item["task_id"] for item in results[family]["rejected"]] for family in TASK_FAMILIES},
        "route_errors": sum("error" in route for route in routes.values()),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "access_log": str(access_log),
    }

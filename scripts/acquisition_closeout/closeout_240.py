"""RQ1 complete 240-episode acquisition closeout and pre-evaluation packages.

A read-only integrity audit of the completed parent run (units 1-180) and extension
run (181-240).  It writes only
  artifacts/acquisition-closeout/rq1-acquisition-240-final/   (closeout manifest, raw
      50-skill pool snapshot, yield report, library-feasibility analysis)
  artifacts/skill-validation/rq1-acquisition-240/             (human review package)
and refuses to overwrite either directory.  It never modifies scientific results,
performs no library selection, and freezes nothing.
"""
import collections
import csv
import hashlib
import io
import itertools
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path("/workspace/persistent/rq1-protocol-migration")
sys.path.insert(0, str(ROOT / "src"))
from rq1.acquisition.extension import load_parent, starting_pool_path, starting_pool_problems, validate_extension_queue_manifest  # noqa: E402
from rq1.acquisition.extension_protocol import (  # noqa: E402
    EXTENSION_ENVIRONMENT_FREEZE, EXTENSION_FROZEN_DIR, EXTENSION_PROTOCOL_FREEZE, EXTENSION_RUN_ID, PARENT,
    extension_protocol_definition, extension_protocol_sha256,
)
from rq1.acquisition.gates import ENVIRONMENT_FREEZE, FROZEN_TASK_DIR, PROTOCOL_FREEZE, load_task_manifest, queue_identity_sha256, validate_queue_manifest  # noqa: E402
from rq1.acquisition.launch import attempt_lineage  # noqa: E402
from rq1.acquisition.protocol import protocol_definition, protocol_sha256  # noqa: E402
from rq1.acquisition.skill_pool import pool_hash, rebuild_pool, verify_snapshot  # noqa: E402
from rq1.experiment.persistence import CRITICAL_FILES  # noqa: E402
from rq1.freeze.validation import read_freeze  # noqa: E402
from rq1.retrieval.text import build_skill_text  # noqa: E402
from rq1.skills.library import ACCUM_EXTRAS_PER_FAMILY, CORE_PER_FAMILY, TASK_FAMILIES  # noqa: E402
from rq1.utils.hashing import sha256_text  # noqa: E402

PARENT_ID = PARENT.run_id
EXT_ID = EXTENSION_RUN_ID
PARENT_RUN = ROOT / "results" / "final" / PARENT_ID
EXT_RUN = ROOT / "results" / "final" / EXT_ID
BACKUPS = pathlib.Path("/workspace/persistent/backups")
LOGS = pathlib.Path("/workspace/persistent/logs")
EXPORTS = pathlib.Path("/workspace/persistent/exports")
PARENT_CLOSEOUT_DIR = ROOT / "artifacts" / "acquisition-closeout" / PARENT_ID
PARENT_APPROVALS = ROOT / "artifacts" / "approvals" / "acquisition" / "8bd452e76120"
EXT_APPROVALS = ROOT / "artifacts" / "approvals" / "acquisition-extension" / "670627a9cd88"
EXT_EVIDENCE = ROOT / "artifacts" / "prelaunch" / "acquisition-extension-check" / "prelaunch-acquisition-extension-check-20260914-a" / "acquisition-extension-check-report.json"
EXT_PREFLIGHT = ROOT / "artifacts" / "prelaunch" / "extension-preflight" / "preflight-20260913T230802Z.json"
OUT = ROOT / "artifacts" / "acquisition-closeout" / "rq1-acquisition-240-final"
VAL = ROOT / "artifacts" / "skill-validation" / "rq1-acquisition-240"
EXPECTED = {
    "parent_commit": "8bd452e76120da21d721c6e894d1ce5af4912ca9",
    "extension_commit": "670627a9cd8800ce64e369a17263e3a3b0486dd1",
    "parent_queue": "1f401869ea877969072f4d73e314db5841ff3f9aae875057ec3e4b5fe468ee09",
    "extension_queue": "7469d4f2c542cc17eb86c7acc57bbe7f91649127ebe5b3c9d6f2fd570f73c243",
    "model": "gemma4:12b",
    "digest": "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c",
    "quantization": "Q4_K_M",
    "family_skills": {"pick_and_place": 11, "pick_two_and_place": 8, "look_at_object": 4,
                      "clean_and_place": 9, "heat_and_place": 15, "cool_and_place": 3},
}
EVALUATION_TASKS, EVALUATION_SEEDS = 30, (11, 29, 47)


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: pathlib.Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def rel(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def ref(path: pathlib.Path) -> dict:
    return {"path": rel(path), "sha256": sha(path)}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


for directory in (OUT, VAL):
    if directory.exists():
        sys.exit(f"refusing to overwrite existing outputs: {directory}")

checks: dict = {}


def check(name: str, condition, detail=None) -> None:
    checks[name] = {"pass": bool(condition), **({"detail": detail} if detail is not None else {})}


def retrieval_events(run: pathlib.Path, rows: list) -> int:
    count = 0
    for row in rows:
        for relpath in row.get("log_paths") or []:
            if relpath.endswith("episode-events.jsonl"):
                count += sum("retriev" in str(event.get("event", "")) for event in jsonl(run / relpath))
    return count


def listing(run: pathlib.Path) -> tuple[str, int, int]:
    lines, total = [], 0
    for path in sorted(run.rglob("*")):
        if path.is_file():
            total += path.stat().st_size
            lines.append(f"{sha(path)}  {rel(path)}")
    return "\n".join(lines) + "\n", len(lines), total


def mirror_identical(run: pathlib.Path, backup: pathlib.Path) -> dict:
    result = {}
    for name in [*CRITICAL_FILES, *(f"manifests/{path.name}" for path in sorted((run / "manifests").glob("*.json")))]:
        if (run / name).is_file():
            result[name] = (backup / name).is_file() and sha(backup / name) == sha(run / name)
    return result


def run_counts(run: pathlib.Path, rows: list) -> dict:
    return {
        "records": len(rows),
        "distinct_run_keys": len({row["run_key"] for row in rows}),
        "distinct_task_ids": len({row["task_id"] for row in rows}),
        "successful": sum(row.get("success") is True for row in rows),
        "scientific_failures": sum(row.get("status") == "completed" and row.get("success") is not True for row in rows),
        "infrastructure_failures": sum(row.get("status") == "failed" for row in rows),
        "error_records": len(jsonl(run / "errors.jsonl")),
        "retrieval_count_field": sum(int(row.get("scientific_retrieval_count") or 0) for row in rows),
        "retrieved_skill_ids": sum(len(row.get("retrieved_skill_ids") or []) for row in rows),
        "retrieval_events": retrieval_events(run, rows),
        "max_actions": max(len(row.get("episode_actions") or []) for row in rows),
    }


# ============================================================== repository, processes, evaluation boundary
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
check("repository_at_extension_commit_and_clean", head == EXPECTED["extension_commit"] and not dirty, {"head": head})
live = subprocess.run(["pgrep", "-f", "^/opt/rq1-venv/bin/python -m rq1.cli acquisition"], capture_output=True, text=True).stdout.split()
sessions = {name: subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0 for name in ("rq1-acquisition", "rq1-acquisition-extension")}
check("no_acquisition_process_alive", not live, {"pids": live})
check("no_acquisition_tmux_runner", not any(sessions.values()), sessions)
final_runs = sorted(path.name for path in (ROOT / "results" / "final").iterdir() if path.is_dir())
check("no_evaluation_started", final_runs == sorted([PARENT_ID, EXT_ID])
      and not list((ROOT / "artifacts" / "task_manifests").rglob("evaluation-*.json"))
      and not (ROOT / "artifacts" / "freezes" / "environment-freeze.json").exists()
      and not (ROOT / "artifacts" / "freezes" / "protocol-freeze.json").exists(), {"results_final": final_runs})

# ============================================================== parent run 1-180 (read-only re-verification)
parent_closeout_path = ROOT / PARENT.closeout_manifest
check("parent_closeout_manifest_unchanged", sha(parent_closeout_path) == PARENT.closeout_manifest_sha256)
parent_closeout = json.loads(parent_closeout_path.read_text(encoding="utf-8"))
parent_state = load_parent(ROOT)
check("parent_verified", not parent_state.problems, parent_state.problems)
parent_rows = jsonl(PARENT_RUN / "results.jsonl")
parent_counts = run_counts(PARENT_RUN, parent_rows)
parent_manifest_path = sorted((ROOT / FROZEN_TASK_DIR).glob("acquisition-*.json"))[0]
parent_manifest = load_task_manifest(parent_manifest_path)
parent_queue = sorted(parent_manifest.tasks, key=lambda task: task.order_index)
check("parent_180_records_180_unique_units", parent_counts["records"] == 180 and parent_counts["distinct_run_keys"] == 180 and parent_counts["distinct_task_ids"] == 180
      and [row["task_id"] for row in parent_rows] == [task.task_id for task in parent_queue], parent_counts)
check("parent_86_successes_94_failures", (parent_counts["successful"], parent_counts["scientific_failures"]) == (86, 94))
check("parent_0_infrastructure_failures", parent_counts["infrastructure_failures"] == 0 and parent_counts["error_records"] == 0
      and all(row["status"] == "completed" and row.get("attempt_index") == 1 for row in parent_rows))
check("parent_0_retrievals", parent_counts["retrieval_count_field"] == parent_counts["retrieved_skill_ids"] == parent_counts["retrieval_events"] == 0)
parent_lineage = attempt_lineage(parent_rows)
check("parent_no_unauthorized_retry_lineage", parent_lineage["authorized"] and parent_lineage["duplicate_completed_units"] == 0 and not parent_lineage["retried_units"])
parent_checkpoint = json.loads((PARENT_RUN / "checkpoint.json").read_text(encoding="utf-8"))
check("parent_checkpoint_completed", parent_checkpoint.get("status") == "completed" and parent_checkpoint.get("completed_run_count") == 180)
parent_mirror = mirror_identical(PARENT_RUN, BACKUPS / PARENT_ID)
check("parent_backup_valid", all(parent_mirror.values()) and len(jsonl(BACKUPS / PARENT_ID / "results.jsonl")) == 180, parent_mirror)
parent_pool = rebuild_pool(parent_rows)
verify_snapshot(PARENT_RUN, parent_pool)
check("parent_final_34_skill_pool_valid", len(parent_pool) == 34 and pool_hash(parent_pool) == PARENT.pool_hash)
unchanged = {name: sha(ROOT / item["path"]) == item["sha256"] for name, item in parent_closeout["authoritative_artifacts"].items()}
unchanged["checkpoint.backup.json"] = sha(PARENT_RUN / "checkpoint.backup.json") == parent_closeout["non_authoritative_files"]["checkpoint.backup.json"]["sha256"]
parent_listing_text, parent_files, parent_bytes = listing(PARENT_RUN)
unchanged["result_directory_listing"] = hashlib.sha256(parent_listing_text.encode("utf-8")).hexdigest() == parent_closeout["result_directory"]["listing_sha256"]
check("parent_artifacts_unchanged_since_closeout", all(unchanged.values()), unchanged)
parent_environment, _ = read_freeze(ROOT / ENVIRONMENT_FREEZE, "acquisition-environment")
parent_protocol, _ = read_freeze(ROOT / PROTOCOL_FREEZE, "acquisition-protocol")
parent_approvals = {}
for path in sorted(PARENT_APPROVALS.glob("*.approval.json")):
    document = json.loads(path.read_text(encoding="utf-8"))
    parent_approvals[path.name] = {"path": rel(path), "sha256": sha(path), "status": document.get("status") if "task-freeze" in path.name else (document.get("approval") or {}).get("status")}
check("parent_freezes_and_approvals_valid",
      parent_environment is not None and parent_protocol is not None
      and parent_environment.repository_commit == parent_protocol.repository_commit == EXPECTED["parent_commit"]
      and parent_protocol.inputs.get("protocol") == protocol_definition() and parent_protocol.inputs.get("protocol_sha256") == protocol_sha256()
      and not validate_queue_manifest(parent_manifest, require_frozen=True) and queue_identity_sha256(parent_manifest) == EXPECTED["parent_queue"]
      and len(parent_approvals) == 3 and all(item["status"] == "APPROVED" for item in parent_approvals.values())
      and all(item["sha256"] == parent_closeout["approvals"][name]["sha256"] for name, item in parent_approvals.items()))

# ============================================================== extension run 181-240
ext_rows = jsonl(EXT_RUN / "results.jsonl")
ext_counts = run_counts(EXT_RUN, ext_rows)
ext_manifest_paths = sorted((ROOT / EXTENSION_FROZEN_DIR).glob("acquisition-extension-*.json"))
ext_manifest = load_task_manifest(ext_manifest_paths[0])
ext_queue = sorted(ext_manifest.tasks, key=lambda task: task.order_index)
check("extension_60_records_60_unique_units", ext_counts["records"] == 60 and ext_counts["distinct_run_keys"] == 60 and ext_counts["distinct_task_ids"] == 60, ext_counts)
check("extension_queue_order_and_logical_positions",
      [row["task_index"] for row in ext_rows] == list(range(1, 61))
      and [row["task_id"] for row in ext_rows] == [task.task_id for task in ext_queue]
      and [row.get("logical_acquisition_index") for row in ext_rows] == list(range(181, 241))
      and all(row.get("parent_run_id") == PARENT_ID for row in ext_rows))
check("extension_30_successes_30_failures", (ext_counts["successful"], ext_counts["scientific_failures"]) == (30, 30))
check("extension_0_infrastructure_failures", ext_counts["infrastructure_failures"] == 0 and ext_counts["error_records"] == 0
      and all(row["status"] == "completed" and row.get("attempt_index") == 1 for row in ext_rows))
check("extension_0_retrievals", ext_counts["retrieval_count_field"] == ext_counts["retrieved_skill_ids"] == ext_counts["retrieval_events"] == 0)
ext_lineage = attempt_lineage(ext_rows)
check("extension_no_duplicate_or_unauthorized_units", ext_lineage["authorized"] and ext_lineage["duplicate_completed_units"] == 0 and not ext_lineage["retried_units"])
check("extension_no_parent_unit_rerun", not ({row["task_id"] for row in ext_rows} & {row["task_id"] for row in parent_rows}))
check("extension_actions_within_50", ext_counts["max_actions"] <= 50 and parent_counts["max_actions"] <= 50)
ext_config = json.loads((EXT_RUN / "manifests" / "acquisition.json").read_text(encoding="utf-8"))
ext_environment, ext_environment_errors = read_freeze(ROOT / EXTENSION_ENVIRONMENT_FREEZE, "acquisition-extension-environment")
ext_protocol, ext_protocol_errors = read_freeze(ROOT / EXTENSION_PROTOCOL_FREEZE, "acquisition-extension-protocol")
configuration, runtime = ext_config["configuration"], ext_config["configuration"]["runtime_settings"]
bound = configuration.get("extension") or {}
check("extension_frozen_settings_in_run_configuration",
      configuration["model_name"] == EXPECTED["model"] and configuration["scientific_evidence"] is True and configuration["queue_sha256"] == EXPECTED["extension_queue"]
      and (runtime["acquisition_action_budget"], runtime["output_token_cap"], runtime["model_context_length"], runtime["temperature"], runtime["seed"],
           runtime["action_selection_protocol"], runtime["max_selection_attempts"]) == (50, 2048, 32768, 0, 42, "action-index-history-v3", 3)
      and configuration["protocol_sha256"] == protocol_sha256()
      and (bound.get("parent_run_id"), bound.get("starting_pool_hash"), bound.get("starting_pool_size"), bound.get("logical_index_offset"), bound.get("protocol_sha256"))
      == (PARENT_ID, PARENT.pool_hash, 34, 180, extension_protocol_sha256())
      and all((row.get("skill_candidate") or {}).get("model") in (None, EXPECTED["model"]) for row in [*parent_rows, *ext_rows]), runtime)
check("extension_run_bound_to_approved_freezes", ext_environment is not None and ext_protocol is not None
      and configuration.get("freeze_fingerprints") == {"task_manifest_sha256": ext_manifest.manifest_sha256,
                                                       "acquisition-extension-environment": ext_environment.input_fingerprint,
                                                       "acquisition-extension-protocol": ext_protocol.input_fingerprint})
ext_checkpoint = json.loads((EXT_RUN / "checkpoint.json").read_text(encoding="utf-8"))
ext_secondary = json.loads((EXT_RUN / "checkpoint.backup.json").read_text(encoding="utf-8"))
ext_differences = sorted(key for key in set(ext_checkpoint) | set(ext_secondary) if ext_checkpoint.get(key) != ext_secondary.get(key))
check("extension_checkpoint_completed", ext_checkpoint.get("status") == "completed" and ext_checkpoint.get("completed_run_count") == 60
      and ext_checkpoint.get("failed_run_count") == 0 and not ext_checkpoint.get("blocking_error"))
check("extension_checkpoint_backup_is_previous_generation", ext_differences == ["status"] and ext_secondary.get("status") == "running", {"differing_fields": ext_differences})
ext_mirror = mirror_identical(EXT_RUN, BACKUPS / EXT_ID)
check("extension_backup_valid", all(ext_mirror.values()) and len(jsonl(BACKUPS / EXT_ID / "results.jsonl")) == 60, ext_mirror)
check("extension_frozen_queue_valid", len(ext_manifest_paths) == 1
      and not validate_extension_queue_manifest(ext_manifest, parent_manifest, PARENT, require_frozen=True)
      and queue_identity_sha256(ext_manifest) == EXPECTED["extension_queue"] and ext_manifest.repository_commit == EXPECTED["extension_commit"])
ext_approvals = {}
for path in sorted(EXT_APPROVALS.glob("*.approval.json")):
    document = json.loads(path.read_text(encoding="utf-8"))
    meta = document if "status" in document else document.get("approval") or {}
    ext_approvals[path.name] = {"path": rel(path), "sha256": sha(path), "status": meta.get("status"), "approved_by": meta.get("approved_by"), "approved_at": meta.get("approved_at")}
check("extension_freezes_and_approvals_valid",
      not ext_environment_errors and not ext_protocol_errors
      and ext_environment.repository_commit == ext_protocol.repository_commit == EXPECTED["extension_commit"]
      and ext_environment.approval.get("status") == ext_protocol.approval.get("status") == "APPROVED"
      and ext_environment.inputs.get("model_digest") == EXPECTED["digest"] and ext_environment.inputs.get("model_quantization") == EXPECTED["quantization"]
      and ext_environment.inputs.get("provider_settings") == parent_environment.inputs.get("provider_settings")
      and ext_protocol.inputs.get("protocol") == extension_protocol_definition() and ext_protocol.inputs.get("protocol_sha256") == extension_protocol_sha256()
      and ext_protocol.inputs.get("inherited_protocol_sha256") == protocol_sha256()
      and len(ext_approvals) == 3 and all(item["status"] == "APPROVED" for item in ext_approvals.values()))
check("extension_starting_pool_snapshot_unchanged", not starting_pool_problems(ROOT, parent_state))
ext_log = (LOGS / f"{EXT_ID}.log").read_text(encoding="utf-8", errors="replace")
ext_launch = re.search(r"RQ1 scientific acquisition extension launch (\S+)", ext_log)
ext_exit = re.search(r"exited with status (\d+) at (\S+)", ext_log)
check("extension_process_exited_zero", bool(ext_exit) and ext_exit.group(1) == "0")
ext_listing_text, ext_files, ext_bytes = listing(EXT_RUN)

# ============================================================== combined pool and provenance
pool = rebuild_pool(ext_rows, base=parent_pool)
verify_snapshot(EXT_RUN, pool)
ext_snapshot = json.loads((EXT_RUN / "skill_pool.json").read_text(encoding="utf-8"))
final_hash = pool_hash(pool)
family_counts = {family: sum(skill.task_family == family for skill in pool) for family in TASK_FAMILIES}
check("final_combined_pool_loads_50_skills", len(pool) == 50 and ext_snapshot.get("pool_size") == 50 and ext_snapshot.get("pool_hash") == final_hash
      and (ext_checkpoint.get("phase_state") or {}).get("skill_pool", {}).get("hash") == final_hash, final_hash)
check("final_skills_by_family", family_counts == EXPECTED["family_skills"], family_counts)
check("parent_skills_preserved_inside_final_pool", [skill.identity() for skill in pool[:34]] == [skill.identity() for skill in parent_pool]
      and ext_rows[0].get("skill_pool_size_before") == 34 and ext_rows[0].get("skill_pool_hash_before") == PARENT.pool_hash)
rows_by_key = {PARENT_ID: {row["run_key"]: row for row in parent_rows}, EXT_ID: {row["run_key"]: row for row in ext_rows}}
entries, problems, family_rank = [], [], collections.Counter()
for skill in pool:
    origin_run = PARENT_ID if skill.pool_index <= 34 else EXT_ID
    row = rows_by_key[origin_run].get(skill.source_run_key)
    logical = skill.source_task_index + (180 if origin_run == EXT_ID else 0)
    family_rank[skill.task_family] += 1
    candidate = (row or {}).get("skill_candidate") or {}
    if (row is None or row.get("task_id") != skill.source_task_id or row.get("success") is not True or candidate.get("status") != "accepted"
            or candidate.get("skill") != skill.to_dict() or skill.provenance.get("experiment_id") != origin_run
            or skill.text != build_skill_text(title=skill.title, body=skill.body) or skill.text_sha256 != sha256_text(skill.text)
            or (origin_run == EXT_ID and (skill.provenance.get("parent_run_id") != PARENT_ID or skill.provenance.get("logical_acquisition_index") != logical))):
        problems.append(skill.skill_id)
    events_log = next((path for path in (row or {}).get("log_paths") or [] if path.endswith("episode-events.jsonl")), None)
    entries.append({
        "pool_index": skill.pool_index,
        "origin": "parent_units_1_180" if origin_run == PARENT_ID else "extension_units_181_240",
        "source_run_id": origin_run,
        "logical_acquisition_index": logical,
        "family_chronological_rank": family_rank[skill.task_family],
        "source_result_record": f"results/final/{origin_run}/results.jsonl#run_key={skill.source_run_key}",
        "source_episode_events_log": f"results/final/{origin_run}/{events_log}" if events_log else None,
        "skill": skill.to_dict(),
    })
check("skill_provenance_parent_to_extension_valid", not problems, problems)
check("chronological_acquisition_order_preserved", [entry["pool_index"] for entry in entries] == list(range(1, 51))
      and all(a["logical_acquisition_index"] < b["logical_acquisition_index"] for a, b in zip(entries, entries[1:])))
check("skill_ids_and_texts_unique", len({skill.skill_id for skill in pool}) == 50 and len({skill.text for skill in pool}) == 50)
accepted_candidates = sum((row.get("skill_candidate") or {}).get("status") == "accepted" for row in [*parent_rows, *ext_rows])
planned = {run: json.loads((ROOT / "results" / "final" / run / "manifests" / "acquisition.json").read_text(encoding="utf-8"))["planned_run_count"] for run in (PARENT_ID, EXT_ID)}
check("no_hidden_result_loss", accepted_candidates == 50 and planned == {PARENT_ID: 180, EXT_ID: 60} and len(parent_rows) + len(ext_rows) == 240, {"accepted_candidates": accepted_candidates, "planned_units": planned})
combined_rows = [*parent_rows, *ext_rows]
family_episodes = dict(collections.Counter(row["task_family"] for row in combined_rows))
check("combined_240_distinct_units_40_per_family", len({row["task_id"] for row in combined_rows}) == 240 and family_episodes == {family: 40 for family in TASK_FAMILIES}, family_episodes)
totals = {
    "episodes": len(combined_rows),
    "successful": parent_counts["successful"] + ext_counts["successful"],
    "scientific_failures": parent_counts["scientific_failures"] + ext_counts["scientific_failures"],
    "infrastructure_failures": parent_counts["infrastructure_failures"] + ext_counts["infrastructure_failures"],
    "retrieval_events": parent_counts["retrieval_events"] + ext_counts["retrieval_events"],
    "final_skill_count": len(pool),
}
check("combined_116_successes_124_failures_0_infra_0_retrieval", (totals["successful"], totals["scientific_failures"], totals["infrastructure_failures"], totals["retrieval_events"]) == (116, 124, 0, 0), totals)
check("queue_lineage_parent_to_extension_valid", (ext_manifest.lineage or {}).get("parent_run_id") == PARENT_ID
      and (ext_manifest.lineage or {}).get("parent_queue_sha256") == EXPECTED["parent_queue"]
      and (ext_manifest.lineage or {}).get("starting_pool_hash") == PARENT.pool_hash and (ext_manifest.lineage or {}).get("logical_index_offset") == 180)
passed = all(item["pass"] for item in checks.values())
if not passed:
    for name, item in checks.items():
        print("CHECK", name, item["pass"], json.dumps(item.get("detail")) if "detail" in item else "")
    sys.exit("closeout integrity audit FAILED; no outputs were written")

# ============================================================== acquisition yield (240)


def yield_block(rows: list, skills: list) -> dict:
    good = [row for row in rows if row.get("success") is True]
    statuses = collections.Counter((row.get("skill_candidate") or {}).get("status") for row in good)
    no_skill = sum((row.get("skill_candidate") or {}).get("status") == "declined" and ((row.get("skill_candidate") or {}).get("response") or "").strip() == "NO_SKILL" for row in good)
    reasons = collections.Counter(reason for row in good if (row.get("skill_candidate") or {}).get("status") == "rejected"
                                  for reason in (row["skill_candidate"].get("rejection_reasons") or []))
    accepted = statuses.get("accepted", 0)
    return {
        "episodes": len(rows),
        "successes": len(good),
        "scientific_failures": len(rows) - len(good),
        "terminations": dict(sorted(collections.Counter(row.get("termination_reason") for row in rows).items())),
        "accepted_skills": accepted,
        "no_skill": no_skill,
        "other_declines": statuses.get("declined", 0) - no_skill,
        "exact_normalized_duplicate_rejections": reasons.get("exact_normalized_duplicate", 0),
        "other_rejections": sum(value for key, value in reasons.items() if key != "exact_normalized_duplicate"),
        "other_rejection_reasons": {key: value for key, value in reasons.items() if key != "exact_normalized_duplicate"},
        "skill_yield_per_successful_episode": round(accepted / len(good), 4) if good else None,
        "skill_yield_per_episode": round(accepted / len(rows), 4) if rows else None,
        "skills_in_final_pool": len(skills),
    }


families = {}
for family in TASK_FAMILIES:
    block = yield_block([row for row in combined_rows if row["task_family"] == family], [skill for skill in pool if skill.task_family == family])
    block["parent_skills"] = sum(skill.task_family == family for skill in pool[:34])
    block["extension_skills"] = sum(skill.task_family == family for skill in pool[34:])
    block["total_final_skills"] = family_counts[family]
    block["by_run"] = {
        PARENT_ID: yield_block([row for row in parent_rows if row["task_family"] == family], [skill for skill in pool[:34] if skill.task_family == family]),
        EXT_ID: yield_block([row for row in ext_rows if row["task_family"] == family], [skill for skill in pool[34:] if skill.task_family == family]),
    }
    families[family] = block
global_block = yield_block(combined_rows, list(pool))
selection = collections.Counter()
for row in combined_rows:
    selection.update(row.get("selection_rejections") or {})
raw_counts = " / ".join(str(family_counts[family]) for family in TASK_FAMILIES)
minimum, maximum = min(family_counts.values()), max(family_counts.values())
yield_report = {
    "schema_version": 1,
    "report_kind": "rq1-acquisition-skill-yield-240",
    "generated_at": now(),
    "runs": {"parent": PARENT_ID, "extension": EXT_ID},
    "global": {
        **global_block,
        "parent_skills": 34,
        "extension_skills": len(pool) - 34,
        "by_run": {PARENT_ID: yield_block(parent_rows, list(pool[:34])), EXT_ID: yield_block(ext_rows, list(pool[34:]))},
        "action_selection_attempt_rejections": dict(sorted(selection.items())),
        "minimum_skills_per_family": minimum,
        "maximum_skills_per_family": maximum,
        "bottleneck_families": sorted(family for family, count in family_counts.items() if count == minimum),
    },
    "families": families,
    "imbalance_statement": (f"The acquisition process itself produced an imbalanced skill distribution. Raw final counts: {raw_counts} "
                            f"by the six families ({', '.join(TASK_FAMILIES)})."),
}

# ============================================================== original library feasibility on the RAW pool
accum60 = CORE_PER_FAMILY + ACCUM_EXTRAS_PER_FAMILY["Accum-60"]
accum96 = CORE_PER_FAMILY + ACCUM_EXTRAS_PER_FAMILY["Accum-96"]
feasibility = {
    "basis": "raw acquired pool only (no human validation applied; validation can only reduce eligibility)",
    "clean_24_raw_feasible": all(count >= CORE_PER_FAMILY for count in family_counts.values()),
    "clean_24_families_below_4": {family: count for family, count in family_counts.items() if count < CORE_PER_FAMILY},
    "accum_60_raw_feasible": all(count >= accum60 for count in family_counts.values()),
    "accum_60_needs_per_family": accum60,
    "accum_60_families_below": {family: count for family, count in family_counts.items() if count < accum60},
    "accum_96_raw_feasible": all(count >= accum96 for count in family_counts.values()),
    "accum_96_needs_per_family": accum96,
    "accum_96_families_below": {family: count for family, count in family_counts.items() if count < accum96},
    "raw_maximum_balanced_skills_per_family": minimum,
    "raw_maximum_balanced_library": minimum * len(TASK_FAMILIES),
    "raw_maximum_is_frozen": False,
}
yield_report["original_library_feasibility_raw"] = feasibility

# ============================================================== write closeout outputs
OUT.mkdir(parents=True, exist_ok=False)
VAL.mkdir(parents=True, exist_ok=False)
(OUT / "extension-result-directory.sha256").write_text(ext_listing_text, encoding="utf-8")
pool_snapshot = {
    "schema_version": 1,
    "kind": "rq1-final-raw-acquired-skill-pool",
    "label": ("RAW ACQUIRED POOL after the complete 240-episode acquisition. It is NOT a human-validated library and NOT an evaluation "
              "condition. No semantic deduplication, manual edit, or quality filtering has been applied."),
    "generated_at": now(),
    "authority": {
        "results": [ref(PARENT_RUN / "results.jsonl"), ref(EXT_RUN / "results.jsonl")],
        "rule": "parent pool rebuilt from parent results.jsonl; extension results appended to it (rq1.acquisition.skill_pool.rebuild_pool with base)",
        "extension_snapshot": ref(EXT_RUN / "skill_pool.json"),
    },
    "pool_size": len(pool),
    "pool_hash": final_hash,
    "pool_hash_definition": "sha256 of canonical JSON of [PoolSkill.identity() in pool_index order] (identical to the extension skill_pool.json pool_hash)",
    "per_family": family_counts,
    "per_origin": {"parent_units_1_180": 34, "extension_units_181_240": len(pool) - 34},
    "chronological_order": "pool_index 1-50 is the acquisition order; logical_acquisition_index is the source episode position 1-240",
    "entries": entries,
}
(OUT / "final-skill-pool-50.json").write_text(json.dumps(pool_snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
(OUT / "acquisition-yield-240.json").write_text(json.dumps(yield_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

markdown = [
    "# RQ1 acquisition yield — complete 240 episodes", "",
    f"Parent `{PARENT_ID}` (units 1–180) and extension `{EXT_ID}` (units 181–240). No human validation has been applied.", "",
    f"- Episodes: {totals['episodes']} (40 per family)",
    f"- Successes: {totals['successful']} · scientific failures: {totals['scientific_failures']} · infrastructure failures: 0 · acquisition retrieval events: 0",
    f"- Accepted skills: {len(pool)} (parent 34, extension {len(pool) - 34})",
    f"- NO_SKILL: {global_block['no_skill']} · other declines: {global_block['other_declines']} · exact normalized duplicate rejections: "
    f"{global_block['exact_normalized_duplicate_rejections']} · other rejections: {global_block['other_rejections']}",
    f"- Skill yield per success: {global_block['skill_yield_per_successful_episode']} · per episode: {global_block['skill_yield_per_episode']}", "",
    "| Family | Episodes | Successes | Failures | Accepted | NO_SKILL | Dup. rejections | Other rejections | Yield / success | Yield / episode | Parent skills | Extension skills | Final skills |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for family, block in families.items():
    markdown.append(f"| {family} | {block['episodes']} | {block['successes']} | {block['scientific_failures']} | {block['accepted_skills']} | {block['no_skill']} | "
                    f"{block['exact_normalized_duplicate_rejections']} | {block['other_rejections']} | {block['skill_yield_per_successful_episode']} | "
                    f"{block['skill_yield_per_episode']} | {block['parent_skills']} | {block['extension_skills']} | {block['total_final_skills']} |")
markdown += [
    "", f"**{yield_report['imbalance_statement']}**", "",
    "## Original library conditions on the raw pool", "",
    f"- Clean-24 raw feasible: {'YES' if feasibility['clean_24_raw_feasible'] else 'NO'} — needs 4 per family; below: {feasibility['clean_24_families_below_4']}",
    f"- Accum-60 raw feasible: {'YES' if feasibility['accum_60_raw_feasible'] else 'NO'} — needs {accum60} per family; below: {feasibility['accum_60_families_below']}",
    f"- Accum-96 raw feasible: {'YES' if feasibility['accum_96_raw_feasible'] else 'NO'} — needs {accum96} per family; below: {feasibility['accum_96_families_below']}",
    f"- Raw maximum balanced library: {minimum} per family × 6 = {minimum * 6} skills (not frozen; human validation may reduce it)", "",
]
(OUT / "acquisition-yield-240.md").write_text("\n".join(markdown), encoding="utf-8")

# ============================================================== human skill-validation package
HUMAN_COLUMNS = ["reviewer_quality_pass", "reviewer_quality_reason", "reviewer_notes", "reviewed_at"]
review_rows = []
for number, entry in enumerate(entries, 1):
    skill = entry["skill"]
    provenance = skill["provenance"]
    review_rows.append({
        "review_row": number,
        "skill_id": skill["skill_id"],
        "pool_index": entry["pool_index"],
        "task_family": skill["task_family"],
        "family_chronological_rank": entry["family_chronological_rank"],
        "logical_acquisition_index": entry["logical_acquisition_index"],
        "origin": entry["origin"],
        "source_run_id": entry["source_run_id"],
        "source_task_id": skill["source_task_id"],
        "source_task_index_in_run": skill["source_task_index"],
        "source_run_key": skill["source_run_key"],
        "source_attempt_id": skill["source_attempt_id"],
        "source_episode_success": True,
        "source_result_record": entry["source_result_record"],
        "source_episode_events_log": entry["source_episode_events_log"],
        "skill_created_at": skill["created_at"],
        "generation_model": provenance.get("model"),
        "skill_title": skill["title"],
        "skill_body": skill["body"],
        "skill_text": skill["text"],
        "skill_text_sha256": skill["text_sha256"],
        "duplicate_status": "exact_normalized_unique",
        "acquisition_automated_validation": "passed",
        **{column: "" for column in HUMAN_COLUMNS},
    })
columns = list(review_rows[0].keys())
csv_buffer = io.StringIO()
writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\r\n")
writer.writeheader()
writer.writerows(review_rows)
(VAL / "skill-validation-review.csv").write_bytes(("﻿" + csv_buffer.getvalue()).encode("utf-8"))

instructions = f"""# Human skill-quality validation — RQ1 acquisition (240 episodes)

Status: **PENDING HUMAN REVIEW.** Nothing in this package has been judged, approved, or selected.

## What you review

- `skill-validation-review.csv` (open in Excel or LibreOffice) and the same rows in `skill-validation-review.json`.
- 50 rows, one per acquired skill, in chronological acquisition order (`pool_index` 1–50).
- Source: the raw pool `artifacts/acquisition-closeout/rq1-acquisition-240-final/final-skill-pool-50.json`, pool hash `{final_hash}`.
- Fill **only** the four empty columns: `reviewer_quality_pass`, `reviewer_quality_reason`, `reviewer_notes`, `reviewed_at`.
- Do not edit, reorder, add, or delete any other cell or row. Before the results are used, every other column is checked against `skill_text_sha256` and the pool hash.

## Where the criteria come from

The repository has no separate written quality rubric. `rq1.skills.library` refers to a "frozen human quality-validation rubric", and the approved methodology defines its content in these records, quoted exactly:

- Decision 003 (`docs/decisions/003-skill-creation-policy.md`): "Acquire general skills from successful training episodes only; no task IDs, room/object instance numbers, memorised trajectories, duplicate equivalents, or positive skills after failure."
- `hermes/prompts/skill_validation.md`: "Reject task IDs, room/object instance numbers, memorised trajectories, duplicates, and skills created after failures."
- `hermes/prompts/post_success_learning.md`: "Create at most one new reusable, non-task-specific skill after a successful `train` episode only."
- Decision 007: duplicates are rejected only when the normalized skill text is exactly equal; "Near-duplicates and independently generated similar skills are preserved"; no semantic or embedding deduplication.

The checklist below turns those criteria into review questions. Q2, Q6, and Q7 come from the researcher's closeout instruction of 2026-09-14 and follow from "general skills".

**Confirm this checklist before you start reviewing.** Library construction will treat it as the frozen rubric.

## Checklist: a skill PASSES only if every answer is yes

- **Q1 Generalized and reusable.** Is it transferable guidance for this kind of task, rather than one memorised episode or trajectory?
- **Q2 Family relevance.** Is it relevant to its recorded `task_family`?
- **Q3 No task identifiers.** Does it avoid task or trial IDs, split names with numbers, and source task names?
- **Q4 No instance memorization.** Does it avoid room or object instance numbers (for example "cabinet 3" or "countertop 1") and other memorised instance details?
- **Q5 Not a trajectory copy.** Is it more than a copy of the source episode's raw action sequence?
- **Q6 Understandable and actionable.** Is it clear enough to act on?
- **Q7 Not empty or corrupt.** Is the text complete, not empty, and not truncated?

Acquisition already applied automatic checks for task-ID and instance-number patterns, verbatim instance actions, and exact normalized duplicates. Every row passed them (`acquisition_automated_validation = passed`). Your review is independent: a skill may still FAIL any question.

## Not criteria (do not use)

- How useful, good, or elegant the skill seems compared with others; ranking; personal or model preference.
- Similarity to other skills. Near-duplicates are judged one at a time and are never failed for resembling another skill (no semantic deduplication).
- Expected evaluation performance, embedding similarity, or any `valid_seen` / `valid_unseen` information. No evaluation has run.
- Wording style or length, unless it makes the skill fail Q6 or Q7.

## How to fill the columns

- `reviewer_quality_pass`: exactly `PASS` or `FAIL`.
- `reviewer_quality_reason`: required. For FAIL, name the question(s), e.g. `Q4: names a numbered receptacle`. For PASS, `meets Q1-Q7` is enough.
- `reviewer_notes`: optional.
- `reviewed_at`: the UTC time you judged the row, ISO-8601, e.g. `2026-09-15T10:00:00Z`.

Review every row once, in the given order, and judge each skill on its own text. `source_task_id` and the linked episode logs are there for auditing leakage or trajectory copying, not for judging usefulness.

## Chronological selection rule (preserved, not yet applied)

- The original rule takes the earliest qualifying skills per family.
- `family_chronological_rank` (with `pool_index`) gives that order.
- After review, a later and separately approved step can deterministically select the earliest N PASS skills per family.
- That selection has **not** been performed, and no library size has been chosen or frozen.

## Context

- The raw pool is imbalanced: {raw_counts} skills for {', '.join(TASK_FAMILIES)}.
- One extension candidate (logical unit 202, look_at_object) was rejected at acquisition as an exact normalized duplicate. It is not in the pool.
- `duplicate_status = exact_normalized_unique` means no two pool skills have identical normalized text. Near-duplicates were not assessed and are retained.
- Using a second independent reviewer is the researcher's decision; the existing methodology does not require one. Cohen's kappa in the methodology applies to retrieval relevance labels.

## Return

Return the completed `skill-validation-review.csv` (preferred) or JSON, with only the four review columns filled. The amended nested library sizes are decided and frozen only after that, and evaluation is prepared only after the freeze.
"""
(VAL / "skill-validation-instructions.md").write_text(instructions, encoding="utf-8")
review_document = {
    "schema_version": 1,
    "kind": "rq1-skill-quality-validation-review",
    "status": "PENDING_HUMAN_REVIEW",
    "generated_at": now(),
    "source_pool": {"path": rel(OUT / "final-skill-pool-50.json"), "sha256": sha(OUT / "final-skill-pool-50.json"), "pool_hash": final_hash, "pool_size": len(pool)},
    "instructions": {"path": rel(VAL / "skill-validation-instructions.md"), "sha256": sha(VAL / "skill-validation-instructions.md")},
    "skills_requiring_review": len(review_rows),
    "per_family": family_counts,
    "immutable_columns": [column for column in columns if column not in HUMAN_COLUMNS],
    "human_review_columns": HUMAN_COLUMNS,
    "allowed_reviewer_quality_pass": ["PASS", "FAIL"],
    "selection_rule_after_review": "earliest N PASS skills per family by pool_index / family_chronological_rank; not applied; N not chosen",
    "rows": review_rows,
}
(VAL / "skill-validation-review.json").write_text(json.dumps(review_document, indent=2, sort_keys=False, ensure_ascii=False) + "\n", encoding="utf-8")

# ============================================================== pre-evaluation library feasibility (analysis only)
levels = tuple(range(1, minimum + 1))


def chains(max_core: int, max_total: int) -> list:
    options = []
    for size in range(1, max_total + 1):
        for chain in itertools.combinations(range(1, max_total + 1), size):
            if chain[0] <= max_core:
                options.append(chain)
    return options


def option_table(options: list) -> list:
    if not options:
        return ["No balanced nested library condition is possible (some family would have no validated core skill).", ""]
    lines = ["| Per-family sizes | Library sizes (NoLib first) | Core per family | Conditions incl. NoLib | Evaluation episodes (30 tasks × 3 seeds × conditions) |", "|---|---|---|---|---|"]
    for chain in options:
        lines.append(f"| {' ⊂ '.join(str(value) for value in chain)} | 0 / {' / '.join(str(value * 6) for value in chain)} | {chain[0]} | {len(chain) + 1} | "
                     f"{EVALUATION_TASKS * len(EVALUATION_SEEDS) * (len(chain) + 1)} |")
    return lines + [""]


family_table = ["| Family | Parent skills | Extension skills | Raw final skills | Clean-24 (4) | Accum-60 (10) | Accum-96 (16) |", "|---|---|---|---|---|---|---|"]
for family in TASK_FAMILIES:
    count = family_counts[family]
    family_table.append(f"| {family} | {families[family]['parent_skills']} | {families[family]['extension_skills']} | {count} | "
                        f"{'ok' if count >= 4 else 'short'} | {'ok' if count >= accum60 else 'short'} | {'ok' if count >= accum96 else 'short'} |")
report = [
    "# Pre-evaluation library feasibility (analysis only)", "",
    f"Generated {now()} from the closed 240-episode acquisition. **This report freezes nothing and selects no skills.** "
    "Any new library design needs a separate, explicitly approved protocol amendment, frozen before any evaluation outcome exists.", "",
    "## 1. Where the study stands", "",
    "- Acquisition ended at the pre-approved hard cap of 240 TRAIN episodes (40 per family): the parent run (units 1–180, Decision 007) and the balanced extension (units 181–240, Decision 011).",
    f"- Totals: {totals['successful']} successes, {totals['scientific_failures']} scientific failures, 0 infrastructure failures, 0 acquisition retrieval events, {len(pool)} accepted skills (pool hash `{final_hash}`).",
    "- No evaluation episode has run and no evaluation outcome exists. No evaluation task manifest or final environment/protocol freeze exists.",
    "- No further acquisition is authorized.", "",
    "## 2. Raw acquired pool", "",
    *family_table, "",
    f"{yield_report['imbalance_statement']}", "",
    "## 3. The original conditions cannot be built", "",
    "`configs/libraries.yaml` (status FROZEN) and `rq1.skills.library` define NoLib 0, Clean-24 (4 validated core skills per family), Accum-60 (core + 6 chronological extras per family) and Accum-96 (core + 12 per family). Construction fails closed when a quota is unmet.", "",
    f"- **Clean-24 raw feasible: NO.** cool_and_place has only {family_counts['cool_and_place']} raw skills, below the required 4, even before human validation.",
    f"- **Accum-60 raw feasible: NO.** It needs 10 per family; below: {feasibility['accum_60_families_below']}.",
    f"- **Accum-96 raw feasible: NO.** It needs 16 per family; every family is below: {feasibility['accum_96_families_below']}.",
    "- No synthetic, manually authored, relabelled, or double-counted skills may be used to reach these quotas.", "",
    "## 4. Raw balanced limit", "",
    f"Balanced libraries need the same number of skills per family. The smallest family has {minimum} raw skills, so the **raw maximum balanced library is {minimum} × 6 = {minimum * 6} skills**. "
    "This is not frozen. Human validation can reduce what is feasible (section 5).", "",
    "## 5. Candidate balanced nested designs (after human validation)", "",
    "Let *v* be the minimum number of PASS skills over the six families (0–3, because cool_and_place has 3 raw skills). Every design below keeps NoLib = 0, balanced families, nested libraries whose smallest non-empty library is the core, the same core inside every larger library, chronological extras, no semantic deduplication, and acquired skills only.", "",
    "Which skills may serve as extras is an **open decision** for the amendment:",
    "- **Rule A (as currently implemented).** `rq1.skills.library` requires human validation only for the core. Accumulated extras are \"the earliest 6 / 12 additional successful acquired skills per family (after the core)\", whether or not they pass review. Core ≤ *v*; largest library ≤ 3 per family.",
    "- **Rule B (stricter alternative).** Every skill in every library must PASS. All sizes ≤ *v*.", "",
]
for v in (3, 2, 1, 0):
    report += [f"### If v = {v}", "", "**Rule A:**", "", *option_table(chains(v, minimum) if v else []), "**Rule B:**", "", *option_table(chains(v, v) if v else [])]
report += [
    "Notes:",
    f"- Only the three-level chain 1 ⊂ 2 ⊂ 3 (libraries 0 / 6 / 12 / 18) keeps four conditions and the planned {EVALUATION_TASKS * len(EVALUATION_SEEDS) * 4} evaluation episodes. It needs v ≥ 1 under Rule A, or v = 3 under Rule B.",
    f"- Each condition adds {EVALUATION_TASKS * len(EVALUATION_SEEDS)} evaluation episodes (30 tasks × seeds 11, 29, 47).",
    "- Under Rule A a FAIL skill can still enter a larger library as an accumulated extra; under Rule B it cannot. The choice changes what the Clean-versus-accumulated comparison means and must be made explicitly.",
    "- With a 6-skill library, top-3 retrieval returns half of the library; Precision@3 and Retrieval Noise keep their definitions. This is an analytic consideration, not a recommendation.", "",
    "## 6. Unchanged evaluation and retrieval protocol", "",
    "None of the following changes unless a later explicit amendment says so:", "",
    "- `configs/tasks/evaluation.yaml`: 30 `valid_unseen` tasks, 5 per family, seeds 11, 29, 47, conditions NoLib / Clean-24 / Accum-60 / Accum-96, 360 core episodes, status BLOCKED_UNTIL_REAL_PILOT_AND_PROTOCOL_FREEZE. Changing the conditions requires amending this file together with `configs/libraries.yaml`.",
    "- Controlled reversible action perturbation; exactly one post-failure retrieval for library conditions; NoLib has no retrieval.",
    "- Sentence-BERT `sentence-transformers/all-mpnet-base-v2`, revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`; cosine ranking; top-3 exactly once after the controlled failure (`configs/base.yaml`).",
    "- Retrieval query `query-v2` (`rq1/retrieval/query.py`): task goal, current observation, inventory, canonical failure message. No action history.",
    "- Human relevance labels; Precision@3; Retrieval Noise = 1 − Precision@3; Cohen's kappa process; recovery metric definitions (`docs/METRICS.md`, `docs/RECOVERY_METRICS.md`).", "",
    "## 7. Decisions that remain with the researcher, in order", "",
    "1. Confirm the quality checklist in `artifacts/skill-validation/rq1-acquisition-240/skill-validation-instructions.md` as the frozen rubric.",
    "2. Complete human validation of all 50 skills.",
    "3. From the validated counts (v), decide Rule A or Rule B and the nested sizes and labels. Record them in a decision record and protocol amendment (`configs/libraries.yaml`, `configs/tasks/evaluation.yaml`, `rq1.skills.library`, `docs/SNAPSHOT_PROTOCOL.md`) and freeze them before evaluation.",
    "4. Only then build the libraries (earliest N PASS skills per family) and prepare evaluation.", "",
]
(OUT / "pre-evaluation-library-feasibility.md").write_text("\n".join(report), encoding="utf-8")

# ============================================================== combined closeout manifest (written last)
decisions = {rel(path): sha(path) for path in sorted((ROOT / "docs" / "decisions").glob("*.md"))}
config_files = ["configs/acquisition/protocol.yaml", "configs/acquisition/extension.yaml", "configs/libraries.yaml", "configs/tasks/acquisition.yaml",
                "configs/tasks/evaluation.yaml", "configs/base.yaml", "configs/relevance_rules.yaml"]
environment_inputs = ext_environment.inputs
manifest = {
    "schema_version": 1,
    "manifest_kind": "rq1-acquisition-240-combined-closeout",
    "closeout_timestamp": now(),
    "closeout_passed": passed,
    "scientific_evidence": True,
    "acquisition_hard_cap": 240,
    "further_acquisition_authorized": False,
    "evaluation_started": False,
    "new_library_sizes_frozen": False,
    "runs": {
        "parent": {"run_id": PARENT_ID, "git_sha": EXPECTED["parent_commit"], "queue_sha256": EXPECTED["parent_queue"], "logical_positions": [1, 180],
                   "closeout_manifest": ref(parent_closeout_path), "counts": parent_counts},
        "extension": {"run_id": EXT_ID, "git_sha": EXPECTED["extension_commit"], "queue_sha256": EXPECTED["extension_queue"], "logical_positions": [181, 240],
                      "starting_pool_size": 34, "starting_pool_hash": PARENT.pool_hash, "counts": ext_counts},
    },
    "acquisition_protocol": {
        "policy_version": protocol_definition()["policy_version"],
        "protocol_sha256": protocol_sha256(),
        "extension_policy_version": extension_protocol_definition()["policy_version"],
        "extension_protocol_sha256": extension_protocol_sha256(),
        "config_hashes": {path: sha(ROOT / path) for path in config_files},
    },
    "totals": {**totals, "family_episode_counts": family_episodes},
    "final_skill_pool": {"size": len(pool), "hash": final_hash, "per_family": family_counts, "snapshot": ref(OUT / "final-skill-pool-50.json")},
    "model": {"tag": environment_inputs.get("model_tag"), "digest": environment_inputs.get("model_digest"),
              "quantization": environment_inputs.get("model_quantization"), "provider_settings": environment_inputs.get("provider_settings"),
              "identical_to_parent_environment_freeze": all(environment_inputs.get(key) == parent_environment.inputs.get(key)
                                                            for key in ("model_tag", "model_digest", "model_quantization", "provider_settings"))},
    "environment": {key: environment_inputs.get(key) for key in ("os", "kernel", "hostname", "gpu", "gpu_driver", "cuda_version", "python_version", "torch_version",
                                                                  "torch_cuda_version", "alfworld_version", "alfworld_data_identity", "hermes_version",
                                                                  "hermes_commit", "ollama_version", "dependency_lock_sha256", "sbert_model", "sbert_revision")},
    "timestamps": {
        "parent": {"start": parent_closeout["start_time"], "completion": parent_closeout["completion_time"]},
        "extension": {"launch": ext_launch.group(1) if ext_launch else None,
                      "experiment_start_time": json.loads((EXT_RUN / "run_manifest.json").read_text(encoding="utf-8")).get("experiment_start_time"),
                      "first_result": ext_rows[0]["timestamp"], "last_result": ext_rows[-1]["timestamp"],
                      "final_checkpoint": ext_checkpoint.get("timestamp"), "process_exit": ext_exit.group(2) if ext_exit else None},
    },
    "artifacts": {
        "parent_results": {name: ref(PARENT_RUN / name) for name in ("results.jsonl", "checkpoint.json", "skill_pool.json", "run_manifest.json", "errors.jsonl", "manifests/acquisition.json")},
        "extension_results": {name: ref(EXT_RUN / name) for name in ("results.jsonl", "checkpoint.json", "skill_pool.json", "run_manifest.json", "errors.jsonl", "manifests/acquisition.json")},
        "extension_result_directory": {"path": rel(EXT_RUN), "files": ext_files, "bytes": ext_bytes, "listing": ref(OUT / "extension-result-directory.sha256")},
        "parent_result_directory": {"path": rel(PARENT_RUN), "files": parent_files, "bytes": parent_bytes, "listing": ref(PARENT_CLOSEOUT_DIR / "result-directory.sha256")},
        "non_authoritative": {
            "parent_checkpoint_backup": {**ref(PARENT_RUN / "checkpoint.backup.json"), "classification": "NON-AUTHORITATIVE previous checkpoint generation (status running)"},
            "extension_checkpoint_backup": {**ref(EXT_RUN / "checkpoint.backup.json"), "classification": "NON-AUTHORITATIVE previous checkpoint generation (status running); identical except status"},
        },
        "parent_freezes": {"task": ref(parent_manifest_path), "environment": {**ref(ROOT / ENVIRONMENT_FREEZE), "input_fingerprint": parent_environment.input_fingerprint},
                           "protocol": {**ref(ROOT / PROTOCOL_FREEZE), "input_fingerprint": parent_protocol.input_fingerprint}},
        "parent_approvals": parent_approvals,
        "extension_freezes": {"task": {**ref(ext_manifest_paths[0]), "manifest_sha256": ext_manifest.manifest_sha256},
                              "environment": {**ref(ROOT / EXTENSION_ENVIRONMENT_FREEZE), "input_fingerprint": ext_environment.input_fingerprint},
                              "protocol": {**ref(ROOT / EXTENSION_PROTOCOL_FREEZE), "input_fingerprint": ext_protocol.input_fingerprint}},
        "extension_approvals": ext_approvals,
        "extension_starting_pool": ref(starting_pool_path(ROOT)),
        "extension_check_evidence": ref(EXT_EVIDENCE),
        "extension_final_preflight": ref(EXT_PREFLIGHT),
        "logs": {"parent": ref(LOGS / f"{PARENT_ID}.log"), "extension": ref(LOGS / f"{EXT_ID}.log")},
        "export_180": {"archive_sha256_file": ref(EXPORTS / "rq1-acquisition-180-final-20260914.tar.gz.sha256"),
                       "validation": ref(EXPORTS / "rq1-acquisition-180-final-20260914.validation.json")},
        "decision_records": decisions,
    },
    "outputs": {
        "final_skill_pool_50": ref(OUT / "final-skill-pool-50.json"),
        "acquisition_yield_240_json": ref(OUT / "acquisition-yield-240.json"),
        "acquisition_yield_240_md": ref(OUT / "acquisition-yield-240.md"),
        "pre_evaluation_library_feasibility": ref(OUT / "pre-evaluation-library-feasibility.md"),
        "skill_validation_review_csv": ref(VAL / "skill-validation-review.csv"),
        "skill_validation_review_json": ref(VAL / "skill-validation-review.json"),
        "skill_validation_instructions": ref(VAL / "skill-validation-instructions.md"),
    },
    "original_library_feasibility_raw": feasibility,
    "human_validation": {"status": "PENDING", "skills_requiring_review": len(review_rows)},
    "integrity_checks": checks,
    "generator": {"script": str(pathlib.Path(__file__).resolve()), "sha256": sha(pathlib.Path(__file__).resolve()), "repository_head": head},
}
(OUT / "combined-closeout-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("CLOSEOUT_PASSED", passed)
for name, item in checks.items():
    print("CHECK", name, item["pass"])
print("TOTALS", json.dumps(totals))
print("FAMILY_SKILLS", json.dumps(family_counts))
print("FINAL_POOL_HASH", final_hash)
print("YIELD_GLOBAL", json.dumps({key: global_block[key] for key in ("no_skill", "other_declines", "exact_normalized_duplicate_rejections", "other_rejections", "skill_yield_per_successful_episode", "skill_yield_per_episode")}))
print("FEASIBILITY", json.dumps(feasibility))
for path in (OUT / "combined-closeout-manifest.json", OUT / "final-skill-pool-50.json", OUT / "acquisition-yield-240.json", OUT / "acquisition-yield-240.md",
             OUT / "pre-evaluation-library-feasibility.md", VAL / "skill-validation-review.csv", VAL / "skill-validation-review.json", VAL / "skill-validation-instructions.md"):
    print("OUTPUT", rel(path), sha(path))

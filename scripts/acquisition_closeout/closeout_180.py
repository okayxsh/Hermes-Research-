"""RQ1 180-episode acquisition closeout: read-only integrity audit, closeout manifest, and skill-yield report.

Reads the completed scientific run and its freezes; writes only under
artifacts/acquisition-closeout/<run id>/ and refuses to overwrite existing outputs.
"""
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path("/workspace/persistent/rq1-protocol-migration")
sys.path.insert(0, str(ROOT / "src"))
from rq1.acquisition.gates import ENVIRONMENT_FREEZE, FROZEN_TASK_DIR, PROTOCOL_FREEZE, load_task_manifest, queue_identity_sha256, validate_queue_manifest  # noqa: E402
from rq1.acquisition.launch import attempt_lineage  # noqa: E402
from rq1.acquisition.protocol import protocol_definition, protocol_sha256  # noqa: E402
from rq1.acquisition.skill_pool import pool_hash, rebuild_pool, verify_snapshot  # noqa: E402
from rq1.experiment.persistence import CRITICAL_FILES  # noqa: E402
from rq1.freeze.validation import read_freeze  # noqa: E402
from rq1.skills.library import ACCUM_EXTRAS_PER_FAMILY, CORE_PER_FAMILY, TASK_FAMILIES  # noqa: E402

RUN_ID = "rq1-acquisition-gemma4-12b"
RUN = ROOT / "results" / "final" / RUN_ID
BACKUP = pathlib.Path("/workspace/persistent/backups") / RUN_ID
SCIENTIFIC_LOG = pathlib.Path("/workspace/persistent/logs") / f"{RUN_ID}.log"
OUT = ROOT / "artifacts" / "acquisition-closeout" / RUN_ID
APPROVAL_DIR = ROOT / "artifacts" / "approvals" / "acquisition" / "8bd452e76120"
EVIDENCE = ROOT / "artifacts" / "prelaunch" / "acquisition-check" / "prelaunch-acquisition-check-20260913-final-a" / "acquisition-check-report.json"
BUNDLE = pathlib.Path("/workspace/persistent/backups/rq1-protocol-migration-final-8bd452e.bundle")
EXPECTED = {
    "commit": "8bd452e76120da21d721c6e894d1ce5af4912ca9",
    "queue": "1f401869ea877969072f4d73e314db5841ff3f9aae875057ec3e4b5fe468ee09",
    "model": "gemma4:12b",
    "digest": "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c",
    "quantization": "Q4_K_M",
    "total": 180,
    "successes": 86,
    "failures": 94,
    "skills": 34,
}


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


if OUT.exists():
    sys.exit(f"refusing to overwrite existing closeout outputs: {OUT}")

checks: dict = {}


def check(name: str, condition: bool, detail=None) -> None:
    checks[name] = {"pass": bool(condition), **({"detail": detail} if detail is not None else {})}


# ------------------------------------------------------------------ repository and processes
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
check("repository_at_frozen_commit_and_clean", head == EXPECTED["commit"] and not dirty, {"head": head})
live = subprocess.run(["pgrep", "-f", "^/opt/rq1-venv/bin/python -m rq1.cli acquisition"], capture_output=True, text=True).stdout.split()
tmux = subprocess.run(["tmux", "has-session", "-t", "rq1-acquisition"], capture_output=True).returncode == 0
check("no_scientific_process_alive", not live, {"pids": live})
check("no_rq1_acquisition_tmux_runner", not tmux)

# ------------------------------------------------------------------ results journal
rows = jsonl(RUN / "results.jsonl")
frozen_paths = sorted((ROOT / FROZEN_TASK_DIR).glob("acquisition-*.json"))
manifest = load_task_manifest(frozen_paths[0])
queue = sorted(manifest.tasks, key=lambda task: task.order_index)
check("exactly_180_result_records", len(rows) == EXPECTED["total"], len(rows))
check("180_distinct_units", len({row["run_key"] for row in rows}) == 180 and len({row["task_id"] for row in rows}) == 180)
check("all_frozen_queue_units_represented", {task.task_id for task in queue} == {row["task_id"] for row in rows})
check("queue_order_consistent", [row["task_index"] for row in rows] == list(range(1, 181))
      and all(row["task_id"] == queue[row["task_index"] - 1].task_id for row in rows))
check("all_completed_single_attempt", all(row["status"] == "completed" and row.get("attempt_index") == 1 for row in rows))
successes = sum(row.get("success") is True for row in rows)
failures = sum(row.get("status") == "completed" and row.get("success") is not True for row in rows)
infrastructure = sum(row.get("status") == "failed" for row in rows)
errors = jsonl(RUN / "errors.jsonl")
check("success_count_86", successes == EXPECTED["successes"], successes)
check("scientific_failure_count_94", failures == EXPECTED["failures"], failures)
check("infrastructure_failures_0", infrastructure == 0 and not errors, {"failed_rows": infrastructure, "error_records": len(errors)})
lineage = attempt_lineage(rows)
check("no_duplicates_or_unauthorized_retries", lineage["authorized"] and lineage["duplicate_completed_units"] == 0 and not lineage["retried_units"])
retrieval_counts = sum(int(row.get("scientific_retrieval_count") or 0) for row in rows)
retrieved_ids = sum(len(row.get("retrieved_skill_ids") or []) for row in rows)
retrieval_events = 0
models = collections.Counter()
for row in rows:
    for relpath in row.get("log_paths") or []:
        if relpath.endswith("episode-events.jsonl"):
            for event in jsonl(RUN / relpath):
                retrieval_events += "retriev" in event.get("event", "")
    candidate = row.get("skill_candidate") or {}
    if candidate.get("model"):
        models[candidate["model"]] += 1
check("retrieval_events_0", retrieval_counts == 0 and retrieved_ids == 0 and retrieval_events == 0,
      {"retrieval_count_field": retrieval_counts, "retrieved_skill_ids": retrieved_ids, "retrieval_events": retrieval_events})
check("all_units_scientific_evidence", all(row.get("scientific_evidence") is True for row in rows))
configuration = json.loads((RUN / "manifests" / "acquisition.json").read_text(encoding="utf-8")).get("configuration", {})
check("model_identity_in_run", configuration.get("model_name") == EXPECTED["model"] and set(models) <= {EXPECTED["model"]},
      {"configuration_model": configuration.get("model_name"), "skill_generation_models": dict(models)})
check("actions_within_50", max(len(row.get("episode_actions") or []) for row in rows) <= 50)

# ------------------------------------------------------------------ skill pool
pool = rebuild_pool(rows)
snapshot = json.loads((RUN / "skill_pool.json").read_text(encoding="utf-8"))
verify_snapshot(RUN, pool)
checkpoint = json.loads((RUN / "checkpoint.json").read_text(encoding="utf-8"))
check("final_pool_34_skills", len(pool) == EXPECTED["skills"] and snapshot.get("pool_size") == 34, len(pool))
check("final_pool_hash_stable", pool_hash(pool) == snapshot.get("pool_hash") == (checkpoint.get("phase_state") or {}).get("skill_pool", {}).get("hash"),
      pool_hash(pool))
check("pool_snapshot_loads_and_matches_results", True)

# ------------------------------------------------------------------ checkpoints and backup mirror
check("primary_checkpoint_completed", checkpoint.get("status") == "completed" and checkpoint.get("completed_run_count") == 180
      and checkpoint.get("failed_run_count") == 0)
secondary = json.loads((RUN / "checkpoint.backup.json").read_text(encoding="utf-8"))
differences = sorted(key for key in set(checkpoint) | set(secondary) if checkpoint.get(key) != secondary.get(key))
check("checkpoint_backup_is_previous_generation", differences == ["status"] and secondary.get("status") == "running"
      and secondary.get("completed_run_count") == 180, {"differing_fields": differences})
mirror = {}
for name in [*CRITICAL_FILES, *(f"manifests/{path.name}" for path in sorted((RUN / "manifests").glob("*.json")))]:
    primary, copy = RUN / name, BACKUP / name
    if primary.is_file():
        mirror[name] = copy.is_file() and sha(copy) == sha(primary)
check("backup_mirror_identical", all(mirror.values()) and len(jsonl(BACKUP / "results.jsonl")) == 180, mirror)

# ------------------------------------------------------------------ freezes and approvals
environment, environment_errors = read_freeze(ROOT / ENVIRONMENT_FREEZE, "acquisition-environment")
protocol, protocol_errors = read_freeze(ROOT / PROTOCOL_FREEZE, "acquisition-protocol")
check("environment_freeze_validates", environment is not None and not environment_errors and environment.repository_commit == EXPECTED["commit"]
      and environment.approval.get("status") == "APPROVED" and environment.inputs.get("model_digest") == EXPECTED["digest"]
      and environment.inputs.get("model_quantization") == EXPECTED["quantization"])
check("protocol_freeze_validates", protocol is not None and not protocol_errors and protocol.repository_commit == EXPECTED["commit"]
      and protocol.approval.get("status") == "APPROVED" and protocol.inputs.get("protocol") == protocol_definition()
      and protocol.inputs.get("protocol_sha256") == protocol_sha256())
check("task_freeze_validates", len(frozen_paths) == 1 and not validate_queue_manifest(manifest, require_frozen=True)
      and queue_identity_sha256(manifest) == EXPECTED["queue"] and manifest.repository_commit == EXPECTED["commit"])
approvals = {}
for path in sorted(APPROVAL_DIR.glob("*.approval.json")):
    document = json.loads(path.read_text(encoding="utf-8"))
    status = document.get("status") if "task-freeze" in path.name else (document.get("approval") or {}).get("status")
    approvals[path.name] = {"sha256": sha(path), "status": status}
check("approvals_validate", len(approvals) == 3 and all(item["status"] == "APPROVED" for item in approvals.values()))

# ------------------------------------------------------------------ timing
launch_log = SCIENTIFIC_LOG.read_text(encoding="utf-8", errors="replace")
launch_match = re.search(r"RQ1 scientific acquisition launch (\S+)", launch_log)
exit_match = re.search(r"exited with status (\d+) at (\S+)", launch_log)
run_manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
check("process_exited_zero", bool(exit_match) and exit_match.group(1) == "0")

# ------------------------------------------------------------------ result directory hash listing
listing_lines, total_bytes = [], 0
for path in sorted(RUN.rglob("*")):
    if path.is_file():
        total_bytes += path.stat().st_size
        listing_lines.append(f"{sha(path)}  {rel(path)}")
listing_text = "\n".join(listing_lines) + "\n"

# ------------------------------------------------------------------ skill-yield report
families = {}
for family in TASK_FAMILIES:
    family_rows = [row for row in rows if row.get("task_family") == family]
    good = [row for row in family_rows if row.get("success") is True]
    statuses = collections.Counter((row.get("skill_candidate") or {}).get("status") for row in good)
    no_skill = sum((row.get("skill_candidate") or {}).get("status") == "declined"
                   and ((row.get("skill_candidate") or {}).get("response") or "").strip() == "NO_SKILL" for row in good)
    reasons = collections.Counter(reason for row in good if (row.get("skill_candidate") or {}).get("status") == "rejected"
                                  for reason in (row["skill_candidate"].get("rejection_reasons") or []))
    accepted = statuses.get("accepted", 0)
    families[family] = {
        "scientific_episodes": len(family_rows),
        "successful_episodes": len(good),
        "scientific_failures": len(family_rows) - len(good),
        "terminations": dict(collections.Counter(row.get("termination_reason") for row in family_rows)),
        "successes_with_accepted_skill": accepted,
        "successes_returning_no_skill": no_skill,
        "other_declines": statuses.get("declined", 0) - no_skill,
        "exact_normalized_duplicate_rejections": reasons.get("exact_normalized_duplicate", 0),
        "other_rejection_reasons": {key: value for key, value in reasons.items() if key != "exact_normalized_duplicate"},
        "accepted_skills_in_pool": sum(skill.task_family == family for skill in pool),
        "skill_yield_per_successful_episode": round(accepted / len(good), 4) if good else None,
        "skill_yield_per_episode": round(accepted / len(family_rows), 4) if family_rows else None,
    }
accepted_counts = {family: value["accepted_skills_in_pool"] for family, value in families.items()}
minimum, maximum = min(accepted_counts.values()), max(accepted_counts.values())
clean_short = {family: count for family, count in accepted_counts.items() if count < CORE_PER_FAMILY}
accum60_need = CORE_PER_FAMILY + ACCUM_EXTRAS_PER_FAMILY["Accum-60"]
accum96_need = CORE_PER_FAMILY + ACCUM_EXTRAS_PER_FAMILY["Accum-96"]
total_no_skill = sum(value["successes_returning_no_skill"] for value in families.values())
all_reasons = collections.Counter()
for value in families.values():
    all_reasons["exact_normalized_duplicate"] += value["exact_normalized_duplicate_rejections"]
    all_reasons.update(value["other_rejection_reasons"])
projection = {family: round(accepted_counts[family] + families[family]["skill_yield_per_episode"] * 10, 1) for family in TASK_FAMILIES}
selection_rejections = collections.Counter()
for row in rows:
    selection_rejections.update(row.get("selection_rejections") or {})
yield_report = {
    "schema_version": 1,
    "report_kind": "rq1-acquisition-skill-yield",
    "run_id": RUN_ID,
    "scientific_episodes": len(rows),
    "successful_episodes": successes,
    "scientific_failures": failures,
    "accepted_skills": len(pool),
    "families": families,
    "global": {
        "proportion_of_successes_with_accepted_skill": round(len(pool) / successes, 4),
        "proportion_of_episodes_with_accepted_skill": round(len(pool) / len(rows), 4),
        "no_skill_count": total_no_skill,
        "other_declines": sum(value["other_declines"] for value in families.values()),
        "exact_normalized_duplicate_rejections": all_reasons.get("exact_normalized_duplicate", 0),
        "other_rejection_categories": {key: value for key, value in all_reasons.items() if key != "exact_normalized_duplicate" and value},
        "minimum_accepted_skills_per_family": minimum,
        "maximum_accepted_skills_per_family": maximum,
        "bottleneck_families": sorted(family for family, count in accepted_counts.items() if count == minimum),
        "termination_reasons": dict(collections.Counter(row.get("termination_reason") for row in rows)),
        "action_selection_attempt_rejections": dict(selection_rejections),
    },
    "library_feasibility_on_accepted_counts": {
        "note": "Necessary conditions only: Clean-24 uses the earliest four skills per family that pass the frozen human quality rubric, which has not been applied; the rubric can only reduce eligibility.",
        "clean_24_currently_feasible": not clean_short,
        "clean_24_families_below_four": clean_short,
        "accum_60_needs_per_family": accum60_need,
        "accum_60_families_below": {family: count for family, count in accepted_counts.items() if count < accum60_need},
        "accum_96_needs_per_family": accum96_need,
        "accum_96_families_below": {family: count for family, count in accepted_counts.items() if count < accum96_need},
    },
    "projection_after_60_extension_units_at_observed_family_yield": {
        "note": "Arithmetic projection (accepted + 10 x observed skills per episode); not a result.",
        "projected_accepted_skills": projection,
    },
}

passed = all(item["pass"] for item in checks.values())
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
closeout = {
    "schema_version": 1,
    "manifest_kind": "rq1-acquisition-closeout",
    "closeout_timestamp": now,
    "closeout_passed": passed,
    "run_id": RUN_ID,
    "scientific_evidence": True,
    "start_time": {"launch": launch_match.group(1) if launch_match else None, "experiment_start_time": run_manifest.get("experiment_start_time"),
                   "first_result": rows[0].get("timestamp")},
    "completion_time": {"last_result": rows[-1].get("timestamp"), "final_checkpoint": checkpoint.get("timestamp"),
                        "process_exit": exit_match.group(2) if exit_match else None},
    "frozen_git_sha": EXPECTED["commit"],
    "queue_hash": EXPECTED["queue"],
    "frozen_task_manifest": {"path": rel(frozen_paths[0]), "sha256": sha(frozen_paths[0]), "manifest_sha256": manifest.manifest_sha256},
    "model": {"tag": environment.inputs.get("model_tag"), "digest": environment.inputs.get("model_digest"),
              "quantization": environment.inputs.get("model_quantization"), "provider_settings": environment.inputs.get("provider_settings")},
    "protocol_sha256": protocol.inputs.get("protocol_sha256"),
    "counts": {"total": len(rows), "successful": successes, "scientific_failures": failures, "infrastructure_failures": infrastructure,
               "retrieval_events": retrieval_events, "final_skill_count": len(pool)},
    "final_pool_hash": pool_hash(pool),
    "skills_per_family": accepted_counts,
    "authoritative_artifacts": {name: {"path": rel(RUN / name), "sha256": sha(RUN / name)}
                                for name in ("results.jsonl", "checkpoint.json", "skill_pool.json", "run_manifest.json", "errors.jsonl", "manifests/acquisition.json")},
    "result_directory": {"path": rel(RUN), "files": len(listing_lines), "bytes": total_bytes,
                         "listing": "result-directory.sha256", "listing_sha256": hashlib.sha256(listing_text.encode("utf-8")).hexdigest()},
    "non_authoritative_files": {
        "checkpoint.backup.json": {
            "path": rel(RUN / "checkpoint.backup.json"),
            "sha256": sha(RUN / "checkpoint.backup.json"),
            "classification": "NON-AUTHORITATIVE stale previous checkpoint generation",
            "explanation": ("ExperimentStore.write_checkpoint copies the current primary checkpoint to checkpoint.backup.json before atomically "
                            "installing each new primary. After the final write the backup holds the generation immediately before it: identical "
                            "except status 'running' (all 180 completed run keys present). load_checkpoint reads it only if checkpoint.json is "
                            "unreadable, and resume reconciles from results.jsonl, the recovery authority. Preserved unchanged; not an incomplete run."),
        }
    },
    "original_freezes": {
        "task_freeze": {"path": rel(frozen_paths[0]), "sha256": sha(frozen_paths[0])},
        "environment_freeze": {"path": rel(ROOT / ENVIRONMENT_FREEZE), "sha256": sha(ROOT / ENVIRONMENT_FREEZE), "input_fingerprint": environment.input_fingerprint},
        "protocol_freeze": {"path": rel(ROOT / PROTOCOL_FREEZE), "sha256": sha(ROOT / PROTOCOL_FREEZE), "input_fingerprint": protocol.input_fingerprint},
    },
    "approvals": {name: {**value, "path": rel(APPROVAL_DIR / name)} for name, value in approvals.items()},
    "evidence_report": {"path": rel(EVIDENCE), "sha256": sha(EVIDENCE)},
    "scientific_log": {"path": str(SCIENTIFIC_LOG), "sha256": sha(SCIENTIFIC_LOG)},
    "backup_mirror": {"path": str(BACKUP), "verified_identical": all(mirror.values()), "files": mirror},
    "git_bundle": {"path": str(BUNDLE), "sha256": sha(BUNDLE)},
    "integrity_checks": checks,
    "yield_report": "yield-report.json",
    "generator": {"script": str(pathlib.Path(__file__).resolve()), "sha256": sha(pathlib.Path(__file__).resolve()), "repository_head": head},
}

OUT.mkdir(parents=True, exist_ok=False)
(OUT / "result-directory.sha256").write_text(listing_text, encoding="utf-8")
(OUT / "yield-report.json").write_text(json.dumps(yield_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
closeout["yield_report_sha256"] = sha(OUT / "yield-report.json")
(OUT / "closeout-manifest.json").write_text(json.dumps(closeout, indent=2, sort_keys=True) + "\n", encoding="utf-8")

markdown = ["# RQ1 acquisition skill yield — rq1-acquisition-gemma4-12b", "",
            f"180 scientific episodes, {successes} successful, {failures} scientific failures, {len(pool)} accepted skills.", "",
            "| Family | Episodes | Successes | Failures | Accepted | NO_SKILL | Dup. rejections | Other rejections | Yield / success | Yield / episode |",
            "|---|---|---|---|---|---|---|---|---|---|"]
for family, value in families.items():
    markdown.append(f"| {family} | {value['scientific_episodes']} | {value['successful_episodes']} | {value['scientific_failures']} | "
                    f"{value['accepted_skills_in_pool']} | {value['successes_returning_no_skill']} | {value['exact_normalized_duplicate_rejections']} | "
                    f"{sum(value['other_rejection_reasons'].values())} | {value['skill_yield_per_successful_episode']} | {value['skill_yield_per_episode']} |")
markdown += ["", f"Clean-24 currently feasible on accepted counts: {'YES' if not clean_short else 'NO'} (below four: {clean_short or 'none'}).",
             f"Bottleneck: {yield_report['global']['bottleneck_families']} with {minimum} accepted skills.", ""]
(OUT / "yield-report.md").write_text("\n".join(markdown), encoding="utf-8")

print("CLOSEOUT_PASSED", passed)
for name, item in checks.items():
    print("CHECK", name, item["pass"], json.dumps(item.get("detail")) if "detail" in item else "")
print("MANIFEST", OUT / "closeout-manifest.json", sha(OUT / "closeout-manifest.json"))
print("YIELD", OUT / "yield-report.json", sha(OUT / "yield-report.json"))
print("FAMILIES", json.dumps({family: {k: families[family][k] for k in ("successful_episodes", "accepted_skills_in_pool", "successes_returning_no_skill", "exact_normalized_duplicate_rejections", "other_rejection_reasons")} for family in TASK_FAMILIES}))
print("GLOBAL", json.dumps(yield_report["global"]))
print("LIBRARY", json.dumps(yield_report["library_feasibility_on_accepted_counts"]))
print("PROJECTION", json.dumps(projection))
sys.exit(0 if passed else 2)

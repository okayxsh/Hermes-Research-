"""Offline candidate packaging. Validates supplied evidence; never supplies approvals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rq1.evaluation.amended_protocol import TOTAL_EPISODE_ACTION_BUDGET
from rq1.skills.library import TASK_FAMILIES
from rq1.university_gpu.models import atomic_write_json, content_hash, load_json, sha256_file


def budget_problems(task: dict) -> list[str]:
    prefix = task.get("prefix_actions", [])
    route = task.get("reference_actions", [])
    if (not isinstance(prefix, (list, tuple)) or not isinstance(route, (list, tuple))
            or any(not isinstance(action, str) or not action.strip() for action in [*prefix, *route])):
        return ["reference route and prefix must be arrays of nonempty action strings"]
    remaining = TOTAL_EPISODE_ACTION_BUDGET - len(prefix) - 1
    problems = []
    if not prefix or not route or route[:len(prefix)] != prefix or len(prefix) >= len(route):
        problems.append("invalid reference route/checkpoint prefix")
    if remaining < 1:
        problems.append("checkpoint and detour exhaust the 50-action total budget")
    if len(route) - len(prefix) > remaining:
        problems.append("frozen reference continuation plus detour exceeds the 50-action total budget")
    if type(task.get("recovery_action_budget")) is not int or task.get("recovery_action_budget") != remaining:
        problems.append("recovery_action_budget differs from 50 - prefix length - 1")
    return problems


def package_candidates(source: Path, evidence_root: Path, destination: Path) -> dict:
    """Consume normalized real task_preparation definitions plus independently saved evidence.

    oracle_evidence paths are relative to evidence_root; the recorded result must include
    validated=true, total_action_limit=50, success=true and total_actions (including prefix
    and detour). This adapter cannot establish that a supplied record is authentic: human
    evidence review and real replay remain separate prerequisites.
    """
    data = load_json(source)
    tasks = data.get("tasks", [])
    root = evidence_root.resolve()
    problems = []
    if len(tasks) != 6 or {t.get("task_family") for t in tasks} != set(TASK_FAMILIES):
        problems.append("requires six tasks, one per family")
    if len({t.get("task_id") for t in tasks}) != len(tasks):
        problems.append("duplicate task IDs")
    reports = []
    for task in tasks:
        issues = budget_problems(task)
        if task.get("split") != "valid_seen" or not str(task.get("task_id", "")).startswith("valid_seen:"):
            issues.append("pilot task must be valid_seen")
        for field in ("checkpoint_digest", "post_detour_digest"):
            digest = task.get(field, "")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                issues.append("missing/invalid " + field)
        if not task.get("checkpoint_id") or not str(task.get("detour_action", "")).startswith("go to "):
            issues.append("checkpoint identity/navigation detour missing")
        references = task.get("source_evidence", [])
        oracle_ref = task.get("oracle_evidence")
        if not references or not oracle_ref:
            issues.append("source task files and separate real oracle evidence are required")
        for ref in [*references, *([oracle_ref] if oracle_ref else [])]:
            path = root / ref["path"]
            if path.is_symlink() or root not in path.resolve().parents or not path.is_file() or sha256_file(path) != ref.get("sha256"):
                issues.append("missing, unsafe or mismatched evidence: " + ref["path"])
                continue
            if ref is oracle_ref:
                oracle = load_json(path)
                count = oracle.get("total_actions")
                if (oracle.get("success") is not True or oracle.get("validated") is not True
                        or oracle.get("total_action_limit") != TOTAL_EPISODE_ACTION_BUDGET
                        or not isinstance(count, int) or isinstance(count, bool)
                        or count != len(task.get("reference_actions", [])) + 1 or count > TOTAL_EPISODE_ACTION_BUDGET
                        or oracle.get("task_id") != task.get("task_id")
                        or oracle.get("checkpoint_digest") != task.get("checkpoint_digest")
                        or oracle.get("post_detour_digest") != task.get("post_detour_digest")):
                    issues.append("real oracle record does not prove the same task/states within the scientific budget")
        reports.append({"task_id": task.get("task_id"), "status": "BLOCKED" if issues else "PASS_OFFLINE_CONSISTENCY",
                        "recovery_action_budget": 49 - len(task.get("prefix_actions", [])), "problems": issues})
    report = {"schema": "rq1-pilot-candidate-evidence-v1", "status": "BLOCKED" if problems or any(r["problems"] for r in reports) else "CANDIDATE_REVIEW_REQUIRED",
              "approved": False, "source_sha256": sha256_file(source), "problems": problems, "tasks": reports,
              "candidate_definitions": tasks, "model_calls": 0, "live_replay_performed": False}
    report["content_sha256"] = content_hash(report)
    atomic_write_json(destination, report, overwrite=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = package_candidates(args.failures, args.evidence_root, args.output)
    print(json.dumps({k: report[k] for k in ("status", "problems", "tasks")}, indent=2))
    raise SystemExit(1 if report["status"] == "BLOCKED" else 0)


if __name__ == "__main__":
    main()

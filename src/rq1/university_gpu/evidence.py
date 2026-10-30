"""Strict evidence checks for the new engineering profile; historical records are untouched."""
from __future__ import annotations
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from rq1.university_gpu.models import sha256_file

REQUIRED_LOGS = ("result/retrieval.jsonl", "result/failure.json", "result/recovery_steps.jsonl", "episode/episode-events.jsonl")


def inventory_attempt(attempt_dir: Path, result: Mapping[str, Any]) -> dict:
    """Inventory closed executor logs; paths remain portable after export."""
    files = []
    for name in REQUIRED_LOGS:
        path = attempt_dir / name
        if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents if p != attempt_dir.parent):
            raise ValueError("missing or unsafe mandatory trace: " + name)
        files.append({"path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"schema": "rq1-feasibility-attempt-evidence-v1", "unit_key": result["unit_key"],
            "attempt_id": result["attempt_id"], "task_id": result["task_id"], "condition": result["condition"],
            "files": files}


def validate_attempt_evidence(attempt_dir: Path, result: Mapping[str, Any], task: Mapping[str, Any], run_id: str) -> list[str]:
    try:
        evidence = result.get("raw_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("schema") != "rq1-feasibility-attempt-evidence-v1":
            return ["missing required attempt evidence inventory"]
        for key in ("unit_key", "attempt_id", "task_id", "condition"):
            if evidence.get(key) != result.get(key):
                return ["attempt evidence identity mismatch: " + key]
        if result.get("log_paths") != list(REQUIRED_LOGS):
            return ["result trace references differ from mandatory bounded inventory"]
        rows = evidence.get("files")
        if not isinstance(rows, list) or len(rows) != len(REQUIRED_LOGS) or {r.get("path") for r in rows} != set(REQUIRED_LOGS):
            return ["mandatory raw trace inventory differs"]
        for row in rows:
            name = row["path"]
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts or "\\" in name:
                return ["unsafe trace path"]
            path = attempt_dir / name
            if (path.is_symlink() or any(p.is_symlink() for p in path.parents)
                    or attempt_dir.resolve() not in path.resolve().parents or not path.is_file()
                    or path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get("sha256")):
                return ["missing, mutated or unsafe mandatory trace: " + name]
        def jsonlines(name):
            return [json.loads(x) for x in (attempt_dir / name).read_text(encoding="utf-8").splitlines() if x.strip()]
        retrieval = jsonlines("result/retrieval.jsonl")
        if len(retrieval) != 1:
            return ["expected exactly one retrieval boundary log"]
        event = retrieval[0]
        for key in ("attempt_id", "task_id", "condition"):
            if event.get(key) != result.get(key):
                return ["retrieval trace identity mismatch: " + key]
        if event.get("episode_id") != result["unit_key"] or event.get("run_id") != run_id:
            return ["retrieval run/episode identity mismatch"]
        top = event.get("top", [])
        if ([r["skill_id"] for r in top] != result["retrieved_skill_ids"]
                or [r["score"] for r in top] != result["retrieval_similarity_scores"]
                or event.get("retrieved") is not (result["condition"] == "D18")):
            return ["retrieval trace disagrees with result"]
        steps = jsonlines("result/recovery_steps.jsonl")
        actions = [s for s in steps if s.get("action")]
        if (len(steps) != result["steps"] or len(actions) != result["actions"]
                or sum(s.get("action_valid") is False for s in steps) != result["invalid_actions"]
                or [s["action"] for s in actions] != result["measurements"]["recovery_actions"]):
            return ["recovery trace counts/actions disagree with result"]
        if bool(any(s.get("done") is True and s.get("success") is True for s in steps)) != result["recovery_success"]:
            return ["recovery terminal trace disagrees with result"]
        failure = json.loads((attempt_dir / "result/failure.json").read_text(encoding="utf-8"))
        if failure.get("failure", {}).get("post_state_digest") != result["measurements"]["perturbation_digest"]:
            return ["failure trace differs from perturbation"]
        events = jsonlines("episode/episode-events.jsonl")
        if not events or sum(e.get("event") == "recovery_memory_injected" for e in events) != 1:
            return ["episode trace lacks exactly one recovery boundary"]
        starts = [e.get("payload", {}) for e in events if e.get("event") == "tool_result" and e.get("payload", {}).get("tool") == "alfworld_start"]
        if len(starts) != 1:
            return ["episode trace requires one environment start"]
        request = starts[0].get("request", {})
        if (request.get("task_id") != result["task_id"] or request.get("split") != "valid_seen"
                or request.get("seed") != result["seed"] or request.get("action_limit") != 200):
            return ["episode start identity/budget mismatch"]
        dispatched = [e.get("payload", {}).get("request", {}).get("action") for e in events
                      if e.get("event") == "tool_result" and e.get("payload", {}).get("tool") == "alfworld_step"]
        expected_actions = list(task["prefix_actions"]) + [task["detour_action"]] + result["measurements"]["recovery_actions"]
        if dispatched != expected_actions or len(dispatched) > 200:
            return ["episode dispatched actions differ from prefix/detour/recovery accounting"]
        start_response = starts[0].get("response")
        if not isinstance(start_response, dict) or not isinstance(start_response.get("episode_id"), str) or not start_response["episode_id"]:
            return ["environment start response lacks episode identity"]
        step_events = [e["payload"] for e in events if e.get("event") == "tool_result" and e.get("payload", {}).get("tool") == "alfworld_step"]
        prefix_count = len(task["prefix_actions"]) + 1
        for index, payload in enumerate(step_events):
            response = payload.get("response")
            if not isinstance(response, dict) or response.get("episode_id") != start_response["episode_id"]:
                return ["environment step response lacks matching episode identity"]
            if index < prefix_count:
                if response.get("action_valid") is not True or response.get("done") is not False or response.get("success") is True:
                    return ["checkpoint/detour environment response is not valid and live"]
            else:
                record = actions[index - prefix_count]
                for field in ("done", "success", "action_valid", "observation"):
                    if response.get(field) != record.get(field):
                        return ["environment recovery response disagrees with recorded step: " + field]
        return []
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        return ["invalid attempt evidence: " + type(exc).__name__]

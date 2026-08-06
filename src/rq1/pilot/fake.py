"""Deterministic fake runtime for all Phase 6 pilot tests."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from rq1.bridge.environment import FakeALFWorldAdapter
from rq1.bridge.episode_manager import EpisodeManager
from rq1.bridge.models import EpisodeStartRequest
from rq1.pilot.models import EvidenceLevel, PilotStatus, PilotTestSpec, RuntimeExecution
from rq1.logging.run_registry import Run, RunRegistry
from rq1.recovery.checkpoints import create_manifest
from rq1.recovery.context import build_recovery_context
from rq1.recovery.fake import FakeRecoveryEnvironment
from rq1.recovery.models import CheckpointPolicy, to_dict
from rq1.recovery.perturbations import fake_target_relocation
from rq1.recovery.replay import replay_checkpoint
from rq1.recovery.solvability import validate_fake_solvability
from rq1.recovery.state_digest import observable_digest


FAKE_TASKS = (
    ("fake-pick-and-place-001", "pick_and_place"),
    ("fake-pick-two-001", "pick_two_and_place"),
    ("fake-look-at-001", "look_at_object"),
    ("fake-clean-and-place-001", "clean_and_place"),
    ("fake-heat-and-place-001", "heat_and_place"),
    ("fake-cool-and-place-001", "cool_and_place"),
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class FakePilotRuntime:
    simulated = True

    def __init__(self, root: Path) -> None:
        self.root = root
        self.results: dict[str, dict[str, Any]] = {}

    def execute(self, spec: PilotTestSpec, *, run_id: str, attempt_id: str, output_dir: Path) -> RuntimeExecution:
        index = int(spec.test_id.split("_")[1])
        details = self._details(index, run_id, attempt_id, output_dir)
        details.update({"simulated": True, "evidence_scope": "runner_contract_only"})
        self.results[spec.test_id] = details
        level = EvidenceLevel.STATIC if index == 0 else EvidenceLevel.MOCK
        return RuntimeExecution(PilotStatus.PASSED, level, details)

    def _details(self, index: int, run_id: str, attempt_id: str, output_dir: Path) -> dict[str, Any]:
        if index == 0:
            schemas = list((self.root / "data" / "schemas").glob("*.json")) if (self.root / "data" / "schemas").exists() else []
            valid = 0
            for path in schemas:
                json.loads(path.read_text(encoding="utf-8")); valid += 1
            suite = {"executed": False, "reason": "fixture root has no tests directory"}
            if (self.root / "tests").is_dir() and os.environ.get("RQ1_PILOT_SELF_TEST_CHILD") != "1":
                environment = os.environ.copy()
                environment["RQ1_PILOT_SELF_TEST_CHILD"] = "1"
                environment["PYTHONPATH"] = str(self.root / "src")
                completed = subprocess.run(
                    (sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"),
                    cwd=self.root, env=environment, capture_output=True, text=True,
                    timeout=170, check=False,
                )
                suite = {
                    "executed": True, "returncode": completed.returncode,
                    "stdout_sha256": _sha(completed.stdout), "stderr_sha256": _sha(completed.stderr),
                }
                if completed.returncode != 0:
                    raise RuntimeError("non-external repository self-test failed")
            return {"imports": True, "configuration": True, "schemas_validated": valid, "writable_output": True, "self_test_child_guard": True, "non_external_suite": suite}
        if index == 1:
            return {"os": "deterministic-fake", "cpu": "fake-cpu", "ram_gb": 24, "gpu": "fake-gpu", "repository_commit": None}
        if index == 2:
            return {"installation_ready": True, "components": ["python", "ollama", "hermes", "alfworld", "model"], "installed": False}
        if index == 3:
            return {"model": "hermes3:8b", "digest": _sha("fake-hermes3:8b"), "responses": ["ready", "ready", "ready"], "latency_ms": [10, 10, 10]}
        if index == 4:
            return {"tool": "echo_test", "expected": {"message": "phase6"}, "observed": {"message": "phase6"}, "repetitions": 3, "malformed": 0}
        if index == 5:
            return {"profile": "rq1-pilot", "repository": "$REPO", "bundled_skills": 0, "memory": 0, "curator": False, "fresh_sessions": 1}
        if index == 6:
            return {"plugin": "alfworld-experiment", "tools": ["alfworld_start", "alfworld_step", "alfworld_status", "alfworld_abort", "alfworld_reset"], "hooks": ["pre_tool_call", "post_tool_call"]}
        if index == 7:
            return {"tool": "echo_test", "dispatched": True, "consumed": True, "run_id": run_id, "attempt_id": attempt_id, "logs_reconcile": True}
        if index == 8:
            return {"profiles": ["rq1-test-pilot-a", "rq1-test-pilot-b"], "skills_isolated": True, "sessions_isolated": True, "memory_isolated": True, "repository_shared": True}
        if index == 9:
            return {"events": [
                {"event": "skill_loaded", "skill_id": "pilot-relevant", "relevance": "relevant", "simulated": True},
                {"event": "skill_loaded", "skill_id": "pilot-distractor", "relevance": "irrelevant", "simulated": True},
                {"event": "skill_loaded", "skill_id": "pilot-relevant", "relevance": "relevant", "simulated": True},
            ]}
        if index == 10:
            content = "deterministic temporary pilot skill\n"
            return {"skill": "pilot-persistent", "before_hash": _sha(content), "after_hash": _sha(content), "fresh_session": True}
        if index == 11:
            snapshot_hash = _sha("read-only-pilot-snapshot")
            return {"write_refused": True, "before_hash": snapshot_hash, "after_hash": snapshot_hash, "shadow_skill": False}
        if index == 12:
            manager = EpisodeManager(FakeALFWorldAdapter, output_dir / "bridge")
            families: list[str] = []
            episodes: list[str] = []
            for task_id, expected_family in FAKE_TASKS:
                started = manager.start(EpisodeStartRequest(task_id, "valid_seen", 1, 4))
                families.append(started.task_family); episodes.append(started.episode_id)
                if started.task_family != expected_family:
                    raise RuntimeError(f"fake family mismatch for {task_id}")
                manager.step(started.episode_id, "invalid fake action")
                manager.reset(started.episode_id)
                manager.abort(started.episode_id, "phase6 fake contract")
            return {"task_families": families, "unique_episode_ids": len(set(episodes)) == len(episodes), "correlation_supported": True, "raw_logs": 6}
        if index == 13:
            return {"start_tool": True, "step_tool": True, "observation_returned": True, "logs_reconcile": True}
        if index == 14:
            return {"episode_preserved": True, "steps": [1, 2, 3], "duplicate_starts": 0, "action_limit_enforced": True}
        if index == 15:
            return {"episodes": [{"task_family": family, "success": True, "actions": 3, "invalid_actions": 0} for _, family in FAKE_TASKS]}
        if 16 <= index <= 23:
            return self._recovery_details(index, run_id, attempt_id)
        if index == 24:
            return {"sources": ["hermes", "plugin", "bridge", "recovery", "run_registry", "pilot"], "mismatches": [], "reconciled": True}
        if index == 25:
            return {"fresh_sessions": 3, "conversation_leakage": False, "observation_leakage": False, "personal_memory": False, "intended_skills_only": True}
        if index == 26:
            failures = ["ollama_unavailable", "hermes_unavailable", "bridge_unavailable", "malformed_tool", "timeout", "invalid_replay", "state_mismatch", "perturbation_failure", "unsolvable", "interrupted", "action_limit"]
            return {"injected_failures": failures, "classified": len(failures), "evidence_retained": True, "attempts_merged": False, "live_services_stopped": False}
        if index == 27:
            skills = [{"skill_id": f"pilot-skill-{i:02d}", "source_task": f"fake-train-{i:02d}", "created_after_success": True, "operations": 1} for i in range(6)]
            library = output_dir / "pilot-mini-library"
            for skill in skills:
                skill_dir = library / skill["skill_id"]; skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text(f"# {skill['skill_id']}\n\nDisposable Phase 6 pilot skill.\n", encoding="utf-8")
            (library / "manifest.json").write_text(json.dumps({"schema_version": 1, "skills": skills}, indent=2) + "\n", encoding="utf-8")
            return {"split": "train", "task_count": 6, "skills": skills, "failed_episode_skills": 0, "promoted_to_acquisition": False}
        if index == 28:
            skills = [f"pilot-skill-{i:02d}" for i in range(6)]
            snapshots = {"L0": [], "Lsmall": skills[:3], "Llarge": skills}
            snapshot_root = output_dir / "pilot-mini-snapshots"
            for name, contents in snapshots.items():
                path = snapshot_root / name; path.mkdir(parents=True)
                (path / "manifest.json").write_text(json.dumps({"schema_version": 1, "snapshot": name, "skills": contents, "read_only": True}, indent=2) + "\n", encoding="utf-8")
            return {"snapshots": snapshots, "nested": set(snapshots["Lsmall"]).issubset(snapshots["Llarge"]), "read_only": True, "final_snapshot_created": False}
        if index == 29:
            digest = _sha("identical-fake-recovery-context")
            conditions = [{"snapshot": name, "context_digest": digest, "checkpoint_digest": _sha("cp"), "perturbation_digest": _sha("pert"), "recovery_success": True, "skill_writes": 0} for name in ("L0", "Lsmall", "Llarge")]
            return {"split": "valid_seen", "conditions": conditions, "paired_state_equal": True, "final_evaluation": False}
        if index == 30:
            registry = RunRegistry(output_dir / "worker-queue.sqlite")
            for item in range(4):
                registry.plan(Run(f"pilot-worker-{item}", f"fake-{item}", "valid_seen", "L0", "rq1-test-worker", 1, "planned"))
            claims: list[tuple[str, str | None]] = []
            lock = threading.Lock()
            def claim(worker: str) -> None:
                while True:
                    value = registry.claim_next(worker)
                    if value is None: return
                    with lock: claims.append((value.run_id, value.attempt_id))
            threads = [threading.Thread(target=claim, args=(f"worker-{index}",)) for index in range(2)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            unique = len({run for run, _ in claims})
            return {"workers": 2, "claims": [run for run, _ in claims], "duplicate_run_ids": len(claims) - unique, "separate_attempts": len({attempt for _, attempt in claims}) == len(claims), "worker_failure_isolated": True}
        if index == 31:
            return {"interrupted_attempt": "attempt-old", "resumed_attempt": attempt_id, "completed_tests_skipped": True, "partial_preserved": True, "successful_run_duplicates": 0}
        if index == 32:
            return {"model_call_latency_ms": 10, "tool_call_latency_ms": 2, "replay_ms": 4, "perturbation_ms": 1, "recovery_ms": 20, "ram_mb": 64, "vram_mb": 0, "projected_acquisition_minutes": 12, "projected_evaluation_minutes": 24}
        if index == 33:
            return {"labels": {"relevant": 2, "irrelevant": 1, "no_retrieval": 1, "ambiguous": 1}, "audit_sample_size": 5, "used_valid_unseen": False}
        if index == 34:
            return {"candidate": "hermes3:8b", "decision": "simulated_accept", "fallback_tested": False, "deepseek_tested": False, "final_model_frozen": False}
        if index == 35:
            return {"checkpoint_source": "deterministic_valid_prefix", "checkpoint_policy": "trajectory_fraction_provisional", "perturbation_type": "target_object_relocation_provisional", "solvability_method": "known_route_provisional", "recovery_context": "phase5_versioned_context", "action_limit": 12, "timeout_seconds": 900, "exclusion_policy": "invalid_checkpoint_or_perturbation_excluded_separately", "approval_state": "unapproved"}
        return {"report_generated": True, "real_evidence_promoted": False, "phase7_required": True}

    def _recovery_details(self, index: int, run_id: str, attempt_id: str) -> dict[str, Any]:
        env = FakeRecoveryEnvironment(task_id="phase6-recovery")
        trajectory = env.reference_trajectory(); env.reset(); checkpoint_state = env.step("go to countertop 1")
        checkpoint = create_manifest(trajectory, CheckpointPolicy("prefix_length", 1), checkpoint_state, "phase6-cp")
        replay_state, replay = replay_checkpoint(env, checkpoint)
        if replay_state is None: raise RuntimeError("deterministic replay unexpectedly failed")
        perturbed, perturbation = fake_target_relocation(env, checkpoint.checkpoint_id, "phase6-pert")
        solvability = validate_fake_solvability(env)
        context = build_recovery_context(perturbed, checkpoint, perturbation, run_id=run_id, attempt_id=attempt_id, profile="rq1-pilot", snapshot=None, action_budget=12)
        common = {"checkpoint": to_dict(checkpoint), "replay": to_dict(replay), "perturbation": to_dict(perturbation), "solvability": to_dict(solvability), "context": to_dict(context), "context_digest": observable_digest(perturbed)}
        if index == 16: return {"checkpoint_id": checkpoint.checkpoint_id, "prefix_length": checkpoint.prefix_length, "manifest_valid": True, "meaningful_remaining_work": True}
        if index == 17: return {"replay_valid": replay.valid, "expected_digest": replay.expected_observable_digest, "actual_digest": replay.actual_observable_digest, "repetitions": 3}
        if index == 18: return {"perturbation": to_dict(perturbation), "verified": perturbation.solvable is True}
        if index == 19: return {"solvability": to_dict(solvability), "object_presence_only": False}
        if index == 20: return {"context": to_dict(context), "context_digest": common["context_digest"]}
        if index == 21: return {"condition": "no_library", "recovery_success": True, "post_failure_actions": 3, "skills_loaded": [], "checkpoint_digest": replay.actual_observable_digest, "context_digest": common["context_digest"]}
        if index == 22: return {"condition": "relevant_skill", "recovery_success": True, "post_failure_actions": 2, "skills_loaded": ["pilot-relevant"], "checkpoint_digest": replay.actual_observable_digest, "context_digest": common["context_digest"]}
        return {"condition": "relevant_plus_distractors", "recovery_success": True, "post_failure_actions": 3, "skills_loaded": ["pilot-distractor", "pilot-relevant"], "checkpoint_digest": replay.actual_observable_digest, "context_digest": common["context_digest"]}

"""Explicit, non-scientific RunPod pre-launch evidence pilot.

This is intentionally not part of the final acquisition/evaluation commands.
It exercises one train acquisition and paired valid_seen recovery conditions,
writing only disposable artifacts under ``artifacts/prelaunch``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from rq1.acquisition.real_executor import run_acquisition_episode
from rq1.evaluation.recovery_executor import RecoveryEpisodeResult, RecoveryEpisodeSpec, run_recovery_episode
from rq1.hermes.episode_driver import (
    ACTION_SELECTION_PROTOCOL,
    EXPERIMENT_MODEL,
    OUTPUT_TOKEN_CAP,
    INFERENCE_SEED,
    MAX_SELECTION_ATTEMPTS,
    RealEpisodeDriver,
)
from rq1.pilot.real_runtime.harnesses import RealAcquisitionHarness, RealRecoveryHarness
from rq1.recovery.reference_route import derive_handcoded_reference, midpoint_candidates
from rq1.retrieval import SentenceBERTEmbedder, build_retrieval_boundary
from rq1.utils.time import utc_now


DEFAULT_TRAIN_TASK = (
    "train:look_at_obj_in_light-Bowl-None-DeskLamp-304/"
    "trial_T20190907_232204_574574"
)
DEFAULT_VALID_SEEN_TASK = (
    "valid_seen:look_at_obj_in_light-AlarmClock-None-DeskLamp-323/"
    "trial_T20190909_044715_250790"
)
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
SBERT_CACHE_DIR = Path("/workspace/persistent/hf-cache/hub")


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pilot_skills() -> list[tuple[str, str]]:
    """Tiny non-final library used solely to exercise the top-3 boundary."""
    return [
        ("pilot-navigation", "TITLE: Navigation\nBODY: Inspect the current room and move to the named receptacle."),
        ("pilot-object-handling", "TITLE: Object handling\nBODY: Locate the required object before taking it."),
        ("pilot-lighting", "TITLE: Lighting\nBODY: Use the required light source after the object is in place."),
    ]


def _library_hash(skills: list[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(skills, sort_keys=True).encode("utf-8")).hexdigest()


def _recovery_summary(result: RecoveryEpisodeResult) -> dict[str, object]:
    return {
        "success": result.outcome.success,
        "retrieval_count": result.retrieval_count,
        "no_retrieval": result.no_retrieval,
        "task_goal": result.failure_context["task_instruction"],
        # Dispatched model actions only; exhausted selections are invalid steps.
        "post_failure_actions": result.outcome.actions,
        "invalid_action_selections": result.outcome.invalid_actions,
        "log_paths": list(result.log_paths),
    }


def _select_checkpoint(
    driver: RealEpisodeDriver,
    *,
    output: Path,
    run_id: str,
    task_id: str,
    seed: int,
    reference_actions: tuple[str, ...],
) -> tuple[str, ...]:
    """Freeze the first midpoint-near candidate proven by the real oracle."""
    from rq1.recovery.reference_route import ReferenceRoute

    route = ReferenceRoute(task_id, "valid_seen", reference_actions)
    failures: list[dict[str, object]] = []
    for index, prefix, _continuation in midpoint_candidates(route):
        harness = RealRecoveryHarness(
            driver,
            output_dir=output / "checkpoint-probes" / f"candidate-{index}",
            run_id=run_id + "-checkpoint-probe",
            attempt_id=f"checkpoint-{index}",
            reference_actions=reference_actions,
        )
        try:
            harness.start_and_replay(task_id, "valid_seen", seed, prefix)
            harness.apply_controlled_action_perturbation(f"pilot-checkpoint-{index}")
            return prefix
        except Exception as exc:
            failures.append({"index": index, "error_type": type(exc).__name__, "error": str(exc)})
        finally:
            harness.close()
    _write(output / "checkpoint-selection-failures.json", {"failures": failures})
    raise RuntimeError("No midpoint-near checkpoint admitted a proven reversible action perturbation")


def run_prelaunch_pilot(
    root: Path,
    *,
    data_dir: Path,
    train_task_id: str = DEFAULT_TRAIN_TASK,
    valid_seen_task_id: str = DEFAULT_VALID_SEEN_TASK,
    seed: int = 1,
) -> dict[str, object]:
    """Run the requested one-acquisition/two-condition non-scientific pilot."""
    root = root.resolve()
    run_id = "prelaunch-" + uuid4().hex
    output = root / "artifacts" / "prelaunch" / run_id
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "schema_version": 1,
        "mode": "non_scientific_prelaunch",
        "run_id": run_id,
        "generated_at": utc_now(),
        "train_task_id": train_task_id,
        "valid_seen_task_id": valid_seen_task_id,
        "seed": seed,
        "model_inference": {
            "model": EXPERIMENT_MODEL,
            "output_token_cap": OUTPUT_TOKEN_CAP,
            "temperature": 0,
            "seed": INFERENCE_SEED,
            "action_selection_protocol": ACTION_SELECTION_PROTOCOL,
            "max_selection_attempts": MAX_SELECTION_ATTEMPTS,
        },
    }
    _write(output / "run-manifest.json", report)
    route = derive_handcoded_reference(data_dir, valid_seen_task_id, "valid_seen")
    report["reference_action_count"] = len(route.actions)
    report["reference_actions"] = list(route.actions)

    with RealEpisodeDriver(root, data_dir=data_dir) as driver:
        acquisition = RealAcquisitionHarness(
            driver,
            output_dir=output / "acquisition" / "episode",
            run_id=run_id,
            attempt_id="acquisition-smoke",
        )
        acquisition_result = run_acquisition_episode(
            acquisition,
            task_id=train_task_id,
            task_family="look_at_obj_in_light",
            attempt_id="acquisition-smoke",
            log_dir=output / "acquisition" / "result",
            seed=seed,
            action_limit=20,
        )
        report["acquisition"] = {
            "success": acquisition_result.outcome.success,
            "retrieval_count": acquisition_result.retrieval_count,
            "candidate_created": acquisition_result.skill_candidate is not None,
            "log_paths": list(acquisition_result.log_paths),
        }

        prefix = _select_checkpoint(
            driver,
            output=output,
            run_id=run_id,
            task_id=valid_seen_task_id,
            seed=seed,
            reference_actions=route.actions,
        )
        report["checkpoint"] = {"prefix_actions": list(prefix), "prefix_length": len(prefix)}

        skills = _pilot_skills()
        library_hash = _library_hash(skills)
        embedder = SentenceBERTEmbedder(
            SBERT_MODEL,
            cache_folder=SBERT_CACHE_DIR,
            revision=SBERT_REVISION,
            local_files_only=True,
        )
        memory_boundary = build_retrieval_boundary(
            skills,
            embedder,
            embedding_model=SBERT_MODEL,
            top_k=3,
            embedding_model_revision=SBERT_REVISION,
        )
        memory_harness = RealRecoveryHarness(
            driver,
            output_dir=output / "recovery-memory" / "episode",
            run_id=run_id,
            attempt_id="recovery-memory",
            reference_actions=route.actions,
        )
        memory_spec = RecoveryEpisodeSpec(
            run_id=run_id,
            attempt_id="recovery-memory",
            task_id=valid_seen_task_id,
            task_family="look_at_obj_in_light",
            split="valid_seen",
            seed=seed,
            condition="Pilot-Memory",
            library_name="pilot-nonfinal-3",
            library_size=len(skills),
            library_hash=library_hash,
            checkpoint_id="pilot-checkpoint-" + str(len(prefix)),
            prefix_actions=prefix,
            action_budget=20,
        )
        try:
            memory = run_recovery_episode(
                memory_harness, memory_spec, memory_boundary, log_dir=output / "recovery-memory" / "result"
            )
        finally:
            memory_harness.close()
        report["memory_recovery"] = _recovery_summary(memory)

        nolib_boundary = build_retrieval_boundary(
            [],
            embedder,
            embedding_model=SBERT_MODEL,
            top_k=3,
            embedding_model_revision=SBERT_REVISION,
        )
        nolib_harness = RealRecoveryHarness(
            driver,
            output_dir=output / "recovery-nolib" / "episode",
            run_id=run_id,
            attempt_id="recovery-nolib",
            reference_actions=route.actions,
        )
        nolib_spec = RecoveryEpisodeSpec(
            run_id=run_id,
            attempt_id="recovery-nolib",
            task_id=valid_seen_task_id,
            task_family="look_at_obj_in_light",
            split="valid_seen",
            seed=seed,
            condition="NoLib",
            library_name="NoLib",
            library_size=0,
            library_hash=None,
            checkpoint_id="pilot-checkpoint-" + str(len(prefix)),
            prefix_actions=prefix,
            action_budget=20,
        )
        try:
            nolib = run_recovery_episode(
                nolib_harness, nolib_spec, nolib_boundary, log_dir=output / "recovery-nolib" / "result"
            )
        finally:
            nolib_harness.close()
        report["nolib_recovery"] = _recovery_summary(nolib)
    _write(output / "pilot-report.json", report)
    return {**report, "output_directory": str(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the non-scientific real RQ1 pre-launch pilot")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-task-id", default=DEFAULT_TRAIN_TASK)
    parser.add_argument("--valid-seen-task-id", default=DEFAULT_VALID_SEEN_TASK)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(run_prelaunch_pilot(Path.cwd(), data_dir=args.data_dir, train_task_id=args.train_task_id, valid_seen_task_id=args.valid_seen_task_id, seed=args.seed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

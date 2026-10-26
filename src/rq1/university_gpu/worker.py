"""Long-lived, one-worker episode process used by the central coordinator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from rq1.evaluation.amended_executor import AmendedEvaluationExecutor
from rq1.evaluation.amended_libraries import AmendedLibrary
from rq1.evaluation.amended_matrix import EvaluationTask
from rq1.experiment.models import ExperimentUnit, RunExecutionContext
from rq1.hermes.episode_driver import RealEpisodeDriver
from rq1.university_gpu.interventions import LocalGemmaTokenizer, TokenizerIdentity, intervene
from rq1.university_gpu.models import load_json, sha256_file
from rq1.university_gpu.ollama import probe_endpoint
from rq1.university_gpu.manifest import MODEL_DIGEST, MODEL_TAG, SBERT_MODEL, SBERT_REVISION, SBERT_SNAPSHOT_SHA256


class WorkerStartupError(RuntimeError):
    pass


class EpisodeWorkerRuntime:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.worker_id = str(config["worker_id"])
        if config.get("model_tag") != MODEL_TAG or config.get("model_digest") != MODEL_DIGEST:
            raise WorkerStartupError("worker configuration does not match the frozen Gemma 4 12B identity")
        if config.get("sbert_model") != SBERT_MODEL or config.get("sbert_revision") != SBERT_REVISION:
            raise WorkerStartupError("worker configuration does not match the frozen Sentence-BERT identity")
        if config.get("sbert_snapshot_sha256") != SBERT_SNAPSHOT_SHA256:
            raise WorkerStartupError("worker configuration does not match the frozen Sentence-BERT snapshot")
        probe = probe_endpoint(str(config["ollama_endpoint"]))
        if probe.get("status") != "PASS":
            raise WorkerStartupError(f"worker Ollama preflight failed: {probe}")
        tokenizer_path = Path(str(config["tokenizer_path"]))
        tokenizer_sha = str(config["tokenizer_sha256"])
        if sha256_file(tokenizer_path) != tokenizer_sha:
            raise WorkerStartupError("worker tokenizer hash differs from the frozen manifest")
        provenance = config.get("provenance", {})
        if (config.get("tokenizer_revision") != provenance.get("tokenizer_revision")
                or tokenizer_sha != provenance.get("tokenizer_sha256")):
            raise WorkerStartupError("worker tokenizer revision/hash differs from frozen provenance")
        self.tokenizer = LocalGemmaTokenizer(TokenizerIdentity(tokenizer_path, tokenizer_sha, str(config["tokenizer_revision"])))
        self.tasks = {
            str(item["task_id"]): EvaluationTask(
                task_id=str(item["task_id"]), task_family=str(item["task_family"]), order_index=int(item.get("order_index", 0)),
                checkpoint_id=str(item["checkpoint_id"]), checkpoint_digest=str(item["checkpoint_digest"]),
                post_detour_digest=str(item["post_detour_digest"]), detour_action=str(item["detour_action"]),
                expected_next_action=str(item["expected_next_action"]), prefix_actions=tuple(item["prefix_actions"]),
                reference_actions=tuple(item["reference_actions"]), recovery_action_budget=int(item["recovery_action_budget"]),
            ) for item in config["tasks"]
        }
        raw_libraries = config["libraries"]
        self.libraries = {
            str(name): AmendedLibrary(str(name), tuple(dict(skill) for skill in payload["skills"]))
            for name, payload in raw_libraries.items()
        }
        self.skill_texts = {str(key): str(value) for key, value in config["skill_texts"].items()}
        self.selected_skill_ids = {str(task): dict(value) for task, value in config["selected_skill_ids"].items()}
        self.filler_unit = str(config["filler_unit"])
        self.tau = float(config["gate_threshold"])
        try:
            from rq1.retrieval.embedder import SentenceBERTEmbedder

            embedder = SentenceBERTEmbedder(SBERT_MODEL, cache_folder=Path(str(config["sbert_cache"])),
                                             revision=SBERT_REVISION, local_files_only=True)
            dims = len(embedder.encode(["university GPU worker readiness check"])[0])
            if dims != 768:
                raise WorkerStartupError(f"Sentence-BERT dimension mismatch: {dims}")
        except Exception as exc:
            if isinstance(exc, WorkerStartupError):
                raise
            raise WorkerStartupError(f"local frozen Sentence-BERT could not be loaded: {type(exc).__name__}") from exc
        self.embedder = embedder
        root = Path(str(config["repository_root"])).resolve()
        self.driver = RealEpisodeDriver(root, data_dir=Path(str(config["alfworld_data_dir"])),
                                        model_name=MODEL_TAG, ollama_url=str(config["ollama_endpoint"]),
                                        bridge_log_root=Path(str(config["worker_output_dir"])) / "bridge")
        self.driver.__enter__()

        def policy(unit: ExperimentUnit, task: EvaluationTask, retrieval_outcome: Any, _failure_context: Mapping[str, Any]):
            top = [{"skill_id": candidate.skill_id, "score": float(candidate.score)} for candidate in retrieval_outcome.result.top]
            return intervene(arm=unit.condition, task_id=task.task_id, raw_memory=retrieval_outcome.recovery_memory.to_dict(),
                             retrieval_top=top, skill_texts=self.skill_texts, selected_skill_ids=self.selected_skill_ids,
                             tokenizer=self.tokenizer, gate_threshold=self.tau, filler_unit=self.filler_unit)

        self.executor = AmendedEvaluationExecutor(
            self.driver,
            tasks=self.tasks,
            libraries=self.libraries,
            embedder=self.embedder,
            scientific=True,
            split=str(config["split"]),
            provenance=dict(config["provenance"]),
            memory_intervention=policy,
        )

    def execute(self, assignment: Mapping[str, Any]) -> dict[str, Any]:
        raw_unit = assignment["unit"]
        unit = ExperimentUnit(
            phase=str(self.config["stage"]), task_id=str(raw_unit["task_id"]), task_index=int(raw_unit["global_order"]),
            condition=str(raw_unit["condition"]), seed=int(raw_unit["seed"]), identity=dict(raw_unit["identity"]),
            payload=dict(raw_unit), library_name=str(raw_unit["condition"]),
            library_size=self.libraries[str(raw_unit["condition"])].size,
            library_hash=self.libraries[str(raw_unit["condition"])].content_sha256,
        )
        attempt_dir = Path(str(assignment["attempt_dir"]))
        context = RunExecutionContext(str(self.config["run_id"]), str(assignment["attempt_id"]), 1, attempt_dir, lambda: False)
        result = self.executor(unit, context)
        return {
            "unit_key": str(raw_unit["unit_key"]),
            "attempt_id": str(assignment["attempt_id"]),
            "status": "completed",
            "task_id": unit.task_id,
            "task_family": self.tasks[unit.task_id].task_family,
            "condition": unit.condition,
            "seed": unit.seed,
            "success": result.success,
            "recovery_success": result.recovery_success,
            "retrieved_skill_ids": list(result.retrieved_skill_ids),
            "retrieval_similarity_scores": list(result.retrieval_similarity_scores),
            "actions": result.actions,
            "steps": result.steps,
            "invalid_actions": result.invalid_actions,
            "latency_ms": result.latency_ms,
            "runtime_seconds": result.runtime_seconds,
            "log_paths": list(result.log_paths),
            "measurements": dict(result.measurements),
        }

    def close(self) -> None:
        self.driver.__exit__(None, None, None)


def serve(config_path: Path, *, input_stream: Any = sys.stdin, output_stream: Any = sys.stdout) -> int:
    runtime: EpisodeWorkerRuntime | None = None
    try:
        config = load_json(config_path)
        runtime = EpisodeWorkerRuntime(config)
        output_stream.write(json.dumps({"ready": True, "worker_id": runtime.worker_id, "gpu_uuid": config["gpu_uuid"]}) + "\n")
        output_stream.flush()
        for line in input_stream:
            if not line.strip():
                continue
            message = json.loads(line)
            if message.get("command") == "shutdown":
                break
            if message.get("command") != "execute":
                output_stream.write(json.dumps({"ok": False, "error": "unsupported worker command"}) + "\n")
                output_stream.flush()
                continue
            try:
                result = runtime.execute(message["assignment"])
                output_stream.write(json.dumps({"ok": True, "result": result}, ensure_ascii=False) + "\n")
            except BaseException as exc:
                output_stream.write(json.dumps({
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failure_details": dict(getattr(exc, "details", {}) or {}),
                    "safe_to_continue": getattr(exc, "safe_to_continue", None),
                    "mutation_state_known": getattr(exc, "mutation_state_known", None),
                }, ensure_ascii=False) + "\n")
            output_stream.flush()
        return 0
    except BaseException as exc:
        output_stream.write(json.dumps({"ready": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False) + "\n")
        output_stream.flush()
        return 1
    finally:
        if runtime is not None:
            runtime.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    return serve(args.config)


if __name__ == "__main__":
    raise SystemExit(main())

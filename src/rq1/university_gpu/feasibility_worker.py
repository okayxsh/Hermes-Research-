"""Worker entry point for the isolated non-scientific 200-action profile."""
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
from rq1.retrieval.embedder import SentenceBERTEmbedder
from rq1.university_gpu.feasibility import (
    MODEL_DIGEST,
    MODEL_QUANTIZATION,
    MODEL_SETTINGS,
    MODEL_TAG,
    POLICY_VERSION,
    PROFILE_ID,
    RESULT_SCHEMA,
    SBERT_MODEL,
    SBERT_REVISION,
    SBERT_SNAPSHOT_SHA256,
    TOTAL_ACTION_BUDGET,
    check_selected_task_sources,
)
from rq1.university_gpu.interventions import LocalGemmaTokenizer, TokenizerIdentity
from rq1.university_gpu.manifest import MODEL_DIGEST as HISTORICAL_WORKER_DIGEST
from rq1.university_gpu.models import load_json, sha256_file
from rq1.university_gpu.ollama import probe_endpoint


class FeasibilityWorkerStartupError(RuntimeError):
    pass


class FeasibilityEpisodeWorkerRuntime:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.worker_id = str(config["worker_id"])
        if config.get("profile_id") != PROFILE_ID or config.get("scientific_evidence") is not False:
            raise FeasibilityWorkerStartupError("worker must use the non-scientific feasibility profile")
        if config.get("model_tag") != MODEL_TAG or config.get("model_digest") != MODEL_DIGEST:
            raise FeasibilityWorkerStartupError("worker does not match the prospective Gemma digest")
        if config.get("model_digest") == HISTORICAL_WORKER_DIGEST:
            raise FeasibilityWorkerStartupError("prospective worker cannot impersonate the historical model identity")
        if config.get("model_quantization") != MODEL_QUANTIZATION:
            raise FeasibilityWorkerStartupError("worker does not match the prospective quantization identity")
        if config.get("inference_settings") != MODEL_SETTINGS:
            raise FeasibilityWorkerStartupError("worker inference settings differ from the versioned feasibility profile")
        if (config.get("sbert_model") != SBERT_MODEL or config.get("sbert_revision") != SBERT_REVISION
                or config.get("sbert_snapshot_sha256") != SBERT_SNAPSHOT_SHA256):
            raise FeasibilityWorkerStartupError("worker does not match the pinned Sentence-BERT snapshot")
        probe = probe_endpoint(str(config["ollama_endpoint"]), expected_model=MODEL_TAG,
                               expected_digest=MODEL_DIGEST, expected_quantization=MODEL_QUANTIZATION)
        if probe.get("status") != "PASS":
            raise FeasibilityWorkerStartupError(f"prospective Ollama identity probe failed: {probe}")

        tokenizer_path = Path(str(config["tokenizer_path"]))
        tokenizer_sha = str(config["tokenizer_sha256"])
        revision = str(config["tokenizer_revision"])
        if sha256_file(tokenizer_path) != tokenizer_sha:
            raise FeasibilityWorkerStartupError("local tokenizer bytes differ from the frozen feasibility manifest")
        if (revision != str(config.get("provenance", {}).get("tokenizer_revision"))
                or tokenizer_sha != str(config.get("provenance", {}).get("tokenizer_sha256"))):
            raise FeasibilityWorkerStartupError("tokenizer revision/hash differs from reviewed provenance")
        self.tokenizer = LocalGemmaTokenizer(TokenizerIdentity(tokenizer_path, tokenizer_sha, revision))

        self.tasks = {
            str(item["task_id"]): EvaluationTask(
                task_id=str(item["task_id"]), task_family=str(item["task_family"]), order_index=int(item["source_task"]["order_index"]),
                checkpoint_id=str(item["checkpoint_id"]), checkpoint_digest=str(item["checkpoint_digest"]),
                post_detour_digest=str(item["post_detour_digest"]), detour_action=str(item["detour_action"]),
                expected_next_action=str(item["expected_next_action"]), prefix_actions=tuple(item["prefix_actions"]),
                reference_actions=tuple(item["reference_actions"]), recovery_action_budget=int(item["recovery_action_budget"]),
            ) for item in config["tasks"]
        }
        if len(self.tasks) != 6 or any(len(task.prefix_actions) + 1 + task.recovery_action_budget != TOTAL_ACTION_BUDGET
                                       for task in self.tasks.values()):
            raise FeasibilityWorkerStartupError("worker task matrix does not account for exactly 200 total actions")
        source_check = check_selected_task_sources(Path(str(config["alfworld_data_dir"])), list(config["tasks"]))
        if source_check.get("status") != "PASS":
            raise FeasibilityWorkerStartupError("selected ALFWorld task files changed or are missing")

        library_path = Path(str(config["accum18_library_path"]))
        if sha256_file(library_path) != config.get("accum18_library_sha256"):
            raise FeasibilityWorkerStartupError("Accum-18 library file differs from its frozen reference")
        library_payload = load_json(library_path)
        raw_libraries = library_payload.get("libraries", library_payload)
        skills = raw_libraries.get("Accum-18", {}).get("skills")
        if not isinstance(skills, list) or len(skills) != 18:
            raise FeasibilityWorkerStartupError("the D18 library must contain exactly 18 original skills")
        skill_ids = [str(skill.get("skill_id", "")) for skill in skills if isinstance(skill, Mapping)]
        expected_ids = config.get("accum18_skill_ids")
        if len(skill_ids) != 18 or len(set(skill_ids)) != 18 or sorted(skill_ids) != expected_ids:
            raise FeasibilityWorkerStartupError("D18 skill IDs differ from the reviewed frozen reference")
        libraries = {
            "D0": AmendedLibrary("D0", ()),
            "D18": AmendedLibrary("D18", tuple(dict(skill) for skill in skills)),
        }
        skill_texts = {str(skill["skill_id"]): str(skill["text"]) for skill in skills}
        if any(not text.strip() for text in skill_texts.values()):
            raise FeasibilityWorkerStartupError("D18 contains an empty skill text")
        cache = Path(str(config["sbert_cache"]))
        embedder = SentenceBERTEmbedder(SBERT_MODEL, cache_folder=cache, revision=SBERT_REVISION,
                                        local_files_only=True)
        if len(embedder.encode(["RQ1 feasibility readiness check"])[0]) != 768:
            raise FeasibilityWorkerStartupError("Sentence-BERT embedding dimension differs from the frozen model")

        root = Path(str(config["repository_root"])).resolve()
        self.driver = RealEpisodeDriver(root, data_dir=Path(str(config["alfworld_data_dir"])),
                                        model_name=MODEL_TAG, ollama_url=str(config["ollama_endpoint"]),
                                        bridge_log_root=Path(str(config["worker_output_dir"])) / "bridge")
        self.driver.__enter__()
        self.libraries = libraries
        self.executor = AmendedEvaluationExecutor(
            self.driver, tasks=self.tasks, libraries=libraries, embedder=embedder, scientific=False,
            split="valid_seen", provenance=dict(config["provenance"]), evaluation_profile=PROFILE_ID,
            evaluation_policy_version=POLICY_VERSION, model_digest=MODEL_DIGEST,
            result_schema=RESULT_SCHEMA, total_action_budget=TOTAL_ACTION_BUDGET,
        )

    def execute(self, assignment: Mapping[str, Any]) -> dict[str, Any]:
        raw_unit = assignment["unit"]
        condition = str(raw_unit["condition"])
        if condition not in {"D0", "D18"}:
            raise ValueError("feasibility worker accepts only D0 and D18")
        library = self.libraries[condition]
        unit = ExperimentUnit(
            phase=PROFILE_ID, task_id=str(raw_unit["task_id"]), task_index=int(raw_unit["global_order"]),
            condition=condition, seed=int(raw_unit["seed"]), identity=dict(raw_unit["identity"]),
            payload=dict(raw_unit), library_name=condition, library_size=library.size, library_hash=library.content_sha256,
        )
        attempt_dir = Path(str(assignment["attempt_dir"]))
        context = RunExecutionContext(str(self.config["run_id"]), str(assignment["attempt_id"]), 1,
                                      attempt_dir, lambda: False)
        outcome = self.executor(unit, context)
        result = {
            "unit_key": str(raw_unit["unit_key"]), "attempt_id": str(assignment["attempt_id"]),
            "status": "completed", "profile_id": PROFILE_ID, "scientific_evidence": False,
            "model_tag": MODEL_TAG,
            "model_digest": MODEL_DIGEST, "model_quantization": MODEL_QUANTIZATION,
            "total_action_budget": TOTAL_ACTION_BUDGET,
            "task_id": unit.task_id, "task_family": self.tasks[unit.task_id].task_family,
            "condition": condition, "seed": unit.seed, "success": outcome.success,
            "recovery_success": outcome.recovery_success,
            "retrieved_skill_ids": list(outcome.retrieved_skill_ids),
            "retrieval_similarity_scores": list(outcome.retrieval_similarity_scores),
            "actions": outcome.actions, "steps": outcome.steps, "invalid_actions": outcome.invalid_actions,
            "latency_ms": outcome.latency_ms, "runtime_seconds": outcome.runtime_seconds,
            "log_paths": list(outcome.log_paths), "measurements": dict(outcome.measurements),
        }

        from rq1.university_gpu.evidence import inventory_attempt, REQUIRED_LOGS
        result["log_paths"] = list(REQUIRED_LOGS)
        result["raw_evidence"] = inventory_attempt(attempt_dir, result)
        return result

    def close(self) -> None:
        self.driver.__exit__(None, None, None)


def serve(config_path: Path, *, input_stream: Any = sys.stdin, output_stream: Any = sys.stdout) -> int:
    runtime: FeasibilityEpisodeWorkerRuntime | None = None
    try:
        config = load_json(config_path)
        runtime = FeasibilityEpisodeWorkerRuntime(config)
        output_stream.write(json.dumps({"ready": True, "worker_id": runtime.worker_id,
                                        "gpu_uuid": config["gpu_uuid"], "profile_id": PROFILE_ID,
                                        "scientific_evidence": False}) + "\n")
        output_stream.flush()
        for line in input_stream:
            if not line.strip():
                continue
            message = json.loads(line)
            if message.get("command") == "shutdown":
                break
            if message.get("command") != "execute":
                output_stream.write(json.dumps({"ok": False, "error": "unsupported worker command"}) + "\n")
            else:
                try:
                    result = runtime.execute(message["assignment"])
                    output_stream.write(json.dumps({"ok": True, "result": result}, ensure_ascii=False) + "\n")
                except BaseException as exc:
                    output_stream.write(json.dumps({"ok": False, "error_type": type(exc).__name__,
                                                    "error": str(exc),
                                                    "safe_to_continue": getattr(exc, "safe_to_continue", None),
                                                    "mutation_state_known": getattr(exc, "mutation_state_known", None)},
                                                   ensure_ascii=False) + "\n")
            output_stream.flush()
        return 0
    except BaseException as exc:
        output_stream.write(json.dumps({"ready": False, "error_type": type(exc).__name__, "error": str(exc)},
                                       ensure_ascii=False) + "\n")
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

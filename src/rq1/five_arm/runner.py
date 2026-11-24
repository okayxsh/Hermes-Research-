"""Five-arm runner adapter for the existing ``AmendedEvaluationExecutor`` (``memory_intervention`` hook).

Execution is gated BEFORE any provider, environment or model component is constructed:
readiness (scientific_pilot / confirmatory) -> approved + observed runtime identity -> frozen manifest bound to the
approved runtime configuration.  Every arm retrieves from the frozen Accum-18 library (natural retrieval is always
computed and verified against the frozen ranking); the memory policy alone decides what the model sees.
Real-provider behaviour is NOT verified locally; the gate and the policy are unit-tested with synthetic doubles.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from rq1.five_arm import readiness, runtime
from rq1.five_arm.measurements import make_record
from rq1.five_arm.memory import FillerCorpus, Skill, Tokenizer, construct_arm_memory
from rq1.five_arm.protocol import CONDITIONS

TERMINATION_MAP = {"success": "success", "recovery_budget_exhausted": "budget_exhausted",
                   "action_selection_exhausted": "invalid_action_limit",
                   "environment_terminated_without_success": "environment_terminal_failure"}


class InterventionIntegrityError(RuntimeError):
    pass


def assert_execution_permitted(*, manifest: Mapping[str, Any], readiness_state: Mapping[str, Any],
                               observed_runtime: Mapping[str, Any] | None, approved_runtime: Mapping[str, Any]) -> None:
    level = "scientific_pilot" if manifest.get("stage") == "pilot" else "confirmatory"
    readiness.require(level, readiness_state)
    runtime.require_runtime(observed_runtime, approved_runtime)
    if manifest.get("status") != "frozen" or not readiness.valid_approval(manifest.get("approval")):
        raise readiness.ExecutionBlocked("the five-arm manifest is not frozen with a valid human approval")
    if manifest["input_hashes"].get("runtime_config_sha256") != runtime.config_sha256(approved_runtime):
        raise readiness.ExecutionBlocked("the manifest is bound to a different runtime configuration")


class FiveArmMemoryPolicy:
    """Callable for ``AmendedEvaluationExecutor(memory_intervention=...)``: (unit, task, retrieval_outcome, failure_context)."""

    def __init__(self, *, manifest: Mapping[str, Any], skill_texts: Mapping[str, str], corpus: FillerCorpus,
                 tokenizer: Tokenizer, tolerance: Mapping[str, Any], d0_representation: str) -> None:
        self.manifest, self.skill_texts, self.corpus = manifest, dict(skill_texts), corpus
        self.tokenizer, self.tolerance, self.d0_representation = tokenizer, dict(tolerance), d0_representation
        self.memory_hashes = {(u["task_id"], u["condition"]): u["memory_block_sha256"] for u in manifest["units"]}

    def __call__(self, unit: Any, task: Any, retrieval_outcome: Any, _failure_context: Mapping[str, Any]):
        arm, task_id = str(unit.condition), str(task.task_id)
        if arm not in CONDITIONS:
            raise InterventionIntegrityError(f"unsupported arm {arm!r}")
        selection = self.manifest["selections"][task_id]
        frozen_top3 = [item["skill_id"] for item in selection["natural_top3"]]
        live = [(c.skill_id, float(c.score)) for c in retrieval_outcome.result.top]
        if [skill_id for skill_id, _ in live] != frozen_top3:
            raise InterventionIntegrityError(f"{task_id}: live natural top-3 {live} differs from the frozen ranking {frozen_top3}")
        natural = [Skill(skill_id, self.skill_texts[skill_id], score, rank) for rank, (skill_id, score) in enumerate(live, 1)]
        active = None
        if arm in ("Rel", "Irr"):
            chosen = selection[arm.lower()]
            if chosen is None:
                raise InterventionIntegrityError(f"{task_id}: {arm} has no frozen selection; the unit must not exist")
            active = Skill(chosen["skill_id"], self.skill_texts[chosen["skill_id"]], chosen.get("score"), chosen.get("rank"))
        memory, audit = construct_arm_memory(arm, natural_top3=natural, active_skill=active,
                                             corpus=self.corpus if arm in ("Rel", "Irr", "Fill") else None,
                                             tokenizer=self.tokenizer, tolerance=self.tolerance,
                                             d0_representation=self.d0_representation, salt=task_id)
        if arm in ("Rel", "Irr", "Fill") and (audit["within_token_tolerance"] is not True or audit["filler_corpus_approved"] is not True):
            raise InterventionIntegrityError(f"{task_id}/{arm}: unapproved filler or token mismatch; intervention refused")
        if audit["memory_block_sha256"] != self.memory_hashes.get((task_id, arm)):
            raise InterventionIntegrityError(f"{task_id}/{arm}: memory block differs from the frozen manifest")
        audit["natural_top3_skill_ids"] = frozen_top3
        audit["natural_top3_scores"] = [score for _, score in live]
        return memory, audit


class ProviderTokenRecorder:
    """Wraps ``RealEpisodeDriver._model_response`` to keep provider-side prompt token counts and latency, which the
    frozen driver parses but does not log."""

    def __init__(self, driver: Any, tokenizer: Tokenizer | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        original: Callable[[str], Any] = driver._model_response

        def wrapped(prompt: str) -> Any:
            started = time.monotonic()
            response = original(prompt)
            self.calls.append({"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                               "local_raw_tokens": len(tokenizer.encode(prompt)) if tokenizer is not None else None,
                               "prompt_eval_count": getattr(response, "prompt_eval_count", None),
                               "eval_count": getattr(response, "eval_count", None),
                               "seconds": round(time.monotonic() - started, 6)})
            return response

        driver._model_response = wrapped

    def take(self) -> list[dict[str, Any]]:
        calls, self.calls = self.calls, []
        return calls


def to_measurement(*, unit: Mapping[str, Any], task: Mapping[str, Any], outcome: Mapping[str, Any] | None,
                   failure: Mapping[str, Any] | None, audit: Mapping[str, Any], provider_calls: list[Mapping[str, Any]],
                   runtime_identity_sha256: str, first_prompt_sha256: str | None, local_tokens: Mapping[str, Any],
                   analysis_labels: Mapping[str, str], synthetic: bool, attempt: int, exposure_policy: str,
                   provider_context_limit: int | None = None) -> dict[str, Any]:
    """Map an executor outcome (or an infrastructure failure) onto the uniform five-arm measurement schema.
    Any provider call whose prompt reached ``provider_context_limit`` tokens may have been silently truncated by the
    provider: the episode becomes INVALID_MEASUREMENT (never a valid outcome)."""
    arm = unit["condition"]
    if outcome is not None and provider_context_limit is not None and any(
            (c.get("prompt_eval_count") or 0) >= provider_context_limit for c in provider_calls):
        outcome = {**outcome, "termination_reason": "provider_context_limit_reached"}
    injected = list(audit["injected_skill_ids"])
    base = dict(
        synthetic=synthetic, unit_key=unit["unit_key"], attempt=attempt, task_id=unit["task_id"],
        goal_template=task["goal_template"], scene=task["scene"], family=task["family"], exposure_policy=exposure_policy,
        oracle_id=task.get("oracle_id", task["oracle_sha256"]), oracle_sha256=task["oracle_sha256"],
        checkpoint_digest=task["checkpoint_digest"], post_detour_digest=task["post_detour_digest"], condition=arm,
        seed=unit["seed"], runtime_identity_sha256=runtime_identity_sha256,
        natural_top3_skill_ids=list(audit.get("natural_top3_skill_ids", [])), natural_top3_scores=list(audit.get("natural_top3_scores", [])),
        injected_skill_ids=injected, injected_skill_count=len(injected),
        analysis_only_injected_labels=[analysis_labels.get(s) for s in injected],
        filler_corpus_sha256=audit["filler_corpus_sha256"], filler_sentence_indices=list(audit["filler_sentence_indices"]),
        memory_block_sha256=audit["memory_block_sha256"], first_recovery_prompt_sha256=first_prompt_sha256,
        tokens={"raw_text_tokens": local_tokens.get("raw_text_tokens"), "memory_block_tokens": audit["memory_block_tokens"],
                "chat_template_tokens": local_tokens.get("chat_template_tokens"),
                "provider_prompt_tokens": provider_calls[0]["prompt_eval_count"] if provider_calls else None},
        total_action_cap=int(unit["recovery_action_budget"]) + int(task["prefix_actions"]) + int(task["detour_actions"]),
        prefix_actions=int(task["prefix_actions"]), detour_actions=int(task["detour_actions"]),
        inference_latency_seconds=round(sum(c["seconds"] for c in provider_calls), 6),
        not_applicable=(["memory"] if arm == "D0" else []) + list(audit.get("not_applicable", [])),
    )
    if outcome is not None:
        reason = TERMINATION_MAP.get(str(outcome.get("termination_reason")))
        if reason is None:
            return make_record(**base, recovery_actions=0, success=None, termination_reason="invalid_measurement",
                               invalid_actions=0, wall_clock_seconds=float(outcome.get("runtime_seconds", 0.0)),
                               outcome_status="INVALID_MEASUREMENT")
        return make_record(**base, recovery_actions=int(outcome["post_failure_actions"]), success=bool(outcome["recovery_success"]),
                           termination_reason=reason, invalid_actions=int(outcome["invalid_action_selections"]),
                           wall_clock_seconds=float(outcome.get("runtime_seconds", 0.0)), outcome_status="VALID")
    stage = str((failure or {}).get("stage", ""))
    klass = str((failure or {}).get("failure_class", ""))
    if klass == "infrastructure":
        reason = "environment_error" if stage == "controlled_failure" else "infrastructure_interruption"
        status = "MISSING_INFRASTRUCTURE"
    else:
        reason, status = "invalid_measurement", "INVALID_MEASUREMENT"
    return make_record(**base, recovery_actions=0, success=None, termination_reason=reason, invalid_actions=0,
                       wall_clock_seconds=float((failure or {}).get("runtime_seconds", 0.0)), outcome_status=status)


class FiveArmWorker:
    """University-side worker.  The gate runs FIRST; provider, environment and model components are only imported and
    built after it passes.  Not executable locally (no approvals exist); real-provider wiring is unverified until the
    university integration step."""

    def __init__(self, config: Mapping[str, Any], *, readiness_state: Mapping[str, Any],
                 observed_runtime: Mapping[str, Any] | None, approved_runtime: Mapping[str, Any]) -> None:
        manifest = config["manifest"]
        assert_execution_permitted(manifest=manifest, readiness_state=readiness_state,
                                   observed_runtime=observed_runtime, approved_runtime=approved_runtime)
        # --- only reachable after every scientific and runtime gate has passed ---------------------------------
        from rq1.evaluation.amended_executor import AmendedEvaluationExecutor
        from rq1.evaluation.amended_libraries import AmendedLibrary
        from rq1.evaluation.amended_matrix import EvaluationTask
        from rq1.hermes.episode_driver import RealEpisodeDriver
        from rq1.retrieval.embedder import SentenceBERTEmbedder
        from rq1.university_gpu.interventions import LocalGemmaTokenizer, TokenizerIdentity
        from rq1.university_gpu.ollama import probe_endpoint
        from rq1.five_arm.memory import load_filler_corpus

        self.config, self.manifest, self.approved_runtime = dict(config), manifest, dict(approved_runtime)
        probe = probe_endpoint(str(config["ollama_endpoint"]), expected_model=approved_runtime["model_tag"],
                               expected_digest=approved_runtime["model_manifest_digest"])
        if probe.get("status") != "PASS":
            raise runtime.RuntimeIdentityError(f"provider preflight failed: {probe}")
        self.tokenizer = LocalGemmaTokenizer(TokenizerIdentity(Path(str(config["tokenizer_path"])),
                                                               approved_runtime["tokenizer_sha256"],
                                                               approved_runtime["tokenizer_revision"]))
        corpus = load_filler_corpus(Path(str(config["filler_corpus_path"])))
        if corpus.sha256 != manifest["input_hashes"]["filler_corpus_sha256"] or not corpus.approved:
            raise InterventionIntegrityError("filler corpus differs from the manifest or is not approved")
        self.tasks_meta = {str(t["task_id"]): t for t in manifest["tasks"]}
        self.tasks = {str(t["task_id"]): EvaluationTask(**config["evaluation_tasks"][str(t["task_id"])]) for t in manifest["tasks"]}
        accum18 = AmendedLibrary("Accum-18", tuple(dict(s) for s in config["accum18_library"]["skills"]))
        embedder = SentenceBERTEmbedder(approved_runtime["sbert_model"], cache_folder=Path(str(config["sbert_cache"])),
                                        revision=approved_runtime["sbert_revision"], local_files_only=True)
        self.driver = RealEpisodeDriver(Path(str(config["repository_root"])).resolve(),
                                        data_dir=Path(str(config["alfworld_data_dir"])), model_name=approved_runtime["model_tag"],
                                        ollama_url=str(config["ollama_endpoint"]),
                                        bridge_log_root=Path(str(config["worker_output_dir"])) / "bridge")
        self.driver.__enter__()
        self.recorder = ProviderTokenRecorder(self.driver, self.tokenizer)
        self.policy = FiveArmMemoryPolicy(manifest=manifest, skill_texts=config["skill_texts"], corpus=corpus,
                                          tokenizer=self.tokenizer, tolerance=config["token_tolerance"],
                                          d0_representation=config["d0_representation"])
        self._last_audit: dict[str, Any] = {}

        def policy(unit, task, outcome, failure_context):
            memory, audit = self.policy(unit, task, outcome, failure_context)
            self._last_audit = dict(audit)
            return memory, audit

        self.executor = AmendedEvaluationExecutor(
            self.driver, tasks=self.tasks, libraries={arm: accum18 for arm in CONDITIONS}, embedder=embedder,
            scientific=True, split=manifest["split"], provenance=dict(config["provenance"]), memory_intervention=policy)

    def execute(self, unit_entry: Mapping[str, Any], *, attempt: int, attempt_dir: Path, analysis_labels: Mapping[str, str],
                exposure_policy: str) -> dict[str, Any]:
        from rq1.experiment.models import ExperimentUnit, RunExecutionContext, RunFailure

        unit = ExperimentUnit(phase=self.manifest["stage"], task_id=unit_entry["task_id"], task_index=int(unit_entry["global_order"]),
                              condition=unit_entry["condition"], seed=int(unit_entry["seed"]), identity=dict(unit_entry["identity"]),
                              payload=dict(unit_entry), library_name="Accum-18", library_size=18,
                              library_hash=self.config["accum18_library"]["content_sha256"])
        context = RunExecutionContext(self.manifest["run_id"], f"{unit_entry['unit_key'][:16]}-a{attempt}", attempt, attempt_dir, lambda: False)
        self.recorder.take()
        self._last_audit = {}
        started = time.monotonic()
        outcome, failure = None, None
        try:
            result = self.executor(unit, context)
            outcome = {**dict(result.measurements), "runtime_seconds": result.runtime_seconds}
        except RunFailure as exc:
            failure = {**dict(getattr(exc, "details", {}) or {}), "runtime_seconds": round(time.monotonic() - started, 3)}
        calls = self.recorder.take()
        audit = self._last_audit or {"injected_skill_ids": [], "filler_corpus_sha256": None, "filler_sentence_indices": [],
                                     "memory_block_sha256": None, "memory_block_tokens": None,
                                     "not_applicable": ["memory_not_constructed_before_failure"]}
        return to_measurement(unit=unit_entry, task=self.tasks_meta[unit_entry["task_id"]], outcome=outcome, failure=failure,
                              audit=audit, provider_calls=calls, runtime_identity_sha256=runtime.config_sha256(self.approved_runtime),
                              first_prompt_sha256=calls[0]["prompt_sha256"] if calls else None,
                              local_tokens={"raw_text_tokens": calls[0]["local_raw_tokens"] if calls else None,
                                            "chat_template_tokens": None},
                              analysis_labels=analysis_labels, synthetic=False, attempt=attempt, exposure_policy=exposure_policy,
                              provider_context_limit=int(self.approved_runtime["num_ctx"]))

    def close(self) -> None:
        self.driver.__exit__(None, None, None)
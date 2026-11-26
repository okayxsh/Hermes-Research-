"""Five-arm and pilot-diagnostic episode executor, composed ONLY from the frozen execution components:

``RealRecoveryHarness`` (checkpoint replay + digest-verified frozen detour), ``run_recovery_episode`` (authoritative
order: replay -> detour -> failure context -> exactly ONE retrieval -> memory injection -> model recovery loop),
``build_retrieval_boundary`` (single-crossing Sentence-BERT boundary) and the unchanged ``RealEpisodeSession`` prompt
builder / bridge loop.  Nothing in those components is modified.

Retrieval-freezing (verified statically, see protocol SD-14): retrieval happens exactly once, on the post-detour failure
context; the same memory block is passed to every recovery decision; there is no re-retrieval, query update or memory
reload during recovery.  This executor additionally asserts at run time that the memory section of every recovery
prompt is byte-identical to the first one.

Modes:
* ``arm``                     -- one of D0/D18/Rel/Irr/Fill; retrieval from the frozen Accum-18 library for every arm;
                                 the memory policy alone decides what the model sees.
* ``diagnostic_detoured``     -- pilot only: frozen checkpoint + frozen detour, no retrieval, D0 memory presentation.
* ``diagnostic_undisturbed``  -- pilot only: frozen checkpoint, NO detour, no retrieval, D0 memory presentation,
                                 identical post-checkpoint model-action allowance (SD-15: total limit = cap - 1).

Provider calls are captured at the SESSION level (``RealEpisodeSession._model_response`` is per session, not per
driver): provider ``prompt_eval_count``/``eval_count``/``done_reason`` and errors are persisted next to local token
counts.  Leakage of arm names, labels, skill IDs, scores or oracle steps in the first recovery prompt is a protocol
violation raised BEFORE the provider is called.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rq1.acquisition.environment import SBERT_MODEL, SBERT_REVISION
from rq1.evaluation.amended_matrix import EvaluationTask
from rq1.evaluation.recovery_executor import RecoveryEpisodeSpec, run_recovery_episode
from rq1.five_arm.measurements import (DIAGNOSTIC_CONDITIONS, make_diagnostic_record, make_record)
from rq1.five_arm.memory import HISTORICAL_NOLIB_BLOCK, D0_REPRESENTATIONS
from rq1.five_arm.prompting import LeakageProbe, scan_leakage, split_sections
from rq1.five_arm.protocol import CONDITIONS, UNDISTURBED_CONDITIONS
from rq1.hermes.episode_driver import (EpisodeDriverError, MODEL_CONTEXT_LENGTH, OUTPUT_TOKEN_CAP, RealEpisodeDriver,
                                       ollama_chat_payload)
from rq1.pilot.real_runtime.harnesses import RealRecoveryHarness, bridge_state_digest
from rq1.recovery.controlled_failure import ControlledFailureError, select_reversible_navigation_action
from rq1.retrieval.controller import build_retrieval_boundary
from rq1.retrieval.query import CANONICAL_FAILURE_MESSAGE

MODES = ("arm", "diagnostic_undisturbed", "diagnostic_detoured")
DIAGNOSTIC_CONDITION = {"diagnostic_undisturbed": "UNDISTURBED", "diagnostic_detoured": "DETOURED"}
MEMORY_SOURCE = {"D0": "none", "D0U": "none", "D18": "natural_retrieval_at_post_detour_state",
                 "D18U": "frozen_block_from_detoured_natural_retrieval", "Rel": "controlled_injection_from_frozen_selection",
                 "Irr": "controlled_injection_from_frozen_selection", "Fill": "controlled_injection_from_frozen_selection"}


class ProtocolViolation(RuntimeError):
    """Never retried: the queue stops.  (Leakage, frozen-ranking drift, memory drift, real/synthetic mixing.)"""


class InterventionIntegrityError(ProtocolViolation):
    """The constructed memory differs from the frozen manifest (ranking, filler, hash): never run, never retried."""


class MeasurementInvalid(RuntimeError):
    """The episode cannot yield a valid measurement (e.g. context overflow); recorded INVALID_MEASUREMENT."""


class Tokenizer:  # protocol only
    def encode(self, text: str) -> Sequence[int]: ...


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- provider capture -------------------------------------------------------------------------------------------------
class ProviderCallCapture:
    """Wraps each session's ``_model_response``; records every call and guards prompt integrity before sending."""

    def __init__(self, *, tokenizer: Any, num_ctx: int, max_output_tokens: int,
                 chat_template_tokens: Callable[[str], int] | None = None) -> None:
        self.tokenizer, self.num_ctx, self.max_output_tokens = tokenizer, int(num_ctx), int(max_output_tokens)
        self.chat_template_tokens = chat_template_tokens
        self.reset(None)

    def reset(self, probe: LeakageProbe | None, request_builder: Callable[[str], Mapping[str, Any]] | None = None) -> None:
        self.request_builder = request_builder
        self.calls: list[dict[str, Any]] = []
        self.prompt_hashes: list[str] = []
        self.probe = probe
        self._memory_section: tuple[bool, str | None] | None = None

    def _check(self, prompt: str) -> dict[str, Any]:
        sections = split_sections(prompt)
        memory = ("RECOVERY MEMORY" in sections, sections.get("RECOVERY MEMORY"))
        if self._memory_section is None:
            self._memory_section = memory
            if self.probe is not None:
                problems = scan_leakage(prompt, self.probe)
                if problems:
                    raise ProtocolViolation("first recovery prompt leaks protected information: " + "; ".join(problems))
        elif memory != self._memory_section:
            raise ProtocolViolation("the memory section changed during recovery (memory must be static after the single retrieval)")
        raw = len(self.tokenizer.encode(prompt)) if self.tokenizer is not None else None
        chat = self.chat_template_tokens(prompt) if self.chat_template_tokens is not None else None
        local = chat if chat is not None else raw
        if local is not None and local + self.max_output_tokens > self.num_ctx:
            raise MeasurementInvalid(f"local prompt tokens {local} + output cap {self.max_output_tokens} exceed num_ctx {self.num_ctx}")
        return {"local_raw_tokens": raw, "local_chat_template_tokens": chat}

    def install(self, session: Any) -> None:
        original = session._model_response

        def wrapped(prompt: str):
            local = self._check(prompt)
            digest = sha256_text(prompt)
            request = self.request_builder(prompt) if self.request_builder is not None else {"prompt": prompt}
            record = {"call": len(self.calls) + 1, "prompt_sha256": digest, "prompt_chars": len(prompt), **local,
                      "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest(),
                      "response_sha256": None, "prompt_eval_count": None, "eval_count": None, "done_reason": None,
                      "seconds": None, "error": None}
            started = time.monotonic()
            try:
                response = original(prompt)
            except Exception as exc:
                record.update(seconds=round(time.monotonic() - started, 6), error=f"{type(exc).__name__}: {exc}"[:300])
                self.calls.append(record)
                self.prompt_hashes.append(digest)
                raise
            record.update(seconds=round(time.monotonic() - started, 6),
                          response_sha256=sha256_text(str(response)),
                          prompt_eval_count=getattr(response, "prompt_eval_count", None),
                          eval_count=getattr(response, "eval_count", None),
                          done_reason=getattr(response, "done_reason", None))
            self.calls.append(record)
            self.prompt_hashes.append(digest)
            return response

        session._model_response = wrapped


class _CapturingDriver:
    """Proxy handed to the frozen harness: every session it opens gets the provider capture installed."""

    def __init__(self, driver: Any, capture: ProviderCallCapture) -> None:
        self._driver, self._capture = driver, capture

    def session(self, **kwargs: Any) -> Any:
        session = self._driver.session(**kwargs)
        self._capture.install(session)
        return session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._driver, name)


# --- undisturbed diagnostic harness -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class NoDetour:
    checkpoint_id: str
    would_be_detour_action: str
    post_state_digest: str
    failure_message: str = CANONICAL_FAILURE_MESSAGE    # used only for the (empty-library) retrieval query; not model-visible

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint_id": self.checkpoint_id, "type": "no_detour_undisturbed_continuation",
                "action": None, "would_be_detour_action": self.would_be_detour_action,
                "post_state_digest": self.post_state_digest}


class UndisturbedContinuationHarness(RealRecoveryHarness):
    """PILOT DIAGNOSTIC ONLY.  Reaches and verifies the same frozen checkpoint, verifies that the deterministic detour rule
    would select the frozen detour (provenance), but performs NO detour: the model continues from the checkpoint."""

    def apply_controlled_action_perturbation(self, checkpoint_id: str, failure_message: str = CANONICAL_FAILURE_MESSAGE):
        if self.session is None or self.session.state is None:
            raise EpisodeDriverError("cannot continue from an inactive session")
        frozen = self.frozen_perturbation or {}
        self.checkpoint_digest = bridge_state_digest(self.session.state)
        if self.checkpoint_digest != frozen.get("checkpoint_digest"):
            raise ControlledFailureError("episode did not reproduce the frozen checkpoint state", code="checkpoint_digest_mismatch")
        admissible = self.session.state.get("admissible_actions")
        if not isinstance(admissible, list):
            raise ControlledFailureError("checkpoint has no observable admissible actions", code="admissible_actions_unavailable")
        would_be = select_reversible_navigation_action(admissible, self._continuation[0])
        if would_be != frozen.get("detour_action"):
            raise ControlledFailureError("deterministic detour differs from the frozen detour", code="detour_mismatch")
        self.post_detour_state = None
        return NoDetour(checkpoint_id, would_be, self.checkpoint_digest)


class _NoEmbedder:
    def encode(self, texts):
        if texts:
            raise ProtocolViolation("diagnostic episodes must not retrieve")
        return []


# --- trace from the persisted episode events ---------------------------------------------------------------------------
def recovery_trace(events_path: Path) -> list[dict[str, Any]]:
    """Per model decision after memory injection: rejected attempts, the dispatched action and its environment result.
    Built from the session's fsync'd ``episode-events.jsonl`` (primary provenance)."""
    if not events_path.is_file():
        return []
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trace: list[dict[str, Any]] = []
    started = False
    pending: list[str] = []
    for event in events:
        name, payload = event.get("event"), event.get("payload") or {}
        if name == "recovery_memory_injected":
            started = True
            continue
        if not started:
            continue
        if name == "model_selection" and payload.get("rejection_reason"):
            pending.append(str(payload["rejection_reason"]))
        elif name == "tool_result" and payload.get("tool") == "alfworld_step":
            response = payload.get("response") or {}
            trace.append({"decision": len(trace) + 1, "action": (payload.get("request") or {}).get("action"),
                          "rejections": pending, "env_action_valid": response.get("action_valid"),
                          "done": bool(response.get("done")),
                          "success": response.get("success") if isinstance(response.get("success"), bool) else None,
                          "selection_exhausted": False})
            pending = []
        elif name == "tool_result" and payload.get("tool") == "alfworld_abort":
            trace.append({"decision": len(trace) + 1, "action": None, "rejections": pending, "env_action_valid": None,
                          "done": True, "success": None, "selection_exhausted": True})
            pending = []
    return trace


# --- executor ---------------------------------------------------------------------------------------------------------
class FiveArmEpisodeExecutor:
    def __init__(self, driver: Any, *, mode: str, tasks: Mapping[str, EvaluationTask], task_meta: Mapping[str, Mapping[str, Any]],
                 split: str, synthetic: bool, tokenizer: Any, runtime_identity_sha256: str, exposure_policy: str,
                 d0_representation: str, library: Any = None, embedder: Any = None, memory_policy: Any = None,
                 leakage_probes: Mapping[str, LeakageProbe] | None = None, analysis_labels: Mapping[str, str] | None = None,
                 num_ctx: int = MODEL_CONTEXT_LENGTH, max_output_tokens: int = OUTPUT_TOKEN_CAP,
                 chat_template_tokens: Callable[[str], int] | None = None,
                 provider_token_tolerance: int | None = None,
                 execution_provenance: Mapping[str, Any] | None = None) -> None:
        if mode not in MODES:
            raise ProtocolViolation(f"unknown executor mode {mode!r}")
        real = isinstance(driver, RealEpisodeDriver)
        if real and synthetic:
            raise ProtocolViolation("a synthetic configuration can never drive the real provider/environment")
        if not real and not synthetic:
            raise ProtocolViolation("only the real episode driver may produce non-synthetic (scientific) records")
        if not synthetic and chat_template_tokens is None:
            raise ProtocolViolation("real execution requires the verified chat-template token counter")
        if d0_representation not in D0_REPRESENTATIONS:
            raise ProtocolViolation(f"d0_representation must be one of {D0_REPRESENTATIONS}")
        if mode == "arm" and (library is None or embedder is None or memory_policy is None):
            raise ProtocolViolation("arm mode needs the frozen library, the embedder and the memory policy")
        if mode != "arm" and (library is not None or memory_policy is not None):
            raise ProtocolViolation("diagnostic modes use no library and no memory policy")
        self.driver, self.mode, self.tasks, self.task_meta = driver, mode, dict(tasks), dict(task_meta)
        self.split, self.synthetic, self.tokenizer = split, synthetic, tokenizer
        self.runtime_identity_sha256, self.exposure_policy = runtime_identity_sha256, exposure_policy
        self.d0_representation, self.library, self.embedder, self.memory_policy = d0_representation, library, embedder, memory_policy
        self.leakage_probes = dict(leakage_probes or {})
        self.analysis_labels = dict(analysis_labels or {})
        self.provider_token_tolerance = provider_token_tolerance
        self.execution_provenance = {"mode": "sequential", "endpoint": getattr(driver, "ollama_url", None),
                                     "model": getattr(driver, "model_name", None), "host": socket.gethostname(),
                                     "pid": os.getpid(), "python": platform.python_version(),
                                     **dict(execution_provenance or {})}
        self.num_ctx = int(num_ctx)
        self.capture = ProviderCallCapture(tokenizer=tokenizer, num_ctx=num_ctx, max_output_tokens=max_output_tokens,
                                           chat_template_tokens=chat_template_tokens)
        self.proxy = _CapturingDriver(driver, self.capture)

    # -- helpers ------------------------------------------------------------------------------------------------------
    def _condition(self, unit: Mapping[str, Any]) -> str:
        condition = str(unit["condition"])
        if self.mode == "arm":
            if condition not in CONDITIONS:
                raise ProtocolViolation(f"arm executor received non-five-arm condition {condition!r}")
        elif condition != DIAGNOSTIC_CONDITION[self.mode]:
            raise ProtocolViolation(f"{self.mode} executor received condition {condition!r}")
        return condition

    def _d0_memory(self) -> dict[str, Any]:
        return {"recovery_memory": None if self.d0_representation == "absent_memory_section" else dict(HISTORICAL_NOLIB_BLOCK)}

    def _diagnostic_audit(self) -> dict[str, Any]:
        inner = self._d0_memory()["recovery_memory"]
        return {"injected_skill_ids": [], "filler_corpus_sha256": None, "filler_sentence_indices": [],
                "memory_block_sha256": None if inner is None else sha256_text(json.dumps(inner, ensure_ascii=False, sort_keys=True)),
                "memory_block_tokens": 0 if inner is None else len(self.tokenizer.encode(json.dumps(inner, ensure_ascii=False, sort_keys=True))),
                "natural_top3_skill_ids": [], "natural_top3_scores": [],
                "not_applicable": ["memory", "retrieval", "token_anchor_match", "neutral_content", "slot_layout"]}

    def _arithmetic(self, unit: Mapping[str, Any], condition: str, prefix: int) -> tuple[bool, int, int]:
        """(undisturbed, detour actions, model-action allowance).  Arm mode: the unit's total cap is 50 for detoured and 49
        for undisturbed conditions, so the allowance (cap - prefix - detour) is identical.  Diagnostic modes keep their
        frozen Phase 3A arithmetic (one study cap; undisturbed bridge limit = cap - 1)."""
        cap = int(unit["total_action_cap"])
        if self.mode == "arm":
            undisturbed = condition in UNDISTURBED_CONDITIONS
            detour = 0 if undisturbed else 1
            allowance = cap - prefix - detour
        else:
            undisturbed = self.mode == "diagnostic_undisturbed"
            detour = 0 if undisturbed else 1
            allowance = cap - prefix - 1
        if allowance < 1 or allowance != int(unit["recovery_action_allowance"]):
            raise ProtocolViolation(f"{unit['task_id']}: recovery allowance {unit.get('recovery_action_allowance')} "
                                    f"differs from cap arithmetic {allowance}")
        return undisturbed, detour, allowance

    # -- one attempt --------------------------------------------------------------------------------------------------
    def __call__(self, unit: Mapping[str, Any], *, attempt: int, attempt_dir: Path) -> dict[str, Any]:
        condition = self._condition(unit)
        task_id = str(unit["task_id"])
        task, meta = self.tasks[task_id], self.task_meta[task_id]
        seed, cap = int(unit["seed"]), int(unit["total_action_cap"])
        prefix = len(task.prefix_actions)
        undisturbed, detour, allowance = self._arithmetic(unit, condition, prefix)
        total_limit = prefix + detour + allowance
        self.driver.inference_seed = seed               # the real driver (not the proxy): decision 012 inference seed
        model = getattr(self.driver, "model_name", "")
        self.capture.reset(self.leakage_probes.get(task_id) if self.mode == "arm" else None,
                           request_builder=lambda prompt: ollama_chat_payload(model, prompt, seed))
        harness_cls = UndisturbedContinuationHarness if undisturbed else RealRecoveryHarness
        harness = harness_cls(self.proxy, output_dir=attempt_dir / "episode", run_id=str(unit.get("run_id", "five-arm")),
                              attempt_id=f"{str(unit['unit_key'])[:16]}-a{attempt}", reference_actions=task.reference_actions,
                              total_action_limit=total_limit, allowed_split=self.split, profile_prefix="rq1-five-arm",
                              frozen_perturbation=task.frozen_perturbation())
        if self.mode == "arm" and not undisturbed:
            boundary = build_retrieval_boundary(self.library.retrieval_documents(), self.embedder, embedding_model=SBERT_MODEL,
                                                top_k=3, embedding_model_revision=SBERT_REVISION)
            library_name, library_size, library_hash = "Accum-18", self.library.size, self.library.content_sha256
        else:
            boundary = build_retrieval_boundary((), _NoEmbedder(), embedding_model=SBERT_MODEL, top_k=3,
                                                embedding_model_revision=SBERT_REVISION)
            library_name, library_size, library_hash = "none", 0, None
        spec = RecoveryEpisodeSpec(run_id=str(unit.get("run_id", "five-arm")), attempt_id=f"{str(unit['unit_key'])[:16]}-a{attempt}",
                                   task_id=task_id, task_family=task.task_family, split=self.split, seed=seed, condition=condition,
                                   library_name=library_name, library_size=library_size, library_hash=library_hash,
                                   checkpoint_id=task.checkpoint_id, prefix_actions=task.prefix_actions, action_budget=allowance,
                                   episode_id=str(unit["unit_key"]))
        audit_box: dict[str, Any] = {}

        def transform(retrieval_outcome, failure_context, _spec):
            if self.mode == "arm":
                memory, audit = self.memory_policy(unit, task, retrieval_outcome, failure_context)
            else:
                if retrieval_outcome.result.retrieved:
                    raise ProtocolViolation("diagnostic episode performed a retrieval")
                memory, audit = self._d0_memory(), self._diagnostic_audit()
            audit_box.update(audit)
            return memory, audit

        started = time.monotonic()
        session = None
        outcome: dict[str, Any] | None = None
        failure: dict[str, Any] | None = None
        try:
            result = run_recovery_episode(harness, spec, boundary, log_dir=attempt_dir / "result", memory_transform=transform)
            session = harness.session
            records = [r for r in session.records if r.phase == "recovery"]
            success = harness.recovery_succeeded()
            if success:
                reason = "success"
            elif session.selection_failures:
                reason = "invalid_action_limit"
            elif len(records) >= allowance:
                reason = "budget_exhausted"
            else:
                reason = "environment_terminal_failure"
            outcome = {"success": success, "termination_reason": reason, "recovery_actions": len(records),
                       "invalid_actions": sum(session.rejection_counts.values()),
                       "detour_action": (result.failure or {}).get("action"),
                       "retrieval_performed": not result.no_retrieval}
        except ProtocolViolation:
            raise
        except MeasurementInvalid as exc:
            failure = {"status": "INVALID_MEASUREMENT", "reason": "invalid_measurement", "detail": f"context_overflow: {exc}"}
        except ControlledFailureError as exc:
            failure = {"status": "MISSING_INFRASTRUCTURE", "reason": "environment_error", "detail": f"controlled_failure:{exc.code}: {exc}"}
        except EpisodeDriverError as exc:
            provider = any(c["error"] for c in self.capture.calls) or "Ollama" in str(exc)
            failure = {"status": "MISSING_INFRASTRUCTURE", "reason": "provider_error" if provider else "environment_error",
                       "detail": f"{type(exc).__name__}: {exc}"[:500]}
        except Exception as exc:  # noqa: BLE001 -- anything else interrupted the episode; never a model outcome
            failure = {"status": "MISSING_INFRASTRUCTURE", "reason": "infrastructure_interruption",
                       "detail": f"{type(exc).__name__}: {exc}"[:500]}
        finally:
            harness.close()
        wall = round(time.monotonic() - started, 6)
        calls = list(self.capture.calls)
        if outcome is not None:
            problem = self._provider_token_problem(calls)
            if problem:
                failure, outcome = {"status": "INVALID_MEASUREMENT", "reason": "invalid_measurement", "detail": problem}, None
        trace = recovery_trace(attempt_dir / "episode" / "episode-events.jsonl")
        return self._record(unit, condition, meta, prefix, detour, allowance, cap, seed, attempt, audit_box, calls, trace,
                            outcome, failure, wall, attempt_dir)

    def interruption_record(self, unit: Mapping[str, Any], attempt: int, detail: str) -> dict[str, Any]:
        """MISSING_INFRASTRUCTURE record for an attempt whose process died before it could record itself."""
        condition = self._condition(unit)
        task_id = str(unit["task_id"])
        task, meta = self.tasks[task_id], self.task_meta[task_id]
        cap, prefix = int(unit["total_action_cap"]), len(task.prefix_actions)
        _undisturbed, detour, allowance = self._arithmetic(unit, condition, prefix)
        failure = {"status": "MISSING_INFRASTRUCTURE", "reason": "infrastructure_interruption", "detail": detail}
        return self._record(unit, condition, meta, prefix, detour, allowance, cap, int(unit["seed"]), attempt, {}, [], [],
                            None, failure, 0.0, None)

    def _provider_token_problem(self, calls: Sequence[Mapping[str, Any]]) -> str | None:
        for call in calls:
            provider = call.get("prompt_eval_count")
            if provider is not None and provider >= self.num_ctx:
                return f"provider_context_limit_reached: call {call['call']} prompt_eval_count {provider} >= num_ctx {self.num_ctx}"
            local = call.get("local_chat_template_tokens")
            if self.provider_token_tolerance is not None and provider is not None and local is not None \
                    and abs(provider - local) > self.provider_token_tolerance:
                return f"provider_token_discrepancy: call {call['call']} provider {provider} vs local chat-template {local}"
        if not self.synthetic and any(c.get("prompt_eval_count") is None and not c.get("error") for c in calls):
            return "provider_prompt_eval_count_missing"
        return None

    def _record(self, unit, condition, meta, prefix, detour, allowance, cap, seed, attempt, audit, calls, trace, outcome,
                failure, wall, attempt_dir) -> dict[str, Any]:
        audit = dict(audit) if audit else {"injected_skill_ids": [], "filler_corpus_sha256": None, "filler_sentence_indices": [],
                                           "memory_block_sha256": None, "memory_block_tokens": None,
                                           "not_applicable": ["memory_not_constructed_before_failure"]}
        injected = list(audit.get("injected_skill_ids", []))
        sequence = hashlib.sha256("\n".join(c["prompt_sha256"] for c in calls).encode()).hexdigest() if calls else None
        base = dict(
            synthetic=self.synthetic, executor_mode=self.mode, unit_key=unit["unit_key"], attempt=attempt, task_id=unit["task_id"],
            goal_template=meta["goal_template"], scene=meta["scene"], family=meta["family"], exposure_policy=self.exposure_policy,
            oracle_id=meta.get("oracle_id", meta["oracle_sha256"]), oracle_sha256=meta["oracle_sha256"],
            checkpoint_digest=meta["checkpoint_digest"], post_detour_digest=meta["post_detour_digest"], condition=condition,
            seed=seed, runtime_identity_sha256=self.runtime_identity_sha256,
            natural_top3_skill_ids=list(audit.get("natural_top3_skill_ids", [])),
            natural_top3_scores=list(audit.get("natural_top3_scores", [])),
            injected_skill_ids=injected, injected_skill_count=len(injected),
            analysis_only_injected_labels=[self.analysis_labels.get(s) for s in injected],
            filler_corpus_sha256=audit.get("filler_corpus_sha256"), filler_sentence_indices=list(audit.get("filler_sentence_indices", [])),
            memory_block_sha256=audit.get("memory_block_sha256"),
            first_recovery_prompt_sha256=calls[0]["prompt_sha256"] if calls else None, prompt_sequence_sha256=sequence,
            tokens={"raw_text_tokens": calls[0]["local_raw_tokens"] if calls else None,
                    "memory_block_tokens": audit.get("memory_block_tokens"),
                    "chat_template_tokens": calls[0]["local_chat_template_tokens"] if calls else None,
                    "provider_prompt_tokens": calls[0]["prompt_eval_count"] if calls else None},
            provider_calls=calls,
            requested_provider_options=ollama_chat_payload(getattr(self.driver, "model_name", ""), "", seed)["options"],
            total_action_cap=cap, prefix_actions=prefix, detour_actions=detour, recovery_action_allowance=allowance,
            inference_latency_seconds=round(sum(c["seconds"] or 0.0 for c in calls), 6), wall_clock_seconds=wall,
            not_applicable=(["memory"] if condition in ("D0", "D0U", *DIAGNOSTIC_CONDITIONS) else []) + list(audit.get("not_applicable", [])),
        )
        if self.mode == "arm":
            base["memory_source"] = MEMORY_SOURCE[condition]
            base["execution_provenance"] = {**self.execution_provenance, "attempt_dir": str(attempt_dir) if attempt_dir else None,
                                            "bridge_action_limit": prefix + detour + allowance}
        if outcome is not None:
            fields = dict(base, recovery_actions=outcome["recovery_actions"], recovery_trace=trace, success=bool(outcome["success"]),
                          termination_reason=outcome["termination_reason"], invalid_actions=outcome["invalid_actions"],
                          outcome_status="VALID", failure_detail=None)
        else:
            fields = dict(base, recovery_actions=sum(1 for s in trace if s["action"] is not None), recovery_trace=trace, success=None,
                          termination_reason=failure["reason"], invalid_actions=sum(len(s["rejections"]) for s in trace),
                          outcome_status=failure["status"], failure_detail=failure["detail"])
        maker = make_record if self.mode == "arm" else make_diagnostic_record
        return maker(**fields)


class InterruptionRecordFactory(FiveArmEpisodeExecutor):
    """Coordinator-side builder of MISSING_INFRASTRUCTURE records for attempts that never reported (revoked on restart
    or lease expired).  It has no driver and can never execute an episode."""

    def __init__(self, *, tasks: Mapping[str, EvaluationTask], task_meta: Mapping[str, Mapping[str, Any]], synthetic: bool,
                 runtime_identity_sha256: str, exposure_policy: str, model_name: str,
                 execution_provenance: Mapping[str, Any] | None = None) -> None:
        from types import SimpleNamespace
        self.mode, self.tasks, self.task_meta, self.synthetic = "arm", dict(tasks), dict(task_meta), synthetic
        self.runtime_identity_sha256, self.exposure_policy, self.analysis_labels = runtime_identity_sha256, exposure_policy, {}
        self.driver = SimpleNamespace(model_name=model_name)
        self.execution_provenance = {"mode": "parallel-coordinator", "host": socket.gethostname(), "pid": os.getpid(),
                                     **dict(execution_provenance or {})}

    def __call__(self, unit: Mapping[str, Any], *, attempt: int, attempt_dir: Path) -> dict[str, Any]:
        raise ProtocolViolation("the interruption-record factory cannot execute episodes")

"""Frozen RQ1 acquisition execution protocol.

Approved as pre-run Decision 007 on 2026-09-13, before any scientific
acquisition data existed.  Changing a value requires a new decision record,
new freezes, and a new acquisition run.  ``configs/acquisition/protocol.yaml``
must equal :func:`protocol_definition`.
"""
from __future__ import annotations

from typing import Any

from rq1.experiment.models import canonical_hash
from rq1.hermes.episode_driver import ACTION_SELECTION_PROTOCOL, INFERENCE_SEED, MAX_SELECTION_ATTEMPTS
from rq1.retrieval.text import SKILL_TEXT_VERSION
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.selection import (
    ACQUISITION_HARD_CAP,
    ACQUISITION_HARD_CAP_PER_FAMILY,
    ACQUISITION_INITIAL_TASKS,
    ACQUISITION_TASKS_PER_FAMILY,
)

ACQUISITION_POLICY_VERSION = "acquisition-execution-v1"
DECISION_RECORD = "docs/decisions/007-acquisition-execution-policy.md"
PROTOCOL_CONFIG = "configs/acquisition/protocol.yaml"
ACQUISITION_SPLIT = "train"
ACQUISITION_PROFILE = "rq1-acquisition"
ACQUISITION_MODEL = "hermes3:8b"
ACQUISITION_TEMPERATURE = 0
ACQUISITION_ACTION_BUDGET = 50
TASK_SELECTION_VERSION = "task-selection-v1"
# The repository's default task-proposal seed (`rq1.cli tasks propose --seed`).
TASK_SELECTION_SEED = 1
TASK_SELECTION_BALANCING = "round_robin_families"
# The observed ALFWorld 0.4.2 adapter does not consume an environment seed, but
# the bridge contract requires one, so a constant is sent for every task.
ACQUISITION_ENVIRONMENT_SEED = 0
MAX_CANDIDATES_PER_SUCCESS = 1
SKILL_GENERATION_PROTOCOL = "post-success-skill-v1"
SKILL_GENERATION_ATTEMPTS = 1
VALIDATION_RULES = (
    "format_title_body_or_no_skill",
    "no_task_id_pattern",
    "no_room_number",
    "no_object_instance_number",
    "no_source_task_identifier",
    "no_executed_instance_action_verbatim",
    "no_exact_normalized_duplicate",
)


def protocol_definition() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_version": ACQUISITION_POLICY_VERSION,
        "decision_record": DECISION_RECORD,
        "frozen_before_scientific_acquisition": True,
        "split": ACQUISITION_SPLIT,
        "task_families": list(TASK_FAMILIES),
        "initial_task_count": ACQUISITION_INITIAL_TASKS,
        "tasks_per_family": ACQUISITION_TASKS_PER_FAMILY,
        "automatic_extension": False,
        "later_hard_cap": {"total": ACQUISITION_HARD_CAP, "per_family": ACQUISITION_HARD_CAP_PER_FAMILY},
        "task_selection": {
            "version": TASK_SELECTION_VERSION,
            "seed": TASK_SELECTION_SEED,
            "requested_count": ACQUISITION_INITIAL_TASKS,
            "balancing": TASK_SELECTION_BALANCING,
            "information": "train_metadata_only",
        },
        "acquisition_action_budget": ACQUISITION_ACTION_BUDGET,
        "action_budget_varies_by_task_or_family": False,
        "environment_seed": ACQUISITION_ENVIRONMENT_SEED,
        "fresh_session_per_task": True,
        "profile": ACQUISITION_PROFILE,
        "model": ACQUISITION_MODEL,
        "inference": {
            "temperature": ACQUISITION_TEMPERATURE,
            "seed": INFERENCE_SEED,
            "action_selection_protocol": ACTION_SELECTION_PROTOCOL,
            "max_selection_attempts": MAX_SELECTION_ATTEMPTS,
        },
        "scientific_retrieval_during_acquisition": False,
        "skill_creation": {
            "author": "same_experimental_agent",
            "separate_summariser": False,
            "trigger": "successful_train_episode_only",
            "mode": "create_only",
            "patching": False,
            "max_candidates_per_successful_episode": MAX_CANDIDATES_PER_SUCCESS,
            "failed_episode_candidates": 0,
            "infrastructure_failure_candidates": 0,
            "generation_protocol": SKILL_GENERATION_PROTOCOL,
            "generation_attempts": SKILL_GENERATION_ATTEMPTS,
            "prompt": "hermes/prompts/post_success_learning.md",
            "validation_prompt": "hermes/prompts/skill_validation.md",
            "text_format": SKILL_TEXT_VERSION,
            "validation_rules": list(VALIDATION_RULES),
        },
        "duplicate_policy": {
            "acquisition_rejection": "exact_normalized_duplicate_only",
            "normalization": "skill-text-v1 whitespace normalization: trim and collapse whitespace runs, including line endings; case preserved",
            "semantic_deduplication": False,
            "embedding_deduplication": False,
            "llm_duplicate_judge": False,
            "near_duplicates": "preserved",
            "retrospective_library_deduplication": False,
        },
        "persistence": {
            "completion_authority": "results.jsonl",
            "skill_pool": "append-only; rebuilt from committed completed results in queue order; skill_pool.json is a derived atomic snapshot",
            "checkpoint": "atomic after every unit",
            "infrastructure_failure": "failed result without skill; run halts for chronological retry-failed",
            "resume": "fail closed on commit, runtime, configuration, queue, or skill-pool drift",
        },
    }


def protocol_sha256() -> str:
    return canonical_hash(protocol_definition())

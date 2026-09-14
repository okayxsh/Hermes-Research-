"""Pre-evaluation amendment (Decision 012) and the frozen controlled-recovery evaluation protocol.

The acquisition ended at the pre-approved hard cap of 240 TRAIN episodes before any
evaluation, with a raw pool (11/8/4/9/15/3 skills per family) that cannot fill
Clean-24 / Accum-60 / Accum-96.  This protocol keeps the approved structure (NoLib,
a human-validated core, chronological accumulated extras, four nested balanced
conditions, 30 valid_unseen tasks x 3 seeds) at the feasible sizes 0 / 6 / 12 / 18.
``configs/evaluation/amended.yaml`` must equal :func:`evaluation_protocol_definition`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rq1.acquisition.environment import SBERT_MODEL, SBERT_REVISION
from rq1.experiment.models import canonical_hash
from rq1.hermes.episode_driver import (
    ACTION_HISTORY_POLICY,
    ACTION_INDEX_PARSING_POLICY,
    ACTION_SELECTION_PROTOCOL,
    DETERMINISM_POLICY,
    EXPERIMENT_MODEL,
    INFRASTRUCTURE_FAILURE_POLICY,
    INITIAL_OBSERVATION_POLICY,
    INVENTORY_POLICY,
    MAX_SELECTION_ATTEMPTS,
    MODEL_CONTEXT_LENGTH,
    MODEL_OUTPUT_FAILURE_POLICY,
    MODEL_QUANTIZATION,
    MODEL_TIMEOUT_SECONDS,
    OUTPUT_TOKEN_CAP,
)
from rq1.retrieval.query import CANONICAL_FAILURE_MESSAGE, INVENTORY_NOT_OBSERVED_MARKER, QUERY_TEMPLATE_VERSION, query_template_hash
from rq1.retrieval.text import SKILL_TEXT_VERSION
from rq1.skills.library import TASK_FAMILIES

AMENDMENT_VERSION = "pre-evaluation-library-amendment-v1"
EVALUATION_POLICY_VERSION = "controlled-recovery-evaluation-v1"
AMENDMENT_DECISION_RECORD = "docs/decisions/012-pre-evaluation-library-amendment.md"
PROTOCOL_DOCUMENT = "docs/EVALUATION_AMENDED_PROTOCOL.md"
RUBRIC_DOCUMENT = "docs/SKILL_QUALITY_RUBRIC.md"
RELEVANCE_RUBRIC_DOCUMENT = "docs/RETRIEVAL_RELEVANCE_RUBRIC.md"
PROTOCOL_CONFIG = "configs/evaluation/amended.yaml"
MODEL_DIGEST = "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
SBERT_SNAPSHOT_SHA256 = "bbfecb04a834241144af31e9200a3dd0df83ff0fbb2453d56a4bb383a6bfbe11"
EMBEDDING_DIMENSION = 768

CONDITIONS = ("NoLib", "Core-6", "Accum-12", "Accum-18")
PER_FAMILY_SIZES = {"NoLib": 0, "Core-6": 1, "Accum-12": 2, "Accum-18": 3}
LIBRARY_SIZES = {condition: size * len(TASK_FAMILIES) for condition, size in PER_FAMILY_SIZES.items()}
CORE_PER_FAMILY = 1
RAW_POOL_HASH = "7fb4b72df6f197bd097e79ae346e5ad6498efc24ed6aa6b30c63408daf1196ba"
RAW_POOL_SNAPSHOT = "artifacts/acquisition-closeout/rq1-acquisition-240-final/final-skill-pool-50.json"
COMBINED_CLOSEOUT_MANIFEST = "artifacts/acquisition-closeout/rq1-acquisition-240-final/combined-closeout-manifest.json"

EVALUATION_SPLIT = "valid_unseen"
TASKS_PER_FAMILY = 5
EVALUATION_TASK_COUNT = TASKS_PER_FAMILY * len(TASK_FAMILIES)
EVALUATION_SEEDS = (11, 29, 47)
EVALUATION_UNIT_COUNT = EVALUATION_TASK_COUNT * len(EVALUATION_SEEDS) * len(CONDITIONS)
TOTAL_EPISODE_ACTION_BUDGET = 50
RETRIEVAL_TOP_K = 3
SHARD_COUNT = 6
EVALUATION_RUN_PREFIX = "rq1-evaluation-gemma4-12b"
EVALUATION_PROFILE = "rq1-evaluation"
MANIFEST_TYPE = "evaluation-amended"

PREPARATION_DIR = Path("artifacts") / "evaluation-preparation"
FROZEN_TASK_DIR = Path("artifacts") / "task_manifests" / "frozen_evaluation"
PROPOSAL_ARCHIVE_DIR = Path("artifacts") / "task_manifests" / "evaluation_proposal_archive"
LIBRARY_DIR = Path("artifacts") / "evaluation-libraries"
APPROVAL_DIR = Path("artifacts") / "approvals" / "evaluation-amended"
CHECK_BASE = Path("artifacts") / "prelaunch" / "evaluation-check"
CHECK_PREFIX = "prelaunch-evaluation-check-"
CHECK_REPORT = "evaluation-check-report.json"
PREFLIGHT_BASE = Path("artifacts") / "prelaunch" / "evaluation-preflight"
REPORT_DIR = Path("artifacts") / "evaluation-reports"
CORE_REVIEW_FILE = Path("artifacts") / "skill-validation" / "rq1-acquisition-240" / "core-validation-fast.csv"
AMENDMENT_FREEZE = Path("artifacts") / "freezes" / "evaluation-amendment-freeze.json"
ENVIRONMENT_FREEZE = Path("artifacts") / "freezes" / "evaluation-environment-freeze.json"
PROTOCOL_FREEZE = Path("artifacts") / "freezes" / "evaluation-protocol-freeze.json"

TASK_SELECTION = {
    "version": "task-selection-v2",
    "rule": "five_longest_handcoded_expert_routes_per_family",
    "length_definition": "number of actions in the ALFWorld 0.4.2 hand-coded expert (TextWorld) route from reset to success",
    "tie_break": "task_id ascending",
    "tasks_per_family": TASKS_PER_FAMILY,
    "replacement": "next-longest task of the same family when no checkpoint admits a validated controlled failure before freeze",
    "information": "valid_unseen task metadata and hand-coded expert routes only; no model, agent, or acquisition outcome",
}
CHECKPOINT_POLICY = {
    "version": "midpoint-navigation-detour-v1",
    "candidate_order": "reference route of length L: index L//2, then L//2+1, L//2-1, L//2+2, ... (prefix non-empty, continuation non-empty)",
    "eligible_checkpoint": "the next reference action is navigation ('go to ...')",
    "perturbation": "controlled reversible action perturbation: dispatch the lexicographically first admissible 'go to ...' action that differs from the next reference action",
    "perturbation_requirements": ["legal admissible navigation", "differs from the expected next action", "valid non-terminal transition",
                                  "changes the observation (a different location)", "non-goal and non-destructive (navigation only)",
                                  "deterministic and identical for every condition and seed of the task"],
    "oracle_validation": "in the real bridge: replay prefix -> detour -> remaining reference route (whose first action is navigation, so it rejoins) must complete with ALFWorld success",
    "oracle_visibility": "oracle and reference actions are never shown to the model",
    "object_relocation": False,
    "runtime_verification": "each episode must reproduce the frozen checkpoint and post-detour observable digests; a mismatch fails the unit closed",
    "state_digest": "sha256 of canonical JSON {observation, admissible_actions, step_number} of the bridge state",
}
SCHEDULE_POLICY = {
    "version": "balanced-interleaved-v1",
    "cells": "task x seed (90), ordered by family (canonical order), frozen task order, seed ascending",
    "condition_order_within_cell": "CONDITIONS rotated left by ((cell_index // SHARD_COUNT) mod 4)",
    "shards": "cell_index mod SHARD_COUNT; each shard holds 15 cells x all 4 conditions (60 units); conditions stay together per cell",
    "outcome_dependent_reordering": False,
}


def evaluation_protocol_definition() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "amendment_version": AMENDMENT_VERSION,
        "policy_version": EVALUATION_POLICY_VERSION,
        "decision_record": AMENDMENT_DECISION_RECORD,
        "frozen_before_evaluation": True,
        "amendment": {
            "reason": "acquisition ended at the pre-approved hard cap of 240 before evaluation; the raw pool cannot fill Clean-24/Accum-60/Accum-96",
            "raw_pool_hash": RAW_POOL_HASH,
            "raw_skills_per_family": {"pick_and_place": 11, "pick_two_and_place": 8, "look_at_object": 4, "clean_and_place": 9, "heat_and_place": 15, "cool_and_place": 3},
            "original_conditions": {"NoLib": 0, "Clean-24": 24, "Accum-60": 60, "Accum-96": 96},
            "original_conditions_feasible": False,
            "evaluation_outcomes_observed": False,
            "skills_fabricated": False,
            "acquisition_episodes_discarded": False,
            "quality_rule_loosened": False,
            "semantic_deduplication": False,
        },
        "libraries": {
            "conditions": list(CONDITIONS),
            "per_family_sizes": dict(PER_FAMILY_SIZES),
            "library_sizes": dict(LIBRARY_SIZES),
            "rule": "A",
            "core": "per family, the chronologically earliest skill marked PASS under skill-quality-rubric-v1",
            "extras": "per family, the chronologically earliest acquired skills not in the core, whether or not reviewed or passing",
            "nesting": "Core-6 subset Accum-12 subset Accum-18, identical core in every non-empty library",
            "balanced_per_family": True,
            "semantic_deduplication": False,
            "near_duplicates": "preserved",
            "source": "the raw 50-skill acquisition pool only",
            "rubric": RUBRIC_DOCUMENT,
            "skill_text_format": SKILL_TEXT_VERSION,
        },
        "tasks": {
            "split": EVALUATION_SPLIT,
            "task_families": list(TASK_FAMILIES),
            "tasks_per_family": TASKS_PER_FAMILY,
            "task_count": EVALUATION_TASK_COUNT,
            "selection": dict(TASK_SELECTION),
        },
        "controlled_failure": {**CHECKPOINT_POLICY, "canonical_failure_message": CANONICAL_FAILURE_MESSAGE},
        "matrix": {
            "conditions": list(CONDITIONS),
            "seeds": list(EVALUATION_SEEDS),
            "units": EVALUATION_UNIT_COUNT,
            "condition_labels_visible_to_model": False,
            "schedule": dict(SCHEDULE_POLICY),
            "shard_count": SHARD_COUNT,
        },
        "episode": {
            "fresh_session_per_unit": True,
            "order": ["reset", "replay frozen checkpoint prefix", "controlled detour", "verify frozen digests",
                      "build failure context", "single retrieval (library conditions)", "inject recovery memory", "recovery actions"],
            "total_action_budget": TOTAL_EPISODE_ACTION_BUDGET,
            "recovery_action_budget": "total_action_budget - checkpoint prefix length - 1 (the detour), fixed per task",
            "skill_writes": False,
            "post_success_learning": False,
        },
        "retrieval": {
            "model": SBERT_MODEL,
            "revision": SBERT_REVISION,
            "snapshot_sha256": SBERT_SNAPSHOT_SHA256,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "normalized_embeddings": True,
            "ranking": "cosine similarity, descending; ties by skill_id",
            "top_k": RETRIEVAL_TOP_K,
            "timing": "exactly once, immediately after the controlled failure and before the first recovery action",
            "pre_failure_retrieval": False,
            "repeated_or_per_step_retrieval": False,
            "nolib": "no retrieval; the same recovery boundary injects an explicit no_retrieval marker",
            "query_version": QUERY_TEMPLATE_VERSION,
            "query_template_sha256": query_template_hash(),
            "query_fields": ["task goal", "current observation", "inventory field", "canonical failure message"],
            "inventory_unobserved_marker": INVENTORY_NOT_OBSERVED_MARKER,
            "action_history_in_query": False,
            "logged": ["candidate skill ids", "ranks", "cosine scores", "query text hash"],
            "scores_visible_to_model": False,
        },
        "recovery_memory": {
            "placement": "one structured RECOVERY MEMORY block in every post-failure action-selection prompt; never the system prompt",
            "content": "rank, skill_id, and skill text of each top-3 skill; no cosine score",
            "nolib": "the same block with no_retrieved_skills_available = true",
        },
        "model": {
            "tag": EXPERIMENT_MODEL,
            "digest": MODEL_DIGEST,
            "quantization": MODEL_QUANTIZATION,
            "temperature": 0,
            "think": False,
            "num_predict": OUTPUT_TOKEN_CAP,
            "num_ctx": MODEL_CONTEXT_LENGTH,
            "inference_seed": "the unit's evaluation replicate seed (11, 29, or 47), identical across conditions",
            "determinism": DETERMINISM_POLICY,
            "action_selection_protocol": ACTION_SELECTION_PROTOCOL,
            "action_history": ACTION_HISTORY_POLICY,
            "initial_observation": INITIAL_OBSERVATION_POLICY,
            "inventory": INVENTORY_POLICY,
            "action_index_parsing": ACTION_INDEX_PARSING_POLICY,
            "max_selection_attempts": MAX_SELECTION_ATTEMPTS,
            "model_timeout_seconds": MODEL_TIMEOUT_SECONDS,
            "model_output_failure_policy": MODEL_OUTPUT_FAILURE_POLICY,
            "infrastructure_failure_policy": INFRASTRUCTURE_FAILURE_POLICY,
        },
        "metrics": {
            "primary": "conditional_recovery_rate",
            "eligible_unit": "a unit that reproduced the frozen checkpoint and post-detour digests and finished its post-failure phase without an infrastructure failure",
            "recovery_success": "ALFWorld reports done and won within the recovery action budget after the controlled failure",
            "conditional_recovery_rate": "recovery successes / eligible units, per condition",
            "task_completion_rate": "successful units / all scheduled units of the condition (equal to the recovery success count because the checkpoint precedes completion)",
            "recovery_latency_actions": "post-failure environment actions up to and including the successful action (successful units only)",
            "recovery_latency_seconds": "wall time from recovery-memory injection to the successful step's bridge result (successful units only)",
            "post_failure_actions": "environment actions dispatched after the detour (all eligible units)",
            "invalid_action_selections": "rejected action-selection attempts after the failure",
            "retries": "re-asked action-selection attempts after a rejection; selection_exhausted marks three rejections at one decision",
            "infrastructure_failure_rate": "units whose final attempt failed for a genuine execution reason / scheduled units",
            "precision_at_3": "adjudicated human-relevant skills among the retrieved top-3 / 3, per retrieval",
            "retrieval_noise": "1 - precision_at_3",
            "nolib_retrieval_metrics": "reported separately as no retrieval, never coerced to 0 or 1",
            "breakdowns": ["condition", "task family", "seed"],
            "uncertainty": "percentile bootstrap, 2000 replicates, seed 20260806, resampling task-seed cells",
            "association": "Spearman rank correlation between per-retrieval noise and recovery outcomes; descriptive and non-causal",
        },
        "relevance_labelling": {
            "rubric": RELEVANCE_RUBRIC_DOCUMENT,
            "raters": 2,
            "labels": ["RELEVANT", "IRRELEVANT"],
            "independent": True,
            "blinded_to": ["condition", "outcome", "cosine score"],
            "agreement": "Cohen's kappa on the original independent labels",
            "adjudication": "after independent rating; final Precision@3 uses adjudicated labels",
        },
    }


def evaluation_protocol_sha256() -> str:
    return canonical_hash(evaluation_protocol_definition())

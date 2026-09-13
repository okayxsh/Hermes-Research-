"""Frozen balanced acquisition extension, logical positions 181-240 (Decision 011).

The initial acquisition protocol froze 180 TRAIN tasks (30 per family) with a
later hard cap of 240 tasks (40 per family) and ``automatic_extension: false``,
so the cap is reached only by an explicit decision.  This module is that
activation for the completed parent run: 60 more TRAIN tasks (10 per family)
under the identical scientific settings, continuing the parent's exact final
skill pool in a separate run.  ``configs/acquisition/extension.yaml`` must equal
:func:`extension_protocol_definition`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rq1.acquisition.protocol import (
    ACQUISITION_ACTION_BUDGET,
    ACQUISITION_MODEL,
    ACQUISITION_TEMPERATURE,
    DECISION_RECORDS,
    MAX_CANDIDATES_PER_SUCCESS,
    SKILL_GENERATION_ATTEMPTS,
    SKILL_GENERATION_PROTOCOL,
    TASK_SELECTION_BALANCING,
    TASK_SELECTION_SEED,
    TASK_SELECTION_VERSION,
    protocol_definition,
    protocol_sha256,
)
from rq1.experiment.models import canonical_hash
from rq1.hermes.episode_driver import (
    ACTION_HISTORY_POLICY,
    ACTION_INDEX_PARSING_POLICY,
    ACTION_SELECTION_PROTOCOL,
    INFERENCE_SEED,
    INITIAL_OBSERVATION_POLICY,
    INVENTORY_POLICY,
    MAX_SELECTION_ATTEMPTS,
    MODEL_CONTEXT_LENGTH,
    MODEL_QUANTIZATION,
    OUTPUT_TOKEN_CAP,
)
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.selection import (
    ACQUISITION_HARD_CAP,
    ACQUISITION_HARD_CAP_PER_FAMILY,
    ACQUISITION_INITIAL_TASKS,
    ACQUISITION_TASKS_PER_FAMILY,
)

EXTENSION_POLICY_VERSION = "acquisition-extension-v1"
EXTENSION_DECISION_RECORD = "docs/decisions/011-acquisition-extension-181-240.md"
EXTENSION_DECISION_RECORDS = (*DECISION_RECORDS, EXTENSION_DECISION_RECORD)
EXTENSION_PROTOCOL_CONFIG = "configs/acquisition/extension.yaml"
EXTENSION_MANIFEST_TYPE = "acquisition-extension"
EXTENSION_RUN_ID = "rq1-acquisition-gemma4-12b-ext-181-240"
EXTENSION_TASK_COUNT = ACQUISITION_HARD_CAP - ACQUISITION_INITIAL_TASKS
EXTENSION_TASKS_PER_FAMILY = ACQUISITION_HARD_CAP_PER_FAMILY - ACQUISITION_TASKS_PER_FAMILY
# The model digest the parent run used, as enforced by its approved environment freeze.
FROZEN_MODEL_DIGEST = "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"

EXTENSION_PROPOSAL_DIR = Path("artifacts") / "task_manifests" / "extension_proposals"
EXTENSION_PROPOSAL_ARCHIVE_DIR = Path("artifacts") / "task_manifests" / "extension_proposal_archive"
EXTENSION_FROZEN_DIR = Path("artifacts") / "task_manifests" / "frozen_extension"
EXTENSION_ENVIRONMENT_FREEZE = Path("artifacts") / "freezes" / "acquisition-extension-environment-freeze.json"
EXTENSION_PROTOCOL_FREEZE = Path("artifacts") / "freezes" / "acquisition-extension-protocol-freeze.json"
EXTENSION_APPROVAL_DIR = Path("artifacts") / "approvals" / "acquisition-extension"
EXTENSION_STATE_DIR = Path("artifacts") / "acquisition-extension"
EXTENSION_CHECK_BASE = Path("artifacts") / "prelaunch" / "acquisition-extension-check"
EXTENSION_CHECK_PREFIX = "prelaunch-acquisition-extension-check-"
EXTENSION_CHECK_REPORT = "acquisition-extension-check-report.json"
EXTENSION_PREFLIGHT_BASE = Path("artifacts") / "prelaunch" / "extension-preflight"


@dataclass(frozen=True)
class ParentReference:
    """The completed parent acquisition, as certified by its closeout manifest."""

    run_id: str
    results_directory: str
    completed_units: int
    successful_units: int
    repository_commit: str
    queue_sha256: str
    task_manifest: str
    task_manifest_sha256: str
    pool_size: int
    pool_hash: str
    closeout_manifest: str
    closeout_manifest_sha256: str

    @property
    def first_logical_index(self) -> int:
        return self.completed_units + 1

    @property
    def last_logical_index(self) -> int:
        return self.completed_units + EXTENSION_TASK_COUNT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PARENT = ParentReference(
    run_id="rq1-acquisition-gemma4-12b",
    results_directory="results/final/rq1-acquisition-gemma4-12b",
    completed_units=180,
    successful_units=86,
    repository_commit="8bd452e76120da21d721c6e894d1ce5af4912ca9",
    queue_sha256="1f401869ea877969072f4d73e314db5841ff3f9aae875057ec3e4b5fe468ee09",
    task_manifest="artifacts/task_manifests/frozen/acquisition-b64c383e0f335b90.json",
    task_manifest_sha256="717aa855c4947c2503aa019d014a474b77fbabcf0d45825de8d1aa41ab414ef2",
    pool_size=34,
    pool_hash="11579cfe7ae0b232f233a9527fe96bae5985b50416b7408f2be89af2a4031341",
    closeout_manifest="artifacts/acquisition-closeout/rq1-acquisition-gemma4-12b/closeout-manifest.json",
    closeout_manifest_sha256="98c54bb3d33816da4771d7185a23a7cf2b739e550d87f59998214bf454d629de",
)


def extension_selection_policy(parent: ParentReference = PARENT) -> dict[str, Any]:
    """The frozen selection recomputed at the combined count; the extension is its tail."""
    return {
        "version": TASK_SELECTION_VERSION,
        "seed": TASK_SELECTION_SEED,
        "requested_count": parent.completed_units + EXTENSION_TASK_COUNT,
        "balancing": TASK_SELECTION_BALANCING,
        "selected_positions": {"first": parent.first_logical_index, "last": parent.last_logical_index},
    }


def extension_protocol_definition(parent: ParentReference = PARENT) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_version": EXTENSION_POLICY_VERSION,
        "decision_record": EXTENSION_DECISION_RECORD,
        "frozen_before_extension_acquisition": True,
        "activation": {
            "contingency": "later_hard_cap of the frozen acquisition protocol",
            "initial_task_count": ACQUISITION_INITIAL_TASKS,
            "initial_tasks_per_family": ACQUISITION_TASKS_PER_FAMILY,
            "hard_cap": {"total": ACQUISITION_HARD_CAP, "per_family": ACQUISITION_HARD_CAP_PER_FAMILY},
            "automatic_extension": False,
            "trigger": "planned accumulated-library quotas unmet by the completed initial acquisition",
            "final_evaluation_started": False,
            "evaluation_outcomes_exist": False,
            "parent_results_discarded_or_rerun": False,
        },
        "parent": parent.to_dict(),
        "inherited_acquisition_protocol_sha256": protocol_sha256(),
        "inherited_acquisition_protocol": protocol_definition(),
        "scientific_settings_changed": False,
        "split": "train",
        "task_families": list(TASK_FAMILIES),
        "extension_task_count": EXTENSION_TASK_COUNT,
        "extension_tasks_per_family": EXTENSION_TASKS_PER_FAMILY,
        "logical_acquisition_positions": {"first": parent.first_logical_index, "last": parent.last_logical_index},
        "combined_task_count_after_completion": parent.completed_units + EXTENSION_TASK_COUNT,
        "combined_tasks_per_family_after_completion": ACQUISITION_TASKS_PER_FAMILY + EXTENSION_TASKS_PER_FAMILY,
        "task_selection": {
            **extension_selection_policy(parent),
            "information": "train_metadata_only",
            "rule": "recompute the frozen selection at the combined count; positions 1 to the parent count must equal the parent frozen queue in order; the remaining positions are the extension queue",
            "parent_queue_tasks_excluded": True,
            "difficulty_or_outcome_based_selection": False,
        },
        "starting_skill_pool": {
            "source": "parent results.jsonl (authority), verified against the parent skill_pool.json snapshot, the closeout manifest, and the recorded hash",
            "size": parent.pool_size,
            "hash": parent.pool_hash,
            "empty_start": False,
        },
        "continuation": {
            "persistence": "append-only; separate run directory; parent results, checkpoint, and skill pool are read-only",
            "skill_mode": "create_only",
            "max_candidates_per_successful_episode": MAX_CANDIDATES_PER_SUCCESS,
            "first_new_pool_index": parent.pool_size + 1,
            "duplicate_rejection": "exact_normalized_duplicate_only, against every parent and extension accepted skill",
            "parent_skills_modified": False,
            "parent_skills_deleted": False,
            "semantic_deduplication": False,
            "near_duplicates": "preserved",
            "retrospective_deduplication": False,
            "provenance_rewritten": False,
            "patching": False,
        },
        "fresh_session_per_task": True,
        "acquisition_action_budget": ACQUISITION_ACTION_BUDGET,
        "model": ACQUISITION_MODEL,
        "model_digest": FROZEN_MODEL_DIGEST,
        "model_quantization": MODEL_QUANTIZATION,
        "inference": {
            "temperature": ACQUISITION_TEMPERATURE,
            "seed": INFERENCE_SEED,
            "think": False,
            "output_token_cap": OUTPUT_TOKEN_CAP,
            "model_context_length": MODEL_CONTEXT_LENGTH,
            "action_selection_protocol": ACTION_SELECTION_PROTOCOL,
            "action_history": ACTION_HISTORY_POLICY,
            "initial_observation": INITIAL_OBSERVATION_POLICY,
            "inventory": INVENTORY_POLICY,
            "action_index_parsing": ACTION_INDEX_PARSING_POLICY,
            "max_selection_attempts": MAX_SELECTION_ATTEMPTS,
        },
        "skill_generation": {
            "author": "same_experimental_agent",
            "protocol": SKILL_GENERATION_PROTOCOL,
            "attempts": SKILL_GENERATION_ATTEMPTS,
        },
        "scientific_retrieval_during_acquisition": False,
        "failures": {
            "model_output_failure": "invalid or output-capped responses consume action-selection attempts inside the episode; a scientific outcome, never an infrastructure failure",
            "infrastructure_failure": "genuine execution failures only; failed result without skill; run halts for chronological retry-failed",
        },
        "resume": {
            "checkpoint": "atomic after every unit; results.jsonl is the recovery authority",
            "completed_units_rerun": False,
            "fail_closed_on": "commit, runtime, configuration, queue, parent, or skill-pool drift",
        },
    }


def extension_protocol_sha256(parent: ParentReference = PARENT) -> str:
    return canonical_hash(extension_protocol_definition(parent))

"""Lightweight validation for recovery manifests without extra dependencies."""
from __future__ import annotations
from typing import Mapping

def validate_checkpoint_payload(value: Mapping[str, object]) -> list[str]:
    required = {"checkpoint_id", "task_id", "split", "task_family", "prefix_actions", "prefix_length", "observable_state_digest", "validation_result"}
    errors = [f"missing {field}" for field in sorted(required - set(value))]
    if isinstance(value.get("prefix_actions"), list) and value.get("prefix_length") != len(value["prefix_actions"]): errors.append("prefix_length does not match prefix_actions")
    return errors

def validate_perturbation_payload(value: Mapping[str, object]) -> list[str]:
    required = {"perturbation_id", "checkpoint_id", "type", "observable_post_state_digest", "solvable", "visible_message"}
    return [f"missing {field}" for field in sorted(required - set(value))]

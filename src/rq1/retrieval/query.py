"""Frozen, deterministic post-failure query template and canonical constants."""
from __future__ import annotations

import hashlib

QUERY_TEMPLATE_VERSION = "query-v1"
EMPTY_INVENTORY_MARKER = "<empty>"

# The perturbation layer must emit exactly this observable message so the
# post-failure query is reproducible across every memory condition.
CANONICAL_FAILURE_MESSAGE = (
    "The required object is no longer where it was expected. "
    "Reassess the current state and continue."
)

# Exact field order is a frozen scientific choice. No action history, library
# condition, future information, or new object location may be added.
_QUERY_TEMPLATE = (
    "TASK:\n{task_instruction}\n"
    "OBSERVATION:\n{observation}\n"
    "INVENTORY:\n{inventory}\n"
    "FAILURE:\n{failure_message}"
)


def _normalize(value: str) -> str:
    return " ".join(value.split())


def build_query_text(
    *,
    task_instruction: str,
    observation: str,
    inventory: tuple[str, ...] | list[str] | None,
    failure_message: str = CANONICAL_FAILURE_MESSAGE,
) -> str:
    items = tuple(_normalize(item) for item in (inventory or ()))
    inventory_text = ", ".join(items) if items else EMPTY_INVENTORY_MARKER
    return _QUERY_TEMPLATE.format(
        task_instruction=_normalize(task_instruction),
        observation=_normalize(observation),
        inventory=inventory_text,
        failure_message=_normalize(failure_message),
    )


def query_template_hash() -> str:
    return hashlib.sha256(_QUERY_TEMPLATE.encode("utf-8")).hexdigest()

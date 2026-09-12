"""Deterministic, frozen skill-retrieval text representation."""
from __future__ import annotations

import hashlib

SKILL_TEXT_VERSION = "skill-text-v1"

# Only title and body are embedded. Skill IDs, timestamps, provenance,
# statistics, condition/library names, and evaluation outcomes are excluded.
_SKILL_TEXT_TEMPLATE = "TITLE: {title}\nBODY: {body}"


def _normalize(value: str) -> str:
    return " ".join(value.split())


def build_skill_text(*, title: str, body: str) -> str:
    return _SKILL_TEXT_TEMPLATE.format(
        title=_normalize(title),
        body=_normalize(body),
    )


def skill_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

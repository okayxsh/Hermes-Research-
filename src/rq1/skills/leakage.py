from __future__ import annotations

import re


PROHIBITED_PATTERNS = {
    "task_id": re.compile(r"\b(?:train|valid_seen|valid_unseen)[_-]\d+\b", re.I),
    "room_number": re.compile(r"\b(?:room|kitchen|bedroom|bathroom)\s+\d+\b", re.I),
    "object_instance": re.compile(r"\b[a-z][a-z _-]*\s+\d+\b", re.I),
}


def find_leakage(text: str) -> list[str]:
    return [name for name, pattern in PROHIBITED_PATTERNS.items() if pattern.search(text)]

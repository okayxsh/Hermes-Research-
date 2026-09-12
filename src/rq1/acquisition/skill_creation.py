"""Post-success candidate skill creation by the experimental agent (Decisions 003 and 007).

The same model that acted in the episode writes at most one candidate after a
successful episode.  Acceptance is purely deterministic: there is no model,
embedding, or semantic judgement in validation or duplicate rejection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Sequence

from rq1.retrieval.text import build_skill_text
from rq1.skills.leakage import find_leakage
from rq1.utils.hashing import sha256_file

NO_SKILL = "NO_SKILL"
LEARNING_PROMPT = Path("hermes") / "prompts" / "post_success_learning.md"
VALIDATION_PROMPT = Path("hermes") / "prompts" / "skill_validation.md"
OUTPUT_CONTRACT = (
    "Return exactly NO_SKILL if no reusable skill should be created. Otherwise return exactly:\n"
    "TITLE: <short general title>\n"
    "BODY: <reusable general procedure>\n"
    "Do not use digits, task identifiers, or a copy of the episode actions."
)


def prompt_text(root: Path, relative: Path) -> str:
    lines = (root / relative).read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.startswith("#")).strip()


def prompt_hashes(root: Path) -> dict[str, str]:
    return {relative.as_posix(): sha256_file(root / relative) for relative in (LEARNING_PROMPT, VALIDATION_PROMPT)}


def render_skill_prompt(
    *,
    learning_instruction: str,
    validation_rules: str,
    task_goal: str,
    actions: Sequence[str],
    final_observation: str,
) -> str:
    return "\n\n".join(
        [
            "POST-SUCCESS LEARNING:\n" + learning_instruction,
            "SKILL VALIDATION RULES:\n" + validation_rules,
            "TASK GOAL:\n" + task_goal,
            "SUCCESSFUL EPISODE ACTIONS:\n" + "\n".join(actions),
            "FINAL OBSERVATION:\n" + final_observation,
            OUTPUT_CONTRACT,
        ]
    )


@dataclass(frozen=True)
class ParsedSkill:
    declined: bool
    title: str | None = None
    body: str | None = None


def parse_skill_response(response: str) -> ParsedSkill | None:
    """Accept exactly ``NO_SKILL`` or a ``TITLE:`` line followed by ``BODY:``."""
    if response.strip() == NO_SKILL:
        return ParsedSkill(True)
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith("TITLE:") or not lines[1].startswith("BODY:"):
        return None
    title = lines[0][len("TITLE:"):].strip()
    body = " ".join([lines[1][len("BODY:"):].strip(), *lines[2:]]).strip()
    if not title or not body:
        return None
    return ParsedSkill(False, title, body)


def _flat(value: str) -> str:
    return " ".join(value.split()).casefold()


def validate_skill(
    *,
    title: str,
    body: str,
    source_task_id: str,
    executed_actions: Sequence[str],
    existing_texts: Collection[str],
) -> list[str]:
    """Return deterministic Decision 003 rejection reasons (empty means accepted)."""
    text = build_skill_text(title=title, body=body)
    reasons = [f"leakage_{name}" for name in find_leakage(text)]
    flat = _flat(text)
    relative = source_task_id.split(":", 1)[-1]
    identifiers = {source_task_id, relative, *relative.split("/")}
    if any(item and _flat(item) in flat for item in identifiers):
        reasons.append("source_task_identifier")
    # A raw trajectory command names object instances; generalized wording does not.
    if any(re.search(r"\d", action) and _flat(action) in flat for action in executed_actions):
        reasons.append("executed_instance_action_verbatim")
    # Exact normalized duplicates only; near-duplicates are preserved (Decision 007).
    if text in existing_texts:
        reasons.append("exact_normalized_duplicate")
    return reasons

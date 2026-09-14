"""Fast human core-validation package (Rule A): review each family only until its first PASS."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rq1.evaluation.amended_libraries import by_family, load_raw_pool, raw_feasibility, render_review_csv, review_rows
from rq1.evaluation.amended_protocol import CORE_REVIEW_FILE, RAW_POOL_HASH, RAW_POOL_SNAPSHOT, RUBRIC_DOCUMENT
from rq1.utils.hashing import sha256_file


def write_core_review_package(root: Path) -> dict[str, Any]:
    csv_path = root / CORE_REVIEW_FILE
    md_path = csv_path.with_suffix(".md")
    if csv_path.exists() or md_path.exists():
        return {"ok": False, "status": "blocked", "reason": f"core review package already exists and is immutable: {csv_path}"}
    pool = load_raw_pool(root / RAW_POOL_SNAPSHOT)
    feasibility = raw_feasibility(pool)
    if not feasibility["feasible"]:
        return {"ok": False, "status": "blocked", "reason": "raw pool cannot supply 3 skills per family", "feasibility": feasibility}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(render_review_csv(review_rows(pool)))
    lines = [
        "# Fast core validation (Rule A): six human PASS decisions",
        "",
        "Status: **PENDING HUMAN REVIEW**. Nothing here has been judged or selected.",
        "",
        f"- File to fill: `{CORE_REVIEW_FILE.as_posix()}` (sha256 `{sha256_file(csv_path)}`).",
        f"- Source: the raw 50-skill pool `{RAW_POOL_SNAPSHOT}` (pool hash `{RAW_POOL_HASH}`).",
        f"- Rubric: `{RUBRIC_DOCUMENT}` (skill-quality-rubric-v1). A skill PASSES only if Q1–Q7 are all yes.",
        "",
        "## What to do",
        "",
        "1. Work through one family at a time, in the order the CSV lists its skills (`family_chronological_rank` 1, 2, 3, ...).",
        "2. For each skill you judge, fill `reviewer_quality_pass` (`PASS` or `FAIL`), `reviewer_quality_reason`",
        "   (for example `PASS_Q1_Q7` or `Q4_INSTANCE_IDENTIFIER`), optionally `reviewer_notes`, and `reviewed_at`",
        "   (UTC, for example `2026-09-14T12:00:00Z`).",
        "3. **Stop a family at its first PASS.** That skill is the family's core. Later skills in the family do not need",
        "   review for core selection; leave their review cells empty.",
        "4. If every skill in a family FAILS, record them all. Evaluation then cannot start.",
        "5. Do not edit any other cell, reorder rows, or delete rows. Immutable columns are verified against the pool.",
        "",
        "Under Rule A the extras of Accum-12 and Accum-18 are the earliest remaining acquired skills of each family,",
        "whether or not they were reviewed. Your decisions select only the core.",
        "",
        "## Skills in review order",
        "",
    ]
    for family, entries in by_family(pool).items():
        lines += [f"### {family} ({len(entries)} raw skills)", "", "| Rank | Skill ID | Origin | Skill text |", "|---|---|---|---|"]
        for entry in entries:
            text = entry.text.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {entry.family_rank} | `{entry.skill_id}` | {entry.origin} | {text} |")
        lines.append("")
    lines += ["## Rubric (verbatim)", "", (root / RUBRIC_DOCUMENT).read_text(encoding="utf-8")]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "ok": True,
        "status": "PENDING_HUMAN_REVIEW",
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "instructions": str(md_path),
        "instructions_sha256": sha256_file(md_path),
        "families": feasibility["raw_counts"],
        "required_core_passes": 6,
    }

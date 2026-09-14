# Skill-quality rubric (skill-quality-rubric-v1)

Status: proposed with the pre-evaluation amendment (Decision 012). It takes effect only
when the researcher approves that amendment. A human reviewer applies it; an LLM never
makes the final PASS/FAIL decision.

## Sources

Before this document the repository had no standalone rubric; `rq1.skills.library` only
referred to a "frozen human quality-validation rubric". Every criterion below comes from
these approved records:

- **Decision 003:** "Acquire general skills from successful training episodes only; no
  task IDs, room/object instance numbers, memorised trajectories, duplicate equivalents,
  or positive skills after failure."
- **`hermes/prompts/skill_validation.md`:** "Reject task IDs, room/object instance
  numbers, memorised trajectories, duplicates, and skills created after failures."
- **`hermes/prompts/post_success_learning.md`:** "Create at most one new reusable,
  non-task-specific skill after a successful `train` episode only."
- **Decision 007:** duplicates are rejected only when the normalized skill text is exactly
  equal; near-duplicates are preserved; no semantic, embedding, or LLM deduplication.

The acquisition already enforced the mechanical parts: success-only creation, exact
normalized duplicates, and task-ID, room-number and object-instance patterns. Every skill in
the raw pool passed those checks. The human review is independent and may still FAIL a skill.

## Criteria

A skill **PASSES only if every question Q1–Q7 is answered yes.**

| Question | Asks | Source |
|---|---|---|
| Q1 Generalized | Is the skill reusable, generalized guidance rather than a description tied to one exact trajectory? | Decision 003 "general skills", "memorised trajectories"; post-success prompt "reusable, non-task-specific" |
| Q2 Family relevance | Is it relevant to the skill's recorded ALFWorld task family? | Decision 003 "general skills" (acquired per family) |
| Q3 No task identifiers | Does it avoid task IDs, trial IDs, and source task names? | Decision 003 "no task IDs"; validation prompt |
| Q4 No instance memorization | Does it avoid room or object instance identifiers (for example "cabinet 3") that would leak a specific training trajectory? | Decision 003 "no room/object instance numbers"; validation prompt |
| Q5 Not a trajectory copy | Does it avoid simply copying the raw successful action sequence? | Decision 003 "memorised trajectories"; validation prompt |
| Q6 Actionable | Is the content understandable and actionable as reusable guidance? | Post-success prompt "reusable" skill |
| Q7 Structurally valid | Is the skill non-empty, non-corrupt, and a complete TITLE/BODY skill? | skill-text-v1 format of Decisions 003/007 |

## Failure reason categories

A FAIL records one or more of these codes in `reviewer_quality_reason`, optionally with a
short explanation:

- `Q1_NOT_GENERALIZED`
- `Q2_WRONG_OR_UNRELATED_FAMILY`
- `Q3_TASK_IDENTIFIER`
- `Q4_INSTANCE_IDENTIFIER`
- `Q5_TRAJECTORY_COPY`
- `Q6_NOT_ACTIONABLE`
- `Q7_EMPTY_CORRUPT_OR_MALFORMED`

A PASS may record `PASS_Q1_Q7`.

## Not criteria

- **Uniqueness.** Similarity to other skills is not a quality criterion. Near-duplicates are
  allowed and are judged one at a time. There is no semantic deduplication.
- **Preference.** Usefulness ranking, style or length preference, researcher or model
  preference, expected evaluation performance, and embedding similarity are not criteria.
- **Evaluation information.** No `valid_seen` or `valid_unseen` information is used.

## Procedure

- Review skills in strict chronological acquisition order within each family.
- **For core selection (Rule A),** review a family's skills in order until the first PASS. That
  skill is the family's core skill, and later skills in the family need no review for core
  selection.
- If every skill in a family FAILS, no core exists and evaluation must not start.
- **Recording.** `reviewer_quality_pass` is exactly `PASS` or `FAIL`; `reviewer_quality_reason`
  is required; `reviewed_at` is a UTC ISO-8601 time ending in `Z`.
- **Integrity.** Immutable columns are verified against the raw pool before use.

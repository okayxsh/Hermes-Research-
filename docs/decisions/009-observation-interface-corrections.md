# 009 — Observation-interface corrections (pre-run amendment)

- Status: approved methodology decision by the study owner
- Date: 2026-09-13
- Timing: frozen before the scientific acquisition run; no scientific acquisition data existed when it was made
- Amends: the Decision 008 action-selection prompt (`action-index-history-v1` becomes `action-index-history-v2`) and the unobserved-inventory text of the retrieval query (`query-v1` becomes `query-v2`)
- Applies identically to: acquisition and every recovery condition (NoLib, Clean-24, Accum-60, Accum-96)

## Why

A harness root-cause audit during NON-SCIENTIFIC prelaunch validation found
three interface defects. These are corrections to how observable state reaches
the agent, not responses to scientific results: no scientific acquisition or
evaluation data had been collected. The same audit confirmed that the execution
path, state/action synchronisation, goal extraction, and admissible-action
mapping are correct, and that the expert trajectory succeeds through the real
bridge for all six audited tasks.

1. The prompt and the retrieval query showed the inventory as `<empty>` even
   while the agent was holding an object.
2. The initial room description was visible only at the first decision.
3. The parser rejected responses that contained a valid `ACTION_INDEX` line
   because other prose surrounded it.

## Decisions

1. **Initial observation retained.** Every action-selection prompt contains
   `INITIAL OBSERVATION:` followed by the exact, unmodified observation ALFWorld
   returned at episode reset, placed before the unchanged chronological episode
   history. It is never paraphrased, summarised, or supplemented.
2. **Inventory is never inferred.** ALFWorld 0.4.2 (TextWorld 1.7.0 PDDL engine)
   exposes inventory only as the observation of the normal `inventory` action.
   Executing that action applies a PDDL action (it adds `checked(agent)`),
   increments the move counter, counts against the environment step limit, and
   consumes one of the 50 budgeted actions. TextWorld's read-only inventory
   information is not implemented for these games (it returns null). The
   inventory field of the prompt and of the retrieval query therefore shows the
   fixed marker `<not observed — use the inventory action>`. The harness never
   queries inventory automatically, never spends hidden actions, never reads
   world facts or any other simulator state, never infers inventory from
   previous actions, and keeps no synthetic inventory tracker. An agent that
   wants its inventory must choose `inventory` from the admissible actions and
   pay the normal action cost; the result then appears only as that step's
   observation and in the episode history.
3. **Deterministic, prose-tolerant ACTION_INDEX parsing.** A response is
   accepted when the token `ACTION_INDEX` occurs exactly once in it, on a line
   that is exactly `ACTION_INDEX: <integer>` (surrounding whitespace allowed)
   naming a listed index; prose on other lines is ignored. It is rejected when
   the token is absent, when that line is malformed, when the token occurs more
   than once (even with the same index), or when the index is out of range.
   Action names are never matched from prose, there is no fallback action, and
   the three-attempt retry limit is unchanged.
4. **Retrieval query structure unchanged.** The Sentence-BERT query keeps the
   same template, field order, and template hash: task goal, current
   observation, inventory, and the canonical failure message. Only the text
   shown for an unobserved inventory changed. Neither history nor the initial
   observation is added, and exactly one retrieval still follows the controlled
   failure.
5. **Nothing else changes.** The model, inference seed 42, temperature 0, the
   admissible-action controller, the 50-action budget, the task set, the
   retrieval count, and the recovery methodology are unchanged.

## Prompt format

```text
TASK GOAL:
<frozen goal>

INITIAL OBSERVATION:
<exact ALFWorld reset observation>

EPISODE HISTORY:
Step 1
Action: <action>
Observation: <resulting observation>

Step 2
...

CURRENT OBSERVATION:
<current observation>

CURRENT INVENTORY:
<not observed — use the inventory action>

[RECOVERY MEMORY: recovery only]

ADMISSIBLE ACTIONS:
0. <action>
...

Return exactly:
ACTION_INDEX: <integer>
```

Before the first executed step, the history section reads `(no previous steps)`.

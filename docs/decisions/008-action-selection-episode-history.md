# 008 — Action-selection episode history (pre-run amendment)

- Status: approved methodology decision by the study owner
- Date: 2026-09-13
- Timing: frozen before the scientific acquisition run; no scientific acquisition data existed when it was made
- Amends: the Hermes action-selection prompt (`action-index-v1` becomes `action-index-history-v1`)
- Applies identically to: acquisition and every recovery condition (NoLib, Clean-24, Accum-60, Accum-96)

No scientific result motivated this change. Multi-step ALFWorld control
requires the agent to retain its own observable interaction history rather than
act as a stateless policy.

## Decisions

1. **Observable trajectory.** At every action-selection step Hermes receives the
   complete observable history of the current episode: for every environment
   step already executed, in chronological order, the action taken and the
   resulting ALFWorld observation. There is no rolling window; acquisition
   episodes are capped at 50 actions.
2. **Nothing else.** The history contains only prior actions and their resulting
   observations. It never contains future actions, expert, oracle, or reference
   trajectories, hidden ALFWorld state, condition or library names, retrieval
   scores, or the task ID. Solvability-oracle episodes are separate sessions and
   never appear.
3. **Recovery episodes.** The history is the episode's own executed steps exactly
   as they happened, which includes the frozen checkpoint-replay steps and the
   controlled detour. The expected action and the reference continuation are
   never shown. Paired conditions therefore see identical history; the only
   intended condition difference remains the recovery-memory content.
4. **Controller unchanged.** The admissible-action index contract
   (`ACTION_INDEX: <integer>`), three attempts per decision, temperature 0, and
   inference seed 42 are unchanged.
5. **Retrieval query unchanged.** The Sentence-BERT query remains `query-v1`:
   task goal, current observation, inventory, and the canonical failure message.
   Action history is never added to it, and exactly one retrieval still follows
   the controlled failure.

## Prompt format

```text
TASK GOAL:
<frozen goal>

EPISODE HISTORY:
Step 1
Action: <action>
Observation: <resulting observation>

Step 2
...

CURRENT OBSERVATION:
<current observation>

CURRENT INVENTORY:
<inventory or <empty>>

[RECOVERY MEMORY: recovery only]

ADMISSIBLE ACTIONS:
0. <action>
...

Return exactly:
ACTION_INDEX: <integer>
```

Before the first executed step, the history section reads `(no previous steps)`.

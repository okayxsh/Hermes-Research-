# Retrieval relevance rubric (retrieval-relevance-rubric-v1)

Status: proposed with the evaluation protocol (Decision 012). It is frozen when the researcher
approves the evaluation protocol freeze, before any evaluation retrieval exists.

## Basis

- **Decision 005:** relevance labels "start with task family, operation, required state, goal
  type, stage, and preconditions; audit a sample and freeze the rule set before unseen
  evaluation."
- **`docs/METRICS.md`:** retrieval quality is measured against human relevance labels, never
  cosine similarity.
  - Precision@3 = adjudicated-relevant skills among the retrieved top-3 / 3.
  - Retrieval Noise = 1 − Precision@3.
  - No-retrieval episodes are reported separately.
  - Cohen's kappa is computed from the original independent ratings.
- **`rq1.analysis.labelling`:** labels are binary (`RELEVANT` / `IRRELEVANT`); disagreements are
  adjudicated explicitly, never coerced.

## What a rater sees

For each item, a rater sees:
- the task goal
- the failure-state observation
- the inventory field
- the canonical failure message
- one retrieved skill (ID and text) with its rank

A rater does **not** see:
- the condition or library size
- the episode outcome
- the cosine score
- the other rater's labels

## Rule

Label the skill **RELEVANT** only if both are true for the situation shown:

1. **Operation or required state.** The skill describes an operation or state change the task
   still requires from this failure state (for example locating or taking the target object,
   heating, cooling, cleaning, using a light source, or placing the object in a receptacle).
2. **Goal type and preconditions.** The guidance is consistent with the task's goal type and its
   preconditions can be met from the current stage.

Otherwise label it **IRRELEVANT**. Specifically:
- A skill about a different operation or goal type, or one that only restates a generic step
  with no bearing on what remains, is IRRELEVANT.
- Task-family match alone is not sufficient.
- Wording quality, length, and similarity to other retrieved skills are not considered.

## Procedure

1. **Calibration.** Before rating evaluation items, both raters label the retrieval items from
   the passed non-scientific valid_seen evaluation check. Disagreements are discussed to align
   application of the rule; the rule text itself is not changed. This is the Decision 005 sample
   audit.
2. **Independent rating.** Two raters label every evaluation retrieval item independently in
   separate files.
3. **Agreement.** Cohen's kappa and its sample size are computed on the original labels, with no
   pass/fail threshold.
4. **Adjudication.** Disagreements are resolved afterwards by recorded adjudication. The final
   Precision@3 uses adjudicated labels.
5. **NoLib.** NoLib has no retrieval items and is reported separately.

# RQ1 Phase 2B: instructions for relevance raters and the adjudicator (DRAFT)

**Status: DRAFT, not approved.** The rubric version is `rq1-five-arm-relevance-rubric-v1-draft`. The tooling is in `src/rq1/five_arm/rating.py`. No real ratings exist yet.

## Roles

| Role | Who | Sees | Never sees |
|---|---|---|---|
| Coordinator | one project member, not a rater | everything needed to run the process; holds the item key and the code-to-person list | does not rate |
| Rater A, Rater B | two people, working separately | their own export file: item ID, task goal, recovery context, skill text, rubric | the other rater's answers, retrieval ranks or scores, which arm a skill is used in, any model outcome, the expert/oracle continuation, results of the experiment |
| Adjudicator | a third person, not A or B | only the conflicting or UNCERTAIN items, with both ratings and rationales | model outcomes, ranks, scores, arms |

## Rating rubric

For each item, answer the question: **"Would this skill's guidance help complete THIS task from THIS situation?"**

- **RELEVANT:** the guidance directly applies to completing this task from this situation. It is the right kind of task, and the steps would actually help now.
- **PARTIAL:** some guidance applies, but only in part or conditionally. For example, the right sub-procedure for the wrong object or receptacle, or something helpful only after other steps.
- **IRRELEVANT:** the guidance does not help complete this task from this situation, even if it is about a similar topic or shares words with the task. Topical similarity is *not* relevance.
- **UNCERTAIN:** you cannot decide from what is shown. Use it sparingly; every UNCERTAIN item goes to the adjudicator.

Write a short rationale (one sentence) for PARTIAL and UNCERTAIN. A rationale for the other labels is optional.

## Rater procedure

1. Open only your own file (`export-<your code>.csv`). The order of the items differs between raters on purpose.
2. Fill in the `rating` column with exactly one of RELEVANT, PARTIAL, IRRELEVANT or UNCERTAIN. Add the optional `rationale`.
3. Do not change any other column, add rows or remove rows.
4. Work alone. Do not discuss items with the other rater or the adjudicator until the coordinator has closed the rating round.
5. Do not use AI tools, chatbots, scripts, search for model outputs, or anything else that produces or suggests labels.
6. Return the file to the coordinator through the agreed channel, together with the declaration below.

## Rater declaration

Fill in this form. Use your **rater code** only: no name or e-mail address in research files. The coordinator keeps the code-to-person list separately.

```json
{
  "declaration_version": "rq1-five-arm-rater-declaration-v1",
  "label_source": "human",
  "rater_id": "<your rater code>",
  "role": "rater",
  "declared_at": "<date and time>",
  "produced_personally": true,
  "no_ai_or_automated_assistance": true,
  "worked_independently": true,
  "saw_other_ratings": false,
  "saw_model_outcomes": false,
  "saw_retrieval_ranks_or_scores": false,
  "saw_treatment_assignments": false,
  "saw_oracle_or_expert_continuations": false,
  "coordinator_receipt": {"received_by_code": "<coordinator code>", "received_at": "<time>",
                          "channel": "<how the file was returned>", "export_sha256": "<hash of the export you received>"}
}
```

The coordinator fills in `coordinator_receipt` on receipt. The tooling refuses submissions:
- whose declaration is incomplete;
- that say AI or automated help was used;
- whose rater code differs from the file;
- that contain a name or e-mail address;
- whose receipt was recorded by the rater rather than the coordinator.

**A declaration is an attestation, not proof.** The coordinator also runs the independence screen in the agreement report (`rating-agreement`). It flags identical label vectors or many identical rationales. Flagged pairs are reviewed with the raters before any label is frozen.

## Adjudicator procedure

1. The coordinator gives you only the items in `requires_adjudication`: the conflicts plus every item both raters marked UNCERTAIN.
2. For each item, choose RELEVANT, PARTIAL or IRRELEVANT. UNCERTAIN is not allowed at this stage. Give a one-sentence reason.
3. Record your adjudicator code. It must differ from both raters' codes.
4. Sign an adjudicator declaration. Use the same form with `"role": "adjudicator"`; you will have seen both ratings, and that is expected.
5. Like the raters, you never see model outcomes, ranks, scores or arms.

## Coordinator procedure

1. **Build exports.** Run `python -m rq1.five_arm rating-export` from the frozen candidate sets:
   - Candidate sets are the natural top 3 per task, from the CPU ranking export.
   - The tool refuses rater-facing text that mentions ranks, scores, arms, outcomes, oracle content or skill identifiers.
   - Record the SHA-256 of each export file.
2. **Distribute and collect.** Send each rater only their own file through separate channels. Collect the submissions and declarations, and fill in the receipts.
3. **Measure agreement.** Run `rating-agreement`. It reports:
   - percent agreement;
   - 4-category and binary κ;
   - the confusion table;
   - the independence screen;
   - the adjudication list.
4. **Adjudicate and freeze.** Collect the adjudications, then freeze the labels with `freeze_labels`. The provenance needs the export hash and the hashes of both submissions, both declarations, the adjudication and the adjudicator declaration.
5. **Hand over for approval.** The frozen file has status `frozen_pending_human_approval`. Its approval is a separate author record bound to `labels_sha256`.

## Planned workload

The natural top-3 candidate set is used (SD-09). Recompute these figures from the frozen rankings with `rating.workload_from_rankings`.

| Population | Distinct pairs | Judgments before adjudication (2 raters) |
|---|---:|---:|
| 12 pilot contexts | 36 | 72 |
| 79 unseen contexts | 237 | 474 |

Adjudication adds one judgment per conflict or agreed UNCERTAIN item.

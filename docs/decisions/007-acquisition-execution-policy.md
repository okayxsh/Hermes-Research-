# 007 — Acquisition execution policy (pre-run amendment)

- Status: approved methodology decision by the study owner
- Date: 2026-09-13
- Timing: frozen before the scientific acquisition run; no scientific acquisition data existed when it was made
- Scope: RQ1 train-only acquisition, initial 180-episode queue
- Clarifies: Decision 003 (skill creation) and `configs/libraries.yaml` (`do_not_deduplicate`)

No acquisition result motivated these decisions. The only earlier acquisition
episodes were non-scientific prelaunch pilots under `artifacts/prelaunch/`,
which are not scientific data. This record is not the human approval of the
acquisition freezes; those remain separate approval files.

## Decisions

1. **Action budget.** Every acquisition episode allows at most 50 ALFWorld
   environment actions (`acquisition_action_budget = 50`), matching ALFWorld's
   standard 50-step episode limit. The budget is identical for every task and
   family and is never increased dynamically.
2. **Skill author.** After a successful `train` episode only, the same
   `hermes3:8b` experimental agent (same model, temperature 0, inference seed 42)
   receives the successful episode context through
   `hermes/prompts/post_success_learning.md` and may produce at most one
   reusable, generalized candidate skill. No separate summariser model is used.
3. **Create-only.** Acquisition never modifies or patches a previously acquired
   skill. A successful episode yields 0 or 1 new candidate; a failed episode
   yields 0; an infrastructure failure yields 0.
4. **Duplicates.** At acquisition, a candidate is rejected only when its
   deterministically normalized skill text exactly equals an already accepted
   skill: leading/trailing whitespace trimmed and whitespace runs, including line
   endings, collapsed (the frozen `skill-text-v1` form). Case is preserved
   because no existing validation policy normalizes case. Embeddings, cosine
   similarity, an LLM judge, or any semantic comparison never decide duplicates.
   Near-duplicates and independently generated similar skills are preserved.
5. **No retrospective deduplication.** Clean-24, Accum-60, and Accum-96 are
   built from the accepted chronological pool without deduplication, so
   `do_not_deduplicate: true` in `configs/libraries.yaml` remains valid.

## Deterministic operationalization

Decision 003 is applied without model judgement. A candidate is rejected when:

- the response is neither exactly `NO_SKILL` nor a `TITLE:` line followed by `BODY:`;
- `rq1.skills.leakage.find_leakage` finds a task-ID pattern, a room number, or an
  object-instance number;
- it contains the source task identifier or one of its path components;
- it reproduces verbatim an executed episode action that names an object
  instance (a raw trajectory command);
- it is an exact normalized duplicate under decision 4.

Skill generation is one deterministic inference with no retries. Raw responses,
prompt hashes, validation outcomes, and provenance are recorded for every
candidate. The machine-readable form is `configs/acquisition/protocol.yaml`,
which must equal `rq1.acquisition.protocol.protocol_definition()`.

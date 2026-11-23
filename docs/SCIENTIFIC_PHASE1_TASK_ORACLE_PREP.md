# Scientific Phase 1 — task eligibility, exposure ledger and controlled-failure oracle preparation

**Status: preparation tooling only.**
- Nothing here is an approved manifest, a task roster, a model outcome or a scientific result.
- No command calls a language model or uses a GPU.
- The closed 500-episode engineering run is not read for outcomes, rerun or reinterpreted. Its six task contexts appear only as exposure evidence.

Package: `src/rq1/scientific_prep/`. Run with `PYTHONPATH=src python -m rq1.scientific_prep <command>`.

## 1. What the existing code already guarantees (reconnaissance, commit 275ae94)

| Guarantee | Existing implementation | Reused here |
|---|---|---|
| Task exists, source identity | `rq1.tasks.discovery` (traj/game SHA-256, `missing_game_file`, `data_root_identity`) | The identity function is reused; the inventory reproduces `discovery_identity` |
| Expert route | `rq1.recovery.reference_route.derive_handcoded_reference` (ALFWorld 0.4.2 hand-coded expert, one-game root) | Yes, unchanged |
| Checkpoint order | `midpoint_candidates` (CHECKPOINT_POLICY midpoint-navigation-detour-v1) | Yes, unchanged |
| Checkpoint reconstruction | `RealEpisodeSession.apply_scripted`: every prefix action valid, non-terminal | Same rule (admissible and valid; terminal only where allowed) |
| Detour | `select_reversible_navigation_action` (first sorted admissible `go to …` ≠ expected) | Yes, unchanged |
| Post-detour state | `bridge_state_digest` {observation, admissible actions, step number} | Yes, unchanged |
| Oracle completes | `RealRecoveryHarness._oracle`: second session; prefix → detour → continuation → `done` and `success` | Same, plus the additions below |
| Action budget | Hard-coded 50 (`TOTAL_EPISODE_ACTION_BUDGET`) | Generalised to caps 50/100/150/200 (accounting only) |

**Gaps found and closed (flagged for scientific review; the failure definition is unchanged):**
1. The historical oracle session never compared its own checkpoint or post-detour digests with the first session's. The validator now requires an **independent second session to reproduce both digests** before the continuation runs.
2. The complete expert route is **replayed from reset to environment-reported success** before any checkpoint is tried.
3. The source identity is **re-hashed at validation time** against the inventory.
4. **Persisted bridge evidence must confirm success.** In-memory session state alone never yields `ENVIRONMENT_VALIDATED`:
   - the `episode-events.jsonl` written by every real session (route replay and oracle session B) must show exactly one `alfworld_start`;
   - it must show the expected number of `alfworld_step` results;
   - the last environment response must be `done` and `success`.

   Every oracle summary also records `environment_backend`, the driver class, so fixture results are identifiable as fixtures.

## 2. Exposure screening: three relations, named policies

The ledger keeps three relations apart. Each row has `exposure_kinds`, `template_overlap_roles` (for every template definition) and `exact_trial_reuse_roles`.

| Relation | Meaning | Effect |
|---|---|---|
| Model-outcome exposure | A declared source records the task itself in a model episode or preparation step | `exposure_class` EXPOSED / POSSIBLY_EXPOSED; never in a pool |
| Exact task reuse | The same ALFRED trial identifier appears under another task ID or split | Excludes only through a policy with `exclude_trial_reuse_roles` |
| Procedural-template overlap | Shares a template (named definition) with referenced tasks | Recorded for every definition. **Not contamination by itself**; excludes only through a named policy |

**Policies in the template** (all relative to the outcome-screened pool, with the removed tasks and the matched roles listed; pairwise set differences are written to `candidate-screening-summary.json`):

| Policy | Definition | Excludes templates of |
|---|---|---|
| `historical_template_screened` | goal | evaluation, evaluation_preparation |
| `known_template_screened` | goal | + an explicit `template_list` (role `known_template_match`) |
| `acquisition_template_screened` | goal | + acquisition |
| `acquisition_engineering_template_screened` | goal | + acquisition, engineering, demonstration, smoke, pilot |
| `exact_trial_reuse_screened` | goal + trial | historical + trial-ID reuse by any exposed role |
| `semantic_type_object_screened` | type + object | all exposed roles |

### The 79 → 77 vs 79 → 29 discrepancy (laptop reconstruction from the frozen manifests)

**The two numbers come from different policies; neither is wrong under its own definition.**

- **79 → 29 = `acquisition_template_screened`.** It removes every historically screened task whose goal template occurs anywhere among the 240 acquisition tasks:
  - 14 of the 27 remaining goal templates (50 tasks) match acquisition tasks;
  - engineering and demonstration templates match none;
  - exact trial reuse is 0.
- **79 → 77 is a known source-template exclusion** of one template, *not* a full acquisition screen. The university per-family counts (clean 23, cool 15, heat 15, look 9, pick 15) equal the 79 minus the two trials of `pick_heat_then_place_in_recep|Potato|None|GarbageCan`.
  - This template matches four acquisition tasks: acq-180 #39 and #141, extension #9 and #15.
  - It is the only heat template among the 79 with exactly two trials, and the only single-template removal consistent with that distribution.
  - **This is a reconstruction from counts.** The audit's own list is not in local evidence. Add it as a `template_list` source only once the audit confirms it.
- **The count alone is misleading.**
  - `type_object` screening against evaluation roles also gives 77, but with clean 21 / heat 17.
  - The broader semantic screen (`semantic_type_object_screened`) gives 12.

The tests assert these policy *relations* (sets recomputed independently, the hypothesis checked against the reported family distribution). No count is used as an eligibility constant.

## 3. Missing exposure sources: fail closed by default

| Source | Status |
|---|---|
| Gemma memory/no-memory demonstration | **Resolved** from committed `artifacts/prelaunch/prelaunch-a718b4e4…/pilot-report.json` (`json_task_fields` source, hash-checked). It exposed two tasks: `valid_seen:pick_and_place_simple-Pencil-None-Desk-310/trial_T20190909_113054_894334` (memory and no-memory recovery outcomes) and `train:look_at_obj_in_light-Bowl-None-DeskLamp-304/trial_T20190907_232204_574574` (acquisition episode) |
| Kaggle `valid_seen` environment smoke | **Unresolved.** Environment only, with no task episode, but the task ID is not recorded |
| University debugging / smoke / pilot / calibration logs | **Unresolved.** Not on the laptop |

These sources are declared with `"status": "unresolved"`.
- **Default (fail closed):** their scope splits become UNKNOWN, so the pools are empty and the exit code is 1.
- **`exposure --allow-provisional`:**
  - tasks stay screenable for preparation;
  - every affected row has `evidence_complete: false` and `provisional_gaps`;
  - every output carries `provisional: true` and a non-empty `roster_approval_blockers`;
  - oracle summaries copy both.

**Preparation (inventory, provisional screening, CPU-only oracle validation) may proceed provisionally. Scientific roster approval must fail closed** until every unresolved source is replaced with real evidence and `roster_approval_blockers` is empty. Missing evidence is never treated as independence.

Expected hashes are SHA-256 of the canonical (LF) committed bytes. A file also matches if only CRLF→LF differs, as with a Windows `autocrlf` checkout. Which form matched is recorded in `sha256_matched`. Any other difference blocks.

## 4. Modules

| Module | Purpose |
|---|---|
| `identity.py` | Task-ID parsing; cross-format reference normalisation; template definitions `goal_template`, `scene_template`, `type_object` |
| `inventory.py` | Deterministic, read-only split walk; per-file SHA-256; duplicates; template and scene counts; `discovery_identity` |
| `exposure.py` | Declared, hashed evidence sources; three relations; named policies; separate confirmatory and pilot pools; roster-approval blockers |
| `horizons.py` | `remaining = cap − prefix − detour`. ARITHMETICALLY_FEASIBLE / ENVIRONMENT_VALIDATED (only with environment evidence) / FAILED |
| `oracle.py` | Identity → expert route (BLOCKED if absent) → full-route replay → midpoint checkpoints → session A → independent session B → persisted-log confirmation → per-cap horizons |
| `safety.py` | New versioned output directory per run; protected areas refused; write-once files; preparation markers |
| `cli.py` | `check-runtime`, `inventory` (`--dry-run`), `exposure` (`--allow-provisional`), `horizons`, `oracle-validate` |

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | success with complete evidence |
| 1 | partial (candidates failed or blocked, uncertain or provisional evidence, roster blockers) |
| 2 | blocked input |
| 3 | refused (output safety, authorisation or consent) |

## 5. Evidence rules

- **`NO_MODEL_OUTCOME_FOUND` is not "unexposed".**
- **Template overlap uses only tasks with exposure evidence.** Tasks merely listed as excluded never define overlap.
- **Pools are screens, not rosters.**
- **Selection never looks at outcomes.**
  - Candidate order is task-ID order.
  - Checkpoint order is the historical midpoint rule.
  - No D0/D18/memory performance is read.
- **Oracle evidence is never model-visible.** It is written only under the oracle output directory, with `"model_visible": false`.
- **`valid_unseen` stays behind the existing gate** (`RQ1_VALID_UNSEEN_ACCESS=evaluation-task-preparation`). Every access is logged.

## 6. Deployment and university (DESKTOP-AIPC-02) commands

**Laptop → GitHub (only after explicit authorisation).** The local `origin` (the `runpod-preparation-clone`) is kept, and a separately named remote is added:

```sh
git remote add github https://github.com/okayxsh/Hermes-Research-.git
git fetch github feat/rq1-engineering-500-nov09-readiness       # read-only; its head is the base 275ae94
git merge-base --is-ancestor github/feat/rq1-engineering-500-nov09-readiness HEAD && echo base-ok
git status --short                                                # expect no output
git ls-remote --heads github sci/rq1-phase1-task-oracle-prep    # expect no output (branch must not exist yet)
git push github sci/rq1-phase1-task-oracle-prep:sci/rq1-phase1-task-oracle-prep   # new branch; no --force; main untouched
```

**University.** Use a **separate** clone and do not modify the team's existing checkouts.

Assumptions in these commands:
- the data root is `/home/researcher/.cache/alfworld/json_2.1.1`;
- the RQ1 Python 3.11 environment and Hermes bridge runtime are available;
- `HIST` is the read-only root of the historical checkout holding `artifacts/task_manifests/…`, `artifacts/prelaunch/…` and `university-gpu/prospective/…`.

```sh
git clone --branch sci/rq1-phase1-task-oracle-prep https://github.com/okayxsh/Hermes-Research-.git ~/rq1-sci-phase1
cd ~/rq1-sci-phase1
export PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1
OUT=~/rq1-sci-prep-outputs            # outside every repository and protected area
DATA=/home/researcher/.cache/alfworld/json_2.1.1

# 1. offline tests (add RQ1_SCI_PREP_HISTORICAL_EVIDENCE=<dir with the 3 manifests + feasibility-200-candidate.json> for the reproduction tests)
python3.11 -B -m unittest test_scientific_prep_inventory test_scientific_prep_exposure test_scientific_prep_oracle test_scientific_prep_safety_cli

# 2. dependencies and data presence (no downloads, no environment start)
python3.11 -B -m rq1.scientific_prep check-runtime --data-root "$DATA"

# 3. inventory (dry run first, then written)
python3.11 -B -m rq1.scientific_prep inventory --data-root "$DATA" --splits valid_seen --dry-run --output-root "$OUT" --label dry-1
RQ1_VALID_UNSEEN_ACCESS=evaluation-task-preparation \
python3.11 -B -m rq1.scientific_prep inventory --data-root "$DATA" --splits train valid_seen valid_unseen --output-root "$OUT" --label inv-1

# 4. exposure: fail-closed run first (records what is unresolved), then the explicitly provisional run
cp configs/scientific_prep/exposure-sources.template.json ~/rq1-sci-prep-exposure-sources.json
python3.11 -B -m rq1.scientific_prep exposure --inventory "$OUT/inventory-inv-1/task-inventory.json" \
  --sources ~/rq1-sci-prep-exposure-sources.json --sources-base "$HIST" --output-root "$OUT" --label exp-closed-1
python3.11 -B -m rq1.scientific_prep exposure --inventory "$OUT/inventory-inv-1/task-inventory.json" \
  --sources ~/rq1-sci-prep-exposure-sources.json --sources-base "$HIST" --output-root "$OUT" --label exp-prov-1 --allow-provisional

# 5. CPU-only oracle validation (no model). Start with --limit 1, inspect, then widen.
python3.11 -B -m rq1.scientific_prep oracle-validate --inventory "$OUT/inventory-inv-1/task-inventory.json" \
  --ledger "$OUT/exposure-exp-prov-1/exposure-ledger.json" --screening "$OUT/exposure-exp-prov-1/candidate-screening-summary.json" \
  --pool pilot_valid_seen --policy acquisition_engineering_template_screened --data-root "$DATA" \
  --output-root "$OUT" --label oracle-pilot-1 --limit 1 --yes
RQ1_VALID_UNSEEN_ACCESS=evaluation-task-preparation \
python3.11 -B -m rq1.scientific_prep oracle-validate --inventory "$OUT/inventory-inv-1/task-inventory.json" \
  --ledger "$OUT/exposure-exp-prov-1/exposure-ledger.json" --screening "$OUT/exposure-exp-prov-1/candidate-screening-summary.json" \
  --pool confirmatory_valid_unseen --policy historical_template_screened --data-root "$DATA" \
  --output-root "$OUT" --label oracle-conf-1 --limit 1 --yes

# 6. read results
python3.11 -m json.tool "$OUT/exposure-exp-prov-1/candidate-screening-summary.json" | less    # roster_approval_blockers, removed lists
column -s, -t < "$OUT/oracle-conf-1/oracle-validation-summary.csv" | less   # per-task status and cap_50..cap_200
```

**Rules for running `oracle-validate`:**
- It requires a clean, committed worktree. Keep configs and outputs outside the clone.
- Each task uses 3 real sessions: route replay, session A and independent session B.
- `--selection-cap N` reproduces the historical budget filter on checkpoints. **The checkpoint-selection rule for the new protocol is a scientific decision.**
- Which screening policy defines the confirmatory pool is a **scientific decision**. Pass the policy explicitly; the tool never chooses one.

## 7. Not verifiable locally

These are pending university execution:
- real ALFWorld route derivation;
- bridge sessions;
- digest reproduction across sessions;
- environment-reported success.

The local tests use a deterministic fake environment (`tests/scientific_prep_fakes.py`), which is **fixture evidence only**. The historical tests run only when `RQ1_SCI_PREP_HISTORICAL_EVIDENCE` points at the frozen manifests.

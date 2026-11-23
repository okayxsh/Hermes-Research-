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
1. The historical oracle session never compared its own checkpoint or post-detour digests with the first session's. Reproducibility was only checked later, during scientific episodes. The validator now requires an **independent second session to reproduce both digests** before the continuation runs.
2. The complete expert route is **replayed from reset to environment-reported success** before any checkpoint is tried, which proves the route applies to this exact task instance.
3. The source identity is **re-hashed at validation time** against the inventory.

**Gaps found and documented as limits:**
- No historical code defines template screening. The 94 → 79 screen is reproduced by `goal_template` (task type, object, movable receptacle and receptacle, scene ignored). The university-reported 79 → 77 acquisition/engineering screen is **not** reproducible from local evidence: `goal_template` gives 29. Its rule must be documented before it is used.
- ALFRED's seventh task type (`pick_and_place_with_movable_recep`) has stable IDs but no ALFWorld family. These trials have no game files and are reported as expected, non-runnable entries.

## 2. Modules

| Module | Purpose |
|---|---|
| `identity.py` | Task-ID parsing (type, object, receptacles, scene, trial); cross-format reference normalisation; template definitions `goal_template`, `scene_template`, `type_object` |
| `inventory.py` | Deterministic, read-only split walk: indexed vs installed vs missing vs orphan entries; per-file SHA-256; duplicate file-hash groups within and across splits; template and scene counts; `discovery_identity` |
| `exposure.py` | Declared evidence sources, each hashed. Classes EXPOSED > POSSIBLY_EXPOSED > UNKNOWN > NO_MODEL_OUTCOME_FOUND. INELIGIBLE (not installed) is a separate screening class. Named template-overlap policies; separate confirmatory and pilot pools |
| `horizons.py` | `remaining = cap − prefix − detour`. Statuses: ARITHMETICALLY_FEASIBLE, ENVIRONMENT_VALIDATED (only with environment evidence), FAILED |
| `oracle.py` | Per candidate: identity → expert route (BLOCKED if absent; never synthesised) → full-route replay → checkpoints in midpoint order → session A (checkpoint and detour digests) → independent session B (reproduce digests, continuation to environment success) → per-cap horizons → evidence-file hashes. Statuses: ENVIRONMENT_VALIDATED, FAILED, BLOCKED, UNVERIFIED |
| `safety.py` | New versioned output directory per run (never reused). Refuses repository `artifacts`/`results`/`runs`/`state`/`data`/`university-gpu`/… and any `--protect` or `RQ1_SCI_PREP_PROTECTED_PATHS` path. Write-once files. Preparation markers; refuses approval or manifest keys |
| `cli.py` | `check-runtime`, `inventory` (`--dry-run`), `exposure`, `horizons`, `oracle-validate` |

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | partial (candidates failed or blocked, or uncertainties recorded) |
| 2 | blocked input |
| 3 | refused (output safety, authorisation or consent) |

## 3. Evidence rules (scientific integrity)

- **`NO_MODEL_OUTCOME_FOUND` is not "unexposed".** It means no *declared* source records a model outcome.
  - A declared source that is missing, an unresolvable reference, or an undeclared exclusion reason makes the affected split's tasks UNKNOWN.
  - A required source that is missing or has the wrong hash blocks the run.
- **Template overlap uses only tasks with exposure evidence.** Tasks merely listed as excluded never define overlap.
- **Pools are screens, not rosters.**
  - The confirmatory pool (`valid_unseen`) and the pilot pool (`valid_seen`) are computed separately, and their shared templates, scenes and IDs are reported.
  - Oracle validation runs only on tasks inside a chosen screened pool, never on exposed tasks.
- **Selection never looks at outcomes.**
  - Candidate order is task-ID order.
  - Checkpoint order is the historical midpoint rule.
  - No D0/D18/memory performance is read.
- **Oracle evidence is never model-visible.** Expert routes, detours and continuations are written only under the oracle output directory, with `"model_visible": false`.
- **`valid_unseen` stays behind the existing gate.**
  - Its metadata and routes require `RQ1_VALID_UNSEEN_ACCESS=evaluation-task-preparation`.
  - Every access is logged (`valid-unseen-access.jsonl`).

## 4. University (DESKTOP-AIPC-02) commands

Use a **separate** clone. Do not modify the team's existing checkouts.

Assumptions in these commands (adjust them to the workstation):
- the data root is `/home/researcher/.cache/alfworld/json_2.1.1`;
- the RQ1 Python 3.11 environment and the RQ1 Hermes bridge runtime that the historical preparation used are available;
- `HIST` is the root of the historical RQ1 checkout that holds `artifacts/task_manifests/…` (read only).

```sh
# 0. separate scientific clone (after the branch is pushed with authorisation)
git clone --branch sci/rq1-phase1-task-oracle-prep https://github.com/okayxsh/Hermes-Research- ~/rq1-sci-phase1
cd ~/rq1-sci-phase1
export PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1
OUT=~/rq1-sci-prep-outputs            # outside every repository and protected area
DATA=/home/researcher/.cache/alfworld/json_2.1.1

# 1. offline tests on the workstation
python3.11 -B -m unittest test_scientific_prep_inventory test_scientific_prep_exposure test_scientific_prep_oracle test_scientific_prep_safety_cli

# 2. dependencies and data presence (no downloads, no environment start)
python3.11 -B -m rq1.scientific_prep check-runtime --data-root "$DATA"

# 3. inventory (dry run first, then written)
python3.11 -B -m rq1.scientific_prep inventory --data-root "$DATA" --splits valid_seen --dry-run --output-root "$OUT" --label dry-1
RQ1_VALID_UNSEEN_ACCESS=evaluation-task-preparation \
python3.11 -B -m rq1.scientific_prep inventory --data-root "$DATA" --splits train valid_seen valid_unseen --output-root "$OUT" --label inv-1

# 4. exposure reconciliation (fill every <FILL…> in a copy of the template first)
cp configs/scientific_prep/exposure-sources.template.json ~/rq1-sci-prep-exposure-sources.json
python3.11 -B -m rq1.scientific_prep exposure --inventory "$OUT/inventory-inv-1/task-inventory.json" \
  --sources ~/rq1-sci-prep-exposure-sources.json --sources-base "$HIST" --output-root "$OUT" --label exp-1

# 5. CPU-only oracle validation (no model). Start with --limit 1, inspect, then widen.
python3.11 -B -m rq1.scientific_prep oracle-validate --inventory "$OUT/inventory-inv-1/task-inventory.json" \
  --ledger "$OUT/exposure-exp-1/exposure-ledger.json" --screening "$OUT/exposure-exp-1/candidate-screening-summary.json" \
  --pool pilot_valid_seen --policy acquisition_engineering_template_screened --data-root "$DATA" \
  --output-root "$OUT" --label oracle-pilot-1 --limit 1 --yes
RQ1_VALID_UNSEEN_ACCESS=evaluation-task-preparation \
python3.11 -B -m rq1.scientific_prep oracle-validate --inventory "$OUT/inventory-inv-1/task-inventory.json" \
  --ledger "$OUT/exposure-exp-1/exposure-ledger.json" --screening "$OUT/exposure-exp-1/candidate-screening-summary.json" \
  --pool confirmatory_valid_unseen --policy historical_template_screened --data-root "$DATA" \
  --output-root "$OUT" --label oracle-conf-1 --limit 1 --yes

# 6. read results
python3.11 -m json.tool "$OUT/exposure-exp-1/candidate-screening-summary.json" | less
column -s, -t < "$OUT/oracle-conf-1/oracle-validation-summary.csv" | less   # per-task status and cap_50..cap_200
```

**Rules for running `oracle-validate`:**
- It requires a clean, committed worktree. Keep configs and outputs outside the clone.
- Each task uses 3 real sessions: route replay, session A and independent session B.
- `--selection-cap N` reproduces the historical budget filter on checkpoints. Without it, every checkpoint is tried and each one's per-cap arithmetic is recorded. **The checkpoint-selection rule for the new protocol is a scientific decision.**

## 5. Not verifiable locally

These are pending university execution:
- real ALFWorld route derivation;
- bridge sessions;
- digest reproduction across sessions;
- environment-reported success.

The local tests use a deterministic fake environment (`tests/scientific_prep_fakes.py`), which is **fixture evidence only**. The historical-screening reproduction test runs only when `RQ1_SCI_PREP_HISTORICAL_EVIDENCE` points at the frozen manifests.

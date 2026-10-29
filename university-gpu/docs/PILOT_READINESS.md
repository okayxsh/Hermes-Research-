# Pilot readiness: six-task `valid_seen` set

Captured 2026-10-29 on the development checkout (`feat/rq1-university-4gpu`), before any
university-machine run. This document records a blocked state; it is not a validation report.

## 1. Status

**BLOCKED.** Two independent blockers:

1. **No ALFWorld data locally.** `data/json_2.1.1/valid_seen/` does not exist and no task
   metadata JSON is present under `data/`. `discover_tasks` therefore fails closed with
   `missing ALFWorld split directory: valid_seen` (`src/rq1/tasks/discovery.py:23-24`).
   No `valid_seen` task can be discovered, selected, or validated on this machine.
2. **Pilot artifacts are unfrozen.** `data/task_lists/pilot_seen.json` is
   `{"status":"TO_BE_FROZEN_BY_PILOT","split":"valid_seen","tasks":[]}` and
   `university-gpu/approval-templates/pilot-manifest.json` is
   `{"approved": false, "approved_by": "", "approval_reference": "", "manifest_sha256": ""}`.
   No command in `src/` reads or writes `data/task_lists/pilot_seen.json`.

**Not produced and not validated:** the six-task candidate list, expert routes, checkpoints,
controlled detours, and failure files. They do not exist. Nothing in this repository may be
cited as pilot validation evidence. No ALFWorld/Ollama/model install, GPU work, or real
recovery episode was performed.

## 2. Six canonical families

Family names are the values of `CANONICAL_FAMILIES` (`src/rq1/tasks/discovery.py:8-12`),
keyed there by native ALFWorld `task_type` (`pick_and_place_simple`, `pick_two_obj_and_place`,
`look_at_obj_in_light`, `pick_clean_then_place_in_recep`, `pick_heat_then_place_in_recep`,
`pick_cool_then_place_in_recep`).

| Family | Candidate task ID | Expert route | Checkpoint | Controlled detour | Validation |
|---|---|---|---|---|---|
| `pick_and_place` | TO BE SELECTED | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | REQUIRES REAL VALIDATION |
| `pick_two_and_place` | TO BE SELECTED | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | REQUIRES REAL VALIDATION |
| `look_at_object` | TO BE SELECTED | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | REQUIRES REAL VALIDATION |
| `clean_and_place` | TO BE SELECTED | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | REQUIRES REAL VALIDATION |
| `heat_and_place` | TO BE SELECTED | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | REQUIRES REAL VALIDATION |
| `cool_and_place` | TO BE SELECTED | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | REQUIRES REAL VALIDATION |

## 3. Procedure on the university machine (existing commands only)

Precondition: ALFWorld 0.4.2 installed and `json_2.1.1/valid_seen/` present at the data root.
Confirm with `rq1 tasks capabilities` (`src/rq1/cli.py:747-749`) and
`rq1 alfworld capabilities` (`src/rq1/cli.py:814`). Both are read-only.

1. **Discover metadata** (never instructions or outcomes):
   `rq1 tasks discover --split valid_seen`
   Flags at `src/rq1/cli.py:889`; handler at `src/rq1/cli.py:750-756`. Writes an immutable
   discovery to `artifacts/task_manifests/discoveries/valid_seen-<identity12>.json`.
2. **Propose** (does not freeze):
   `rq1 tasks propose --kind pilot --count 6 --seed <approved-seed>`
   Flags at `src/rq1/cli.py:890-892`; handler at `src/rq1/cli.py:785-795`; selection via
   `propose_manifest` (`src/rq1/tasks/selection.py:99-110`) and `select_tasks`
   (`src/rq1/tasks/selection.py:24-35`). `select_tasks` is a seeded round-robin over sorted
   families: `--count 6` yields one task per family only if all six families are present, and
   it does **not** choose longest trajectories.
3. **Longest-per-family step: no command exists.** `select_longest_per_family`
   (`src/rq1/tasks/selection.py:38-71`) is library-only; no CLI subcommand calls it (checked in
   `src/rq1/cli.py`). If the pilot protocol requires longest-per-family candidates, that step
   must be added or run by a human; it must not be simulated here.
4. **Validate the proposal**:
   `rq1 tasks validate --manifest artifacts/task_manifests/proposals/pilot-<sha16>.json`
   Flag at `src/rq1/cli.py:894`; handler at `src/rq1/cli.py:757-759`. This checks schema,
   type/split, counts, and hash (`src/rq1/tasks/validation.py:11-26`); it does **not** check
   six-family coverage.
5. **Cross-manifest overlap check**:
   `rq1 tasks overlap-check --manifest <pilot.json> --manifest <acquisition.json> --manifest <evaluation.json>`
   Flag at `src/rq1/cli.py:895`; handler at `src/rq1/cli.py:760-762`
   (`src/rq1/tasks/validation.py:28-35`).
6. **Freeze with human approval**:
   `rq1 tasks freeze --kind pilot --approval-file <approval.json> --proposal <proposal.json> --yes`
   Flags at `src/rq1/cli.py:893`; handler at `src/rq1/cli.py:764-784`. Requires approval with
   `approved_by`, `approved_at`, and `status: APPROVED` (`src/rq1/tasks/freeze.py:16-21`) and a
   clean committed repository (`src/rq1/tasks/freeze.py:27-28`). Writes
   `artifacts/task_manifests/frozen/pilot-<sha16>.json`. The working tree must be committed
   first; uncommitted changes block freezing.
7. **Checkpoint/perturbation validation is schema-only at the CLI:**
   `rq1 recovery validate-checkpoint <payload.json>` and
   `rq1 recovery validate-perturbation <payload.json>` (`src/rq1/cli.py:843-844`, `316-321`;
   fields at `src/rq1/recovery/validation.py:5-13`). These validate a payload, not a task, and
   prove nothing about ALFWorld.
8. **No command proves an expert route or reversible detour per frozen task.** The only
   implemented prover runs a live environment inside `_select_checkpoint`
   (`src/rq1/pilot/prelaunch.py:75-106`) and `RealRecoveryHarness._oracle`
   (`src/rq1/pilot/real_runtime/harnesses.py:191-214`). It is not reachable from any `tasks`
   command, so this step has no existing CLI entry point.

## 4. Acceptance criteria for a valid pilot set

- Exactly six frozen tasks, one per canonical family in section 2, with no family repeated.
- Every `TaskRecord.split` is `valid_seen`; zero `valid_unseen` tasks, IDs, or metadata.
- The frozen manifest passes `rq1 tasks validate --manifest <frozen> --require-frozen`
  (`src/rq1/tasks/validation.py:11-26`), and `overlap-check` reports no cross-manifest overlap.
- Each task has a checkpoint with a recorded prefix and observable-state digest, plus a
  controlled detour accepted by the existing live oracle: `solvable=True`, a non-terminal
  post-detour state, successful reference-continuation replay, and matching state digest
  (`src/rq1/pilot/real_runtime/harnesses.py:191-214`).
- Checkpoint and perturbation payloads pass the existing validators
  (`src/rq1/recovery/validation.py:5-13`).
- Evidence is recorded by the existing pipeline: immutable discovery, proposal, frozen manifest
  under `artifacts/task_manifests/`, and an approval referencing the manifest SHA-256. No
  hand-typed task IDs and no edits to `data/task_lists/pilot_seen.json`.

## 5. Validation report template

Fill only from a real university-machine run; leave blank until then.

| Field | Value |
|---|---|
| task_id | |
| family | |
| split | |
| source_sha256 | |
| game_sha256 | |
| discovery_artifact | |
| expert_route_source | |
| route_action_count | |
| checkpoint_id | |
| checkpoint_prefix_length | |
| checkpoint_state_digest | |
| perturbation_id | |
| detour_action | |
| oracle_solvable | |
| observable_post_state_digest | |
| checkpoint_payload_validation | |
| perturbation_payload_validation | |
| frozen_manifest_sha256 | |
| validation_run_id | |
| validated_at | |
| validator | |
| result | |
| notes | |

## 6. Offline tests

Run from the repository root with the project virtualenv:

```
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Interpreter: `.venv\Scripts\python.exe` (Python 3.11.9, project editable install).
Baseline before this document: `Ran 333 tests in 132.955s` — `OK (skipped=4)`.
`tests/test_pilot_readiness_doc.py` adds the family-table check; its result is reported in the
task report, not in this document.

These tests are offline and mock-based. Passing them does not demonstrate ALFWorld
compatibility, task validity, or recovery success.

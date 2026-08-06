# Pilot protocol

Run the recovery-aware master-plan pilot sequence in order: repository/doctor checks, raw model and tool calls, Hermes/plugin/profile/skill tests, bridge and multi-step ALFWorld tests, checkpoint/replay equality, deterministic solvable perturbation, controlled recovery, logging reconciliation, mini acquisition/snapshot runs, capacity/relevance audit, then the model and recovery-protocol decisions. Phase 2's fake bridge validates the HTTP contract only; repeat the bridge contract tests against a capability-confirmed real ALFWorld adapter before considering the integration operational.

Before plugin/profile validation, inspect `rq1 hermes-capabilities` and run the fake Phase 3 check. For installed Hermes, use a temporary `HERMES_HOME`, `HERMES_ENABLE_PROJECT_PLUGINS=1`, and `RQ1_RUN_REAL_HERMES_TESTS=1`; do not create profiles or load an LLM for discovery evidence. Capture native skill events only when a supported native operation is actually observed.

No final acquisition or unseen evaluation is permitted before a passing pilot report.

Real pilot tasks must be proposed from the installed `valid_seen` metadata through `rq1 tasks discover --split valid_seen` and `rq1 tasks propose --kind pilot`. Placeholder or manually typed task IDs cannot satisfy a real-pilot task-manifest gate.

## Phase 6 runner

The typed catalog contains `pilot_00` through `pilot_36`. Fake mode runs all 37 tests with simulated evidence and keeps `pilot_ready`, `real_integrated`, and `experimental_ready` false:

```bash
python -m rq1.cli pilot run --mode fake
```

Use `pilot prerequisites`, `--test`, `--group`, or `--from/--to` for bounded execution. Resume and retry always create new attempt IDs; completed attempt reports are immutable.

Phase 7 real execution is explicitly opted in:

```bash
python -m rq1.cli pilot plan --mode real
RQ1_RUN_REAL_PILOT_TESTS=1 python -m rq1.cli pilot run --mode real --yes
```

Real mode installs, downloads, and pulls nothing. Each pilot has an explicit Phase 7 handler and capability record; missing Hermes dispatch, native skill events, ALFWorld, replay, perturbation, or solvability capabilities block only their dependent checks. Fake fallback is forbidden. `valid_unseen` is rejected before adapter calls. Pilot 27-29 use only disposable `train`/`valid_seen` resources and are not final acquisition or evaluation.

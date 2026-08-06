# Phase 6 pilot runner

`rq1.pilot` owns the ordered `pilot_00`-`pilot_36` catalog, pilot-run state, evidence gates, fake/real runtimes, and aggregate report. It reuses the existing stage lock and SQLite run registry. The ALFWorld bridge remains the environment owner; Hermes remains a bridge client; the recovery package remains responsible for checkpoint/replay/perturbation contracts.

## Evidence and state

Each test attempt has a unique ID and immutable `attempt-report.json` plus hashed raw evidence. Pilot state under `state/pilot_runs/` is resumable control state, not scientific evidence. Episode, pilot-attempt, and recovery bindings are additive SQLite records.

Fake evidence is always `simulated` and cannot satisfy installed or real gates. Real mode re-probes capabilities and blocks if the installed setup, approved task list, Hermes surface, real ALFWorld adapter, or real recovery operation is unavailable.

## Execution

```bash
python -m rq1.cli pilot list
python -m rq1.cli pilot prerequisites --test pilot_17
python -m rq1.cli pilot run --mode fake
python -m rq1.cli pilot resume --run-id <run-id>
python -m rq1.cli pilot retry-failed --run-id <run-id>
python -m rq1.cli pilot report --run-id <run-id>
```

Selections do not run missing prerequisites implicitly unless `--include-prerequisites` is supplied. Resume re-probes blocked tests and restarts interrupted tests at a safe boundary. `retry-failed` additionally reruns failed tests under new attempt IDs.

## University boundary

```bash
python -m rq1.cli pilot plan --mode real
RQ1_RUN_REAL_PILOT_TESTS=1 python -m rq1.cli pilot run --mode real --yes
```

This command installs and downloads nothing, never uses `valid_unseen`, and never modifies personal/default Hermes profiles. Phase 7 supplies real installed-version evidence and performs the manual freeze after a go recommendation.

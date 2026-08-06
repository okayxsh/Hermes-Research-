# Reproducible agent-environment experiment

This repository provides a reproducible foundation for a controlled-recovery experiment: whether a persistent agent's naturally accumulated skill library helps it recover from plan-invalidating ALFWorld failures or creates post-failure retrieval noise. The public documentation intentionally describes implementation boundaries without claiming untested integrations.

The currently tested local layer includes configuration loading, stage state and reports, SQLite run claiming, synthetic episodes, snapshot validation, leakage checks, metrics, schemas, CI, a deterministic fake ALFWorld HTTP bridge, a capability-gated ALFWorld 0.4.2 text-adapter implementation, a capability-gated project-local Hermes plugin boundary, and the typed Phase 6 pilot runner. Hermes, Ollama, models, ALFWorld data/runtime, real bridge execution, and real Hermes plugin dispatch remain unverified.

## Local development quick start

```bash
python -m pip install -e .
python -m rq1.cli preflight
python -m rq1.cli doctor
python -m rq1.cli validate-config
python -m rq1.cli mock-run
python -m unittest discover -s tests -v
```

Run the fake bridge explicitly with:

```bash
python -m rq1.cli bridge-server --host 127.0.0.1 --port 8000
```

This bridge defaults to the fake local contract. A real server requires `--mode real --yes`, a supported installed package, and an indexed local data directory; it never falls back to fake mode.

Inspect the external boundary without installing or downloading anything:

```bash
python -m rq1.cli alfworld capabilities
python -m rq1.cli alfworld index --split valid_seen
```

## Recovery-aware pilot runner

```bash
python -m rq1.cli pilot list
python -m rq1.cli pilot plan --mode fake
python -m rq1.cli pilot run --mode fake
python -m rq1.cli pilot status
```

Fake completion validates `pilot_00` through `pilot_36` orchestration only and must retain `experimental_ready: false`. Real execution is reserved for Phase 7 and requires both `RQ1_RUN_REAL_PILOT_TESTS=1` and `--yes`; it installs and downloads nothing. Each real pilot test has a capability-gated handler and reports its exact blocked dependency rather than using fake fallback.

See [docs/PILOT_RUNNER.md](docs/PILOT_RUNNER.md) for selection, evidence, resume, artifacts, and university boundaries.

## Hermes boundary

The five local bridge tools and fake verification are documented in [docs/HERMES_INTEGRATION.md](docs/HERMES_INTEGRATION.md). The plugin is disabled unless an installed Hermes instance explicitly opts into trusted project plugins; fake verification does not establish Hermes compatibility.

## Isolated profile boundary

Phase 4 defines `rq1-pilot` and `rq1-acquisition` as isolated Hermes state, not repository copies. Use `python -m rq1.cli profiles plan` to inspect plans and `python -m rq1.cli profiles isolation-test` for the fully local fake lifecycle check. Real profile creation requires an installed, capability-confirmed Hermes CLI plus `--yes`; future `rq1-recovery-<snapshot>` profiles remain templates until frozen snapshots exist.

## University machine setup

The Ubuntu 22.04/24.04 machine-setup contract, stages, flags, reports, resume semantics, and pilot gate are documented in [docs/SETUP.md](docs/SETUP.md) and [docs/UNIVERSITY_RUNBOOK.md](docs/UNIVERSITY_RUNBOOK.md).

The intended master command is:

```bash
bash scripts/setup_machine.sh --yes --resume --verbose
```

Typed setup orchestration, the master script, thin stage wrappers, machine-readable schemas, and mocked setup tests are present. No real apt, Ollama, Hermes, model, ALFWorld, ALFWorld-data, profile, or GPU installation test was run during repository development. Do not run ad hoc upstream installers and then treat the repository as verified.

## Readiness boundary

- Installation verification may use the deterministic fake bridge to test health, start, step, status, reset, and abort.
- Real ALFWorld support remains unverified until the real adapter passes an actual start → step → reset test with installed data on the target university machine.
- Missing external capabilities must produce clean failed or blocked reports, never fabricated success or an unhandled traceback.
- Bounded mini acquisition/snapshot/evaluation checks are disposable pilot instrumentation and never populate final experiment outputs.

See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md), [docs/UNVERIFIED_INTEGRATIONS.md](docs/UNVERIFIED_INTEGRATIONS.md), and [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) before attempting external setup.

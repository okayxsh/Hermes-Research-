# Reproducible agent-environment experiment

This repository provides a reproducible foundation for studying agent behavior and task outcomes under controlled procedural guidance. The public documentation intentionally describes the research at a high level.

The currently tested local layer includes configuration loading, stage state and reports, SQLite run claiming, synthetic episodes, snapshot validation, leakage checks, metrics, schemas, CI, a deterministic fake ALFWorld HTTP bridge, and a capability-gated project-local Hermes plugin boundary. Hermes, Ollama, models, ALFWorld data/runtime, the real bridge adapter, and real Hermes plugin dispatch remain unverified.

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

This bridge validates the local HTTP contract only; it does not run ALFWorld.

## Hermes boundary

The five local bridge tools and fake verification are documented in [docs/HERMES_INTEGRATION.md](docs/HERMES_INTEGRATION.md). The plugin is disabled unless an installed Hermes instance explicitly opts into trusted project plugins; fake verification does not establish Hermes compatibility.

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

See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md), [docs/UNVERIFIED_INTEGRATIONS.md](docs/UNVERIFIED_INTEGRATIONS.md), and [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) before attempting external setup.

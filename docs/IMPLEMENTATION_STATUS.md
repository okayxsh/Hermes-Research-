# Implementation status

## Implemented and locally testable

- Repository layout, configuration templates, stage state/report foundations, locks, SQLite run claiming, synthetic episode flow, metrics, snapshot validation, leakage checks, schemas, and CI.
- A standard-library localhost HTTP bridge backed by a deterministic fake ALFWorld adapter.
- Fake bridge health, start, step, status, abort, reset, action limits, unique episode IDs, raw JSONL event logs, contract tests, and concurrent-episode integration tests.
- Typed machine-setup options, stage-result models, setup-stage dependency definitions, command redaction, subprocess abstraction, and a setup-state registry.
- Typed setup orchestration and stage handlers for preflight, system packages, Python, Ollama, Hermes, ALFWorld package/data, candidate models, base profiles, and installation verification.
- CLI surfaces for `setup-machine`, `setup-stage`, `verify-installation`, and `setup-status`.
- A project-local, opt-in Hermes plugin boundary with five bridge-backed tools, strict local validation, local-only HTTP enforcement, correlation headers, fake adapter/event coverage, run-registry bindings, reconciliation, and fake/real verification commands.

The fake bridge is sufficient for installation-plumbing verification only. It is not evidence of real ALFWorld compatibility.

## Installation workflow status

The required Ubuntu 22.04/24.04 setup contract is documented in [SETUP.md](SETUP.md). It includes stages for preflight, apt prerequisites, a locked Python 3.11 environment, Ollama, Hermes, text-only ALFWorld, ALFWorld data, candidate models, isolated profiles, and aggregate verification.

The repository contains `scripts/setup_machine.sh`, the thin 00–09 stage wrappers, dedicated mocked setup tests, report schemas, sanitized examples, and the typed Python orchestration. The scripts are intended to be invoked with `bash` on a supported Ubuntu target. Their external operations remain unverified until the university-machine setup produces evidence.

## Not installed or verified during implementation

No real apt packages, Ollama service, Hermes Agent installation, model pull, ALFWorld package, ALFWorld data, profile materialization, or GPU inference test was run while implementing the repository foundation. No live compatibility claim is made for:

- Ubuntu 22.04 or 24.04 setup execution
- Ollama serving or `hermes3:8b` inference
- Hermes installation, configuration, profiles, real tools, hooks, plugin discovery, or plugin behavior
- ALFWorld 0.4.2 imports, downloader output, task data, or runtime API
- the real bridge adapter

Missing external capabilities must remain clean, structured `failed` or `blocked` states with remediation rather than tracebacks or fabricated passes.

## Readiness definitions

- `installation_ready` may become true only after all required installation stages pass and the fake bridge HTTP workflow succeeds on the target machine.
- `pilot_ready` must remain false until the capability-gated real ALFWorld adapter completes an actual start → step → reset test using the installed package and downloaded data.
- Real Hermes-to-ALFWorld operation remains unverified until later pilot evidence is captured.
- A passing `verify-hermes-integration --mode fake` report is contract evidence only; it never sets a real Hermes or ALFWorld compatibility field.

Generated output paths are listed in [SETUP.md](SETUP.md). Machine-specific manifests and reports are ignored by Git; only schemas, documentation, and sanitized examples belong in the public repository.

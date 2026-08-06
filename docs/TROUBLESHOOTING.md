# Troubleshooting

## First check the implementation status

The complete workflow is wired through `scripts/setup_machine.sh` and the final 00–09 stage wrappers. Do not substitute ad hoc upstream installer commands when a stage fails; preserve the report and use the documented resume or force-stage recovery path. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

Inspect `state/setup_status.json`, `artifacts/stage_reports/installation.json`, and the referenced per-stage report first. Do not delete a lock, report, registry, model, profile, data directory, or environment to bypass a failure.

## Resume and targeted recovery

Use:

```bash
bash scripts/setup_machine.sh --resume --verbose
```

`--resume` revalidates passed stages and continues from the first incomplete stage. If a specific stage must be rerun, preview the effect first:

```bash
bash scripts/setup_machine.sh --dry-run --force-stage <stage> --verbose
```

Then, after reviewing the invalidated downstream stages:

```bash
bash scripts/setup_machine.sh --yes --force-stage <stage> --verbose
```

Forcing a stage must not remove external software, models, ALFWorld data, Hermes profiles, or historical reports.

## Common capability failures

### Unsupported machine

Preflight accepts only x86_64 Ubuntu 22.04/24.04, native or WSL2. Unsupported distribution/version/architecture, less than 16 GiB RAM, less than 25 GiB free disk, unwritable paths, or unavailable required network hosts must fail before installation begins.

GPU absence is recorded but is not a hard failure. A detected NVIDIA GPU with unusable driver/CUDA state should be reported separately so CPU fallback is explicit.

### `uv`, Python, or lockfile failure

The project requires Python 3.11 and `uv sync --locked`. A missing or stale `uv.lock` is an implementation/setup error; do not use an unlocked sync as an automatic repair. Record the failing command and resolve the lock deliberately in repository development.

### Ollama unavailable

Check the recorded install stage, service method, and localhost endpoint. Native Ubuntu should prefer systemd; WSL2 may require the managed fallback when systemd is unavailable. A process or binary alone is insufficient—the local API must respond. Never report the model as ready unless Ollama also reports the expected model metadata.

### Hermes unavailable or command shape changed

Check `hermes --help`, `hermes profile --help`, `hermes profile create --help`, and `hermes doctor`, then review `artifacts/manifests/hermes_capabilities.json`. If an expected flag or configuration command is absent, block the profile stage. Do not guess configuration keys or edit the user's default profile as a workaround.

### ALFWorld package or data missing

Treat these as separate failures:

- Package missing: the Python metadata/import probe or `alfworld-download` discovery fails.
- Data missing: `RQ1_ALFWORLD_DATA_DIR` is absent, unreadable, incomplete, or fails inventory validation.

`--skip-alfworld-data` prevents a download; it does not satisfy the data requirement. Missing package/data must produce structured remediation and keep readiness false.

### Real adapter is unavailable

Run `python -m rq1.cli alfworld capabilities`. The adapter supports exactly installed `alfworld==0.4.2`, indexed `train`/`valid_seen` text data, and no `valid_unseen` access. Do not alter package versions or substitute a random task. When capability evidence is ready, run the explicitly approved `rq1 alfworld smoke-test --split valid_seen --yes`; it records failure evidence rather than falling back to fake mode.

### Model missing or smoke test fails

`--skip-model` prevents a pull but does not create a pass. The primary model remains `hermes3:8b`. Install the fallback only with `--install-fallback-model`; never silently change model selection after a failure.

### Fake bridge passes but pilot is blocked

This is expected until real ALFWorld is verified. The fake bridge confirms the HTTP contract and setup plumbing. The pilot remains blocked until the real adapter performs start → step → reset using actual ALFWorld data on the target machine.

## Pilot runner failures

### Fake Phase 6 passes but returns no-go

This is expected. `mock_orchestration_ready: true` means all 37 runner contracts completed; `pilot_ready` and `experimental_ready` remain false until Phase 7 records real integrated evidence.

### A Phase 7 real pilot test is blocked

Read the attempt's `block_code`, capability snapshot, and remediation rather than rerunning the entire sequence. Common expected blocks are unavailable Ollama, unsupported Hermes dispatch, missing native skill events, missing approved task trajectories, and unsupported target relocation. A block is evidence that a required external surface was not observed; do not work around it with fake mode or a changed recovery protocol.

### Resume or retry a pilot

Use `python -m rq1.cli pilot status`, then `pilot resume --run-id <id>`. Ordinary resume does not rerun failed tests; use `pilot retry-failed --run-id <id>`. Real runs require the same opt-in and `--yes`. Every retry preserves prior evidence under a new attempt ID.

## Safe reporting

Failures should end as `failed` or `blocked` with the probe name, redacted diagnostics, and a next action. If output contains credentials, tokens, usernames, hostnames, IP addresses, serial numbers, or environment values, do not copy it into committed documentation or reports.

## Autopilot blockers

Use `rq1 autopilot status --run-id <id>` and `logs --run-id <id>` to inspect append-only contingency evidence. Do not retry an uncertain environment or skill mutation in place; resolve the blocker and resume from the documented safe boundary.

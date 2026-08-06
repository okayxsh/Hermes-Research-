# Machine setup

This repository is for a reproducible agent-environment experiment. The machine setup contract targets x86_64 Ubuntu 22.04 or 24.04, either natively or under WSL2.

> **Current status:** typed setup orchestration, CLI commands, `scripts/setup_machine.sh`, thin 00–09 wrappers, schemas, and mocked setup tests are implemented. During repository development, no real apt installation, Ollama or Hermes installation, model pull, ALFWorld installation, ALFWorld data download, or GPU test was run. The workflow remains operationally unverified until it runs on the university machine.

## Requirements and boundaries

- Hard minimums: 16 GiB RAM and 25 GiB free disk space. Twenty-four GiB RAM is recommended.
- GPU and CUDA visibility are recorded when available, but absence of a GPU is not by itself an installation failure; local inference may fall back to CPU.
- The project environment uses Python 3.11, `uv`, `.venv`, and a committed `uv.lock`. The installation flow must use `uv sync --locked` rather than silently updating dependency resolutions.
- ALFWorld is pinned to `alfworld==0.4.2` and installed without visual/THOR extras. Package installation and data availability are independent capabilities.
- Ollama and Hermes are installed only through capability-gated stages based on their official Linux installation guidance. The resolved versions and installer hashes must be recorded rather than treated as permanently compatible.
- The primary model is `hermes3:8b`. `llama3.1:8b` is an explicit fallback and must never be substituted silently.
- Hermes profiles are named `rq1-pilot` and `rq1-acquisition`. They must be isolated, created without bundled skills, and must not become the user's default profile.

Official references used to define this setup contract:

- [uv installation](https://docs.astral.sh/uv/) and [locked synchronization](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Ollama Linux installation](https://docs.ollama.com/linux)
- [Hermes Agent installation](https://hermes-agent.nousresearch.com/docs/getting-started/installation) and [profile commands](https://hermes-agent.nousresearch.com/docs/reference/profile-commands/)
- [ALFWorld installation and data download](https://github.com/alfworld/alfworld)

These links describe upstream tools. They do not establish that the versions available on a target machine work with this repository.

## Staged setup contract

The master entry point is `scripts/setup_machine.sh`. It executes these stages in dependency order:

| Stage | Script | Required result |
|---|---|---|
| 00 | `00_preflight.sh` | Confirm supported Ubuntu/WSL2 and x86_64; inspect CPU, RAM, disk, GPU/VRAM, driver/CUDA visibility, network reachability, Git, curl, bootstrap Python, repository revision, writable paths, and sudo availability. Fail before installation if a hard requirement is absent. |
| 01 | `01_install_system_dependencies.sh` | Install only documented apt prerequisites and record resolved package versions. |
| 02 | `02_create_python_environment.sh` | Bootstrap `uv`, provision Python 3.11, create `.venv`, and run `uv sync --locked`. |
| 03 | `03_install_ollama.sh` | Reuse or install Ollama, bind its service to localhost, and verify its local API. Prefer systemd where available; use a managed WSL process only when systemd is unavailable. |
| 04 | `04_install_hermes.sh` | Reuse or perform a per-user Hermes CLI installation with browser setup skipped, then capture help and capability probes. Bundled skills are disabled later when the isolated experiment profiles are created. Never modify an existing personal/default profile. |
| 05 | `05_install_alfworld.sh` | Install and import-test pinned text-only ALFWorld, then verify that `alfworld-download` is discoverable. Do not download data in this stage. |
| 06 | `06_download_alfworld_data.sh` | Reuse valid existing data or explicitly invoke the official downloader. Use `RQ1_ALFWORLD_DATA_DIR` when set; otherwise use an isolated experiment cache. |
| 07 | `07_pull_candidate_models.sh` | Pull and inspect `hermes3:8b`, then run a deterministic raw-inference smoke test. Pull `llama3.1:8b` only when explicitly requested. |
| 08 | `08_create_base_profiles.sh` | Create and validate isolated `rq1-pilot` and `rq1-acquisition` profiles with `--no-skills`; configure only settings confirmed by installed Hermes capability/help probes. |
| 09 | `09_verify_installation.sh` | Re-probe dependencies, validate the profiles, and exercise the deterministic fake bridge over local HTTP. Keep real ALFWorld compatibility unverified. |

No stage may perform an implicit ALFWorld reset or data download. Missing commands, unsupported help output, absent data, unavailable services, and unsupported platforms must produce a structured failed or blocked result with remediation guidance, not an unhandled traceback.

## Master command and flags

The normal university-machine invocation is:

```bash
bash scripts/setup_machine.sh --yes --resume --verbose
```

The exact command-line contract is:

| Flag | Meaning |
|---|---|
| `--dry-run` | Resolve the stage plan and probes without apt installs, installer execution, model pulls, profile creation, data downloads, service mutation, or fake-server startup. Dry-run reports/manifests may still be written under ignored `artifacts/`. |
| `--yes` | Required confirmation for any non-dry-run setup or stage command. It does not authorize destructive replacement of existing data or personal profiles. |
| `--resume` | Revalidate passed stages and continue from the first incomplete or invalid stage. |
| `--skip-system-packages` | Do not run apt. The stage passes only if the required packages are already detected; otherwise setup remains incomplete. |
| `--skip-model` | Do not pull a model. Existing model capability may be reused; otherwise installation readiness remains false. |
| `--skip-alfworld-data` | Do not download ALFWorld data. Existing validated data may be reused; otherwise installation readiness and pilot readiness remain false. |
| `--install-fallback-model` | Also pull `llama3.1:8b`. This never changes the primary model automatically. |
| `--force-stage <stage>` | Rerun the named stage and invalidate its downstream setup results. It must not delete installed software, models, data, profiles, or historical reports. |
| `--verbose` | Print expanded progress and redacted command diagnostics. Secrets must remain redacted. |

Valid `--force-stage` names are `preflight`, `system-packages`, `python-environment`, `ollama`, `hermes`, `alfworld-package`, `alfworld-data`, `candidate-models`, `base-profiles`, and `installation-verification`.

### Resume and force semantics

`--resume` does not trust a prior `passed` label by itself. It recomputes the stage input fingerprint and re-runs non-mutating capability checks. A changed input, missing artifact, absent service, or invalid report makes that stage incomplete and prevents dependent stages from being treated as passed.

`--force-stage` is a non-destructive recovery mechanism. It resets the selected stage and its downstream setup-state entries to pending, while preserving immutable historical reports and externally installed assets. Forcing a stage must require confirmation unless combined with `--dry-run`.

## Reports and manifests

Machine-specific output is generated under ignored `artifacts/` paths:

```text
artifacts/stage_reports/installation.json
artifacts/manifests/machine_manifest.yaml
artifacts/manifests/software_versions.yaml
artifacts/manifests/model_manifest.yaml
artifacts/manifests/alfworld_data_manifest.yaml
artifacts/manifests/hermes_capabilities.json
```

Per-stage reports use `pending`, `running`, `passed`, `failed`, `blocked`, or `skipped`. They record a run ID, attempt ID, timestamps, input fingerprint, redacted commands, probes, produced artifacts, warnings, errors, skip reason, and remediation.

The aggregate installation report distinguishes `installed`, `configured`, `import_tested`, `smoke_tested`, `real_integration_tested`, and `unverified`. It exposes separate booleans:

- `installation_ready`: the required software, data, model, profiles, and fake bridge verification are present.
- `pilot_ready`: `installation_ready` is true and the real ALFWorld adapter has passed an actual start → step → reset test on the target machine.

Reports must omit usernames, hostnames, IP addresses, serial numbers, credentials, tokens, and raw environment-variable values.

## Bridge verification and pilot gate

Stage 09 may start the existing deterministic fake bridge on an ephemeral localhost port and exercise health, start, step, status, reset, and abort. Passing that check validates only the local HTTP contract and installation plumbing.

Before a real pilot may start, the capability-gated real adapter must use the installed ALFWorld package and downloaded data to complete a real episode start, one valid step, and an explicit reset. Until that exact gate passes, the report must keep `pilot_ready: false`, `real_integration_tested: false`, and real ALFWorld compatibility `unverified`.

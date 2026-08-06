# University machine runbook

This runbook provisions and verifies one Ubuntu machine for a reproducible agent-environment experiment. It does not run acquisition, evaluation, or analysis.

> **Implementation gate:** do not follow the execution commands until [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) says the complete setup workflow is runnable. No real external installation was executed during repository implementation.

## 1. Prepare the machine

Use an x86_64 Ubuntu 22.04 or 24.04 installation, either native or under WSL2. Confirm at least 16 GiB RAM and 25 GiB available storage; 24 GiB RAM is recommended. Clone the repository to a stable writable path and check out the intended commit.

Do not manually create or modify Hermes profiles before setup. If Hermes is already installed, preserve the default/personal profile and let capability probes determine whether the installation can be reused.

## 2. Preview the staged setup

From the repository root:

```bash
bash scripts/setup_machine.sh --dry-run --verbose
```

Review the proposed preflight, system package, Python, Ollama, Hermes, ALFWorld, data, model, profile, and verification stages. Dry-run must not install packages, invoke remote installers, pull models, download data, create profiles, or mutate services.

## 3. Run or resume installation

After the preview and local approval:

```bash
bash scripts/setup_machine.sh --yes --resume --verbose
```

The default run installs or reuses:

- documented Ubuntu prerequisites;
- a locked Python 3.11 `.venv` managed by `uv`;
- Ollama serving on localhost;
- a per-user Hermes Agent CLI installation without browser setup, followed by experiment profiles created without bundled skills;
- pinned text-only `alfworld==0.4.2` and its separately downloaded data;
- primary model `hermes3:8b`;
- isolated `rq1-pilot` and `rq1-acquisition` Hermes profiles; and
- the repository's local ALFWorld bridge.

Optional flags are exactly:

```text
--dry-run
--yes
--resume
--skip-system-packages
--skip-model
--skip-alfworld-data
--install-fallback-model
--force-stage <stage>
--verbose
```

Skip flags are non-mutating choices. They leave setup incomplete unless the capability already exists and passes a fresh probe. `--install-fallback-model` additionally installs `llama3.1:8b`; it never silently replaces the primary model.

## 4. Recover without deleting state

Resume after an interruption with the same command:

```bash
bash scripts/setup_machine.sh --yes --resume --verbose
```

To rerun one stage, first preview:

```bash
bash scripts/setup_machine.sh --dry-run --force-stage <stage> --verbose
```

Then execute with `--yes`. A forced stage invalidates only that stage and its downstream setup status. It preserves installed software, models, profiles, ALFWorld data, logs, and historical reports.

## 5. Review installation evidence

Do not rely only on terminal output. Review:

```text
artifacts/stage_reports/installation.json
artifacts/manifests/machine_manifest.yaml
artifacts/manifests/software_versions.yaml
artifacts/manifests/model_manifest.yaml
artifacts/manifests/alfworld_data_manifest.yaml
artifacts/manifests/hermes_capabilities.json
```

Confirm that sensitive machine identifiers and secrets are absent. Verify that each required stage is `passed`, no required capability is merely `skipped`, and the aggregate report distinguishes installation, configuration, import tests, smoke tests, real integration tests, and unverified items.

## 6. Installation verification

Stage 09 starts the deterministic fake bridge on an ephemeral localhost port and exercises health, episode start, step, status, reset, and abort. It re-probes external commands and validates base profiles only through the Phase 4 capability-gated lifecycle; real profile isolation requires separate observed evidence.

A successful fake bridge workflow may set `installation_ready: true` when every other installation requirement passes. It does not establish real ALFWorld or Hermes-to-ALFWorld compatibility.

## 7. Real ALFWorld pilot gate

Before starting any pilot, the capability-gated real adapter must load the installed ALFWorld package and configured dataset, start a real episode, execute one valid step, and explicitly reset that episode. Capture the requests, responses, runtime version, data manifest, and bridge events.

Only that successful start → step → reset test may set:

```text
pilot_ready: true
real_integration_tested: true
```

If the package, downloader, data, real adapter, or expected runtime behavior is missing, malformed, or unsupported, stop with a structured blocked report and remediation. Do not fall back to the fake adapter while labeling the result as real.

## 8. Handoff

Preserve the installation report, manifests, stage reports, repository revision, and raw gate logs. Do not publish credentials or machine-identifying details. Proceed to the separate pilot protocol only after both readiness fields and all required evidence have been reviewed.

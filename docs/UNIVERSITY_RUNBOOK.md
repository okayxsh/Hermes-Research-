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

Before starting any pilot, inspect `python -m rq1.cli alfworld capabilities` and `python -m rq1.cli alfworld index --split valid_seen`. Then run `python -m rq1.cli alfworld smoke-test --split valid_seen --yes`. The capability-gated real adapter must load the installed ALFWorld 0.4.2 package and configured dataset, start one indexed real episode, execute one valid step, report cached status, explicitly reset, and controller-abort it. Capture the immutable report, runtime version, data identity, and bridge events.

Only that successful start → step → reset test may set:

```text
pilot_ready: true
real_integration_tested: true
```

If the package, downloader, data, real adapter, or expected runtime behavior is missing, malformed, or unsupported, stop with a structured blocked report and remediation. Do not fall back to the fake adapter while labeling the result as real.

## 8. Handoff

Preserve the installation report, manifests, stage reports, repository revision, and raw gate logs. Do not publish credentials or machine-identifying details. Proceed to the separate pilot protocol only after both readiness fields and all required evidence have been reviewed.

## 9. Phase 7 real pilot and freeze

Phase 6 prepares the runner but does not create real evidence. On the approved machine:

```bash
python -m rq1.cli pilot plan --mode real
RQ1_RUN_REAL_PILOT_TESTS=1 python -m rq1.cli pilot run --mode real --yes
```

Resume with the reported run ID and never substitute fake evidence for a blocked test. A `go` recommendation permits manual Phase 7 approval; it does not automatically freeze versions, counts, tasks, prompts, or recovery policy.

If target relocation or native skill retrieval is not capability-observed, the relevant real handlers will produce immutable blocked evidence and the report remains `no_go`. Do not replace the controlled perturbation or retrieval-noise metric during execution; any alternative requires manual research-protocol approval.

## 10. Final-stage gate

Do not begin final acquisition or evaluation during the pilot. After a real Phase 7 `go`, create separately approved immutable freezes:

```bash
python -m rq1.cli freeze plan
python -m rq1.cli freeze environment --approval-file <environment.json> --pilot-report <go-report.json> --yes
python -m rq1.cli freeze protocol --approval-file <protocol.json> --pilot-report <go-report.json> --yes
```

Every final runner revalidates both freezes, the clean Git commit, frozen model/prompts/manifests, and capability evidence. It blocks rather than falling back to a fake adapter when recovery-profile, Hermes, or controlled-perturbation support is unavailable.

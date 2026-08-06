# University-machine execution runbook

Use native Ubuntu 22.04/24.04 or WSL2 Ubuntu on x86_64 with at least 16 GiB RAM, 25 GiB free disk, and GPU visibility when available. Start from a fresh checkout and do not reuse untrusted setup state.

## 1. Preflight

```bash
uname -m
python3 --version
git status --short
python3 -m rq1.cli doctor
python3 -m rq1.cli validate-config
bash scripts/setup_machine.sh --dry-run --verbose
```

Confirm x86_64, Python 3.11 availability, a clean intended commit, supported Ubuntu/WSL2, resources, network, and a dry-run with no installer/model/data mutation.

## 2. Full setup

```bash
bash scripts/setup_machine.sh --yes --resume --verbose
```

This runs the existing 00–09 setup sequence: dependencies, Python/uv, Ollama, Hermes, ALFWorld 0.4.2/data, model, isolated profiles, and installation verification.

## 3. Inspect status and logs

```bash
python3 -m rq1.cli setup-status
python3 -m rq1.cli stage-status
find artifacts/stage_reports artifacts/manifests -type f -maxdepth 2 -print
```

Review `artifacts/stage_reports/installation.json` and the machine, software, model, ALFWorld-data, and Hermes manifests.

## 4. Recover a blocked setup

```bash
bash scripts/setup_machine.sh --yes --resume --verbose
bash scripts/setup_machine.sh --dry-run --force-stage <stage> --verbose
bash scripts/setup_machine.sh --yes --force-stage <stage> --resume --verbose
```

Use only the documented stage names in `docs/SETUP.md`; preserve reports and data. Do not bypass a blocked capability.

## 5. Real pilot

```bash
python3 -m rq1.cli alfworld capabilities
python3 -m rq1.cli alfworld index --split valid_seen
python3 -m rq1.cli alfworld smoke-test --split valid_seen --yes
python3 -m rq1.cli pilot plan --mode real
RQ1_RUN_REAL_PILOT_TESTS=1 python3 -m rq1.cli pilot run --mode real --yes
```

The pilot uses only approved `valid_seen` tasks and performs real capability-gated checks.

## 6. Confirm PILOT_GO and readiness

```bash
python3 -m rq1.cli pilot report --run-id <PILOT_RUN_ID>
python3 -c 'import json; p=json.load(open("artifacts/pilot_reports/<PILOT_RUN_ID>/pilot-report.json")); assert p["go_no_go"]["decision"] == "go" and p["pilot_ready"] is True and p["experimental_ready"] is True; print("PILOT_GO and experimental_ready=true")'
```

Do not proceed unless the checks pass and all mandatory real-integrated evidence is reviewed.

## 7. Manual approval and freezes

```bash
python3 -m rq1.cli freeze plan
python3 -m rq1.cli freeze environment --approval-file <ENVIRONMENT_APPROVAL.json> --pilot-report artifacts/pilot_reports/<PILOT_RUN_ID>/pilot-report.json --yes
python3 -m rq1.cli freeze protocol --approval-file <PROTOCOL_APPROVAL.json> --pilot-report artifacts/pilot_reports/<PILOT_RUN_ID>/pilot-report.json --yes
```

Approval files must contain the reviewed frozen model, prompts, policies, task counts, budgets, timeouts, relevance/exclusion rules, and references.

## 8. Final autopilot

```bash
bash scripts/rq1_autopilot.sh plan --mode final
bash scripts/rq1_autopilot.sh final --approval artifacts/approvals/final-run-approved.json --yes
```

The final command revalidates every freeze and approval and blocks if any real adapter or evidence is unavailable.

## 9. Stop and resume safely

```bash
python3 -m rq1.cli autopilot stop --run-id <AUTOPILOT_RUN_ID>
python3 -m rq1.cli autopilot status --run-id <AUTOPILOT_RUN_ID>
python3 -m rq1.cli autopilot logs --run-id <AUTOPILOT_RUN_ID>
python3 -m rq1.cli autopilot resume --run-id <AUTOPILOT_RUN_ID>
```

Uncertain mutations always create a new attempt; never retry them in place.

## 10. Final outputs

Setup evidence is under `artifacts/stage_reports/` and `artifacts/manifests/`; pilot evidence is under `artifacts/pilot_reports/<PILOT_RUN_ID>/`; autopilot state is under `artifacts/autopilot/<AUTOPILOT_RUN_ID>/`; final outputs are under `results/final/<RUN_ID>/` when a validated final run exists.

## 11. Common blockers

- Ollama: service unavailable, model digest mismatch, tool-calling failure, or 4 GiB VRAM OOM; reduce concurrency or remain blocked.
- Hermes: unsupported CLI/profile/plugin/hooks, missing native skill observability, or contaminated profiles; do not invent commands or use personal profiles.
- ALFWorld: missing 0.4.2 package/data, invalid index, unsupported runtime, reset/replay mismatch, perturbation, or solvability failure.
- GPU: absent/incorrect driver, CUDA visibility failure, OOM, or unstable latency; CPU fallback is not a real-readiness substitute.
- Disk/profiles: below 25 GiB free, interrupted writes, path collisions, shared state, or evaluation skill writes; stop and preserve evidence.

## 12. `valid_unseen` protection

Never discover, read, freeze, queue, or run `valid_unseen` before the approved environment/protocol freezes, validated acquisition/snapshots/profiles/recovery evidence, and immutable evaluation activation. The YAML flag alone cannot authorize it; only the explicitly activated final evaluation command may access the frozen list.

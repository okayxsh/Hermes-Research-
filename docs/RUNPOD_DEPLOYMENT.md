# RunPod deployment runbook (pre-provisioning)

This runbook prepares an already-running RunPod Pod for the RQ1 repository. The
repository changes do not create a Pod, start a Pod, rent a GPU, or launch a
scientific phase. There is an intentional manual approval boundary between the
validation gate and the final launch gate.

## Recommended machine

- RunPod Secure Cloud, On-Demand (not Spot/interruptible).
- Preferred: one NVIDIA RTX A6000 with 48 GB VRAM.
- Fallbacks: A40 48 GB, RTX A5000 24 GB, or RTX 4090 24 GB.
- At least 32 GB system RAM, 8 vCPU, and 80–100 GB free storage.
- Attach a network volume when the experiment must be recoverable on a replacement Pod.

`configs/runpod/deployment.example.json` is the reviewable deployment
specification. The documented PyTorch image is only a CUDA-capable base; the
bootstrap creates the locked project environment with Python 3.11, which is the
scientific requirement.

Public Secure Cloud rates are availability-dependent and can change. The
RunPod pricing page currently lists a 48 GB RTX A6000 at roughly $0.53/hour and
an A40 at roughly $0.49/hour; storage and any regional differences are extra.
Treat the live console/CLI quote as authoritative immediately before approval.

## Provisioning command (documented, deliberately not run)

The official agent setup intentionally does not install `runpodctl`, Flash, or
API keys yet; those are installed on demand by the RunPod plugin. If you later
choose CLI control, install/configure `runpodctl` and SSH on the local machine
with the official installer and `runpodctl doctor`. Neither command was run in
this task because no RunPod credentials were provided. The create command below
is the first billable boundary and must be reviewed manually:

```bash
bash <(curl -sL cli.runpod.io)
runpodctl doctor

# DO NOT RUN until you approve the GPU, price, volume, and Pod name.
runpodctl pod create \
  --name "rq1-hermes-alfworld" \
  --gpu-id "NVIDIA RTX A6000" \
  --image "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
  --container-disk-in-gb 60 \
  --volume-in-gb 120 \
  --volume-mount-path /workspace/persistent \
  --gpu-count 1
```

The exact official Codex agent setup is also non-billable. The marketplace has
been registered locally with:

```bash
codex plugin marketplace add https://github.com/runpod/runpod-plugins-official.git
```

The remaining steps are an explicit Codex UI/user action: open `/plugins`,
install **Runpod** from the RunPod marketplace, and reload if prompted. If the
hosted MCP tools do not appear after installation, the official fallback is:

```bash
codex mcp add runpod --transport http https://mcp.getrunpod.io/
```

OAuth sign-in must then be approved by the user in the Codex UI. I did not
install the plugin or perform OAuth in this task, and did not add those tools to
the research repository.

The command intentionally omits an interruptible/Spot flag. For a network
volume that must be reused by a replacement Pod, create/select it in the same
datacenter and pass its ID using the current `runpodctl` reference; do not
create storage automatically as part of experiment launch.

## Checkout and persistent layout

After the Pod exists and SSH is available, place the reviewed, committed
checkout on the mounted volume. Do not use an uncommitted working tree for the
real experiment:

```bash
mkdir -p /workspace/persistent
cd /workspace/persistent
git clone https://github.com/okayxsh/Hermes-Research- rq1-repo
cd /workspace/persistent/rq1-repo
git rev-parse --verify HEAD
```

The intended layout is:

```text
/workspace/persistent/
  rq1-repo/
    artifacts/                 setup, validation, manifests, logs
    results/final/<run-id>/    final checkpoint/results/errors/logs
    state/                     stage and lock state
  models/ollama/               persistent Ollama model store
  alfworld_data/               ALFWorld 0.4.2 data
  backups/<run-id>/             mirrored critical evidence
  logs/                         host/service logs
  manifests/                   machine and deployment evidence
```

The complete directory to copy/download at the end is:

```text
/workspace/persistent/rq1-repo/results/final/<EXPERIMENT_ID>/
```

Copy the matching backup directory as well:

```text
/workspace/persistent/backups/<EXPERIMENT_ID>/
```

Copying only `checkpoint.json` is insufficient; the manifest, JSONL journals,
phase manifests, errors, and logs are all needed to resume safely.

## Bootstrap

The bootstrap is idempotent and target-only. It verifies Linux/x86_64, GPU,
RAM, disk, and a resolvable Git commit before it installs `git`, `curl`,
`build-essential`, `zstd`, `tmux`, `jq`, `ripgrep`, `ffmpeg`, and the existing
repository setup stages. It points ALFWorld data, Ollama models, and backups at
the persistent volume.

Preview without mutation:

```bash
bash scripts/runpod/bootstrap.sh --dry-run --persistent-root /workspace/persistent
```

Run only after the Pod and volume have been manually approved:

```bash
bash scripts/runpod/bootstrap.sh --apply \
  --repo-root /workspace/persistent/rq1-repo \
  --persistent-root /workspace/persistent \
  --verbose
```

The known Hermes profile limitation may leave Stage 08 (`base-profiles`)
blocked. Bootstrap permits the repository's diagnostic Stage 09 to run, but it
does not turn that diagnostic pass into installation, pilot, or scientific
readiness.

## Validation gate

Run this after bootstrap. It performs real, read-only capability checks plus
explicitly labelled synthetic checkpoint/interruption smoke tests. It never
accesses `valid_unseen`, starts acquisition/evaluation, or writes a launch
approval.

```bash
bash scripts/runpod/validate_real_stack.sh \
  --repo-root /workspace/persistent/rq1-repo \
  --persistent-root /workspace/persistent \
  --backup-dir /workspace/persistent/backups
```

The report is `artifacts/runpod/validation-report.json`. A launch marker is
written only when every required check—including real Hermes integration—passes:

```text
/workspace/persistent/rq1-validation-passed.json
```

If real Hermes dispatch or native skill-event observation is unavailable, the
validator fails closed and no marker is written. Fake bridge or synthetic
checkpoint evidence cannot satisfy that gate.

## Scientific launch gate

Review the validation report, frozen approvals, task manifests, and the exact
configuration. Then make the separate manual approval explicit:

```bash
# Acquisition, detached under tmux:
bash scripts/runpod/start_rq1.sh start \
  --phase acquisition --run-id <ACQUISITION_ID> \
  --backup-dir /workspace/persistent/backups \
  --approve-launch

# Evaluation, only after immutable activation exists:
bash scripts/runpod/start_rq1.sh start \
  --phase evaluation --run-id <EVALUATION_ID> \
  --activation-manifest artifacts/evaluation_activation/<ACTIVATION>.json \
  --backup-dir /workspace/persistent/backups \
  --approve-launch
```

The existing CLI still enforces the final freeze, profile-isolation, recovery,
and activation gates. A blocked real adapter therefore stops instead of falling
back to fake execution.

## Unattended operation and recovery

```bash
tmux attach -t rq1-acquisition-<ACQUISITION_ID>
# detach: Ctrl+B, then D

bash scripts/runpod/start_rq1.sh status --run-id <EXPERIMENT_ID>
bash scripts/runpod/start_rq1.sh logs --run-id <EXPERIMENT_ID>
bash scripts/runpod/backup_state.sh <EXPERIMENT_ID> /workspace/persistent/backups
```

For a normal resume after an SSH disconnect, process kill, reboot, or Pod
replacement, copy the complete `results/final/<run-id>` tree to the same
relative location and run:

```bash
bash scripts/runpod/start_rq1.sh resume \
  --phase acquisition --run-id <ACQUISITION_ID> \
  --backup-dir /workspace/persistent/backups \
  --approve-launch
```

The durable `results.jsonl` journal is completion authority. A result persisted
before a crash is detected and is not repeated, even if the checkpoint was not
replaced. An interrupted in-progress episode has no successful result row and
restarts from the beginning. Resume rejects task/order, condition, seed,
model/runtime, configuration, activation, or library-hash drift.

## Kaggle findings carried into this runbook

The prior dry-run report states that Kaggle successfully demonstrated Python
3.11, repository config/checkpoint tests, ALFWorld 0.4.2 and a real
`valid_seen` start → step → reset/abort smoke, Ollama after installing `zstd`,
`hermes3:8b` returning `READY`, and Hermes Agent 0.21.1 installation. The
Hermes installer may open an interactive wizard, so the bootstrap captures the
installer and uses the repository's non-browser/no-skills flags. These are
portable preparation findings, not evidence that the current RunPod machine has
passed the real validation gate.

## Known blockers and approvals

- The current local checkout has no Git commit; validation correctly refuses it.
- Real Hermes dispatch, native skill-event observation, profile isolation, and
  final acquisition/evaluation adapters remain capability-gated in this repo.
- No command in this task installed `runpodctl`, authenticated an account,
  created storage, created/started a Pod, pulled a model, downloaded ALFWorld,
  ran a pilot, or launched a final experiment.
- Approve the RunPod GPU/price, persistent-volume location, reviewed Git commit,
  real validation report, and final scientific approval separately.

## Official references

- [RunPod agent setup instructions](https://docs.runpod.io/agent-setup.md)
- [RunPod agent skills and CLI setup](https://docs.runpod.io/get-started/agent-skills)
- [RunPod CLI installation and `doctor`](https://docs.runpod.io/runpodctl/overview)
- [RunPod Pod create flags](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)
- [RunPod persistent/network volumes](https://docs.runpod.io/runpodctl/reference/runpodctl-network-volume)
- [RunPod pricing](https://www.runpod.io/pricing)

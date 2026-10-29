# RQ1 University 4-GPU Execution Package

This package prepares a reproducible, offline-capable execution path for the prospective six-arm selective-memory study. It does not alter the frozen Oct 20 results, original rater labels, amended evaluation records, or archived packages. It does not install software, download a model, or run inference during local preparation.

## State and evidence boundaries

- Scientific parent commit: `c052e2cbdaa63419db4122cf08b1ccc64b38b3b4`.
- Base frozen evaluation remains the Oct 20 result. This prospective design is separate and exploratory until its protocol, relevance coverage, gate calibration, task-cluster precision simulation, and institutional approval are complete.
- Six arms: `D0`, `D18`, `Rel`, `Irr`, `Fill`, `Gate`. Planned floor: 36 valid-seen pilot episodes plus 540 valid-unseen confirmatory episodes. These are counts in a manifest, not a claim that the sample is adequately powered.
- The 30 confirmatory task identities have historical RQ1 outcomes in the project. They remain `valid_unseen` relative to the acquisition split, but are not a new untouched holdout for the project. Freezing and launch approval require explicit acknowledgment of this exposure.
- Historical NoLib and Accum-18 units stay separate. Current compatibility evidence does not establish that they replace contemporaneous cells. Reuse is sensitivity-only unless a documented compatibility review and author approval say otherwise.
- Four physical university GPUs, Linux/Ubuntu, Python 3.11, 64 GiB RAM, at least 16 GiB free per GPU, the locked dependencies, local ALFWorld data, frozen Sentence-BERT snapshot, exact local Gemma model, tokenizer, and approved filler are required for a real launch.
- A check or mock passing never demonstrates real GPU placement or throughput. Only `nvidia-smi` process evidence mapped to the four UUIDs can establish placement.

## First Friday check

On the university Linux host, from the repository root:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu check --json
```

It is read-only. A blocked result is expected until the target hardware, Python environment, and frozen external inputs are available. It creates no files and performs no model calls.

## Preparation flow

1. Inspect `university-gpu/docs/PROTOCOL.md` and `university-gpu/docs/OPERATIONS.md`.
2. Copy `university-gpu/config.example.json` to an operator-controlled path outside the checkout. Set its environment variables to frozen input paths on the university host. Keep outputs outside the Git checkout.
3. Run `rq1 university-gpu setup` for a dry-run plan. Project-local dependency setup is a separate, approval-gated action; no system install or model pull is supported.
4. Produce the blinded task-skill applicability labels and a separate valid-seen similarity-score file. Keep both source CSVs immutable.
5. Propose `Gate` calibration with `gate-calibration propose`; validate the result, obtain human approval bound to its hash, then freeze it. The rule uses adjudicated applicability labels only, not recovery outcomes.
6. Supply exact local model/tokenizer, library, task, controlled-failure, calibration, and filler files. Confirm filler and token-accounting rules with the authors before freezing.
7. Build a `pilot` proposal first. Confirm task eligibility and all 18 dual ratings per task, inspect the proposed manifest and hash, then freeze with a human approval file. The source repository must be committed and clean.
8. Run the pilot only after the university operator and authors authorize GPU inference. Inspect failure modes, token matching, placement, and protocol deviations before proposing confirmatory work.
9. Do not freeze or launch confirmatory episodes until relevance coverage, task eligibility, outcome-blind Gate calibration, task-cluster precision/power simulation, exposure acknowledgment, and institutional approval are signed off.

Examples are in `docs/OPERATIONS.md`. Immutable proposals and freezes should be written to a secure evidence directory outside the source checkout.

## Offline validation

From the repository root, without dependency installation:

```powershell
$env:PYTHONPATH = "src"
py -3.11 -m unittest discover -s tests -p "test_university_gpu*.py" -v
py -3.11 -m compileall -q src/rq1/university_gpu
```

Run the existing offline test suite separately where its declared test dependencies are already installed. The package uses standard-library tests and fake providers; mocks are not evidence of compatibility with Ollama, ALFWorld, university GPUs, or institutional policies.

## Operator restrictions

- No automatic model/tag fallback, `ollama pull`, system package installation, or implicit inference.
- GPU smoke requests and scientific launch require an approval file plus `--yes`; approvals must name the approver, record, and exact manifest hash.
- Confirmatory approval also records that the selected task outcomes were historically observed.
- The scheduler records unique physical attempts and retries. It offers resumable result writes and one authoritative result per unit; it does not promise exactly-once physical execution.
- Stop requests let active episodes finish and prevent new assignments. Cutoff times prevent starting a cell too close to the cutoff.
- Merge and the complete-only `export` require a complete, hash-valid run. `export-progress` can preserve a stopped incomplete run and marks its missing units explicitly.

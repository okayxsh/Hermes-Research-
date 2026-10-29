# Validation gates and known limits

## Local package validation

Offline validation is intentionally separate from model or hardware readiness. Standard-library tests exercise six-arm assignment/balance, manifest hash and approvals, task/relevance leakage guards, exact fake-tokenizer accounting, endpoint environment construction, GPU UUID parsing/placement attribution, worker ownership, retries, heartbeats, safe stop/cutoff, incomplete/corrupt/duplicate results, export gates, and read-only frozen-input hashes.

Run:

```powershell
py -3.11 -m unittest discover -s tests -p "test_university_gpu*.py" -v
py -3.11 -m compileall -q src/rq1/university_gpu
```

Then run the existing offline tests with the repository's already-available test runner. Do not install dependencies to make local checks pass during preparation. If `pytest` is absent, report that and use only the tests runnable with the current Python environment.

## Prelaunch validation

The machine report requires Linux, Python 3.11, at least eight logical CPUs, 64 GiB RAM, four NVIDIA GPUs with 16 GiB free each, four open local ports, and writable external storage with at least 20 GB free. It records GPU UUID, model, total and free memory, process IDs, platform, and current Git provenance. The final model check verifies the exact installed Ollama digest and quantization at all four private endpoints. No network pull or tag fallback occurs.

The input gate verifies:

- a full 40-character clean source commit matching the frozen manifest;
- each task, controlled-failure, library, relevance, tokenizer, filler, and calibration SHA-256;
- exact Accum-18 size and unique IDs;
- task split/family balance and validated failure-oracle presence;
- eighteen blinded relevance records and final adjudication per task;
- an approved outcome-blind valid-seen gate calibration;
- ALFWorld 0.4.2 data layout and package version;
- the exact pinned local all-mpnet-base-v2 snapshot;
- the exact local Gemma tokenizer and approved filler file;
- the exact content tree of the ALFWorld data directory and its expected 0.4.2 package version;
- the exact Gemma 4 12B model and `Q4_K_M` identity;
- at runtime, the actual mapping from every Ollama process tree to a different expected physical GPU UUID.

The existing follow-up tasks are previously observed by RQ1 even though their split label is valid-unseen relative to acquisition. This exposure cannot be removed by changing run IDs. Confirmatory approval must name it and decide whether the design remains useful.

## Scheduler invariants

- One frozen manifest defines the only valid unit keys. A cell consists of the same task and seed under all six arms.
- A single writer owns the run lock. Each assignment has a unique attempt ID and preserved attempt directory. Heartbeats are append-only events.
- A completed result is written once to its unit-key path, has a canonical result hash, and must match manifest task, seed, condition, and unit identity.
- An interrupted or failed unit is not counted as an outcome. A retry gets a new physical attempt; no exactly-once claim is made.
- A result written just before a process crash is recovered from its hash-valid unit record. Duplicate, unknown or hash-mismatched results block export. Incomplete results block complete-run validation, merge and the complete-only `export` command.
- A safe stop/cutoff does not kill an active episode. Partial paired cells remain incomplete; `export-progress` preserves and explicitly marks them incomplete.
- Run output is required outside the source checkout; no source freeze or archived run is edited.

## Scientific gates before confirmatory execution

The 576-unit total is only a planned floor from the task/seed/arm design. Before calling it adequate, the authors must approve an explicit task-cluster precision/power simulation for the primary contrast(s). Record the minimum scientifically relevant effect, plausible task-level heterogeneity, seed-level correlation, expected eligibility loss, multiplicity handling, target power/interval width, and Monte Carlo precision. Use a range of plausible nuisance values and report the decision if conclusions change across that range. No 720-episode design is presumed necessary; nor is 576 presumed sufficient.

Historical NoLib and Accum-18 results are not accepted as contemporaneous primary cells by default. A fair reuse decision requires a side-by-side compatibility check for task and checkpoint identity, perturbation, model/runtime, prompt/recovery memory, retrieval timing and content, token handling, eligibility, seed/order, outcome fields, and missingness. If any dimension is unverified or differs, keep those records as clearly labeled historical context. Only after complete evidence and author approval could compatible anchors reduce the prospective count; an arithmetic 450/360 is not a scientific justification.

The Gate threshold is calibrated on valid-seen relevance labels only. It is not calibrated on recovery outcomes. The six-task pilot may assess feasibility and failure modes; it cannot certify confirmatory effect size or power. Family-level results and new task-cluster intervals remain exploratory given the small number of task clusters.

## Not established by offline checks

- University permission to use, schedule, monitor, or persist data on the 4-GPU system.
- CUDA/Ollama behavior on university hardware, actual placement, throughput, model memory margin, or long-run stability.
- Availability/permission for ALFWorld data, Hugging Face snapshot, exact Ollama model, tokenizer, and approved neutral filler on that system.
- Whether human labelers were actually blinded, independent, and faithful to the rubric; calibration artifact correctness; or oracle quality beyond the supplied validation record.
- Whether authors approve previous-outcome exposure, final sample size, estimands, exclusions, and interpretation.
- Live Overleaf synchronization, authorship order, funding/AI disclosures, and public artifact/license permissions.

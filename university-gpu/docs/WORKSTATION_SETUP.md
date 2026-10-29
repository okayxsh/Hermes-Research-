# Workstation setup: university Linux host

Status: **preparation only.** Nothing in this document has been executed on the university host,
and no package, model, data set, or service was installed while writing it. The four RTX 5000 Ada
GPUs, the model, throughput, and permissions remain unverified on that machine
(`FRIDAY_FINAL_READINESS.md:3`). Read `OPERATIONS.md` before running any command.

Status labels used below:

| Label | Meaning |
|---|---|
| `AVAILABLE_LOCALLY` | Present in this development checkout, with the path shown. Local presence does not prove the university host has it. |
| `MUST_OBTAIN` | Not present locally and must be supplied/verified on the host before a real launch. |
| `UNKNOWN_ON_UNIVERSITY_HOST` | Not verifiable from this checkout; must be checked on the host. Never pull or install it as part of a check. |

## 1. Requirement inventory

| Requirement | Status | Evidence / path |
|---|---|---|
| Python 3.11 | `AVAILABLE_LOCALLY` | `.venv\Scripts\python.exe` reports 3.11.9; the university target is Python 3.11 (`README.md:12`, `OPERATIONS.md:3`). |
| `uv` and committed `uv.lock` | `AVAILABLE_LOCALLY` (lock) / `UNKNOWN_ON_UNIVERSITY_HOST` (`uv`) | `uv.lock` exists; the `check` output reports `locked_python_dependencies: PASS`; `uv` must already exist before any setup apply (`OPERATIONS.md:12`). |
| Ollama executable and existing model directory | `UNKNOWN_ON_UNIVERSITY_HOST` | `setup` refuses model pulls and only verifies an existing installation (`OPERATIONS.md:12`; `src/rq1/university_gpu/ops.py:142-145`). |
| Model tag `gemma4:12b` | `UNKNOWN_ON_UNIVERSITY_HOST` | Required tag, digest `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`, Q4_K_M (`PROTOCOL.md:13-16`). Its presence on the host is **UNVERIFIED** and must be checked, not pulled. |
| ALFWorld `0.4.2` package | `MUST_OBTAIN` (on host) | Pinned version (`docs/SETUP.md:12`); the local venv lacks `alfworld` (import probe returned `False`). |
| ALFWorld data tree `json_2.1.1` | `MUST_OBTAIN` | `data/json_2.1.1/` is absent locally (`PILOT_READINESS.md:10-12`); preflight requires `alfworld_data_dir/json_2.1.1` (`ops.py:439-441`). |
| Sentence-BERT `all-mpnet-base-v2` snapshot, revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` | `MUST_OBTAIN` | Required (`PROTOCOL.md:18-19`). Local Hugging Face cache contains `all-MiniLM-L6-v2`, not `all-mpnet-base-v2`; `sentence_transformers` is also absent. |
| Hash-pinned Gemma `tokenizer.json` + pinned revision | `MUST_OBTAIN` | Required at manifest freeze (`OPERATIONS.md:24`, `PROTOCOL.md:37`). No `tokenizer.json` found in the checkout; `tokenizers` is absent from the local venv. |
| Author-reviewed neutral filler unit | `MUST_OBTAIN` | Required and not supplied by the package (`PROTOCOL.md:37`). |
| Four physical NVIDIA GPUs (RTX 5000 Ada), ≥16 GiB free each | `UNKNOWN_ON_UNIVERSITY_HOST` | Required (`README.md:12`); check output on this Windows machine reported one RTX 3050 Laptop GPU, 3962 MiB free. Placement is proven only by the launch command's live process-to-UUID check (`README.md:13`, `OPERATIONS.md:101`). |
| Four-worker configuration | `AVAILABLE_LOCALLY` | `university-gpu/config.example.json:2` sets `"worker_count": 4`; local `check` reported `four_worker_configuration: PASS`. |
| External config and output directories | `MUST_OBTAIN` | `run_output_dir` must resolve outside the checkout (`OPERATIONS.md:16`); template at `university-gpu/config.example.json`. |

## 2. Linux setup sequence

1. **Confirm Python 3.11** is the interpreter used for every command. The documented invocations
   are `PYTHONPATH=src python3.11 -m rq1.cli ...` (`OPERATIONS.md:8-9`). If a project-local venv is
   used, activate it first; the package's stage-02 setup contract creates `.venv` and runs a locked
   sync (`docs/SETUP.md:34`), but that script is not part of this task and was not run.
2. **Set environment variables** before any file-inspecting command (`FRIDAY_FINAL_READINESS.md:59`):

   | Variable | Purpose |
   |---|---|
   | `RQ1_CONFIG` | Path to the external copy of `config.example.json`, outside the checkout. Used by `manifest propose`, `preflight`, `launch`, `resume` (`OPERATIONS.md:69,88,97`). |
   | `RQ1_EVIDENCE` | External evidence directory for proposals, freezes, approvals, Gate outputs (`OPERATIONS.md:37-57`). |
   | `RQ1_RUN_ROOT` | Per-run output root, e.g. `$RQ1_RUN_OUTPUT_DIR/rq1-university-pilot-001` (`FRIDAY_FINAL_READINESS.md:70`); used by `status`, `stop`, `validate`, `merge`, `export`, `benchmark-report` (`OPERATIONS.md:106-111`). |
   | `RQ1_ACCESS_END_EPOCH` | Actual access deadline as Unix UTC seconds; assignment stops 30 minutes before it (`OPERATIONS.md:98,114`). |

   The config file also expands `RQ1_FROZEN_TASK_MANIFEST`, `RQ1_CONTROLLED_FAILURES`,
   `RQ1_ACCUM18_LIBRARY`, `RQ1_TASK_SKILL_RELEVANCE_CSV`, the Gate score/label/frozen paths,
   `RQ1_GEMMA_TOKENIZER_JSON`, `RQ1_APPROVED_NEUTRAL_FILLER`, `RQ1_ALFWORLD_DATA_DIR`,
   `RQ1_HUGGINGFACE_CACHE`, `RQ1_OLLAMA_MODELS_DIR`, and `RQ1_RUN_OUTPUT_DIR`
   (`university-gpu/config.example.json:7-20`).
3. **Run the read-only check first** (`README.md:20-23`; `OPERATIONS.md:8`):

   ```bash
   PYTHONPATH=src python3.11 -m rq1.cli university-gpu check --json
   ```

   On this Windows development checkout it returns `"status": "BLOCKED"` and exit code 1, which is
   expected off-target (`README.md:23`). A blocked result is not a setup failure by itself, but every
   FAIL/BLOCKED row must be resolved or explained before launch.
4. **Display the setup plan** (read-only; `OPERATIONS.md:9`, `src/rq1/university_gpu/cli.py:160-163`):

   ```bash
   PYTHONPATH=src python3.11 -m rq1.cli university-gpu setup
   ```

   Local output: `"status": "DRY_RUN_ONLY"`, with the single project-local action
   `uv sync --frozen --extra alfworld --extra retrieval` marked `requires_approval: true`, and
   `refused_actions` containing `sudo`, system package install, driver changes, model pull, and GPU
   inference. Running `setup` **without** `--apply` performs no mutation.
5. **Dependency installation is separate and approval-gated.** `setup --apply --yes
   --approval-file PATH` runs the project-local sync only, requires an approval file, and requires
   `uv` to already exist; it never runs `sudo` or pulls a model (`OPERATIONS.md:12`;
   `ops.py:160-166`). Do not run it without explicit authorization.

## 3. Ollama and Gemma requirements

- Only tag `gemma4:12b` per AGENTS.md Decision 010, digest
  `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`, quantization `Q4_K_M`,
  temperature 0, `think=false`, `num_predict=2048`, `num_ctx=32768` (`PROTOCOL.md:11-16`).
- No substitution, fallback, or `ollama pull` is permitted (`README.md:53`; `OPERATIONS.md:101`).
- The model must already exist in the configured `ollama_models_dir`. Presence on the university
  host is `UNKNOWN_ON_UNIVERSITY_HOST` and must be checked there; this package never downloads it.
- The local Ollama manifest directory contains `deepseek-r1`, `phi3`, and `qwen-uncensored`, not
  `gemma4`; that local state is irrelevant to the host and is recorded only to avoid a false
  "available" claim.

## 4. ALFWorld data

- Package: `alfworld==0.4.2`, text-only (`docs/SETUP.md:12`).
- Data root must contain `json_2.1.1/`; preflight checks both the directory layout and the installed
  package version (`ops.py:439-447`). The tree is frozen as a sorted relative-path/file-hash digest
  and symlinks are rejected (`OPERATIONS.md:27`).
- Split policy: `train` for acquisition, `valid_seen` for pilots, untouched `valid_unseen` only for
  final evaluation (`AGENTS.md`).
- Local status: absent (`PILOT_READINESS.md:10-12`). Status is `MUST_OBTAIN`; host presence is
  unverified.

## 5. Sentence-BERT retrieval snapshot

- Model `sentence-transformers/all-mpnet-base-v2`, revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`, snapshot SHA-256
  `bbfecb04a834241144af31e9200a3dd0df83ff0fbb2453d56a4bb383a6bfbe11` (`PROTOCOL.md:18-19`).
- `gate-scores` performs CPU embedding inference from the local snapshot only and does not call
  Gemma or the GPUs (`OPERATIONS.md:31,34-37`).
- Local status: `MUST_OBTAIN`; the cache has `all-MiniLM-L6-v2` only, and
  `sentence_transformers` is not installed in `.venv`.

## 6. Tokenizer and filler

- `tokenizer.json` must come from the exact pinned Gemma tokenizer revision; both the file hash and
  revision string are bound into the manifest (`OPERATIONS.md:24,27`).
- Every `Rel`/`Irr`/`Fill`/`Gate` memory block must match the `D18` serialized block token count
  under that tokenizer (`PROTOCOL.md:37`).
- The neutral filler unit must be author-reviewed before freezing (`PROTOCOL.md:32,37`).
- Local status: `MUST_OBTAIN`; no tokenizer or filler artifact exists in the checkout, and no
  production artifact is supplied by the code (`PROTOCOL.md:37`).

## 7. GPU and four-worker verification

- Hardware requirement: exactly four physical NVIDIA devices, one UUID per isolated Ollama server,
  four unique endpoints, sufficient free memory, and at least 16 GiB free per GPU
  (`README.md:12`; `OPERATIONS.md:101`; `university-gpu/config.example.json:2-3`).
- Static preflight checks GPU identity, memory, disk, ports, source hashes, ALFWorld data, SBERT
  snapshot, and the local Ollama executable/model directory. Without `--check-endpoints` it can only
  return `STATIC_PASS` (`OPERATIONS.md:84`).
- **GPU placement is verified by the launch command**, not by this document and not by `check`:
  launch starts the isolated endpoints, checks the exact model, runs a separately approved small
  smoke request on each, and verifies process-to-GPU UUID placement before assigning episodes
  (`OPERATIONS.md:84,101`; `README.md:13`).
- Start four workers only through `launch`/`resume`, using the external config with
  `worker_count: 4` (`config.example.json:2`). Do not start Ollama servers by hand for a scientific
  run.

## 8. Benchmark procedure

The benchmark is part of the pilot launch, not a separate performance test:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu launch \
  --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" \
  --approval-file "$RQ1_EVIDENCE/pilot-launch-approval.json" --yes \
  --cutoff-epoch "$RQ1_ACCESS_END_EPOCH" --max-attempts 4
PYTHONPATH=src python3.11 -m rq1.cli university-gpu benchmark-report \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_EVIDENCE/pilot-frozen.json"
```

`--max-attempts 4` is documented at `FRIDAY_FINAL_READINESS.md:75-80`. `benchmark-report` is
file-only and **needs four completed attempts on distinct workers** (`FRIDAY_FINAL_READINESS.md:82`;
`src/rq1/university_gpu/cli.py:99-101`). A paused result after four attempts is expected and exits
nonzero (`FRIDAY_FINAL_READINESS.md:75`). The observed rate excludes setup, recovery, transfer, and
later variability, and four attempts are not a representative sample or a power calculation
(`FRIDAY_FINAL_READINESS.md:37,82`).

## 9. Crash recovery

1. Inspect state before resuming (file-only; `OPERATIONS.md:106`):

   ```bash
   PYTHONPATH=src python3.11 -m rq1.cli university-gpu status \
     --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"
   ```

2. Resume only after review, at the episode-unit boundary (an interrupted episode is not restored
   mid-trajectory; `OPERATIONS.md:114`):

   ```bash
   PYTHONPATH=src python3.11 -m rq1.cli university-gpu resume \
     --config "$RQ1_CONFIG" --manifest "$RQ1_FROZEN_MANIFEST" \
     --approval-file "$RQ1_LAUNCH_APPROVAL" --yes --retry-failed \
     --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
   ```

   `--retry-failed` creates new physical attempts only after operator review; prior attempt logs are
   never overwritten (`OPERATIONS.md:103`; `FRIDAY_FINAL_READINESS.md:98`). Retries are
   at-least-once across failure/retry; no exactly-once claim is made (`OPERATIONS.md:114`).
3. **A stale lock needs manual inspection**; do not delete it blindly (`FRIDAY_FINAL_READINESS.md:98`).
   Resume is designed for the original run location and worker identities; a different host, path,
   or GPU UUID can fail the worker-configuration identity check and needs a reviewed migration
   procedure (`FRIDAY_FINAL_READINESS.md:98`).

## 10. Safe stop

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu stop \
  --run-root "$RQ1_RUN_ROOT" --reason "operator cutoff"
```

`stop` writes a `STOP` marker (`cli.py:231-237`) and lets active episodes finish while assigning no
new units (`OPERATIONS.md:114`; `README.md:57`). **Wait for active attempts to finish** before
exporting or moving files. The cutoff does not kill active episodes, so they can exceed the
30-minute margin; watch the clock (`OPERATIONS.md:114`).

## 11. Results export and verification

After the process exits, for each stopped run and a fresh destination (`OPERATIONS.md:111,114`;
`FRIDAY_FINAL_READINESS.md:100-106`):

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu progress-report \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu export-progress \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST" \
  --relevance "$RQ1_RELEVANCE_CSV" --destination "$RQ1_EXPORT_DIR"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu verify-export \
  --destination "$RQ1_EXPORT_DIR"
```

- Use a new destination; the exporter refuses an existing one (`FRIDAY_FINAL_READINESS.md:108`).
- `export-progress` preserves incomplete work and marks missing units; complete-only `export` and
  `merge` still require a validated complete run (`OPERATIONS.md:114`; `README.md:58`).
- Copy the directory to an **independent drive** and verify the **copied** directory before leaving,
  using `-B` from the export's `analysis-src` directory (`FRIDAY_FINAL_READINESS.md:110-116`).

## 12. What this document does not establish

No university-host command was run. `check`, `setup` (dry-run), and `--help` were exercised only on
this Windows development checkout. Host OS, GPU count/placement, Ollama, `gemma4:12b`, ALFWorld
package/data, SBERT snapshot, tokenizer, and filler remain unverified; see
`FRIDAY_FINAL_READINESS.md:11`. Mocks and local checks are not evidence of host compatibility
(`README.md:49`).

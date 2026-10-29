# Transfer checklist: development checkout → university host

Purpose: move exactly the reviewed, hash-bound inputs and approvals needed for the pilot, without
carrying secrets, results, or machine-specific paths (`OPERATIONS.md:16`). Tick each row and record
the SHA-256 actually transferred. Do not transfer a dirty working tree.

## 1. Code and provenance

| # | Item | Who provides | Check |
|---|---|---|---|
| 1.1 | Repository at a single committed revision on `feat/rq1-university-4gpu`. At capture time HEAD was `153f5cfc9182b76ee5a6a5f7594ea7341c37442c`; the tree also had uncommitted modifications/untracked files, so this is **not** a transferable revision. | Repository owner / authors | Record `git rev-parse HEAD` after the final commit; `git status --porcelain` must be empty (freeze requires a committed, clean repository, `OPERATIONS.md:64`). |
| 1.2 | Scientific parent commit `c052e2cbdaa63419db4122cf08b1ccc64b38b3b4` recorded in the transfer note | Authors | Matches `README.md:7`. |
| 1.3 | `uv.lock`, `pyproject.toml` | Repository owner | Locked dependency resolution; `uv sync --frozen` is the only project setup action (`OPERATIONS.md:12`). |
| 1.4 | `university-gpu/config.example.json` (template only) | Operator | The real config is created on the host, outside the checkout, with host paths (`README.md:28`). |

## 2. Pilot scientific inputs

All input files are SHA-256-bound into the manifest at proposal time; transfer the exact bytes, never
reconstructed copies (`OPERATIONS.md:27`).

| # | Item | Who provides | Check |
|---|---|---|---|
| 2.1 | Frozen six-task `valid_seen` task manifest (one per family) | Authors / task reviewer | No frozen six-task manifest existed locally (`FRIDAY_FINAL_READINESS.md:8`); must be produced and frozen before transfer. |
| 2.2 | Controlled-failure oracle definitions for those six tasks (checkpoint, post-detour digest, prefix, detour, reference route, recovery budget) | Authors / task reviewer | Required per task before freeze (`PROTOCOL.md:65`). |
| 2.3 | Six reviewed pilot contexts JSON (task instruction, observation, inventory, failure message) | Authors | Format and review rules at `OPERATIONS.md:31`. |
| 2.4 | Accum-18 library snapshot (18 unique skill IDs and exact text) | Authors | Historical copy recovered locally; do not alter the original (`FRIDAY_FINAL_READINESS.md:7`). |
| 2.5 | Task-skill relevance CSV with exactly `task_id,skill_id,rater_a,rater_b,adjudicated` | Raters + adjudicator, combined by operator | 108 pilot pairs required (18 skills × 6 tasks); `OPERATIONS.md:22`, `FRIDAY_FINAL_READINESS.md:9`. |
| 2.6 | Valid-seen similarity score CSV (`task_id,skill_id,cosine_similarity`) and adjudicated calibration labels | Operator (CPU `gate-scores`) + adjudicator | Three files must agree and be hashed together (`OPERATIONS.md:23`). |
| 2.7 | Frozen Gate calibration JSON + Gate approval JSON bound to `calibration_sha256` | Operator generates; authors approve | `OPERATIONS.md:47-60`. |
| 2.8 | Hash-pinned Gemma `tokenizer.json` + exact revision string | Model owner / operator | Not found locally (`FRIDAY_FINAL_READINESS.md:11`); both file hash and revision are bound (`OPERATIONS.md:27`). |
| 2.9 | Author-reviewed neutral filler unit | Authors | Required before freezing (`PROTOCOL.md:37`). |

## 3. Rating package

| # | Item | Who provides | Check |
|---|---|---|---|
| 3.1 | Outcome-free rating worksheet | Authors | Must not reveal condition, retrieval output, or recovery outcome (`FRIDAY_FINAL_READINESS.md:45`). |
| 3.2 | `rater_a.csv` | Independent rater A | Complete, blinded, original file preserved. |
| 3.3 | `rater_b.csv` | Independent rater B | Complete, blinded, independent of A. |
| 3.4 | `final-labels.csv` | Human adjudicator | Every disagreement resolved under the approved rubric. |
| 3.5 | Combined `relevance.csv` output | Operator | Produced only by `university-gpu/combine_ratings.py`, which preserves both original ratings (`FRIDAY_FINAL_READINESS.md:45-49`). |
| 3.6 | Confirmatory blank forms (540 rows), only if confirmatory work is authorized | Authors | They contain no ratings (`FRIDAY_FINAL_READINESS.md:9`). |

Pilot needs 108 task–skill pairs; at least 494 pairs and 988 individual ratings are missing project-wide
if all 154 historical pairs are accepted, otherwise up to 648 pairs / 1,296 ratings
(`FRIDAY_FINAL_READINESS.md:9`). Do not transfer historical ratings as if they satisfied the
prospective blinding rules.

## 4. Approval records (human-signed, hash-bound)

| # | Record | Who provides | Check |
|---|---|---|---|
| 4.1 | Gate approval (`gate-approval.json`) | Authors / approver | `approved=true`, `approved_by`, `approval_reference`, exact `calibration_sha256` (`OPERATIONS.md:60`). |
| 4.2 | Pilot manifest approval (`pilot-approval.json`) | Authors / approver | `approved=true`, approver, reference, exact `manifest_sha256` (`OPERATIONS.md:78`). |
| 4.3 | Pilot launch approval (`pilot-launch-approval.json`) | Authors + university operator | Exact `manifest_sha256`, `gpu_inference=true`, `scientific_run=true`, approver and reference (`OPERATIONS.md:93`). |
| 4.4 | Confirmatory-only attestations (`previous_task_outcomes_acknowledged`, `eligibility_verified`, `blinded_relevance_verified`, `leakage_review_complete`, `pilot_go_approved`, `precision_power_review_complete`, power/pilot-validation hashes) | Authors | Required for confirmatory freeze; human attestations, not machine proof (`OPERATIONS.md:78`). |
| 4.5 | Institutional GPU/access permission and access deadline | University staff | Deadline becomes `RQ1_ACCESS_END_EPOCH` (`OPERATIONS.md:98`). |

Blank templates in `university-gpu/approval-templates/` are unapproved examples and must be copied
out and filled only after review (`FRIDAY_FINAL_READINESS.md:43`).

## 5. Model and data artifacts (verify in place; do not copy unless approved)

Large artifacts normally stay on the university host. Record identity, not just a path.

| # | Artifact | Who provides | Hash / identity |
|---|---|---|---|
| 5.1 | Ollama `gemma4:12b` | University host / model owner | Digest `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`, Q4_K_M (`PROTOCOL.md:14-15`). Presence unverified; never pull. |
| 5.2 | Sentence-BERT `all-mpnet-base-v2` snapshot | University host / model owner | Revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, snapshot SHA-256 `bbfecb04a834241144af31e9200a3dd0df83ff0fbb2453d56a4bb383a6bfbe11` (`PROTOCOL.md:18-19`). |
| 5.3 | ALFWorld 0.4.2 data tree (`json_2.1.1`) | University host / data steward | Tree digest computed by `manifest propose` from the sorted relative-path/file-hash tree; symlinks rejected (`OPERATIONS.md:27`). |
| 5.4 | Gemma tokenizer file | Model owner | File hash and revision bound at propose (`OPERATIONS.md:27`). |
| 5.5 | Existing external run output directory | University host / operator | Must be outside the Git checkout (`OPERATIONS.md:16`); ≥20 GB free is checked by `check`. |

Copy only hashes and manifests for 5.1–5.4 unless a separate data-transfer approval exists. Never
transfer model blobs, data trees, labels, secrets, or personal paths into the repository
(`OPERATIONS.md:16`).

## 6. Transfer and verification

- [ ] Transfer code as a Git commit/clone; do not copy a dirty directory.
- [ ] Transfer inputs and approvals through an approved channel; record each SHA-256 on arrival.
- [ ] On the host, run `university-gpu check --json` and `university-gpu setup` before anything else
      (`README.md:20-29`).
- [ ] Confirm `git status --porcelain` is empty on the host before proposing a manifest
      (`OPERATIONS.md:64`).
- [ ] Confirm the manifest proposal's recorded hashes match the transferred files; do not hand-edit
      a proposal or approval.
- [ ] Keep the only copy of approvals and frozen evidence outside the checkout; export results to a
      new destination and verify the copy on an independent drive (`FRIDAY_FINAL_READINESS.md:100-116`).

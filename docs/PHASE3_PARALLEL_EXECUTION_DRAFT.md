# RQ1 Phase 3C: isolated four-GPU execution (DRAFT)

**Status: DRAFT.** It is implemented and tested only with synthetic drivers. No model, GPU or provider has been used. Scientific execution still requires the readiness levels, approvals and separate launch authorisation.

| | |
|---|---|
| Code | `src/rq1/five_arm/parallel.py`, `parallel_cli.py`, `runner.build_university_executor` |
| Tests | `tests/test_five_arm_parallel.py` |

## 1. Architecture

| Element | Implementation |
|---|---|
| Provider endpoints | Four Ollama endpoints, one per GPU. Each lane has its own `endpoint` and `gpu_uuid`; the lane worker requires `CUDA_VISIBLE_DEVICES` to equal its lane's GPU UUID |
| Worker processes | One `parallel_cli worker --lane Lk` process per lane. Each builds its own gated executor, driver and ALFWorld bridge on its endpoint |
| Authoritative coordinator | One `parallel_cli coordinator` process per run directory. It holds a process-exclusive lock, owns `journal.jsonl` and `records.jsonl`, and is the only writer of records |
| Transport | Atomic request and response files (`inbox/` → `outbox/`); no shared memory between processes |
| Durable claims | A claim (unit, attempt, lane, worker, random token) is journalled with fsync before execution. Results are accepted only with the active token of that exact unit and attempt |
| Leases | Heartbeats run during an episode. A claim without a heartbeat for `lease_seconds` (default 900) is closed as `infrastructure_interruption` (worker loss) |
| Safe restart | On start, the record hash chain is verified (corruption halts) and every claim left open by a previous coordinator is **revoked** and closed as an interruption |
| Late and duplicate results | A result for a revoked or expired claim is rejected as **late**. A second result for a completed claim is rejected as a **duplicate**. Unknown tokens are rejected |
| Lane assignment | Frozen, balanced and outcome-blind (`assign_lanes`). Task-seed cells are taken in salted-hash order, each to the least-loaded lane, and **all conditions of a task-seed cell share one lane**. The assignment is bound to the manifest hash |
| Isolation | Per-attempt directories `lanes/<lane>/attempts/<unit>/a<k>-<token>` give a fresh Hermes workspace per session, and each lane has its own bridge-log root |
| Retries | Unchanged: only `MISSING_INFRASTRUCTURE` attempts are retried, at most twice |
| Provenance (per record) | lane, worker, endpoint, GPU UUID, `CUDA_VISIBLE_DEVICES`, provider version, model digest, processor before and after, host, pid, Hermes workspace, claim-token hash, plus per-call request and response SHA-256, prompt hashes, token counts and latency |

## 2. Mandatory halts

A halt writes a persistent `HALTED.json` and stops all claims on every lane. Clearing it needs a recorded human review (`clear_halt`); this is operational, not a scientific approval.

| Trigger | Detection | The triggering result |
|---|---|---|
| CPU fallback or partial offload | `/api/ps` `size_vram < size`, before or after the episode | quarantined |
| Runtime drift | model digest ≠ approved, or runtime identity ≠ approved | quarantined |
| Token discrepancy | record ends `provider_token_discrepancy`, `provider_context_limit_reached` or `provider_prompt_eval_count_missing` | **recorded** (a valid INVALID_MEASUREMENT), then halt |
| Lane/provenance mismatch | endpoint, GPU, `CUDA_VISIBLE_DEVICES`, worker or claim token differ from the frozen lane | quarantined |
| Schema-invalid or mismatched result | record validation fails, or the record does not match its claim | quarantined |
| Protocol violation | leakage, frozen-ranking drift or memory drift, reported by the worker | — |
| Evidence corruption | the hash chain over `records.jsonl` fails on restart | the coordinator refuses to start |

## 3. CPU-to-GPU and cap transition

- **Cap.** The cap stays **50** (undisturbed 49), as recommended by the frozen budget rule from the CPU cap-100 diagnostic, truncated.
- **Hardware.** Execution moves from sequential CPU inference to GPU lanes. A difference between cap-50 GPU trajectories and truncated cap-100 CPU trajectories is **not** evidence of a cap effect: hardware, kernels and provider builds change, and temperature-0 decoding is not guaranteed bit-identical across them.

**Prospective comparability checks** (no extra scientific episodes):
1. **Pilot D0 versus diagnostic.** In the seven-condition pilot, each pilot task's D0 seed-11 outcome and action sequence are compared descriptively with the diagnostic's DETOURED seed-11 trajectory truncated to cap 50. Report agreement as a CPU-to-GPU comparability statistic, explicitly not as a cap test.
2. **Within-run hardware identity.** Every accepted record must show the lane's GPU, full offload and the approved digest (enforced by the halts above).
3. **Seed determinism on GPU.** The 48-episode reproducibility subset (D0 and D18 at seeds 29 and 47) documents seed determinism under memory on the GPU lanes.
4. **Lane balance.** Because cells are lane-pure, condition contrasts are within-lane. Lane is reported, and a lane × condition descriptive table is included to show no lane-specific anomaly.

## 4. Deployment (after authorised push and readiness)

```sh
python3.11 -B -m rq1.five_arm.parallel_cli assign --manifest frozen-manifest.json --lanes lanes.json \
    --salt <recorded salt> --output lane-assignment.json
python3.11 -B -m rq1.five_arm.parallel_cli coordinator --run-config run.json --state state.json \
    --observed-runtime observed.json --approved-runtime approved.json --assignment lane-assignment.json --run-root <run dir>
# one per GPU, each in its own process with its own endpoint:
CUDA_VISIBLE_DEVICES=<GPU UUID of Lk> python3.11 -B -m rq1.five_arm.parallel_cli worker --lane Lk \
    --run-config run.json --state state.json --observed-runtime observed.json --approved-runtime approved.json \
    --assignment lane-assignment.json --run-root <run dir>
```

Each Ollama server must be started on its own GPU and port (`OLLAMA_HOST`, `CUDA_VISIBLE_DEVICES`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`), as in the existing `university_gpu.ollama.server_environment`.

## 5. What remains unverified

- Real multi-process behaviour on the workstation.
- Real `/api/ps` fields for this Ollama build.
- Real lane throughput.
- Real bridge concurrency across four processes.

The tests exercise the coordinator, claims, restart, leases, the transport and the halts with scripted drivers only.

# RQ1 Phase 3C/3D: isolated four-GPU execution (DRAFT)

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
| Leases | Heartbeats run during an episode. A claim without a heartbeat for `lease_seconds` (default 900) is closed as `infrastructure_interruption` (worker loss), and its **lane is fenced** (§1a) |
| Safe restart | On start, the record hash chain is verified (corruption halts). Every claim left open by a previous coordinator is **revoked**, closed as an interruption, and its **lane is fenced** (§1a) |
| Late and duplicate results | A result for a fenced claim that has not yet been verified stopped is rejected as **late**. A result for a claim whose worker was **verified stopped** is overlap evidence: it is quarantined and the **lane is halted**. A second result for a completed claim is rejected as a **duplicate**. Unknown tokens are rejected |
| One unit in flight per lane | A lane can never hold two active claims. A claim must carry complete process identities (host, pid, kernel start time) for the worker **and** the lane's Ollama server |
| Lane assignment | Frozen, balanced and outcome-blind (`assign_lanes`). Task-seed cells are taken in salted-hash order, each to the least-loaded lane, and **all conditions of a task-seed cell share one lane**. The assignment is bound to the manifest hash |
| Isolation | Per-attempt directories `lanes/<lane>/attempts/<unit>/a<k>-<token>` give a fresh Hermes workspace per session, and each lane has its own bridge-log root |
| Retries | Unchanged: only `MISSING_INFRASTRUCTURE` attempts are retried, at most twice |
| Provenance (per record) | lane, worker, endpoint, GPU UUID, `CUDA_VISIBLE_DEVICES`, provider version, model digest, processor before and after, **placement before and after** (§1b), host, pid, Hermes workspace, claim-token hash, plus per-call request and response SHA-256, prompt hashes, token counts and latency |

## 1a. Fencing (Phase 3D): no reassignment before verified termination

**Problem fixed.** In Phase 3C, an expired or revoked unit became claimable at once. Cancelling a claim does not stop the old worker process, and closing an HTTP client does not demonstrably cancel inference already running inside the Ollama server. A replacement attempt could therefore overlap the abandoned one on the same GPU and endpoint.

**Rules (enforced in `Coordinator.claim`, `review_lane` and `submit`):**

| Rule | Behaviour |
|---|---|
| Process-bound identity | Every claim records `{host, pid, start_time}` for the worker and for the lane's Ollama server. The server pid is given in `RQ1_LANE_OLLAMA_PID`; `start_time` is field 22 of `/proc/<pid>/stat`, so a reused pid cannot match |
| Fence | Lease expiry or a restart revocation marks the claim `fenced` and writes the `MISSING_INFRASTRUCTURE` interruption record. **No unit of that lane is claimable** while any of its claims is fenced (`LaneSuspended`) |
| Verified termination | On each claim attempt for the lane, the coordinator probes both identities. Only when the worker **and** the server are `stopped` is the claim marked `terminated_verified` (journalled with evidence), and the lane released |
| Why the server too | The only demonstrable way to end an outstanding provider request is that the serving process (and its runner children) has exited. Recovering a lane therefore means stopping the worker and restarting that lane's Ollama server |
| Runner descendants (Phase 3E) | Ollama runs the model in runner child processes, which can outlive a killed parent. The claim records the server's descendant identities (`provider_descendants`). Heartbeats add any runner spawned later, and the worker sends one heartbeat right after its pre-episode snapshot. Every recorded descendant must also be `stopped` |
| Outstanding provider work (Phase 3E) | Before release, `nvidia-smi` compute apps on the lane's GPU UUID may contain only processes of the **new** claimant's provider tree. Any other process (for example, an unrecorded orphaned runner) keeps the lane suspended. If the GPU cannot be queried, the state is `unknown` and the lane is halted |
| Unknown termination | If either state is `unknown` (another host, no `/proc`, unreadable or missing identity), the **lane is halted** (`LANE_HALTED_<lane>.json`). The other lanes continue. Clearing needs `clear_lane_halt(reviewer, note)` after a human has verified, on the host, that both processes are gone. This is recorded, and is not a scientific approval |
| Late result after verified termination | Contradicts the evidence. The result is quarantined, the lane is halted, and the in-flight replacement's result on that lane is quarantined too (overlap cannot be excluded) |
| Reuse of fenced processes | A claim from the same worker or server identity as a fenced or verified-stopped claim is rejected |
| Durability | `claimed` (with identities), `lease_expired` / `revoked_on_restart`, `termination_verified` and `lane_halted` are journalled with fsync and replayed on restart |

**Operational consequences.**
- The coordinator and the four workers must run on the **same host**, so the coordinator can read `/proc`. Otherwise every fence becomes a lane halt, which is fail-closed.
- A lane worker exits with status 3 on `LaneSuspended` or `LaneHalted` and prints the recovery action.
- The coordinator loop ends when nothing runnable remains on non-halted lanes, and returns 4 if any lane or the run is halted.

## 1b. Scientific placement gate (Phase 3D)

`CUDA_VISIBLE_DEVICES` alone is **not** accepted as evidence of GPU execution. Before and after every episode, the worker records two independent observations. Both must pass, or the result is quarantined and the run halts.

| Evidence | Source | Pass condition |
|---|---|---|
| Model offload | Ollama `GET /api/ps`. Each entry in `models` carries `name`, `model`, `size`, `size_vram`, `digest`, `details` and `expires_at` (documented API; confirm on the workstation's build) | `processor == "gpu_full"`: the approved model tag is loaded and `size_vram >= size > 0`. A missing `size` or `size_vram` gives `unknown` (fail-closed); `0 < size_vram < size` gives `gpu_partial`; `size_vram == 0` gives `cpu` |
| Process-bound CUDA placement | `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits`, matched against the process tree of `RQ1_LANE_OLLAMA_PID` (Ollama runs the model in a runner child process, so the tree is used) | `placement == "lane_gpu_exclusive"`: at least one process of the server tree holds a CUDA context on the lane's GPU UUID, none of the tree is on another GPU, and no foreign process is on the lane's GPU |

Other placement values halt the run: `no_provider_cuda_context_on_lane_gpu`, `provider_on_other_gpu` and `lane_gpu_shared_with_foreign_process`.

If `nvidia-smi` cannot see host PIDs (for example inside a container PID namespace), the gate fails closed. That must be resolved on the workstation before the pilot, not bypassed.

## 2. Mandatory halts

A halt writes a persistent `HALTED.json` and stops all claims on every lane. Clearing it needs a recorded human review (`clear_halt`); this is operational, not a scientific approval.

| Trigger | Detection | The triggering result |
|---|---|---|
| CPU fallback or partial offload | `/api/ps` `size_vram < size` (or the fields are missing), before or after the episode | quarantined |
| Unverified GPU placement | `nvidia-smi` compute apps do not show the lane's Ollama process tree exclusively on the lane GPU (§1b) | quarantined |
| Runtime drift | model digest ≠ approved, or runtime identity ≠ approved | quarantined |
| Token discrepancy | record ends `provider_token_discrepancy`, `provider_context_limit_reached` or `provider_prompt_eval_count_missing` | **recorded** (a valid INVALID_MEASUREMENT), then halt |
| Lane/provenance mismatch | endpoint, GPU, `CUDA_VISIBLE_DEVICES`, worker or claim token differ from the frozen lane | quarantined |
| Schema-invalid or mismatched result | record validation fails, or the record does not match its claim | quarantined |
| Protocol violation | leakage, frozen-ranking drift or memory drift, reported by the worker | — |
| Evidence corruption | the hash chain over `records.jsonl` fails on restart | the coordinator refuses to start |

**Lane halts** (`LANE_HALTED_<lane>.json`) stop one lane only:
- the termination of a fenced worker or server cannot be established;
- a late result arrives after verified termination.

They are cleared with `clear_lane_halt`, which requires a reviewer and a note.

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
CUDA_VISIBLE_DEVICES=<GPU UUID of Lk> RQ1_LANE_OLLAMA_PID=<pid of Lk's ollama serve> \
    python3.11 -B -m rq1.five_arm.parallel_cli worker --lane Lk \
    --run-config run.json --state state.json --observed-runtime observed.json --approved-runtime approved.json \
    --assignment lane-assignment.json --run-root <run dir>
```

Each Ollama server must be started on its own GPU and port (`OLLAMA_HOST`, `CUDA_VISIBLE_DEVICES`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`), as in the existing `university_gpu.ollama.server_environment`.

## 5. What remains unverified

- Real multi-process behaviour on the workstation.
- Real `/api/ps` fields for this Ollama build. The code fails closed if `size` or `size_vram` is absent.
- `nvidia-smi --query-compute-apps` output on the workstation: host PID visibility, and the Ollama runner child holding the CUDA context.
- `/proc/<pid>/stat` start-time probing on the workstation kernel. On other platforms the state is `unknown`, which halts the lane.
- Real lane throughput.
- Real bridge concurrency across four processes.

The tests use scripted drivers and a synthetic process table only. They exercise:
- the coordinator, claims, restart, leases and the transport;
- the halts;
- fencing (expired worker still alive; outstanding provider request at restart; no reassignment before verified termination; late result after termination and reassignment; unknown termination halts the lane; orphaned runner descendant; runner recorded by heartbeat across restart; unrecorded work on the lane GPU; unobservable GPU halts the lane);
- placement (compute-app parsing, process tree, `/api/ps` field handling, unverified placement halts).

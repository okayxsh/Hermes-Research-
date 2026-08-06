# ALFWorld bridge

The deterministic fixture server remains the default:

```bash
python -m rq1.cli bridge-server --host 127.0.0.1 --port 8000
```

Endpoints are `POST /health`, `POST /episode/start`, `POST /episode/step`, `GET /episode/{episode_id}/status`, `POST /episode/{episode_id}/abort`, and `POST /episode/{episode_id}/reset`.

The server writes append-only per-episode JSONL logs beneath `runs/pilot/bridge/`. Reset keeps the same episode ID, starts its deterministic fixture state again, and increments `reset_count`; terminal episodes cannot be reset or stepped.

The repository now contains a version-specific ALFWorld 0.4.2 text adapter. It is selected only with `bridge-server --mode real --yes` after `rq1 alfworld capabilities` confirms the installed package, data, and indexed `train`/`valid_seen` tasks. A real request names a canonical `split:relative/task/path` task ID; no random-task selection or fake fallback is permitted. Real `status` is cached, inventory is explicitly unavailable on the observed text surface, and controller abort is not described as an ALFWorld API.

`rq1 alfworld smoke-test --split valid_seen --yes` writes immutable bridge and response evidence without installing or downloading anything. Its result is execution evidence, not a general compatibility claim; real recovery remains unverified.

Phase 6 adds six deterministic fake task-family fixtures for orchestration coverage. These labels and episodes are mock evidence only; Pilot 12 must repeat the contract against real ALFWorld before any compatibility claim.

Phase 5 recovery replay is a separate explicit controller layered above this bridge contract. It records checkpoint/replay/perturbation evidence but does not extend the HTTP bridge with unverified ALFWorld state mutation.

Phase 3 clients may add optional `X-RQ1-Run-ID`, `X-RQ1-Attempt-ID`, `X-RQ1-Profile`, `X-RQ1-Session-ID`, `X-RQ1-Tool-Call-ID`, and `X-RQ1-Request-ID` headers. They are recorded with the episode's raw events. Missing headers preserve the Phase 2 contract. Conflicting run, attempt, profile, or session values on an existing episode return structured `409` errors; request and tool-call IDs are per-operation correlation values.

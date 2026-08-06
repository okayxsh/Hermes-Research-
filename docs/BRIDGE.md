# ALFWorld bridge

Run only the deterministic fixture server in this phase:

```bash
python -m rq1.cli bridge-server --host 127.0.0.1 --port 8000
```

Endpoints are `POST /health`, `POST /episode/start`, `POST /episode/step`, `GET /episode/{episode_id}/status`, `POST /episode/{episode_id}/abort`, and `POST /episode/{episode_id}/reset`.

The server writes append-only per-episode JSONL logs beneath `runs/pilot/bridge/`. Reset keeps the same episode ID, starts its deterministic fixture state again, and increments `reset_count`; terminal episodes cannot be reset or stepped. This mode does not install, download, import, or run ALFWorld. The real adapter is only package-discoverable and remains `TO_BE_VERIFIED_BY_PILOT`.

Phase 6 adds six deterministic fake task-family fixtures for orchestration coverage. These labels and episodes are mock evidence only; Pilot 12 must repeat the contract against real ALFWorld before any compatibility claim.

Phase 5 recovery replay is a separate explicit controller layered above this bridge contract. It records checkpoint/replay/perturbation evidence but does not extend the HTTP bridge with unverified ALFWorld state mutation.

Phase 3 clients may add optional `X-RQ1-Run-ID`, `X-RQ1-Attempt-ID`, `X-RQ1-Profile`, `X-RQ1-Session-ID`, `X-RQ1-Tool-Call-ID`, and `X-RQ1-Request-ID` headers. They are recorded with the episode's raw events. Missing headers preserve the Phase 2 contract. Conflicting run, attempt, profile, or session values on an existing episode return structured `409` errors; request and tool-call IDs are per-operation correlation values.

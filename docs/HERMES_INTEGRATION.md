# Hermes integration boundary

Phase 3 provides a project-local, capability-gated plugin at `.hermes/plugins/alfworld-experiment/`. Hermes remains a client of the local bridge: it never owns episode state and it cannot select the real ALFWorld adapter.

## Trust and capability gate

The plugin is discovered only when Hermes supports project plugins and the caller explicitly sets `HERMES_ENABLE_PROJECT_PLUGINS=1`. Registration requires the documented `register_tool` and `register_hook` surface; otherwise it raises a clear registration error and Hermes disables the plugin. It does not modify a personal/default profile, enable itself in a profile, install a package, download data, or start an LLM.

Profile creation, isolation, and contamination controls are a separate Phase 4 boundary documented in [HERMES_PROFILES.md](HERMES_PROFILES.md). A successful fake plugin check is not profile evidence.

Probe installed evidence without modifying Hermes:

```bash
python -m rq1.cli hermes-capabilities
```

The command writes `artifacts/manifests/hermes_integration_capabilities.json`. It uses only version/help commands and stores evidence hashes, not raw help output or secrets.

## Tool contract

Toolset: `alfworld_experiment`.

| Tool | Required fields | Bridge operation |
| --- | --- | --- |
| `alfworld_start` | `task_id`, `split`, `seed`, `action_limit` | `POST /episode/start` |
| `alfworld_step` | `episode_id`, `action` | `POST /episode/step` |
| `alfworld_status` | `episode_id` | `GET /episode/{episode_id}/status` |
| `alfworld_abort` | `episode_id`; optional `reason` | `POST /episode/{episode_id}/abort` |
| `alfworld_reset` | `episode_id` | `POST /episode/{episode_id}/reset` |

Unknown fields and malformed values fail local validation before any HTTP call. Each result is one JSON object with `ok`, `tool`, `request_id`, `latency_ms`, `result`, `error`, and correlation metadata. Health/status can retry once. Start, step, reset, and abort never retry following a timeout and return `outcome_unknown: true` instead.

Only `http://127.0.0.1:<port>` and `http://localhost:<port>` bridge URLs are accepted. Remote bridge targets are refused.

## Correlation and logs

The client sends optional `X-RQ1-Run-ID`, `X-RQ1-Attempt-ID`, `X-RQ1-Profile`, `X-RQ1-Session-ID`, `X-RQ1-Tool-Call-ID`, and `X-RQ1-Request-ID` headers. Their absence remains compatible with Phase 2 callers. The bridge persists start correlation in append-only per-episode JSONL records; conflicting stable run/attempt/profile/session values return `409`.

`RunRegistry` has additive episode bindings for run, attempt, episode, session, profile, Hermes log, and bridge log paths. Reconciliation uses these bindings, event IDs, actions, and terminal records rather than filename inference.

Skill events are versioned: `skill_index_available`, `skill_selected`, `skill_loaded`, `skill_managed`, and `unknown_native_skill_operation`. They record `simulated`/`observed` and `relevant`/`irrelevant`/`unknown`. Readers still accept legacy `skill_view` events.

## Verification

Run the fully local contract check:

```bash
python -m rq1.cli verify-hermes-integration --mode fake
```

It starts only the deterministic fake bridge on an ephemeral localhost port, exercises start → step → status → reset → abort, records JSONL evidence, creates a run-registry binding, reconciles evidence, and writes `artifacts/stage_reports/phase3-hermes-integration.json`.

The real command is intentionally opt-in:

```bash
RQ1_RUN_REAL_HERMES_TESTS=1 python -m rq1.cli verify-hermes-integration --mode real
```

It never installs Hermes, modifies normal profiles, downloads models/data, or calls an LLM. It can prove project-plugin discovery only when a compatible Hermes CLI is already installed. Real tool dispatch, native skill capture, real ALFWorld, and Hermes-to-ALFWorld compatibility remain unverified until observed on the university machine.

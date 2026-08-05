# Implementation status

Implemented and locally testable: repository layout, JSON-compatible configuration templates, stage registry, locks, non-overwriting stage reports, machine inspection, SQLite run registry with atomic claiming, mock episode flow, metrics, snapshot validation, leakage checks, schemas, CI, and a standard-library local HTTP bridge backed by a deterministic fake ALFWorld adapter. The bridge supports health, start, step, status, abort, reset, action limits, unique episode IDs, raw JSONL event logs, and localhost contract tests.

Not implemented or verified: Hermes CLI/profile commands, plugin API and hooks, Ollama model serving, ALFWorld installation/data/API, real skill loading, real profile materialization, and GPU capacity. The bridge's real ALFWorld adapter performs package discoverability only and cannot run an environment. Each external boundary must pass the documented pilot before experimental use.

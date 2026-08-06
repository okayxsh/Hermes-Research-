# Pilot protocol

Run the master-plan pilot sequence in order: repository/doctor checks, raw model and tool calls, Hermes/plugin/profile/skill tests, bridge and multi-step ALFWorld tests, logging reconciliation, failure/resume/parallel tests, mini acquisition/snapshot runs, capacity benchmark, relevance audit, then model decision and full report. Phase 2's fake bridge validates the HTTP contract only; repeat the bridge contract tests against a capability-confirmed real ALFWorld adapter before considering the integration operational.

Before plugin/profile validation, inspect `rq1 hermes-capabilities` and run the fake Phase 3 check. For installed Hermes, use a temporary `HERMES_HOME`, `HERMES_ENABLE_PROJECT_PLUGINS=1`, and `RQ1_RUN_REAL_HERMES_TESTS=1`; do not create profiles or load an LLM for discovery evidence. Capture native skill events only when a supported native operation is actually observed.

No final acquisition or unseen evaluation is permitted before a passing pilot report.

# Failure policy

Freeze this policy before final evaluation. Malformed calls get one repair attempt; health/status timeouts may retry once; start, step, reset, and abort timeouts are never retried because their outcome is unknown and require a new experiment attempt; bridge/Ollama crashes restart the episode under a new attempt ID; invalid actions count toward the limit; interruptions restart from the beginning; partial attempts are never merged.

The Phase 6 runner resumes only between pilot tests. Stale running tests become `interrupted`; blocked capabilities are re-probed; and retries create new immutable attempts. Live service termination is never automated by fake failure tests.

If a final-evaluation activation validation detects drift, it writes an immutable invalidation record. The activation is never silently repaired or reactivated; a reviewed approval and a new activation are required.

Analysis fails closed on unvalidated activation/evaluation evidence, missing pairs, snapshot or configuration drift, unreconciled logs, profile contamination, skill writes, or absent post-failure boundaries. Excluded raw data is retained and classified; it is never silently dropped.

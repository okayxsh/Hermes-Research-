# Failure policy

Freeze this policy before final evaluation. Malformed calls get one repair attempt; health/status timeouts may retry once; start, step, reset, and abort timeouts are never retried because their outcome is unknown and require a new experiment attempt; bridge/Ollama crashes restart the episode under a new attempt ID; invalid actions count toward the limit; interruptions restart from the beginning; partial attempts are never merged.

The Phase 6 runner resumes only between pilot tests. Stale running tests become `interrupted`; blocked capabilities are re-probed; and retries create new immutable attempts. Live service termination is never automated by fake failure tests.

# Autopilot contingency policy

Contingencies are append-only records classifying transient, uncertain-mutation, compatibility, scientific, measurement, contamination, resource, user-stop, code, and archive failures. Read-only probes may retry with bounded backoff. Uncertain mutations are never retried in-place: evidence is retained, the attempt is interrupted, contamination is checked, and a fresh attempt is required.

Any drift in frozen model, environment, policy, task, snapshot, profile, worker, or relevance inputs blocks continuation and requires a new reviewed approval.

# Data dictionary

- Run: deterministic task/snapshot/repetition assignment plus profile, model, machine, status, and attempt IDs.
- Step: observation, selected action, validity, reward, timing, and loaded skills.
- Skill event: a legacy `skill_view` or a versioned `skill_index_available`, `skill_selected`, `skill_loaded`, `skill_managed`, or `unknown_native_skill_operation` record. It retains the native operation, skill ID, relevance (`relevant`, `irrelevant`, or `unknown`), and whether it was `simulated` or `observed`.
- Hermes tool result: one typed JSON object containing the tool, request ID, latency, typed bridge result or structured error, and available run/attempt/profile/session correlation metadata.
- Episode binding: additive run-registry evidence linking run, attempt, episode, session, profile, Hermes log, and bridge log paths.
- Pilot test: immutable `pilot_00`-`pilot_36` specification with prerequisites, mode, timeout, retry/cleanup policy, capability requirements, blocking class, and required evidence level.
- Pilot attempt: one execution of one pilot test with a unique attempt ID, input fingerprint, status, evidence references, error/remediation, and next allowed test.
- Pilot evidence: a schema-validated artifact path and SHA-256 hash labelled `static`, `mock`, `installed`, `real_component`, or `real_integrated`; mock evidence cannot satisfy a real gate.
- Recovery evidence binding: additive registry link from run/attempt to checkpoint, perturbation, profile, snapshot, recovery log, and result.
- Snapshot: skill list/count, order, hashes, source library, and Git revision.

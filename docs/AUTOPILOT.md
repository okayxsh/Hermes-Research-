# RQ1 autopilot

`scripts/rq1_autopilot.sh` is the public supervisor entry point. Use `plan --mode bootstrap` before `bootstrap --yes`, and use `final --approval <file> --yes` only after a real pilot GO and reviewed freezes. The supervisor is fail-closed: unverified real adapters produce a structured blocked state, never a simulated completion.

`status`, `logs`, `resume`, and `stop` operate on a run ID under `artifacts/autopilot/`. Stops are cooperative; uncertain environment mutations are preserved as interrupted evidence and must restart with a new attempt.

Final outputs are local only. No command uploads, syncs, or accesses `valid_unseen` before final activation.

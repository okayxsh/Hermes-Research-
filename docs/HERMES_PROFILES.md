# Isolated Hermes profiles

`rq1-pilot` and `rq1-acquisition` are isolated Hermes state directories, not repositories. They may share this repository as their working directory but must not share skills, sessions, memory, plugins, curator state, or profile databases.

Use `python -m rq1.cli profiles plan` for a non-mutating plan and `python -m rq1.cli profiles isolation-test` for local fake-backend verification. `create-base --yes` is capability-gated: it refuses to run unless Hermes advertises JSON profile inspection, safe profile-location discovery, `--no-skills`, and project-plugin activation. It never touches a personal/default profile.

The pilot profile permits explicitly designated temporary pilot skills. The acquisition profile is train-only; later acquisition code may write skills only during an explicit post-success learning stage. Both begin with no skills, no sessions, no memory, curator disabled, and only the experiment plugin/toolset.

`rq1-recovery-<snapshot>` is an uninstantiated Phase 4 template. It will later receive one frozen read-only chronological snapshot; checkpoint and perturbation metadata belong to each run, not mutable profile state. Real profile creation, validation, and isolation remain unverified until a compatible Hermes installation is tested.

Phase 6 uses only `rq1-pilot` plus clearly named disposable `rq1-test-*` profiles. Mini acquisition and snapshot checks never touch `rq1-acquisition` or instantiate final `rq1-recovery-*` profiles. Real temporary-profile cleanup remains explicit and destructive-confirmation gated.

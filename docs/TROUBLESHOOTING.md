# Troubleshooting

If a stage fails, inspect `state/stage_status.json` and its referenced report. Do not delete a lock, report, registry, or snapshot to bypass a failure. Resolve the cause, then use an explicit future recovery command or start a new attempt according to the failure policy.

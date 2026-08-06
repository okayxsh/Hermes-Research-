# Operational state

`stage_status.json`, `run_registry.sqlite`, `pilot_latest.json`, and `pilot_runs/` are generated locally and excluded from Git. Pilot state is atomically updated resumable control data; immutable attempt evidence lives under ignored `artifacts/pilot_reports/` and is never overwritten.

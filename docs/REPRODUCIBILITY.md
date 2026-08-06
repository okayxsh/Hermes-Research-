# Reproducibility

Final-stage commands require immutable environment and protocol freeze manifests. Old generic stage reports are retained as history but cannot satisfy scientific gates because they lack validated scientific artifacts.

After the pilot, freeze the Git commit, Hermes/Ollama/ALFWorld/Python versions, model digest, prompts, tool schemas, task lists, configs, snapshots, and failure policy. Archive checksums, manifests, processed logs, analysis, tables, and figures. Do not commit credentials, model weights, raw large datasets, personal profiles, or large logs.

Phase 6 stores immutable per-test attempts beneath `artifacts/pilot_reports/<run-id>/tests/` and preserves earlier aggregate revisions by content hash. The mutable latest-run pointer is convenience state, not evidence. Only Phase 7 real reports may support an environment/protocol freeze.

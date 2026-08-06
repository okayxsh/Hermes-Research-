# Reproducibility

Final-stage commands require immutable environment and protocol freeze manifests. Old generic stage reports are retained as history but cannot satisfy scientific gates because they lack validated scientific artifacts.

Task manifests are separately hash-bound evidence. A changed ALFWorld metadata/index identity invalidates a proposal or frozen manifest; manually entered IDs, mutable task lists, and simulated fixtures are not reproducible final-task evidence.

Final evaluation has a third immutable evidence layer: an activation manifest that references hashes for all required pilot, freeze, task, acquisition, snapshot, profile, and recovery evidence. It is invalidated rather than silently reused when inputs drift.

After the pilot, freeze the Git commit, Hermes/Ollama/ALFWorld/Python versions, model digest, prompts, tool schemas, task lists, configs, snapshots, and failure policy. Archive checksums, manifests, processed logs, analysis, tables, and figures. Do not commit credentials, model weights, raw large datasets, personal profiles, or large logs.

Phase 6 stores immutable per-test attempts beneath `artifacts/pilot_reports/<run-id>/tests/` and preserves earlier aggregate revisions by content hash. The mutable latest-run pointer is convenience state, not evidence. Only Phase 7 real reports may support an environment/protocol freeze.

Analysis output is hash-addressed beneath `results/analysis/<evaluation-run>/`. Its manifest records input, activation, relevance-rule, code, and generated-artifact hashes plus the deterministic bootstrap seed. It is an offline transformation of saved logs only.

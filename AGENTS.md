# Reproducible agent-environment experiment

## Purpose

Build a reproducible controlled-recovery experiment: determine whether a persistent agent's naturally accumulated skill library improves recovery from controlled plan-invalidating failures in multi-step ALFWorld tasks, or increases post-failure retrieval noise and recovery degradation.

## Hard constraints

- Do not use DeepSeek or DeepSeek-R1 derivatives.
- Model candidates are Hermes 3 Llama 3.1 8B, then Llama 3.1 8B Instruct only if the pilot requires it.
- Use Hermes native skills first. Do not add Sentence-BERT, vector databases, LangChain, LangGraph, or a custom semantic retriever without explicit approval.
- Use ALFWorld text tasks: `train` for acquisition, `valid_seen` for pilots, and untouched `valid_unseen` only for final evaluation.
- Use one repository and isolated Hermes profiles. Experimental profiles contain no bundled skills.
- Do not revert to ordinary task-success-only evaluation: paired conditions must share the checkpoint and controlled perturbation.
- Evaluation snapshots are frozen/read-only; disable persistent memory, curator, unrelated tools, and skill writes during recovery evaluation.
- Never invent Hermes commands, configuration keys, hook payloads, or plugin schemas. Probe capabilities and keep version-specific code in adapters.
- Do not install Hermes, Ollama, models, or ALFWorld—or run GPU tests—unless a later task explicitly authorizes it.

## Implementation rules

- Core orchestration is typed Python; Bash scripts are thin wrappers.
- All stages are idempotent, resumable, observable, fail-fast, and non-destructive by default.
- Never overwrite passed stage reports, snapshots, profiles, logs, or run IDs without an explicit destructive flag.
- Do not claim real Hermes or ALFWorld compatibility from mocks. Capability-gate unverified external behaviour.
- Record Git revision, machine manifest, configuration, attempt ID, outputs, errors, and the next allowed command.
- Analysis must run solely from saved logs.
- Phase 6 fake pilot success is mock orchestration evidence only. Phase 7 is the real university pilot and environment/recovery-protocol freeze.
- Pilot mini acquisition and snapshots are disposable artifacts; never promote them into final acquisition, snapshots, recovery profiles, or evaluation data.

## Validation rules

- Validate configuration and output schemas before a stage is marked passed.
- Run the available tests, write a machine-readable stage report, update stage state, and print the next allowed command.
- Do not fabricate successful tests or integrations. Clearly report untested or unavailable external dependencies.

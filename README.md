# Reproducible agent-environment experiment

This repository is the foundation for a controlled agent-environment study involving persistent procedural guidance, retrieval behavior, and task outcomes.

It currently provides a tested, mock-only orchestration layer: configuration loading, stage state and reports, SQLite run claiming, synthetic episodes, snapshot validation, leakage checks, metrics, schemas, and CI. Hermes, Ollama, and ALFWorld integration are deliberately **unverified adapters** until the pilot.

## Quick start

```bash
python -m pip install -e .
python -m rq1.cli preflight
python -m rq1.cli doctor
python -m rq1.cli validate-config
python -m rq1.cli mock-run
python -m unittest discover -s tests -v
```

Use `python -m rq1.cli stage-status` to inspect progress. On Bash/WSL, the numbered scripts in `scripts/` call the same CLI.

## Safety boundary

The foundation does not install external services, download models/data, or make claims about Hermes APIs. See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) and [docs/UNVERIFIED_INTEGRATIONS.md](docs/UNVERIFIED_INTEGRATIONS.md).

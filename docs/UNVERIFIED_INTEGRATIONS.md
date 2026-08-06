# Unverified integrations

This repository intentionally separates an implemented contract from evidence that an external tool works on the target university machine. Official upstream documentation informs the setup design, but does not prove live compatibility.

| Capability | Current evidence | Evidence required before claiming support |
|---|---|---|
| Ubuntu 22.04/24.04 setup | Setup contract only | Complete staged setup with valid per-stage and aggregate reports on the target machine. |
| `uv`/Python environment | Typed setup foundation only | `uv sync --locked`, project import, and test execution from the resulting Python 3.11 `.venv`. |
| Ollama | No live installation or serving test | Recorded version and installer hash, responsive localhost API, model metadata, and deterministic raw-inference smoke test. |
| Hermes Agent | No live installation or CLI test | Recorded version/help output, clean `hermes doctor`, verified local-model configuration path, and capability manifest. |
| Hermes profiles | Command shape derived from current official docs, not locally executed | Isolated `rq1-pilot` and `rq1-acquisition` homes, `--no-skills` evidence, unchanged default profile, and profile-specific configuration checks. |
| Hermes plugin/tools/hooks | Placeholder boundary only | Installed-version schema probes and raw tool-call/hook evidence from the pilot. |
| ALFWorld package | Package discoverability probe only | Pinned package install, metadata/import checks, and clean downloader discovery. |
| ALFWorld data | No download or dataset validation | Data inventory, size/count checks, digest, and successful task loading from the configured data directory. |
| Fake bridge | Unit, contract, and local HTTP integration coverage | Already suitable for fake installation verification, but not for a real compatibility claim. |
| Real ALFWorld bridge | Capability-gated and unavailable | Actual start → step → reset through the real adapter on the target machine. |
| `hermes3:8b` | Candidate identifier only | Ollama pull metadata/digest and a successful raw inference smoke test. |
| `llama3.1:8b` fallback | Optional candidate identifier only | Explicit fallback installation flag, recorded digest, and separate smoke-test evidence. |

## Clean missing-capability behavior

Missing executables, packages, data, models, unsupported command help, unavailable localhost services, malformed external output, and real-adapter unavailability must be reported as structured `failed` or `blocked` results. Reports must include the failed probe and remediation without exposing secrets or emitting an unhandled traceback.

A skip flag means "do not mutate this capability," not "pretend it passed." A skipped requirement may be accepted only when a fresh probe confirms an existing valid installation; otherwise `installation_ready` and `pilot_ready` remain false.

## Evidence boundary

Fake bridge verification exercises health, start, step, status, reset, and abort over the local HTTP contract. It does not load ALFWorld data, run an ALFWorld environment, validate Hermes tools, or establish end-to-end compatibility.

Before operational use, preserve installed versions, installer hashes, command help, capability probes, model metadata, profile-isolation evidence, bridge logs, ALFWorld data inventory, and machine/software/model manifests. The exact real-ALFWorld pilot gate is an actual start → step → reset cycle; package import alone is insufficient.

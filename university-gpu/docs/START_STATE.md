# University GPU package start state

Captured on 2026-10-26 at approximately 19:20 UTC, before package implementation.

## Git provenance

| Checkout | State before implementation |
|---|---|
| `experiment/rq1-protocol-migration` | Clean at `d1f17d8fd78752e6f4e631e2d4449c99e9586fb2`, branch `rq1-protocol-migration`. The configured origin was fetched and verified to point at `c052e2cbdaa63419db4122cf08b1ccc64b38b3b4`; the new `feat/rq1-university-4gpu` branch was created from that exact commit. |
| `experiment/runpod-preparation-clone` | `bca2e2c0612f53655d41f09a1b186884eb3061e0`, branch `runpod-preparation`, ahead 1 / behind 2 of its local upstream reference. Four tracked files were modified (`src/rq1/hermes/adapter.py`, `plugin_runtime.py`, `verification.py`, and `tests/test_hermes_integration.py`); `src/hermes_alfworld_rq1.egg-info/` was untracked. This checkout was not changed. |
| `experiment/work here` | Unborn `master` repository with the project tree untracked and no commit. This checkout was not changed. |

The scientific feature branch is based on the verified remote commit, not the stale local tracking ref. The legacy protocol instructions in the fetched source were reviewed; the current model freeze is Gemma 4 12B, while the existing 0/6/12/18 protocol remains the historical evaluated protocol.

## Frozen and roadmap input hashes

SHA-256 values captured before implementation:

| Input | SHA-256 |
|---|---|
| Oct 20 `data/evaluation-units-360.csv` | `D6F0CE4B75987096A2E8C550AB9952CBC4939AA8F9AD4F9B5982E8CC5158E215` |
| Oct 20 `data/evaluation-terminal-records-360.jsonl` | `834004F8EC6EE5E0B795CFBA562C8499C12FA2FCE80C120F1BCD65F9A53C1EA0` |
| Oct 20 `data/final-human-relevance-810.csv` | `44D552DE320D65095FFCF6CED8D211E6EC462B48C2075AF1045C64B6013C7DB3` |
| Oct 20 `data/final-human-relevance-810.jsonl` | `66D920AD49019E8927152424ECD178DA2F18E693369EF34F5035C5034019C7BD` |
| Oct 26 `EXPERIMENT_ROADMAP.md` | `D6405342C40B6EE73D83DEE78C0D906BFAE7EFDD1315DC2E1126D1E2280B1B33` |
| Oct 26 `audit/roadmap-anchor-compatibility.md` | `6EB8D962D3D2B24ED713A0ECAB8452A9D3D3E710D92A048A10501247A73EB2A8` |
| Oct 26 `SELECTIVE_MEMORY_EXTENSION.md` | `DE1E1FB4AE1AA5E100C87DAD88C7FEF7898A846472F5FE357B3AD85DF88D675F` |
| Oct 24 `hani_calibration_completed.csv` | `67F79F1B41A787F70689738A31C069730EB5636CF2D0592406296597637ACFBC` |

## Scientific source hashes

| Source | SHA-256 |
|---|---|
| `configs/models/gemma4-12b.yaml` | `9A62EFDD7EA5FA5D8E3552E6B60FA5D16F1DC985796877C75A36FD362050CAB4` |
| `configs/evaluation/amended.yaml` | `9336E7F8FE3C55B0C3F8D880EFE0CBA7C027C99218C839DE19D236740BCC7F11` |
| `docs/decisions/012-pre-evaluation-library-amendment.md` | `948E5EF450D20D3752D5372F7719050A3E210AAB267D33A3DA26DD5698916808` |
| `src/rq1/evaluation/amended_executor.py` | `5459830B37C70A3D43BE1D4F8094F445E9799E6B7AD8B5A0B56BBB2651D7C935` |
| `src/rq1/evaluation/amended_matrix.py` | `16B3FF056D696AE61EAB7CB8D39644796223902362A33838B94BFC40D3C9E49E` |
| `src/rq1/hermes/episode_driver.py` | `9CFBBA659FA4BEBFD69DD6AA237E92B8A9C46DF8D77C7EFB97366A0F35B34601` |

No historical evaluation, annotation, manuscript, calibration, or archived input files were edited as part of this package.

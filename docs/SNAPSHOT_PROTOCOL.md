# Skill-library protocol

The four frozen memory conditions are built by `rq1.skills.library`: `NoLib` (0), `Clean-24` (24 core skills, 4 per family), `Accum-60` (`Clean-24` + 36 accumulated), and `Accum-96` (`Clean-24` + 72 accumulated). The `Clean-24` core is byte-identical across the three non-empty conditions; accumulated near-duplicates are preserved, never deduplicated. Construction fails closed if any per-family quota cannot be satisfied.

The legacy chronological snapshot design (`L0`/`L25`/`L50`/`L75`/`L100`) is deprecated and must not be used on the scientific path.

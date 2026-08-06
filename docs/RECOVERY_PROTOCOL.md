# Controlled recovery protocol

Phase 5 creates contract evidence for the same post-failure state across later chronological skill snapshots. An explicit recovery operation resets the environment, replays a saved valid prefix, compares versioned observable-state digests, applies a stored perturbation, validates solvability, and constructs the recovery-start context.

Observable state equality is not a claim of complete internal ALFWorld equality. Real state serialization, target relocation, and solvability remain `TO_BE_VERIFIED_BY_RECOVERY_PILOT`; fake verification proves only this repository's deterministic recovery contract.

# Experiment boundary

Acquire skills only on `train`; run pilots/calibration only on `valid_seen`; freeze task lists, relevance rules, model, prompts, tool schemas, and failure policy before final evaluation on untouched `valid_unseen`. Evaluation must use one read-only isolated profile per library snapshot.

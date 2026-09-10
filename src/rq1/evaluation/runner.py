from __future__ import annotations
from pathlib import Path
from typing import Mapping

from rq1.evaluation.models import EvaluationQueueItem
from rq1.evaluation.activation import log_task_access, require_runtime_opt_in
from rq1.experiment.models import ExperimentUnit, RunExecutionContext, RunExecutor
from rq1.experiment.persistence import ExperimentStore, bind_repository_configuration
from rq1.experiment.runner import DurableExperimentRunner, RunnerOptions


def evaluation_units(
    queue: list[EvaluationQueueItem],
    library_sizes: Mapping[str, int] | None = None,
) -> list[ExperimentUnit]:
    sizes = dict(library_sizes or {})
    return [
        ExperimentUnit(
            phase="evaluation",
            task_id=item.task_id,
            task_index=index,
            condition=item.snapshot_id,
            seed=item.seed,
            identity={
                "phase": "evaluation",
                "task_id": item.task_id,
                "checkpoint_digest": item.checkpoint_digest,
                "perturbation_digest": item.perturbation_digest,
                "recovery_context_digest": item.recovery_context_digest,
                "snapshot_id": item.snapshot_id,
                "snapshot_hash": item.snapshot_hash,
                "repetition": item.repetition,
                "seed": item.seed,
            },
            payload=item.to_dict(),
            library_name=item.snapshot_id,
            library_size=sizes.get(item.snapshot_id),
            library_hash=item.snapshot_hash,
        )
        for index, item in enumerate(queue, 1)
    ]


def run_resumable_evaluation(
    root: Path,
    activation_manifest: Path,
    experiment_id: str,
    queue: list[EvaluationQueueItem],
    executor: RunExecutor,
    *,
    configuration: dict[str, object],
    options: RunnerOptions | None = None,
    library_sizes: Mapping[str, int] | None = None,
    output_base: Path | None = None,
) -> dict[str, object]:
    activation = require_runtime_opt_in(root, activation_manifest)
    if library_sizes is None:
        raise ValueError("validated evaluation library sizes are required")
    missing_sizes = {item.snapshot_id for item in queue} - set(library_sizes)
    if missing_sizes:
        raise ValueError(
            "missing validated library sizes for snapshots: "
            + ", ".join(sorted(missing_sizes))
        )
    store = ExperimentStore(root, experiment_id, base=output_base)
    bound_configuration = {
        **configuration,
        "activation_id": activation.activation_id,
        "activation_hash": activation.content_sha256,
        "model_digest": activation.model_digest,
    }
    bound_configuration = bind_repository_configuration(root, bound_configuration)

    def activated_executor(unit: ExperimentUnit, context: RunExecutionContext):
        log_task_access(activation_manifest, unit.task_id)
        return executor(unit, context)

    return DurableExperimentRunner(store).run(
        "evaluation", evaluation_units(queue, library_sizes), bound_configuration,
        activated_executor, options,
    )


def run_final_evaluation(root: Path, activation_manifest: Path) -> None:
    """Validate deliberate activation before any unseen task can be loaded."""
    require_runtime_opt_in(root, activation_manifest)
    # No task list is opened before the activation verification above.
    raise RuntimeError("real final evaluation remains blocked until observed recovery-profile, perturbation, solvability, and Hermes dispatch adapters are available")

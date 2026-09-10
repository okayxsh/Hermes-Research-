"""Crash-safe experiment execution primitives.

The package is deliberately independent of Hermes and ALFWorld.  Production
adapters supply one-run executors only after their external surfaces have been
observed and capability-gated.
"""

from rq1.experiment.models import (
    ExperimentUnit,
    RunExecutionContext,
    RunFailure,
    RunOutcome,
)
from rq1.experiment.runner import DurableExperimentRunner, RunnerOptions

__all__ = [
    "DurableExperimentRunner",
    "ExperimentUnit",
    "RunExecutionContext",
    "RunFailure",
    "RunOutcome",
    "RunnerOptions",
]

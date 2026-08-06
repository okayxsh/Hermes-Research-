"""Explicit pilot-ID routing. Missing routes are a programmer error, never a generic block."""
from __future__ import annotations
from collections.abc import Callable
from rq1.pilot.models import RuntimeExecution
from . import alfworld, hermes, installation, machine, mini_workflow, model, profiles, recovery, reporting, repository, resilience
from rq1.pilot.real_runtime.base import RealExecutionContext

Handler = Callable[[RealExecutionContext], RuntimeExecution]

def build_handlers() -> dict[int, Handler]:
    handlers: dict[int, Handler] = {
        0: repository.run, 1: machine.run, 2: installation.run, 3: model.probe, 4: model.tool_format,
        5: profiles.profile, 6: hermes.discovery, 7: hermes.dispatch, 8: profiles.isolation,
        9: hermes.skills, 10: hermes.skills, 11: profiles.write_protection, 12: alfworld.standalone,
        13: hermes.dispatch, 14: lambda ctx: alfworld.trajectory(ctx), 15: lambda ctx: alfworld.trajectory(ctx, complete=True),
        16: lambda ctx: recovery.run(ctx, "checkpoint"), 17: lambda ctx: recovery.run(ctx, "replay"),
        18: lambda ctx: recovery.run(ctx, "perturbation"), 19: lambda ctx: recovery.run(ctx, "solvability"),
        20: lambda ctx: recovery.run(ctx, "context"), 21: lambda ctx: recovery.run(ctx, "conditions"),
        22: lambda ctx: recovery.run(ctx, "conditions"), 23: lambda ctx: recovery.run(ctx, "conditions"),
        24: lambda ctx: recovery.run(ctx, "reconcile"), 25: profiles.isolation, 26: resilience.run,
        27: lambda ctx: mini_workflow.run(ctx, "acquisition"), 28: lambda ctx: mini_workflow.run(ctx, "snapshots"),
        29: lambda ctx: mini_workflow.run(ctx, "paired"), 30: lambda ctx: mini_workflow.run(ctx, "workers"),
        31: lambda ctx: mini_workflow.run(ctx, "resume"), 32: lambda ctx: mini_workflow.run(ctx, "benchmark"),
        33: lambda ctx: mini_workflow.run(ctx, "labels"), 34: lambda ctx: mini_workflow.run(ctx, "benchmark"),
        35: lambda ctx: recovery.run(ctx, "protocol"), 36: reporting.final,
    }
    expected = set(range(37))
    if set(handlers) != expected: raise RuntimeError(f"Real pilot handler routing is incomplete: {sorted(expected - set(handlers))}")
    return handlers

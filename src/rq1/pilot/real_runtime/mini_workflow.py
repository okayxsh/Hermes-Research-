from __future__ import annotations
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked

def run(context: RealExecutionContext, stage: str):
    messages = {
        "acquisition": ("native_skill_write_unobserved", "A safe installed Hermes skill-write operation has not been observed."),
        "snapshots": ("pilot_library_missing", "No disposable real mini-acquisition library exists."),
        "paired": ("paired_recovery_unavailable", "Real controlled perturbation/recovery evidence is required before paired mini evaluation."),
        "workers": ("real_worker_queue_unconfigured", "No approved disposable real worker queue is configured."),
        "resume": ("real_worker_queue_unconfigured", "No approved disposable real worker queue is configured."),
        "benchmark": ("runtime_inputs_incomplete", "Real model and ALFWorld measurements are required for a runtime benchmark."),
        "labels": ("native_skill_events_unobservable", "Observed native retrieval events are required for relevance labels."),
    }
    code, message = messages[stage]
    return blocked(code, message, "Resolve the named pilot-only prerequisite; do not create final experiment artifacts.", {"handler": "mini_" + stage, "disposable": True})

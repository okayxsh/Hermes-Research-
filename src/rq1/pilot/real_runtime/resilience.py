from __future__ import annotations
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked

def run(context: RealExecutionContext):
    return blocked("safe_failure_injection_requires_live_boundaries", "Real failure injection needs an observed local bridge/Hermes/model boundary without stopping services.", "Run after real bridge and Hermes dispatch evidence is available; this handler will never stop live services.", {"handler": "resilience", "planned_failures": ["ollama_unavailable", "hermes_unavailable", "bridge_unavailable", "malformed_tool", "timeout", "replay_mismatch", "perturbation_failure", "unsolvable", "interrupted"]})

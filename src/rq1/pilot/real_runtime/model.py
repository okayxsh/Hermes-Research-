"""Local-only Ollama probes. They never pull, install, or substitute models."""
from __future__ import annotations

import json, time
from urllib.error import URLError
from urllib.request import Request, urlopen

from rq1.pilot.models import EvidenceLevel
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked, failed, passed

APPROVED = {"hermes3:8b", "llama3.1:8b"}

def _api(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request("http://127.0.0.1:11434" + path, data=data, method="POST" if data else "GET")
    if data: request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode())
    if not isinstance(value, dict): raise ValueError("Ollama returned non-object JSON")
    return value

def probe(context: RealExecutionContext):
    if context.candidate_model not in APPROVED:
        return blocked("unapproved_model", "Only approved candidate models may be used.", "Use hermes3:8b or explicitly selected llama3.1:8b.")
    try:
        version = _api("/api/version")
        tags = _api("/api/tags")
        models = tags.get("models", [])
        model = next((item for item in models if isinstance(item, dict) and item.get("name") == context.candidate_model), None)
        if not model: return blocked("model_missing", f"{context.candidate_model} is not installed in local Ollama.", "Install it through the approved setup stage; the pilot runner never pulls models.", {"handler": "model", "ollama_version": version})
        shown = _api("/api/show", {"name": context.candidate_model})
        samples = []
        for _ in range(3):
            started = time.monotonic(); result = _api("/api/generate", {"model": context.candidate_model, "prompt": "Reply exactly: RQ1_READY", "stream": False}); samples.append({"response": result.get("response"), "latency_ms": round((time.monotonic()-started)*1000)})
        if any(item["response"] != "RQ1_READY" for item in samples): return failed("model_response_unstable", "Raw model response did not meet deterministic pilot prompt.", {"handler": "model", "samples": samples})
        return passed(EvidenceLevel.REAL_COMPONENT, {"handler": "model", "operation_executed": True, "ollama_version": version.get("version"), "model": context.candidate_model, "digest": shown.get("digest") or model.get("digest"), "samples": samples})
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        return blocked("ollama_unavailable", "Local Ollama is unavailable or returned malformed output.", "Start only the configured localhost Ollama service and verify the approved model is installed.", {"handler": "model", "error_type": type(exc).__name__})

def tool_format(context: RealExecutionContext):
    result = probe(context)
    if result.status.value != "passed": return result
    return blocked("ollama_tool_surface_unobserved", "The installed Ollama model tool-call surface is not yet version-adapted.", "Capture official installed-version tool-call evidence before enabling this handler.", {**result.details, "handler": "model_tool_format"})

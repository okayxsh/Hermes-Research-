"""Model / runtime identity: historical pin preserved, installed digest PROPOSED, fail-closed verification.

The same tag and quantisation do NOT establish equivalence between digests.  Execution requires (a) an approved
runtime configuration bound by SHA-256 and (b) an observed runtime record (from the university workstation) that matches
it field by field.  This module never contacts a provider.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from rq1.hermes.episode_driver import MODEL_CONTEXT_LENGTH, OUTPUT_TOKEN_CAP
from rq1.university_gpu.manifest import MODEL_DIGEST as HISTORICAL_DIGEST, SBERT_MODEL, SBERT_REVISION
from rq1.evaluation.amended_protocol import SBERT_SNAPSHOT_SHA256

INSTALLED_ENGINEERING_DIGEST = "6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b"
HISTORICAL_PIN = {"model_tag": "gemma4:12b", "model_manifest_digest": HISTORICAL_DIGEST, "quantization": "Q4_K_M",
                  "status": "historical_scientific_pin (unchanged; historical records keep it)"}
REQUIRED_FIELDS = (
    "model_tag", "model_manifest_digest", "weight_blob_digests", "quantization", "tokenizer_sha256", "tokenizer_revision",
    "chat_template_sha256", "provider", "provider_version", "hermes_version", "alfworld_version", "sbert_model",
    "sbert_revision", "sbert_snapshot_sha256", "num_ctx", "temperature", "top_p", "top_k", "num_predict", "think",
    "inference_seeds", "environment_seed_policy",
)


class RuntimeIdentityError(RuntimeError):
    pass


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps({k: config.get(k) for k in REQUIRED_FIELDS}, sort_keys=True).encode()).hexdigest()


def proposed_runtime_config() -> dict[str, Any]:
    """Known values filled from the frozen code; everything that must be OBSERVED on the workstation is None."""
    return {
        "status": "PROPOSED_UNVERIFIED",
        "model_tag": "gemma4:12b",
        "model_manifest_digest": INSTALLED_ENGINEERING_DIGEST,
        "weight_blob_digests": None,
        "quantization": "Q4_K_M",
        "tokenizer_sha256": None,
        "tokenizer_revision": None,
        "chat_template_sha256": None,
        "provider": "ollama",
        "provider_version": None,
        "hermes_version": None,
        "alfworld_version": "0.4.2",
        "sbert_model": SBERT_MODEL,
        "sbert_revision": SBERT_REVISION,
        "sbert_snapshot_sha256": SBERT_SNAPSHOT_SHA256,
        "num_ctx": MODEL_CONTEXT_LENGTH,
        "temperature": 0,
        "top_p": None,
        "top_k": None,
        "num_predict": OUTPUT_TOKEN_CAP,
        "think": False,
        "inference_seeds": [11, 29, 47],
        "environment_seed_policy": "inference_seed_only; ALFWorld state fixed by the frozen checkpoint; seeds are not "
                                   "independent environment randomisations",
        "historical_pin": dict(HISTORICAL_PIN),
        "verification_required": [
            "ollama /api/show for the tag: manifest digest, every referenced blob digest, quantisation, template, parameters",
            "tokenizer.json hash and revision linked to the GGUF tokenizer (vocabulary + special tokens + chat template)",
            "prompt_eval_count from a real request equals local chat-template token count within the approved tolerance",
            "Ollama, Hermes and ALFWorld versions; Sentence-BERT revision and snapshot hash",
            "effective sampling parameters (temperature, top_p, top_k, seed, num_predict, num_ctx, think) as sent and as reported",
        ],
    }


def _approved(config: Mapping[str, Any]) -> list[str]:
    approval = config.get("approval")
    problems = []
    if not isinstance(approval, Mapping) or approval.get("approved") is not True:
        problems.append("runtime configuration is not approved")
    else:
        if not approval.get("approved_by") or not approval.get("approval_reference"):
            problems.append("runtime approval lacks approver or reference")
        if approval.get("config_sha256") != config_sha256(config):
            problems.append("runtime approval is not bound to this exact configuration")
    return problems


def verify_runtime(observed: Mapping[str, Any] | None, approved_config: Mapping[str, Any]) -> list[str]:
    """Return blocking problems (empty list = this runtime may execute under this approved configuration)."""
    problems = _approved(approved_config)
    missing = [f for f in REQUIRED_FIELDS if approved_config.get(f) in (None, "", [])]
    if missing:
        problems.append(f"approved configuration leaves required fields unset: {missing}")
    if approved_config.get("model_manifest_digest") not in (HISTORICAL_DIGEST, INSTALLED_ENGINEERING_DIGEST):
        problems.append("model digest is neither the historical pin nor the reviewed installed digest")
    if approved_config.get("think") is not False:
        problems.append("thinking must be disabled (frozen provider setting)")
    if approved_config.get("temperature") != 0:
        problems.append("temperature must be 0 (frozen provider setting)")
    if observed is None:
        problems.append("no observed runtime record from the execution workstation")
        return problems
    for name in REQUIRED_FIELDS:
        if observed.get(name) != approved_config.get(name):
            problems.append(f"observed {name} differs from the approved configuration")
    return problems


def require_runtime(observed: Mapping[str, Any] | None, approved_config: Mapping[str, Any]) -> None:
    problems = verify_runtime(observed, approved_config)
    if problems:
        raise RuntimeIdentityError("runtime identity not verified: " + "; ".join(problems))

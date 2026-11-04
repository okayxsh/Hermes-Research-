"""Offline verification for externally pinned engineering authorization records."""
from __future__ import annotations

import hmac
import os
import re
from typing import Any, Mapping

from rq1.university_gpu.models import content_hash


TRUSTED_APPROVAL_SHA256_ENV = "RQ1_TRUSTED_ENGINEERING_APPROVAL_SHA256"
TRUSTED_MANIFEST_SHA256_ENV = "RQ1_TRUSTED_ENGINEERING_MANIFEST_SHA256"


class AuthorizationError(RuntimeError):
    pass


def approval_digest(approval: Mapping[str, Any]) -> str:
    """Digest canonical approval content; this digest is not itself a trust anchor."""
    return content_hash(dict(approval))


def verify_engineering_authorization(
    approval: Mapping[str, Any], *, candidate_sha256: str, model_tag: str, model_digest: str,
    profile_id: str, action_budget: int, run_id: str, model_quantization: str,
    conditions: tuple[str, ...] = ("D0", "D18"),
) -> str:
    """Validate schema and match an operator-pinned approval digest from a separate channel."""
    if (approval.get("test_only") is not False or approval.get("schema") != ("rq1-engineering-scale-approval-v1" if profile_id == "rq1-engineering-500-v1" else "rq1-engineering-feasibility-approval-v1")
            or approval.get("authorization_type") != "engineering_gpu"
            or approval.get("approval_scope") != "engineering_feasibility_only"
            or approval.get("gpu_inference") is not True or approval.get("scientific_run") is not False
            or approval.get("not_for_scientific_claims") is not True
            or approval.get("candidate_manifest_sha256") != candidate_sha256
            or approval.get("profile_id") != profile_id or approval.get("model_tag") != model_tag
            or approval.get("model_digest") != model_digest
            or approval.get("model_quantization") != model_quantization
            or approval.get("total_action_budget") != action_budget
            or approval.get("run_id") != run_id
            or approval.get("conditions") != list(conditions)
            or approval.get("prior_pick_place_exposure_acknowledged") is not True
            or not isinstance(approval.get("approver_id"), str) or not approval["approver_id"].strip()
            or approval.get("approved_by") != approval.get("approver_id")
            or not isinstance(approval.get("approval_reference"), str) or not approval["approval_reference"].strip()
            or not isinstance(approval.get("approval_provenance"), str) or not approval["approval_provenance"].strip()):
        raise AuthorizationError("engineering approval schema or candidate/profile/model scope is invalid")
    mode = os.environ.get("RQ1_ENGINEERING_AUTHORITY_MODE", "independent")
    if profile_id == "rq1-engineering-500-v1" and mode not in {"independent", "owner_reviewed"}:
        raise AuthorizationError("scale runs require independent or owner-reviewed authorization")
    if mode == "owner_reviewed" and profile_id != "rq1-engineering-500-v1":
        raise AuthorizationError("owner-reviewed authorization is limited to the engineering scale profile")
    if mode not in {"independent", "self_attested", "owner_reviewed"}:
        raise AuthorizationError("unsupported engineering authorization mode")
    if mode == "owner_reviewed":
        _verify_owner_review(approval)
    elif profile_id == "rq1-engineering-500-v1" and approval.get("review_mode") == "owner_reviewed":
        raise AuthorizationError("owner-reviewed approval cannot be represented as independent")
    if mode == "self_attested" and "self-attestation" not in str(approval.get("approval_provenance", "")).lower():
        raise AuthorizationError("self-attested mode requires explicit self-attestation provenance")
    pin_key = ({"self_attested": "RQ1_SELF_ATTESTED_ENGINEERING_APPROVAL_SHA256",
                "owner_reviewed": "RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256"}.get(mode, TRUSTED_APPROVAL_SHA256_ENV))
    trusted = os.environ.get(pin_key, "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", trusted):
        raise AuthorizationError("owner-reviewed engineering approval integrity pin is not configured" if mode == "owner_reviewed"
                                 else "independent engineering approval hash is not configured; production authorization is blocked")
    observed = approval_digest(approval)
    if not hmac.compare_digest(trusted, observed):
        raise AuthorizationError("approval content does not match the owner-reviewed integrity pin" if mode == "owner_reviewed"
                                 else "approval content does not match the independently configured trusted hash")
    return observed


def _verify_owner_review(approval: Mapping[str, Any]) -> None:
    if (approval.get("review_mode") != "owner_reviewed"
            or approval.get("independent_review") is not False
            or approval.get("owner_reviewer_id") != approval.get("approver_id")
            or not isinstance(approval.get("owner_reviewer_id"), str)
            or not approval["owner_reviewer_id"].strip()
            or "owner-reviewed" not in str(approval.get("approval_provenance", "")).lower()):
        raise AuthorizationError("owner-reviewed mode requires the owner's identity and explicit non-independent provenance")


def verify_frozen_manifest_anchor(manifest: Mapping[str, Any]) -> str:
    """Require separately pinned frozen-run identity before any GPU operation."""
    mode = os.environ.get("RQ1_ENGINEERING_AUTHORITY_MODE", "independent")
    if manifest.get("profile_id") == "rq1-engineering-500-v1" and mode not in {"independent", "owner_reviewed"}:
        raise AuthorizationError("scale runs require independent or owner-reviewed manifest verification")
    if mode == "owner_reviewed" and manifest.get("profile_id") != "rq1-engineering-500-v1":
        raise AuthorizationError("owner-reviewed verification is limited to the engineering scale profile")
    if mode not in {"independent", "self_attested", "owner_reviewed"}:
        raise AuthorizationError("unsupported engineering authorization mode")
    approval = manifest.get("engineering_approval", {})
    if mode == "owner_reviewed":
        if not isinstance(approval, Mapping):
            raise AuthorizationError("manifest lacks owner-reviewed approval")
        _verify_owner_review(approval)
    elif (manifest.get("profile_id") == "rq1-engineering-500-v1" and isinstance(approval, Mapping)
          and approval.get("review_mode") == "owner_reviewed"):
        raise AuthorizationError("owner-reviewed approval cannot be represented as independent")
    if mode == "self_attested":
        approval = manifest.get("engineering_approval", {})
        if "self-attestation" not in str(approval.get("approval_provenance", "")).lower():
            raise AuthorizationError("manifest lacks self-attestation provenance")
    pin_key = ({"self_attested": "RQ1_SELF_ATTESTED_ENGINEERING_MANIFEST_SHA256",
                "owner_reviewed": "RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256"}.get(mode, TRUSTED_MANIFEST_SHA256_ENV))
    trusted = os.environ.get(pin_key, "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", trusted):
        raise AuthorizationError("owner-reviewed frozen manifest integrity pin is not configured" if mode == "owner_reviewed"
                                 else "independent frozen engineering manifest hash is not configured")
    observed = str(manifest.get("manifest_sha256", "")).lower()
    if not hmac.compare_digest(trusted, observed):
        raise AuthorizationError("frozen manifest differs from the owner-reviewed integrity pin" if mode == "owner_reviewed"
                                 else "frozen manifest differs from independently configured trusted hash")
    return observed

"""Offline provenance validation for a candidate model; never authorizes launch."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rq1.university_gpu.models import atomic_write_json, content_hash, load_json, sha256_file
from rq1.university_gpu.interventions import LocalGemmaTokenizer, TokenizerIdentity


def verify_candidate(manifest_path: Path, blobs: Path, tokenizer: Path, provenance: dict, encode, link_evidence: Path) -> dict:
    digest = sha256_file(manifest_path)
    if provenance.get("model_manifest_sha256") != digest:
        raise ValueError("model manifest differs from tokenizer provenance")
    manifest = load_json(manifest_path)
    refs = [manifest["config"], *manifest["layers"]]
    checked = []
    for ref in refs:
        value = ref["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ValueError("unsafe blob digest")
        path = blobs / value.replace(":", "-", 1)
        if (path.is_symlink() or not path.is_file() or sha256_file(path) != value[7:]
                or path.stat().st_size != ref["size"]):
            raise ValueError("model blob missing or corrupt: " + value)
        checked.append(value)
    tokenizer_hash = sha256_file(tokenizer)
    if (tokenizer.is_symlink() or tokenizer_hash != provenance.get("tokenizer_sha256")
            or not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("revision", "")))
            or not str(provenance.get("source_repository", "")).startswith("https://")
            or link_evidence.is_symlink() or not link_evidence.is_file()
            or sha256_file(link_evidence) != provenance.get("model_tokenizer_link_evidence_sha256")):
        raise ValueError("incomplete tokenizer source/revision/hash/link evidence")
    samples = provenance.get("samples", [])
    if len(samples) < 5 or len({s["text"] for s in samples}) != len(samples):
        raise ValueError("at least five distinct recorded reference-token samples required")
    for sample in samples:
        ids = sample.get("token_ids")
        if (not sample.get("reference_source") or not isinstance(sample.get("text"), str)
                or not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids)
                or list(encode(sample["text"])) != ids):
            raise ValueError("reference token comparison failed or lacks source")
    result = {"schema": "rq1-prospective-identity-evidence-v1", "status": "CANDIDATE_REVIEW_REQUIRED",
              "approved": False, "launch_authorized": False, "model_manifest_sha256": digest,
              "checked_blobs": checked, "tokenizer_sha256": tokenizer_hash,
              "provenance_sha256": content_hash(provenance), "reference_sample_count": len(samples),
              "limitations": ["Sample/source authenticity and tokenizer-model linkage require review",
                              "Finite sample agreement is not proof of all-input equivalence",
                              "Quantization and runtime/template config semantics require independent review",
                              "This artifact is not accepted as a launch approval by the historical launcher"]}
    result["content_sha256"] = content_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blobs", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--link-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = load_json(args.provenance)
    tokenizer = LocalGemmaTokenizer(TokenizerIdentity(args.tokenizer, evidence["tokenizer_sha256"], evidence["revision"]))
    report = verify_candidate(args.manifest, args.blobs, args.tokenizer, evidence, tokenizer.encode, args.link_evidence)
    atomic_write_json(args.output, report, overwrite=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

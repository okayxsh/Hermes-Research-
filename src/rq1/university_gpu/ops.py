"""Operator-side checks and immutable study preparation for university hardware."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from rq1.acquisition.environment import sbert_snapshot_sha256
from rq1.evaluation.amended_protocol import MODEL_DIGEST as EXECUTED_MODEL_DIGEST, SBERT_SNAPSHOT_SHA256
from rq1.university_gpu.hardware import machine_report, parse_compute_processes, verify_worker_placements
from rq1.university_gpu.calibration import CalibrationError, propose_gate_calibration, validate_gate_calibration
from rq1.university_gpu.manifest import (
    MODEL_DIGEST,
    MODEL_TAG,
    MODEL_QUANTIZATION,
    SBERT_MODEL,
    SBERT_REVISION,
    _select_skill_ids,
    build_manifest,
    freeze_manifest,
    validate_manifest,
)
from rq1.university_gpu.models import GPUDevice, atomic_write_json, content_hash, load_json, sha256_file
from rq1.university_gpu.ollama import OllamaServer, probe_endpoint, worker_specs
from rq1.university_gpu.relevance import RelevanceError, parse_relevance_csv
from rq1.skills.library import TASK_FAMILIES
from rq1.university_gpu.scheduler import Coordinator, RunLock, RunStore
from rq1.university_gpu.worker_process import WorkerProcessClient
from rq1.university_gpu.interventions import LocalGemmaTokenizer, TokenBudgetError, TokenizerIdentity


class OperatorError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise OperatorError(result.stderr.strip() or f"git command failed: {args}")
    return result.stdout.strip()


def git_provenance(root: Path) -> dict[str, Any]:
    try:
        commit = _git(root, "rev-parse", "HEAD")
        branch = _git(root, "branch", "--show-current")
        status = _git(root, "status", "--porcelain=v2", "--branch")
        clean = not any(line and not line.startswith("#") for line in status.splitlines())
        return {"commit": commit, "branch": branch, "clean": clean, "status": status}
    except OperatorError as exc:
        return {"commit": None, "branch": None, "clean": False, "error": str(exc)}


def _project_freeze_checks(root: Path) -> list[dict[str, Any]]:
    model_path = root / "configs" / "models" / "gemma4-12b.yaml"
    evaluation_path = root / "configs" / "evaluation" / "amended.yaml"
    checks = []
    try:
        model = load_json(model_path)
        inference = model.get("inference", {})
        expected_inference = {"temperature": 0, "seed": 42, "think": False, "num_predict": 2048, "num_ctx": 32768}
        valid = (model.get("model_identifier") == MODEL_TAG and model.get("digest") == MODEL_DIGEST
                 and model.get("quantization") == MODEL_QUANTIZATION and inference == expected_inference)
        checks.append({"name": "frozen_model_identity", "status": "PASS" if valid else "FAIL",
                       "expected": {"tag": MODEL_TAG, "digest": MODEL_DIGEST, "quantization": MODEL_QUANTIZATION,
                                    "inference": expected_inference},
                       "observed": {"tag": model.get("model_identifier"), "digest": model.get("digest"),
                                    "quantization": model.get("quantization"), "inference": inference}})
    except (OSError, ValueError, TypeError) as exc:
        checks.append({"name": "frozen_model_identity", "status": "FAIL", "error": str(exc)})
    try:
        protocol = load_json(evaluation_path)
        model = protocol.get("model", {})
        retrieval = protocol.get("retrieval", {})
        tasks = protocol.get("tasks", {})
        expected_model = {"tag": MODEL_TAG, "digest": MODEL_DIGEST, "quantization": MODEL_QUANTIZATION,
                          "temperature": 0, "think": False, "num_predict": 2048, "num_ctx": 32768}
        expected_retrieval = {"model": SBERT_MODEL, "revision": SBERT_REVISION,
                              "snapshot_sha256": SBERT_SNAPSHOT_SHA256, "top_k": 3}
        valid = (all(model.get(key) == value for key, value in expected_model.items())
                 and all(retrieval.get(key) == value for key, value in expected_retrieval.items())
                 and tasks.get("split") == "valid_unseen" and tasks.get("task_count") == 30
                 and protocol.get("matrix", {}).get("seeds") == [11, 29, 47])
        checks.append({"name": "historical_protocol_identity", "status": "PASS" if valid else "FAIL",
                       "model": model.get("tag"), "sbert_revision": retrieval.get("revision"),
                       "top_k": retrieval.get("top_k"), "task_split": tasks.get("split"),
                       "task_count": tasks.get("task_count"), "seeds": protocol.get("matrix", {}).get("seeds")})
    except (OSError, ValueError, TypeError) as exc:
        checks.append({"name": "historical_protocol_identity", "status": "FAIL", "error": str(exc)})
    checks.append({"name": "frozen_evaluation_digest_constant", "status": "PASS" if EXECUTED_MODEL_DIGEST == MODEL_DIGEST else "FAIL",
                   "digest": EXECUTED_MODEL_DIGEST})
    return checks


def check(root: Path, *, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or {})
    output = Path(str(config.get("run_output_dir") or "results/university-gpu"))
    gpu_count = int(config.get("worker_count", 4))
    hardware = machine_report(output_dir=output, gpu_count=4, min_gpu_free_mib=int(config.get("min_gpu_free_mib", 16_384)),
                              ollama_base_port=int(config.get("ollama_base_port", 11_534)))
    provenance = git_provenance(root)
    source_checks = _project_freeze_checks(root)
    lock = root / "uv.lock"
    uv = shutil.which("uv")
    uv_check = {"name": "locked_python_dependencies", "status": "PASS" if lock.is_file() and uv else "BLOCKED",
                "message": "uv.lock is present; uv is required for the approved project-local setup."}
    execution_config = {"name": "four_worker_configuration", "status": "PASS" if gpu_count == 4 else "FAIL",
                        "observed_worker_count": gpu_count, "required_worker_count": 4}
    return {
        "schema": "rq1-university-gpu-check-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_evidence": False,
        "status": "PASS" if hardware["status"] == "PASS" and provenance.get("clean")
                  and uv_check["status"] == "PASS" and execution_config["status"] == "PASS"
                  and all(item["status"] == "PASS" for item in source_checks) else "BLOCKED",
        "hardware": hardware,
        "git": provenance,
        "source_checks": source_checks,
        "dependency_check": uv_check,
        "execution_config_check": execution_config,
        "requested_worker_count": gpu_count,
        "notes": ["This check performs no model inference, install, download, or system mutation.",
                  "A PASS does not prove Ollama process-to-GPU placement; that requires an approved smoke test and live nvidia-smi evidence."],
    }


def setup_plan(root: Path) -> dict[str, Any]:
    return {
        "schema": "rq1-university-gpu-setup-plan-v1",
        "status": "DRY_RUN_ONLY",
        "scientific_evidence": False,
        "actions": [
            {"scope": "project-local", "command": "uv sync --frozen --extra alfworld --extra retrieval", "requires_approval": True},
            {"scope": "system", "action": "none", "requires_sudo": False},
            {"scope": "ollama", "action": "verify existing installation and exact model only", "download": False},
            {"scope": "ALFWorld", "action": "verify existing 0.4.2 install/data path", "download": False},
        ],
        "refused_actions": ["sudo", "system package install", "driver changes", "model pull", "GPU inference"],
        "project_root": str(root),
    }


def apply_project_setup(root: Path, approval_path: Path) -> dict[str, Any]:
    approval = load_json(approval_path)
    if approval.get("approved") is not True or not approval.get("approved_by") or not approval.get("approval_reference"):
        raise OperatorError("project setup requires an explicit local approval record")
    if approval.get("project_python_environment") is not True or approval.get("project_dependencies") is not True:
        raise OperatorError("approval must explicitly authorize the project-local Python environment and dependencies")
    if approval.get("system_changes") or approval.get("model_download") or approval.get("gpu_inference"):
        raise OperatorError("this setup command will not install system packages, download models, or run inference")
    uv = shutil.which("uv")
    if not uv:
        raise OperatorError("uv must already be available; this command will not install uv")
    result = subprocess.run([uv, "sync", "--frozen", "--extra", "alfworld", "--extra", "retrieval"],
                            cwd=root, capture_output=True, text=True, check=False)
    return {"status": "PASS" if result.returncode == 0 else "FAIL", "returncode": result.returncode,
            "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:],
            "approval_reference": approval["approval_reference"], "project_local_only": True}


def load_external_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise OperatorError("external config must be a JSON object")
    expanded: dict[str, Any] = {}
    unresolved = []
    for key, item in value.items():
        if isinstance(item, str):
            result = os.path.expandvars(item)
            if result.startswith("${") and result.endswith("}"):
                unresolved.append(key)
            expanded[key] = result
        else:
            expanded[key] = item
    if unresolved:
        raise OperatorError("external path variables are unresolved: " + ", ".join(unresolved))
    return expanded


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    value = str(config.get(key, "")).strip()
    if not value:
        raise OperatorError(f"external config lacks required path: {key}")
    path = Path(value).expanduser()
    if not path.exists():
        raise OperatorError(f"required external input is missing ({key}): {path}")
    return path


def directory_sha256(path: Path) -> str:
    """Hash every regular file and relative path in a dataset tree; reject symlinks."""
    if not path.is_dir() or path.is_symlink():
        raise OperatorError(f"expected a real dataset directory: {path}")
    entries = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise OperatorError(f"dataset tree contains a symlink that cannot be frozen safely: {child}")
        if child.is_file():
            entries.append({"path": child.relative_to(path).as_posix(), "sha256": sha256_file(child),
                            "bytes": child.stat().st_size})
    if not entries:
        raise OperatorError(f"dataset directory contains no files: {path}")
    return content_hash(entries)


def _tasks_and_failures(task_path: Path, failure_path: Path, *, stage: str) -> list[dict[str, Any]]:
    manifest = load_json(task_path)
    failures = load_json(failure_path)
    if not isinstance(manifest, dict) or not isinstance(failures, dict):
        raise OperatorError("task and controlled-failure inputs must be JSON objects")
    if manifest.get("status") not in {"frozen", "approved"}:
        raise OperatorError("task manifest must be frozen before a proposal is generated")
    records = manifest.get("tasks")
    failure_records = failures.get("tasks")
    if not isinstance(records, list) or not isinstance(failure_records, list):
        raise OperatorError("task and controlled-failure inputs require tasks arrays")
    definitions = {str(item.get("task_id")): item for item in failure_records if isinstance(item, dict)}
    result = []
    for record in records:
        if not isinstance(record, dict):
            continue
        task_id = str(record.get("task_id", ""))
        definition = definitions.get(task_id)
        if definition is None or (definition.get("oracle") or {}).get("validated") is not True:
            continue
        family = str(record.get("task_family") or record.get("family") or definition.get("task_family") or "")
        result.append({
            "task_id": task_id, "task_family": family, "order_index": int(record.get("order_index", len(result))),
            "checkpoint_id": definition["checkpoint_id"], "checkpoint_digest": definition["checkpoint_digest"],
            "post_detour_digest": definition["post_detour_digest"], "detour_action": definition["detour_action"],
            "expected_next_action": definition["expected_next_action"], "prefix_actions": list(definition["prefix_actions"]),
            "reference_actions": list(definition["reference_actions"]),
            "recovery_action_budget": int(definition["recovery_action_budget"]),
        })
    if stage == "pilot" and any(not item["task_id"].startswith("valid_seen:") for item in result):
        raise OperatorError("pilot task input includes a non-valid_seen task")
    if stage == "confirmatory" and any(not item["task_id"].startswith("valid_unseen:") for item in result):
        raise OperatorError("confirmatory task input includes a non-valid_unseen task")
    return result


def _library_payload(path: Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    value = load_json(path)
    libraries = value.get("libraries", value) if isinstance(value, dict) else None
    payload = libraries.get("Accum-18") if isinstance(libraries, dict) else None
    skills = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(skills, list) or len(skills) != 18:
        raise OperatorError("frozen library file must contain exactly 18 Accum-18 skills")
    texts = {str(skill["skill_id"]): str(skill["text"]) for skill in skills if isinstance(skill, dict) and skill.get("skill_id") and skill.get("text")}
    if len(texts) != 18:
        raise OperatorError("Accum-18 must contain 18 unique skill IDs with non-empty text")
    return sorted(texts), texts, value


def build_proposed_manifest(root: Path, config: Mapping[str, Any], *, stage: str, run_id: str) -> dict[str, Any]:
    tasks_path = _required_path(config, "task_manifest_path")
    failures_path = _required_path(config, "controlled_failures_path")
    library_path = _required_path(config, "accum18_library_path")
    data_path = _required_path(config, "alfworld_data_dir")
    relevance_path = _required_path(config, "relevance_path")
    calibration_path = _required_path(config, "gate_calibration_path")
    gate_scores_path = _required_path(config, "gate_scores_path")
    gate_labels_path = _required_path(config, "gate_labels_path")
    tokenizer_path = _required_path(config, "tokenizer_path")
    filler_path = _required_path(config, "filler_path")
    tasks = _tasks_and_failures(tasks_path, failures_path, stage=stage)
    library_ids, _texts, _library = _library_payload(library_path)
    relevance = parse_relevance_csv(relevance_path.read_bytes(), allowed_tasks={str(task["task_id"]) for task in tasks},
                                    library_skill_ids=set(library_ids))
    calibration = load_json(calibration_path)
    if not isinstance(calibration, dict):
        raise OperatorError("gate calibration must be a JSON object")
    calibration_issues = validate_gate_calibration(calibration, require_frozen=True)
    if calibration_issues:
        raise OperatorError("gate calibration failed validation: " + "; ".join(calibration_issues))
    gate_scores_sha256 = sha256_file(gate_scores_path)
    gate_labels_sha256 = sha256_file(gate_labels_path)
    if (calibration.get("similarity_scores_sha256") != gate_scores_sha256
            or calibration.get("adjudicated_labels_sha256") != gate_labels_sha256):
        raise OperatorError("frozen Gate calibration does not match its supplied score/label source files")
    provenance = git_provenance(root)
    if not provenance.get("commit") or not provenance.get("clean"):
        raise OperatorError("freeze source code in a clean Git commit before building the prospective manifest")
    return build_manifest(
        stage=stage, run_id=run_id, tasks=tasks, relevance=relevance, library_skill_ids=library_ids,
        randomization_seed=int(config.get("randomization_seed", 20261026)), source_commit=str(provenance["commit"]),
        task_manifest_sha256=sha256_file(tasks_path), controlled_failure_sha256=sha256_file(failures_path),
        library_sha256=sha256_file(library_path), alfworld_data_sha256=directory_sha256(data_path),
        relevance_sha256=sha256_file(relevance_path), tokenizer_sha256=sha256_file(tokenizer_path),
        tokenizer_revision=str(config.get("tokenizer_revision", "")), filler_sha256=sha256_file(filler_path),
        gate_scores_sha256=gate_scores_sha256, gate_labels_sha256=gate_labels_sha256,
        gate_threshold=float(calibration["threshold"]), gate_calibration_sha256=sha256_file(calibration_path),
        gate_calibration_split=str(calibration["split"]),
    )


def _input_checks(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    mapping = {
        "task_manifest_path": "task_manifest_sha256",
        "controlled_failures_path": "controlled_failure_sha256",
        "accum18_library_path": "library_sha256",
        "relevance_path": "relevance_sha256",
        "tokenizer_path": "tokenizer_sha256",
        "filler_path": "filler_sha256",
        "gate_scores_path": "gate_scores_sha256",
        "gate_labels_path": "gate_labels_sha256",
        "gate_calibration_path": "gate_calibration_sha256",
    }
    checks = []
    for config_key, manifest_key in mapping.items():
        try:
            path = Path(str(config[config_key]))
            observed = sha256_file(path)
            expected = str(manifest["inputs"][manifest_key])
            checks.append({"name": manifest_key, "status": "PASS" if observed == expected else "FAIL",
                           "expected": expected, "observed": observed})
        except (KeyError, OSError) as exc:
            checks.append({"name": manifest_key, "status": "BLOCKED", "error": str(exc)})
    try:
        observed = directory_sha256(Path(str(config["alfworld_data_dir"])))
        expected = str(manifest["inputs"]["alfworld_data_sha256"])
        checks.append({"name": "alfworld_data_sha256", "status": "PASS" if observed == expected else "FAIL",
                       "expected": expected, "observed": observed})
    except (KeyError, OSError, OperatorError) as exc:
        checks.append({"name": "alfworld_data_sha256", "status": "BLOCKED", "error": str(exc)})
    try:
        source_tasks = _tasks_and_failures(
            Path(str(config["task_manifest_path"])), Path(str(config["controlled_failures_path"])),
            stage=str(manifest["stage"]),
        )
        family_order = {family: index for index, family in enumerate(TASK_FAMILIES)}
        source_tasks.sort(key=lambda item: (family_order.get(str(item["task_family"]), len(family_order)),
                                            int(item.get("order_index", 0))))
        tasks_match = source_tasks == manifest.get("tasks")
        checks.append({"name": "frozen_task_definitions", "status": "PASS" if tasks_match else "FAIL",
                       "expected_task_count": len(manifest.get("tasks", [])), "observed_task_count": len(source_tasks)})
    except (KeyError, OSError, AttributeError, OperatorError, TypeError, ValueError) as exc:
        checks.append({"name": "frozen_task_definitions", "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
    try:
        library_ids, _texts, _payload = _library_payload(Path(str(config["accum18_library_path"])))
        library_match = library_ids == manifest.get("library_skill_ids")
        checks.append({"name": "accum18_library_identity", "status": "PASS" if library_match else "FAIL",
                       "expected_skill_ids": manifest.get("library_skill_ids"), "observed_skill_ids": library_ids})
    except (KeyError, OSError, OperatorError, TypeError, ValueError) as exc:
        checks.append({"name": "accum18_library_identity", "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
    try:
        labels_path = Path(str(config["relevance_path"]))
        relevance = parse_relevance_csv(labels_path.read_bytes(),
                                        allowed_tasks={str(task["task_id"]) for task in manifest["tasks"]},
                                        library_skill_ids=set(manifest["library_skill_ids"]))
        selections = _select_skill_ids(manifest["tasks"], relevance, set(manifest["library_skill_ids"]),
                                        int(manifest["randomization_seed"]))
        matches = selections == manifest.get("selected_skill_ids")
        checks.append({"name": "frozen_blinded_relevance_selections", "status": "PASS" if matches else "FAIL"})
    except (KeyError, OSError, RelevanceError, TypeError, ValueError) as exc:
        checks.append({"name": "frozen_blinded_relevance_selections", "status": "BLOCKED",
                       "error": f"{type(exc).__name__}: {exc}"})
    try:
        calibration_path = Path(str(config["gate_calibration_path"]))
        calibration = load_json(calibration_path)
        if not isinstance(calibration, Mapping):
            raise CalibrationError("frozen Gate calibration must be a JSON object")
        issues = validate_gate_calibration(calibration, require_frozen=True)
        proposal = propose_gate_calibration(Path(str(config["gate_scores_path"])),
                                            Path(str(config["gate_labels_path"])))
        comparable = ("threshold", "calibration_metrics", "candidate_count", "task_count", "task_ids", "pair_count",
                      "similarity_scores_sha256", "adjudicated_labels_sha256")
        matches = (not issues and all(calibration.get(key) == proposal.get(key) for key in comparable)
                   and calibration.get("threshold") == (manifest.get("gate") or {}).get("threshold")
                   and calibration.get("split") == (manifest.get("gate") or {}).get("calibration_split"))
        checks.append({"name": "frozen_gate_calibration_reproduction", "status": "PASS" if matches else "FAIL",
                       "issues": issues, "observed_threshold": calibration.get("threshold") if isinstance(calibration, Mapping) else None,
                       "reproduced_threshold": proposal.get("threshold")})
    except (KeyError, OSError, AttributeError, CalibrationError, TypeError, ValueError) as exc:
        checks.append({"name": "frozen_gate_calibration_reproduction", "status": "BLOCKED",
                       "error": f"{type(exc).__name__}: {exc}"})
    try:
        tokenizer_path = Path(str(config["tokenizer_path"]))
        tokenizer_sha256 = str(manifest["inputs"]["tokenizer_sha256"])
        revision = str((manifest.get("tokenizer") or {})["revision"])
        tokenizer = LocalGemmaTokenizer(TokenizerIdentity(tokenizer_path, tokenizer_sha256, revision))
        filler = Path(str(config["filler_path"])).read_bytes().decode("utf-8-sig").strip()
        filler_tokens = len(tokenizer.encode(filler))
        if not filler or filler_tokens < 1:
            raise TokenBudgetError("approved neutral filler is empty or tokenizes to no tokens")
        checks.append({"name": "local_tokenizer_and_filler", "status": "PASS",
                       "tokenizer_revision": revision, "filler_token_count": filler_tokens})
    except (KeyError, OSError, UnicodeDecodeError, TokenBudgetError, ImportError, TypeError, ValueError) as exc:
        checks.append({"name": "local_tokenizer_and_filler", "status": "BLOCKED",
                       "error": f"{type(exc).__name__}: {exc}"})
    expected_revision = (manifest.get("tokenizer") or {}).get("revision")
    checks.append({"name": "tokenizer_revision", "status": "PASS" if config.get("tokenizer_revision") == expected_revision else "FAIL",
                   "expected": expected_revision, "observed": config.get("tokenizer_revision")})
    if manifest.get("stage") == "confirmatory":
        for key, field in (("power_report_path", "power_report_sha256"), ("pilot_validation_path", "pilot_validation_sha256")):
            try:
                path = Path(str(config[key]))
                observed = sha256_file(path)
                expected = manifest["approval"][field]
                valid = observed == expected
                if key == "pilot_validation_path":
                    result = load_json(path)
                    valid = valid and result.get("passed") is True and result.get("expected") == 36 and result.get("completed") == 36
                checks.append({"name": key, "status": "PASS" if valid else "FAIL", "expected": expected, "observed": observed})
            except (KeyError, OSError, AttributeError, TypeError, ValueError) as exc:
                checks.append({"name": key, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
    return checks


def preflight(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    check_endpoints: bool = True,
) -> dict[str, Any]:
    problems = validate_manifest(manifest, require_frozen=True)
    provenance = git_provenance(root)
    if provenance.get("commit") != manifest.get("source_commit"):
        problems.append("current source commit differs from frozen manifest")
    if not provenance.get("clean"):
        problems.append("repository must be clean at launch")
    machine = machine_report(output_dir=Path(str(config.get("run_output_dir") or "results/university-gpu")), gpu_count=4,
                             min_gpu_free_mib=int(config.get("min_gpu_free_mib", 16_384)),
                             ollama_base_port=int(config.get("ollama_base_port", 11_534)))
    inputs = _input_checks(config, manifest)
    input_failures = [item for item in inputs if item["status"] != "PASS"]
    if input_failures:
        problems.extend(f"input mismatch or missing: {item['name']}" for item in input_failures)
    data_path = Path(str(config.get("alfworld_data_dir", "")))
    if not data_path.is_dir() or not (data_path / "json_2.1.1").is_dir():
        problems.append("ALFWorld 0.4.2 data directory is absent or has the wrong layout")
    try:
        alfworld_version = importlib.metadata.version("alfworld")
        if alfworld_version != "0.4.2":
            problems.append(f"ALFWorld version is {alfworld_version}, expected 0.4.2")
    except importlib.metadata.PackageNotFoundError:
        alfworld_version = None
        problems.append("ALFWorld package is not installed in the active Python environment")
    try:
        cache = Path(str(config["sbert_cache"]))
        actual_sbert = sbert_snapshot_sha256(cache)
        if actual_sbert != SBERT_SNAPSHOT_SHA256:
            problems.append("Sentence-BERT local snapshot differs from the frozen snapshot")
    except (KeyError, OSError) as exc:
        actual_sbert = None
        problems.append(f"Sentence-BERT cache cannot be checked: {exc}")
    tokenizer_path = Path(str(config.get("tokenizer_path", "")))
    if not tokenizer_path.is_file() or sha256_file(tokenizer_path) != manifest["inputs"]["tokenizer_sha256"]:
        problems.append("exact local Gemma tokenizer is unavailable or has a mismatched hash")
    count = int(config.get("worker_count", 4))
    if count != 4:
        problems.append("the university confirmatory protocol requires exactly four isolated GPU workers")
    if not shutil.which(str(config.get("ollama_binary", "ollama"))):
        problems.append("the Ollama executable is not already installed and on PATH")
    model_dir = Path(str(config.get("ollama_models_dir", "")))
    if not model_dir.is_dir():
        problems.append("the existing local Ollama model directory is missing")
    endpoints = [f"http://127.0.0.1:{int(config.get('ollama_base_port', 11_534)) + index}" for index in range(count)]
    model_checks = ([{"endpoint": endpoint, **probe_endpoint(endpoint)} for endpoint in endpoints]
                    if check_endpoints else [{"endpoint": endpoint, "status": "NOT_TESTED"} for endpoint in endpoints])
    if check_endpoints and any(item.get("status") != "PASS" for item in model_checks):
        problems.append("one or more isolated Ollama endpoints do not show the exact frozen model")
    return {
        "schema": "rq1-university-gpu-preflight-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": manifest.get("stage"),
        "run_id": manifest.get("run_id"),
        "status": ("PASS" if check_endpoints else "STATIC_PASS") if not problems and machine["status"] == "PASS" else "BLOCKED",
        "problems": problems,
        "machine": machine,
        "input_hash_checks": inputs,
        "alfworld_version": alfworld_version,
        "sbert_snapshot_sha256": actual_sbert,
        "ollama_model_checks": model_checks,
        "gpu_process_placement": {"status": "NOT_TESTED", "reason": "requires an approved inference smoke test after endpoint startup"},
        "scientific_evidence": False,
    }


def smoke_test_endpoints(endpoints: Sequence[str], *, approval: Mapping[str, Any], model: str = MODEL_TAG) -> dict[str, Any]:
    if approval.get("gpu_inference") is not True or not approval.get("approved_by") or not approval.get("approval_reference"):
        raise OperatorError("GPU smoke test requires explicit approval for inference")
    from urllib.request import Request, urlopen

    rows = []
    for endpoint in endpoints:
        request = Request(endpoint.rstrip("/") + "/api/chat", data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "Reply exactly: RQ1_READY"}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 42, "num_predict": 8, "num_ctx": 32768},
        }).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = ((body.get("message") or {}).get("content") or "").strip()
            rows.append({"endpoint": endpoint, "status": "PASS" if text == "RQ1_READY" else "FAIL",
                         "latency_seconds": round(time.monotonic() - started, 3), "response": text})
        except Exception as exc:
            rows.append({"endpoint": endpoint, "status": "FAIL", "latency_seconds": round(time.monotonic() - started, 3),
                         "error": f"{type(exc).__name__}: {exc}"})
    return {"schema": "rq1-university-gpu-smoke-v1", "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
            "approval_reference": approval["approval_reference"], "model": model, "inference_count": len(rows), "workers": rows,
            "scientific_evidence": False}


def capture_placement(worker_pids: Mapping[str, int], *, runner: Any = subprocess.run) -> dict[str, Any]:
    result = runner(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=False, timeout=20)
    if result.returncode != 0:
        return {"passed": False, "errors": [result.stderr.strip() or "nvidia-smi process query failed"], "assignments": {}}
    rows = parse_compute_processes(result.stdout)
    placement = verify_worker_placements(worker_pids, rows)
    return {**placement, "server_process_ids": dict(worker_pids), "compute_processes": rows}


def run_design_launch(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    approval: Mapping[str, Any],
    yes: bool,
    resume: bool = False,
    retry_failed: bool = False,
    cutoff_epoch: float | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    if not yes or approval.get("gpu_inference") is not True or approval.get("scientific_run") is not True:
        raise OperatorError("launch requires --yes and explicit scientific/GPU approval")
    if approval.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise OperatorError("approval does not match this frozen manifest")
    if manifest.get("stage") == "confirmatory" and approval.get("previous_task_outcomes_acknowledged") is not True:
        raise OperatorError("confirmatory approval must acknowledge that these task outcomes were observed in the historical RQ1 run")
    if not approval.get("approved_by") or not approval.get("approval_reference"):
        raise OperatorError("launch approval requires an approver and reference")
    if max_attempts is not None and max_attempts < 1:
        raise OperatorError("max_attempts must be positive")
    if cutoff_epoch is None or not math.isfinite(cutoff_epoch):
        raise OperatorError("a concrete access deadline is required")
    if cutoff_epoch is not None and time.time() + 1800 >= cutoff_epoch:
        raise OperatorError("access deadline is within the 30-minute assignment reserve")
    checked = preflight(root, config, manifest, check_endpoints=False)
    if checked["status"] != "STATIC_PASS":
        raise OperatorError("launch preflight is blocked: " + "; ".join(checked["problems"]))
    run_root = Path(str(config["run_output_dir"])) / str(manifest["run_id"])
    resolved_output = Path(str(config["run_output_dir"])).expanduser().resolve()
    try:
        resolved_output.relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise OperatorError("run_output_dir must be outside the Git checkout so scientific output cannot dirty the frozen source")
    store = RunStore(run_root, manifest)
    tasks_by_id = {str(task["task_id"]): task for task in manifest["tasks"]}
    library_ids, skill_texts, library_payload = _library_payload(Path(str(config["accum18_library_path"])))
    if library_ids != manifest.get("library_skill_ids"):
        raise OperatorError("the local Accum-18 skill ID set differs from the frozen manifest")
    skills = tuple(dict(item) for item in library_payload.get("libraries", library_payload)["Accum-18"]["skills"])
    libraries: dict[str, dict[str, Any]] = {"D0": {"skills": []}}
    for condition in ("D18", "Rel", "Irr", "Fill", "Gate"):
        libraries[condition] = {"skills": list(skills)}
    # Relevance-derived skill IDs are decision metadata; original ratings remain in external files.
    for task_id, pair in manifest["selected_skill_ids"].items():
        if any(skill_id not in library_ids for skill_id in pair.values()):
            raise OperatorError(f"frozen manifest selected a skill outside Accum-18: {task_id}")
    machine = machine_report(output_dir=run_root, gpu_count=4, min_gpu_free_mib=int(config.get("min_gpu_free_mib", 16_384)),
                             ollama_base_port=int(config.get("ollama_base_port", 11_534)))
    if machine["status"] != "PASS":
        raise OperatorError("hardware, storage, or endpoint ports changed after preflight")
    devices = [GPUDevice(str(item["uuid"]), int(item["index"]), str(item["name"]),
                         int(item["memory_total_mib"]), int(item["memory_free_mib"]))
               for item in machine["devices"]]
    specs = worker_specs(
        devices,
        count=int(config.get("worker_count", 4)), base_port=int(config.get("ollama_base_port", 11_534)), root=run_root / "workers",
    )
    tokenizer_path = Path(str(config["tokenizer_path"]))
    filler_path = Path(str(config["filler_path"]))
    filler_bytes = filler_path.read_bytes()
    if hashlib.sha256(filler_bytes).hexdigest() != manifest["inputs"]["filler_sha256"]:
        raise OperatorError("neutral filler changed after static input preflight")
    try:
        filler_unit = filler_bytes.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise OperatorError("approved neutral filler is not UTF-8") from exc
    if not filler_unit:
        raise OperatorError("approved neutral filler is empty")
    # Reserve and validate the run identity before starting services or the
    # separately approved four-request placement smoke test.
    run_root.mkdir(parents=True, exist_ok=True)
    launch_lock = RunLock(run_root / "RUN.lock", resume=resume)
    launch_lock.__enter__()
    try:
        store.initialize()
    except BaseException:
        launch_lock.__exit__(None, None, None)
        raise
    model_dir = Path(str(config["ollama_models_dir"]))
    servers = [OllamaServer(spec, model_dir, binary=str(config.get("ollama_binary", "ollama"))) for spec in specs]
    clients: list[WorkerProcessClient] = []
    launch_id = uuid4().hex
    previous_unseen_access = os.environ.get("RQ1_VALID_UNSEEN_ACCESS")
    changed_unseen_access = False
    try:
        for server in servers:
            server.start()
        probes = [probe_endpoint(spec.endpoint) for spec in specs]
        if any(item.get("status") != "PASS" for item in probes):
            raise OperatorError("isolated Ollama process failed frozen model digest/quantization check")
        # Verify the PID-to-GPU map using model-loaded compute processes. This minimal request
        # is permitted only under the same explicit approval as the full launch.
        smoke = smoke_test_endpoints([spec.endpoint for spec in specs], approval=approval)
        if smoke["status"] != "PASS":
            raise OperatorError("per-worker model smoke test failed; no scientific episode was started")
        placement = capture_placement({server.spec.worker_id: int(server.process.pid) for server in servers if server.process is not None})
        expected_placement = {spec.worker_id: spec.gpu_uuid for spec in specs}
        if not placement.get("passed") or placement.get("assignments") != expected_placement:
            raise OperatorError("could not verify one distinct physical GPU per Ollama worker")
        atomic_write_json(run_root / "launches" / f"{launch_id}.json", {
            "schema": "rq1-university-launch-evidence-v1", "launch_id": launch_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest["manifest_sha256"],
            "approval": dict(approval), "preflight": checked, "model_probes": probes,
            "smoke_test": smoke, "gpu_placement": placement,
        }, overwrite=False)
        config_dir = run_root / "worker-configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        if manifest["stage"] == "confirmatory":
            os.environ["RQ1_VALID_UNSEEN_ACCESS"] = "scientific-evaluation"
            changed_unseen_access = True
        for spec in specs:
            worker_config = {
                "worker_id": spec.worker_id, "gpu_uuid": spec.gpu_uuid, "ollama_endpoint": spec.endpoint,
                "worker_output_dir": str(spec.worker_dir), "run_id": manifest["run_id"], "stage": manifest["stage"],
                "split": "valid_seen" if manifest["stage"] == "pilot" else "valid_unseen",
                "repository_root": str(root.resolve()), "alfworld_data_dir": str(config["alfworld_data_dir"]),
                "sbert_cache": str(config["sbert_cache"]), "sbert_model": SBERT_MODEL, "sbert_revision": SBERT_REVISION,
                "sbert_snapshot_sha256": manifest["retrieval"]["snapshot_sha256"],
                "model_tag": MODEL_TAG, "model_digest": MODEL_DIGEST,
                "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": manifest["inputs"]["tokenizer_sha256"],
                "tokenizer_revision": str(manifest["tokenizer"]["revision"]),
                "filler_unit": filler_unit, "gate_threshold": manifest["gate"]["threshold"],
                "tasks": list(tasks_by_id.values()), "libraries": libraries, "skill_texts": skill_texts,
                "selected_skill_ids": manifest["selected_skill_ids"],
                "provenance": {"manifest_sha256": manifest["manifest_sha256"], "protocol_version": manifest["protocol_version"],
                               "source_commit": manifest["source_commit"], "tokenizer_sha256": manifest["inputs"]["tokenizer_sha256"],
                               "tokenizer_revision": manifest["tokenizer"]["revision"],
                               "sbert_snapshot_sha256": manifest["retrieval"]["snapshot_sha256"],
                               "alfworld_data_sha256": manifest["inputs"]["alfworld_data_sha256"],
                               "filler_sha256": manifest["inputs"]["filler_sha256"],
                               "placement": {"passed": placement["passed"], "assignments": placement["assignments"]}},
            }
            config_path = config_dir / f"{spec.worker_id}.json"
            if config_path.exists():
                if load_json(config_path) != worker_config:
                    raise OperatorError(f"existing worker config differs from the frozen launch: {config_path}")
            else:
                atomic_write_json(config_path, worker_config, overwrite=False)
            client = WorkerProcessClient(spec.worker_id, config_path, cwd=root, gpu_uuid=spec.gpu_uuid)
            client.start()
            clients.append(client)
        coordinator = Coordinator(store, clients)
        outcome = coordinator.run(resume=resume, retry_failed=retry_failed, cutoff_epoch=cutoff_epoch,
                                  max_attempts=max_attempts, lock_held=True)
        return {"preflight": checked, "smoke_test": smoke, "gpu_placement": placement, "run": outcome}
    except BaseException as exc:
        try:
            atomic_write_json(run_root / "launches" / f"{launch_id}-failure.json", {
                "schema": "rq1-university-launch-failure-v1", "launch_id": launch_id,
                "recorded_at": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__, "error": str(exc),
            }, overwrite=False)
        except OSError:
            pass
        raise
    finally:
        if changed_unseen_access:
            if previous_unseen_access is None:
                os.environ.pop("RQ1_VALID_UNSEEN_ACCESS", None)
            else:
                os.environ["RQ1_VALID_UNSEEN_ACCESS"] = previous_unseen_access
        try:
            for client in reversed(clients):
                client.stop()
            for server in reversed(servers):
                server.stop()
        finally:
            launch_lock.__exit__(None, None, None)


def status(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    store = RunStore(run_root, manifest)
    if not run_root.exists():
        return {"run_id": manifest.get("run_id"), "status": "not_started", "planned_units": len(manifest.get("units", [])), "completed_units": 0}
    completed = store.completed()
    events = store.event_rows()
    attempts = sum(row.get("event") == "attempt_started" for row in events)
    failures = sum(row.get("event") == "attempt_failed" for row in events)
    stop_requested = (run_root / "STOP").exists()
    if stop_requested:
        run_status = "stop_requested"
    elif len(completed) == len(manifest.get("units", [])):
        run_status = "completed"
    elif (run_root / "RUN.lock").exists():
        run_status = "active_or_stale_lock"
    elif events:
        run_status = "paused_or_incomplete"
    else:
        run_status = "not_started"
    return {"run_id": manifest.get("run_id"), "status": run_status,
            "planned_units": len(manifest.get("units", [])), "completed_units": len(completed), "attempts_started": attempts,
            "failed_attempts": failures, "stop_requested": stop_requested}


def export_run(run_root: Path, manifest_path: Path, destination: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    manifest_path = manifest_path.resolve()
    destination = destination.resolve()
    manifest = load_json(manifest_path)
    store = RunStore(run_root, manifest)
    coordinator = Coordinator(store, [type("OfflineWorker", (), {"worker_id": "export-only", "execute": lambda *args: {}})()])
    validation = coordinator.validate()
    if not validation["passed"]:
        raise OperatorError("cannot export an invalid or incomplete run")
    merged_path = run_root / "merged-results.jsonl"
    if not merged_path.is_file():
        raise OperatorError("run must be explicitly merged before export")
    coordinator.merge()  # Verify the existing merge hash against authoritative unit results.
    if destination.exists():
        raise OperatorError(f"export destination already exists: {destination}")
    try:
        destination.relative_to(run_root)
    except ValueError:
        pass
    else:
        raise OperatorError("export destination cannot be inside the run directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.exporting-{os.getpid()}")
    if temporary.exists():
        raise OperatorError(f"stale export staging directory exists and needs review: {temporary}")
    temporary.mkdir()
    try:
        files = []
        sources = (manifest_path, merged_path, *sorted((run_root / "events").glob("*.json")),
                   *sorted((run_root / "results" / "units").glob("*.json")), *sorted((run_root / "attempts").rglob("*")))
        for source in sources:
            if not source.is_file():
                continue
            relative = Path("run") / source.relative_to(run_root) if source.is_relative_to(run_root) else Path("run") / "manifest.json"
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files.append({"path": relative.as_posix(), "sha256": sha256_file(target), "bytes": target.stat().st_size})
        report = {"schema": "rq1-university-gpu-export-v1", "run_id": manifest["run_id"], "manifest_sha256": manifest["manifest_sha256"],
                  "created_at": datetime.now(timezone.utc).isoformat(), "files": files, "files_sha256": content_hash(files)}
        atomic_write_json(temporary / "EXPORT-MANIFEST.json", report, overwrite=False)
        temporary.replace(destination)
        return {"status": "PASS", "destination": str(destination), "files": len(files), "manifest_sha256": report["files_sha256"]}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

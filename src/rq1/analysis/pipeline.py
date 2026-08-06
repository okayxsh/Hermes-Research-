"""Offline, fail-closed analysis of immutable controlled-recovery evidence.

This module deliberately has no client imports: analysis is a reader of saved
evidence, never an executor of models, Hermes, or ALFWorld.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rq1.evaluation.activation import ActivationManifest, EvidenceReference, validate_activation
from rq1.utils.hashing import sha256_file


SCHEMA_VERSION = 1
EXCLUSION_RULE_VERSION = "controlled-recovery-exclusions-v1"
BOOTSTRAP_REPLICATES = 2000


class AnalysisInputError(RuntimeError):
    """Raised when final evidence cannot safely support analysis."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AnalysisInputError(f"invalid JSON evidence: {path}: {type(exc).__name__}") from exc
    if not isinstance(result, dict):
        raise AnalysisInputError(f"JSON evidence must be an object: {path}")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnalysisInputError(f"missing log: {path}") from exc
    values: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise AnalysisInputError(f"invalid JSONL at {path}:{number}") from exc
        if not isinstance(value, dict):
            raise AnalysisInputError(f"non-object JSONL record at {path}:{number}")
        values.append(value)
    return values


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


@dataclass(frozen=True)
class Exclusion:
    run_id: str
    attempt_id: str
    category: str
    reason: str
    scientific_failure: bool
    protocol_invalidation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalEvent:
    run_id: str
    attempt_id: str
    snapshot_id: str
    task_family: str
    event_index: int
    skill_id: str | None
    phase: str
    label: str
    rule: str | None
    observable: bool
    timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisRecord:
    run_id: str
    attempt_id: str
    task_id: str
    task_family: str
    checkpoint_digest: str
    perturbation_digest: str
    recovery_context_digest: str
    repetition: int
    seed: int
    snapshot_id: str
    snapshot_hash: str
    skill_count: int
    cumulative_skill_operations: int
    perturbation_type: str
    recovery_success: bool
    post_failure_budget_complete: bool
    relevant_skill_available: bool | None
    post_failure_actions: int | None
    post_failure_model_calls: int | None
    post_failure_tool_calls: int | None
    invalid_post_failure_actions: int | None
    recovery_latency_ms: float | None
    time_to_first_useful_action_ms: float | None
    time_to_first_relevant_load_ms: float | None
    redundant_action_count: int | None
    phase_boundary_valid: bool
    source_result_path: str

    @property
    def paired_unit(self) -> tuple[str, str, str, str, int, int]:
        return (self.task_id, self.checkpoint_digest, self.perturbation_digest,
                self.recovery_context_digest, self.repetition, self.seed)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["paired_unit"] = "|".join(map(str, self.paired_unit))
        return result


@dataclass(frozen=True)
class ValidationResult:
    report_path: Path
    evaluation_run_id: str
    activation_path: Path
    input_hashes: dict[str, str]
    records: tuple[AnalysisRecord, ...]
    retrieval_events: tuple[RetrievalEvent, ...]
    exclusions: tuple[Exclusion, ...]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def _activation(path: Path) -> ActivationManifest:
    payload = _read_json(path)
    try:
        payload["evidence"] = tuple(EvidenceReference(**item) for item in payload["evidence"])
        return ActivationManifest(**payload)
    except (KeyError, TypeError) as exc:
        raise AnalysisInputError("invalid activation manifest") from exc


def _load_rules(path: Path) -> tuple[dict[str, Any], str]:
    # The frozen file is JSON-compatible YAML in this repository.  A tiny
    # fallback accepts the declared field list without importing a YAML parser.
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except ValueError:
        value = {"raw_rules": text}
    if not isinstance(value, dict):
        raise AnalysisInputError("relevance rules must be an object")
    return value, sha256_file(path)


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else event


def _label_event(event: dict[str, Any], rules: dict[str, Any]) -> tuple[str, str | None, bool]:
    relevance = event.get("relevance", _event_payload(event).get("relevance"))
    if relevance in {"relevant", "irrelevant"}:
        return str(relevance), "observed_event_relevance", True
    if relevance in {"ambiguous", "unknown"}:
        return "ambiguous", "observed_event_relevance", True
    if rules.get("status") == "TO_BE_FROZEN_BEFORE_UNSEEN_EVALUATION":
        return "unavailable", "rules_not_frozen", False
    return "unavailable", "no_auditable_relevance", False


def _is_load(event: dict[str, Any]) -> bool:
    return event.get("event") in {"skill_loaded", "skill_view"}


def _valid_boundary(events: Iterable[dict[str, Any]]) -> bool:
    values = list(events)
    return any(item.get("phase") == "post_failure" for item in values)


def _result_record(item: dict[str, Any], root: Path) -> tuple[AnalysisRecord | None, list[Exclusion], dict[str, Any]]:
    run_id = str(item.get("run_id", "")); attempt_id = str(item.get("attempt_id", ""))
    result_path = Path(str(item.get("result_path", "")))
    if not result_path.is_absolute(): result_path = root / result_path
    if not result_path.is_file():
        return None, [Exclusion(run_id, attempt_id, "infrastructure_failure", "missing_result_log", False)], {}
    payload = _read_json(result_path)
    excluded = payload.get("exclusion_reason") or item.get("exclusion_reason")
    if excluded:
        return None, [Exclusion(run_id, attempt_id, "protocol_invalidation", str(excluded), False, True)], payload
    if payload.get("status") in {"timeout", "interrupted", "abandoned"}:
        return None, [Exclusion(run_id, attempt_id, "infrastructure_failure", str(payload.get("status")), False)], payload
    if payload.get("replay_valid") is False or payload.get("perturbation_valid") is False or payload.get("solvable") is False:
        return None, [Exclusion(run_id, attempt_id, "scientific_task_failure", "invalid_recovery_state", True)], payload
    if payload.get("reconciled") is False or payload.get("profile_read_only") is False or payload.get("skill_writes"):
        return None, [Exclusion(run_id, attempt_id, "protocol_invalidation", "contaminated_or_unreconciled", False, True)], payload
    try:
        record = AnalysisRecord(
            run_id=run_id, attempt_id=attempt_id, task_id=str(item["task_id"]), task_family=str(item["task_family"]),
            checkpoint_digest=str(item["checkpoint_digest"]), perturbation_digest=str(item["perturbation_digest"]),
            recovery_context_digest=str(item["recovery_context_digest"]), repetition=int(item["repetition"]), seed=int(item["seed"]),
            snapshot_id=str(item["snapshot_id"]), snapshot_hash=str(item["snapshot_hash"]),
            skill_count=int(item["skill_count"]), cumulative_skill_operations=int(item.get("cumulative_skill_operations", item["skill_count"])),
            perturbation_type=str(item.get("perturbation_type", "approved")), recovery_success=bool(payload["recovery_success"]),
            post_failure_budget_complete=bool(payload.get("post_failure_budget_complete", True)),
            relevant_skill_available=payload.get("relevant_skill_available"),
            post_failure_actions=payload.get("post_failure_actions"), post_failure_model_calls=payload.get("post_failure_model_calls"),
            post_failure_tool_calls=payload.get("post_failure_tool_calls"), invalid_post_failure_actions=payload.get("invalid_post_failure_actions"),
            recovery_latency_ms=payload.get("recovery_latency_ms"), time_to_first_useful_action_ms=payload.get("time_to_first_useful_action_ms"),
            time_to_first_relevant_load_ms=payload.get("time_to_first_relevant_load_ms"), redundant_action_count=payload.get("redundant_action_count"),
            phase_boundary_valid=bool(payload.get("phase_boundary_valid", False)), source_result_path=_relative(root, result_path),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, [Exclusion(run_id, attempt_id, "protocol_invalidation", f"malformed_result:{type(exc).__name__}", False, True)], payload
    if not record.phase_boundary_valid:
        return None, [Exclusion(run_id, attempt_id, "protocol_invalidation", "invalid_phase_boundary", False, True)], payload
    return record, [], payload


def validate_inputs(root: Path, evaluation_run: str) -> ValidationResult:
    """Validate only persisted evidence; this intentionally has no runtime clients."""
    report_path = root / "artifacts" / "evaluation_reports" / evaluation_run / "evaluation-report.json"
    report = _read_json(report_path)
    errors: list[str] = []; exclusions: list[Exclusion] = []; records: list[AnalysisRecord] = []; retrieval: list[RetrievalEvent] = []
    if report.get("mode") != "real" or report.get("valid") is not True or report.get("status") not in {"validated", "passed"}:
        errors.append("evaluation report is not validated real evidence")
    if not isinstance(report.get("configuration_hashes"), dict) or not report["configuration_hashes"]:
        errors.append("evaluation report lacks frozen configuration hashes")
    if not isinstance(report.get("snapshot_hashes"), dict) or not report["snapshot_hashes"]:
        errors.append("evaluation report lacks frozen snapshot hashes")
    if report.get("queue_sha256") in {None, ""} or report.get("reconciliation_valid") is not True:
        errors.append("evaluation queue or reconciliation evidence is not validated")
    activation_path = Path(str(report.get("activation_manifest_path", "")))
    if not activation_path.is_absolute(): activation_path = root / activation_path
    try:
        activation_errors = validate_activation(root, _activation(activation_path))
        if activation_errors: errors.extend("activation: " + value for value in activation_errors)
    except (AnalysisInputError, OSError) as exc:
        errors.append(f"activation: {exc}")
    rules_path = Path(str(report.get("relevance_rules_path", "")))
    if not rules_path.is_absolute(): rules_path = root / rules_path
    try: rules, rules_hash = _load_rules(rules_path)
    except (OSError, AnalysisInputError) as exc: rules, rules_hash = {}, ""; errors.append(f"relevance rules: {exc}")
    expected = report.get("expected_pairs")
    if not isinstance(expected, list): errors.append("evaluation report lacks expected_pairs")
    seen_pairs: dict[tuple[str, str, str, str, int, int], set[str]] = defaultdict(set)
    hashes: dict[str, str] = {"evaluation_report": sha256_file(report_path)}
    if activation_path.is_file(): hashes["activation_manifest"] = sha256_file(activation_path)
    if rules_hash: hashes["relevance_rules"] = rules_hash
    for item in report.get("attempts", []):
        if not isinstance(item, dict): errors.append("non-object attempt entry"); continue
        try:
            record, found, _payload = _result_record(item, root)
        except AnalysisInputError as exc:
            errors.append(str(exc)); continue
        exclusions.extend(found)
        if record is None: continue
        pair = record.paired_unit
        if record.snapshot_id in seen_pairs[pair]:
            exclusions.append(Exclusion(record.run_id, record.attempt_id, "protocol_invalidation", "duplicate_attempt", False, True)); continue
        seen_pairs[pair].add(record.snapshot_id); records.append(record)
        if report.get("snapshot_hashes", {}).get(record.snapshot_id) != record.snapshot_hash:
            errors.append(f"snapshot hash drift for {record.snapshot_id}")
        log_path = Path(str(item.get("hermes_log_path", "")))
        if not log_path.is_absolute(): log_path = root / log_path
        try:
            events = _read_jsonl(log_path)
        except AnalysisInputError as exc:
            errors.append(str(exc)); continue
        if not _valid_boundary(events):
            errors.append(f"no post_failure phase boundary for {record.run_id}"); continue
        hashes[f"log:{record.run_id}:{record.attempt_id}"] = sha256_file(log_path)
        for index, event in enumerate(events):
            if event.get("phase") != "post_failure" or not _is_load(event): continue
            label, rule, observable = _label_event(event, rules)
            payload = _event_payload(event)
            retrieval.append(RetrievalEvent(record.run_id, record.attempt_id, record.snapshot_id, record.task_family, index,
                str(payload.get("skill_id")) if payload.get("skill_id") is not None else None, "post_failure", label, rule, observable, event.get("timestamp")))
    if isinstance(expected, list):
        for raw in expected:
            if not isinstance(raw, dict): errors.append("non-object expected pair"); continue
            try:
                pair = (str(raw["task_id"]), str(raw["checkpoint_digest"]), str(raw["perturbation_digest"]), str(raw["recovery_context_digest"]), int(raw["repetition"]), int(raw["seed"]))
                needed = set(raw["snapshots"])
            except (KeyError, TypeError, ValueError): errors.append("malformed expected pair"); continue
            missing = needed - seen_pairs.get(pair, set())
            if missing:
                errors.append("missing paired run: " + "|".join(map(str, pair)) + " snapshots=" + ",".join(sorted(missing)))
    return ValidationResult(report_path, evaluation_run, activation_path, hashes, tuple(records), tuple(retrieval), tuple(exclusions), tuple(errors))


def _rate(values: Iterable[bool]) -> float | None:
    entries = list(values)
    return None if not entries else sum(entries) / len(entries)


def _mean(values: Iterable[int | float | None]) -> float | None:
    entries = [float(value) for value in values if value is not None]
    return None if not entries else statistics.fmean(entries)


def _snapshot_metrics(records: list[AnalysisRecord], events: list[RetrievalEvent]) -> list[dict[str, Any]]:
    by_snapshot: dict[str, list[AnalysisRecord]] = defaultdict(list)
    for record in records: by_snapshot[record.snapshot_id].append(record)
    event_map: dict[tuple[str, str], list[RetrievalEvent]] = defaultdict(list)
    for event in events: event_map[(event.run_id, event.attempt_id)].append(event)
    rows=[]
    for snapshot, values in sorted(by_snapshot.items(), key=lambda item: (min(x.skill_count for x in item[1]), item[0])):
        eligible=[item for item in values if item.post_failure_budget_complete]
        loads=[event for item in values for event in event_map[(item.run_id,item.attempt_id)] if event.label in {"relevant","irrelevant"}]
        all_loads=[event for item in values for event in event_map[(item.run_id,item.attempt_id)]]
        relevant_available=[item for item in values if item.relevant_skill_available is True]
        no_retrieval=sum(not event_map[(item.run_id,item.attempt_id)] for item in values)
        invalid_total=sum(item.invalid_post_failure_actions or 0 for item in values if item.post_failure_actions is not None)
        action_total=sum(item.post_failure_actions or 0 for item in values if item.post_failure_actions is not None)
        rows.append({"snapshot_id":snapshot,"skill_count":min(item.skill_count for item in values),"cumulative_skill_operations":min(item.cumulative_skill_operations for item in values),
            "episode_count":len(values),"conditional_recovery_rate":_rate(item.recovery_success for item in eligible),"conditional_recovery_denominator":len(eligible),
            "retrieval_noise_rate":_rate(event.label=="irrelevant" for event in loads),"retrieval_load_count":len(loads),"unauditable_load_count":sum(event.label not in {"relevant","irrelevant"} for event in all_loads),
            "no_retrieval_rate":no_retrieval/len(values) if values else None,"relevant_skill_hit_rate":_rate(any(event.label=="relevant" for event in event_map[(item.run_id,item.attempt_id)]) for item in relevant_available),"relevant_skill_available_count":len(relevant_available),
            "recovery_actions":_mean(item.post_failure_actions for item in values),"recovery_latency_ms":_mean(item.recovery_latency_ms for item in values),"invalid_action_rate":(invalid_total/action_total if action_total else None)})
    return rows


def _paired(records: list[AnalysisRecord], events: list[RetrievalEvent]) -> list[dict[str, Any]]:
    event_map: dict[tuple[str,str], list[RetrievalEvent]]=defaultdict(list)
    for event in events: event_map[(event.run_id,event.attempt_id)].append(event)
    groups: dict[tuple[str,str,str,str,int,int], list[AnalysisRecord]]=defaultdict(list)
    for record in records: groups[record.paired_unit].append(record)
    snapshots=sorted({record.snapshot_id for record in records}, key=lambda value: min(item.skill_count for item in records if item.snapshot_id==value))
    comparisons=[]
    for pair, values in sorted(groups.items()):
        values=sorted(values,key=lambda item:(item.skill_count,item.snapshot_id))
        pairs=[]
        if len(values)>1: pairs.append((values[0],values[-1],"L0_vs_largest"))
        if len(values) > 2:
            pairs += [(left,right,"adjacent") for left,right in zip(values,values[1:])]
        for left,right,kind in pairs:
            def noise(item: AnalysisRecord) -> float | None:
                loads=[event for event in event_map[(item.run_id,item.attempt_id)] if event.label in {"relevant","irrelevant"}]
                return _rate(event.label=="irrelevant" for event in loads)
            comparisons.append({"paired_unit":"|".join(map(str,pair)),"comparison":kind,"left_snapshot":left.snapshot_id,"right_snapshot":right.snapshot_id,
                "recovery_rate_difference":int(right.recovery_success)-int(left.recovery_success),"recovery_rate_relative_difference":None if not left.recovery_success else (int(right.recovery_success)-1),
                "retrieval_noise_difference":None if noise(left) is None or noise(right) is None else noise(right)-noise(left),
                "recovery_action_difference":None if left.post_failure_actions is None or right.post_failure_actions is None else right.post_failure_actions-left.post_failure_actions,
                "latency_difference_ms":None if left.recovery_latency_ms is None or right.recovery_latency_ms is None else right.recovery_latency_ms-left.recovery_latency_ms,
                "relevant_hit_difference":int(any(e.label=="relevant" for e in event_map[(right.run_id,right.attempt_id)]))-int(any(e.label=="relevant" for e in event_map[(left.run_id,left.attempt_id)]))})
    return comparisons


def _bootstrap(records: list[AnalysisRecord], seed: int) -> dict[str, Any]:
    groups: dict[tuple[str,str,str,str,int,int], list[AnalysisRecord]]=defaultdict(list)
    for record in records: groups[record.paired_unit].append(record)
    clusters=list(groups.values())
    if not clusters: return {"method":"cluster_bootstrap_percentile","replicates":BOOTSTRAP_REPLICATES,"seed":seed,"intervals":{}}
    rng=random.Random(seed); samples=[]
    for _ in range(BOOTSTRAP_REPLICATES):
        selected=[item for _ in clusters for item in rng.choice(clusters)]
        eligible=[item for item in selected if item.post_failure_budget_complete]
        samples.append(_rate(item.recovery_success for item in eligible))
    ordered=sorted(value for value in samples if value is not None)
    return {"method":"cluster_bootstrap_percentile","replicates":BOOTSTRAP_REPLICATES,"seed":seed,"intervals":{"conditional_recovery_rate":{"lower":ordered[int(.025*(len(ordered)-1))],"upper":ordered[int(.975*(len(ordered)-1))],"paired_unit_count":len(clusters)}}}


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1]); result=[0.0]*len(values); index=0
    while index < len(ordered):
        end=index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]: end += 1
        rank=(index + end + 2) / 2
        for position in range(index,end+1): result[ordered[position][0]]=rank
        index=end+1
    return result


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2: return None
    xs, ys=zip(*pairs); rx,ry=_rank(list(xs)),_rank(list(ys)); mean_x=statistics.fmean(rx); mean_y=statistics.fmean(ry)
    numerator=sum((x-mean_x)*(y-mean_y) for x,y in zip(rx,ry)); denominator=(sum((x-mean_x)**2 for x in rx)*sum((y-mean_y)**2 for y in ry))**.5
    return None if denominator == 0 else numerator/denominator


def _associations(records: list[AnalysisRecord], events: list[RetrievalEvent]) -> dict[str, Any]:
    event_map: dict[tuple[str,str],list[RetrievalEvent]]=defaultdict(list)
    for event in events: event_map[(event.run_id,event.attempt_id)].append(event)
    rows=[]
    for record in records:
        loads=[event for event in event_map[(record.run_id,record.attempt_id)] if event.label in {"relevant","irrelevant"}]
        noise=_rate(event.label=="irrelevant" for event in loads)
        if noise is not None: rows.append((noise,record))
    def measure(getter): return _spearman([(noise,float(value)) for noise,record in rows if (value:=getter(record)) is not None])
    return {"method":"spearman_descriptive_clustered_observations","observation_count":len(rows),"causal_claim":False,
            "noise_with_recovery_success":measure(lambda item:int(item.recovery_success)),"noise_with_recovery_actions":measure(lambda item:item.post_failure_actions),"noise_with_recovery_latency_ms":measure(lambda item:item.recovery_latency_ms),"noise_with_invalid_actions":measure(lambda item:item.invalid_post_failure_actions)}


def compute_metrics(validation: ValidationResult, seed: int = 20260806) -> dict[str, Any]:
    if not validation.valid: raise AnalysisInputError("analysis inputs are invalid: " + "; ".join(validation.errors))
    records=list(validation.records); events=list(validation.retrieval_events)
    summary=_snapshot_metrics(records,events); paired=_paired(records,events)
    by_family=[]
    for family in sorted({item.task_family for item in records}):
        subset=[item for item in records if item.task_family==family]
        by_family.append({"task_family":family,"episode_count":len(subset),"conditional_recovery_rate":_rate(item.recovery_success for item in subset if item.post_failure_budget_complete)})
    best=max(summary,key=lambda item:(item["conditional_recovery_rate"] if item["conditional_recovery_rate"] is not None else -1,-item["skill_count"])) if summary else None
    final=summary[-1] if summary else None
    return {"schema_version":SCHEMA_VERSION,"evaluation_run_id":validation.evaluation_run_id,"analysis_kind":"controlled_recovery_post_failure","simulated":False,
        "sample_counts":{"episodes":len(records),"paired_units":len({item.paired_unit for item in records}),"excluded":len(validation.exclusions)},"snapshot_summary":summary,"task_family_summary":by_family,"paired_comparisons":paired,
        "uncertainty":_bootstrap(records,seed),"associations":_associations(records,events),"growth":{"best_observed_snapshot":best["snapshot_id"] if best else None,"final_minus_best_degradation":None if not best or not final or best["conditional_recovery_rate"] is None or final["conditional_recovery_rate"] is None else final["conditional_recovery_rate"]-best["conditional_recovery_rate"]},
        "claims":{"association_is_causal":False,"manual_relevance_agreement_available":False}}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values=list(rows); path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({key for item in values for key in item})
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for item in values: writer.writerow({key: json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value for key,value in item.items()})


def output_directory(root: Path, validation: ValidationResult) -> Path:
    fingerprint=_sha_value({"inputs":validation.input_hashes,"records":[item.to_dict() for item in validation.records]})[:16]
    return root/"results"/"analysis"/validation.evaluation_run_id/fingerprint


def write_analysis(root: Path, validation: ValidationResult, metrics: dict[str, Any], seed: int = 20260806) -> Path:
    output=output_directory(root,validation); output.mkdir(parents=True,exist_ok=True)
    _write_csv(output/"validated_runs.csv",[item.to_dict() for item in validation.records]); _write_csv(output/"exclusions.csv",[item.to_dict() for item in validation.exclusions]); _write_csv(output/"snapshot_summary.csv",metrics["snapshot_summary"]); _write_csv(output/"paired_comparisons.csv",metrics["paired_comparisons"]); _write_csv(output/"retrieval_events.csv",[item.to_dict() for item in validation.retrieval_events])
    audit=sorted((item.to_dict() for item in validation.retrieval_events),key=lambda item:_sha_value({"seed":seed,"event":item}))[:min(50,len(validation.retrieval_events))]
    for row in audit: row.update({"manual_label":"","manual_reviewer":"","manual_notes":""})
    _write_csv(output/"relevance_audit_sample.csv",audit)
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    summary="# Controlled-recovery analysis\n\nThis output is descriptive, paired, and post-failure-only. It does not support causal claims or human-label agreement.\n\n"+json.dumps(metrics["sample_counts"],indent=2,sort_keys=True)+"\n"
    (output/"analysis_summary.md").write_text(summary,encoding="utf-8")
    generated={path.name:sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    manifest={"schema_version":SCHEMA_VERSION,"evaluation_run_id":validation.evaluation_run_id,"repository_commit":_git_commit(root),"analysis_code_hash":_code_hash(root),"input_validation_hash":_sha_value({"errors":validation.errors,"hashes":validation.input_hashes}),"input_hashes":validation.input_hashes,"activation_manifest_hash":validation.input_hashes.get("activation_manifest"),"relevance_rule_hash":validation.input_hashes.get("relevance_rules"),"exclusion_rule_version":EXCLUSION_RULE_VERSION,"resampling_seed":seed,"python_version":sys.version.split()[0],"generated_artifacts":generated}
    (output/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return output


def _git_commit(root: Path) -> str | None:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): return None


def _code_hash(root: Path) -> str:
    directory=root/"src"/"rq1"/"analysis"; digest=hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()

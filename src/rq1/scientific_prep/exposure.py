"""Exposure ledger, template-overlap screening and candidate pools (preparation evidence only).

Exposure is reconstructed ONLY from explicitly declared evidence sources (sources config, schema
``rq1-sci-prep-exposure-sources-v1``).  Each source is hashed; a missing/altered *required* source blocks the run.

Three DIFFERENT relations are kept apart and never merged:
  * model-outcome exposure  -- a declared source records the task itself in a model episode / preparation step.
    Classes (most severe wins): EXPOSED > POSSIBLY_EXPOSED > UNKNOWN > NO_MODEL_OUTCOME_FOUND.
    ``NO_MODEL_OUTCOME_FOUND`` means "no declared source shows a model outcome" -- never "unexposed".
  * exact task reuse        -- the same ALFRED trial identifier appears under another task ID / split.
  * procedural-template overlap -- the task shares a template (under a NAMED definition) with referenced tasks.
    Overlap is not contamination by itself; it only matters through an explicitly named screening policy.
``INELIGIBLE`` (not installed/runnable) is the screening class of non-runnable entries; oracle eligibility is a
different, later artifact (oracle.py) and is never inferred here.

Missing evidence never becomes independence: an unavailable or ``status: unresolved`` source turns its scope splits
UNKNOWN (fail closed).  Only the explicit ``allow_provisional`` mode keeps such tasks screenable, and then every
affected row and every output is marked provisional with the gap recorded; provisional screens can never support a
scientific roster approval.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq1.scientific_prep.identity import (TEMPLATE_DEFINITIONS, TaskIdentity, TaskIdentityError, normalize_task_reference,
                                          parse_task_id, template_key)
SOURCES_SCHEMA = "rq1-sci-prep-exposure-sources-v1"
CLASSES = ("EXPOSED", "POSSIBLY_EXPOSED", "UNKNOWN", "NO_MODEL_OUTCOME_FOUND")
SEVERITY = {name: rank for rank, name in enumerate(reversed(CLASSES))}
SOURCE_KINDS = ("task_manifest", "controlled_failures", "task_list", "csv_task_column", "inline_task_ids",
                "json_task_fields", "template_list")
_CLASSED_KINDS = ("task_list", "csv_task_column", "inline_task_ids", "controlled_failures", "json_task_fields")
EXPOSURE_KINDS = ("model_outcome_exposure", "exact_task_reuse", "procedural_template_overlap", "unknown_evidence")


class ExposureConfigError(ValueError):
    pass


def _get(obj: Any, pointer: str) -> Any:
    for part in [p for p in pointer.split("/") if p]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def load_sources(config_path: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_sources(config)
    return config


def validate_sources(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SOURCES_SCHEMA:
        raise ExposureConfigError(f"sources config must declare schema {SOURCES_SCHEMA}")
    if config.get("template_definition", "goal_template") not in TEMPLATE_DEFINITIONS:
        raise ExposureConfigError(f"template_definition must be one of {sorted(TEMPLATE_DEFINITIONS)}")
    ids = [s.get("id") for s in config.get("sources", [])]
    if not ids or len(ids) != len(set(ids)) or not all(ids):
        raise ExposureConfigError("sources need unique non-empty ids")
    for s in config["sources"]:
        if s.get("kind") not in SOURCE_KINDS:
            raise ExposureConfigError(f"source {s['id']}: kind must be one of {SOURCE_KINDS}")
        for key in ("role", "scope_splits"):
            if not s.get(key):
                raise ExposureConfigError(f"source {s['id']}: {key} is required")
        if s.get("status", "resolved") not in ("resolved", "unresolved"):
            raise ExposureConfigError(f"source {s['id']}: status must be 'resolved' or 'unresolved'")
        if s.get("status") == "unresolved":
            if not s.get("unresolved_reason"):
                raise ExposureConfigError(f"source {s['id']}: an unresolved source needs unresolved_reason")
            continue                                   # nothing else can be checked: the evidence is not available
        if s["kind"] not in ("inline_task_ids", "template_list") and not s.get("path"):
            raise ExposureConfigError(f"source {s['id']}: path is required")
        if s["kind"] in _CLASSED_KINDS and s.get("class") not in CLASSES[:2]:
            raise ExposureConfigError(f"source {s['id']}: class must be EXPOSED or POSSIBLY_EXPOSED")
        if s["kind"] in ("inline_task_ids", "template_list") and not s.get("provenance"):
            raise ExposureConfigError(f"source {s['id']}: inline evidence needs a provenance note")
        if s["kind"] == "json_task_fields" and not (isinstance(s.get("pointers"), list) and s["pointers"]):
            raise ExposureConfigError(f"source {s['id']}: json_task_fields needs a non-empty pointers list")
        if s["kind"] == "template_list":
            if s.get("template_definition") not in TEMPLATE_DEFINITIONS:
                raise ExposureConfigError(f"source {s['id']}: template_list needs template_definition in {sorted(TEMPLATE_DEFINITIONS)}")
            if not isinstance(s.get("templates"), list):
                raise ExposureConfigError(f"source {s['id']}: template_list needs a templates list")
        rules = [s.get("selected") or {}, s.get("replaced") or {}, *(s.get("exclusions") or {}).values()]
        for rule in rules:
            if rule.get("class") is not None and rule["class"] not in CLASSES[:3]:
                raise ExposureConfigError(f"source {s['id']}: rule class {rule['class']!r} must be EXPOSED, "
                                          "POSSIBLY_EXPOSED, UNKNOWN or null (INELIGIBLE comes only from the inventory)")
    names = [p.get("name") for p in config.get("screening_policies", [])]
    if len(names) != len(set(names)):
        raise ExposureConfigError("screening policy names must be unique")
    for p in config.get("screening_policies", []):
        if not p.get("name") or not isinstance(p.get("exclude_template_roles"), list):
            raise ExposureConfigError("each screening policy needs a name and exclude_template_roles")
        if p.get("template_definition", "goal_template") not in TEMPLATE_DEFINITIONS:
            raise ExposureConfigError(f"policy {p['name']}: unknown template_definition")
        if not isinstance(p.get("exclude_trial_reuse_roles", []), list):
            raise ExposureConfigError(f"policy {p['name']}: exclude_trial_reuse_roles must be a list")


def _resolve(path_text: str, base: Path) -> Path:
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else (base / p)


def _extract(source: Mapping[str, Any], base: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns (evidence items with raw references, source record)."""
    record: dict[str, Any] = {k: source.get(k) for k in ("id", "kind", "role", "required", "scope_splits", "description",
                                                         "expected_sha256", "provenance")}
    record["status"] = source.get("status", "resolved")
    items: list[dict[str, Any]] = []
    if record["status"] == "unresolved":
        record.update(available=False, sha256=None, path=None, unresolved_reason=source["unresolved_reason"])
        return items, record
    if source["kind"] == "template_list":
        record.update(available=True, sha256=None, path=None, template_definition=source["template_definition"],
                      templates=sorted(source["templates"]))
        return items, record
    if source["kind"] == "inline_task_ids":
        record.update(available=True, sha256=None, path=None)
        for ref in source.get("task_ids", []):
            items.append({"ref": ref, "class": source["class"], "role": source["role"],
                          "reason": source.get("reason", "explicitly supplied exposure")})
        return items, record
    path = _resolve(source["path"], base)
    record["path"] = str(path)
    if not path.is_file():
        record.update(available=False, sha256=None)
        return items, record
    raw = path.read_bytes()
    record.update(available=True, sha256=hashlib.sha256(raw).hexdigest(),
                  sha256_lf_normalized=hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
    if source["kind"] == "csv_task_column":
        reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))
        for row in reader:
            items.append({"ref": row.get(source.get("column", "task_id"), ""), "class": source["class"],
                          "role": source["role"], "reason": source.get("reason", "listed in source")})
        return items, record
    data = json.loads(path.read_text(encoding="utf-8"))
    if source["kind"] == "json_task_fields":
        for pointer in source["pointers"]:
            try:
                ref = _get(data, pointer)
            except (KeyError, IndexError, ValueError, TypeError):
                ref = ""
            items.append({"ref": ref if isinstance(ref, str) else "", "class": source["class"], "role": source["role"],
                          "reason": source.get("reason", f"task identity recorded at {pointer}")})
    elif source["kind"] == "task_list":
        entries = _get(data, source.get("json_pointer", "tasks"))
        field = source.get("id_field", "task_id")
        for entry in entries:
            ref = entry.get(field, "") if isinstance(entry, dict) else entry
            items.append({"ref": ref, "class": source["class"], "role": source["role"],
                          "reason": source.get("reason", "listed in source")})
    elif source["kind"] == "controlled_failures":
        for t in data.get("tasks", []):
            items.append({"ref": t.get("task_id", ""), "class": source["class"], "role": source["role"],
                          "reason": source.get("reason", "controlled-failure oracle prepared")})
        for family_items in (data.get("replaced_candidates") or {}).values():
            for t in family_items:
                rule = source.get("replaced", {})
                if rule.get("class"):
                    items.append({"ref": t.get("task_id", ""), "class": rule["class"], "role": rule.get("role", source["role"]),
                                  "reason": rule.get("reason", "rejected during controlled-failure preparation")})
    else:  # task_manifest (rq1.tasks.models.TaskManifest)
        selected = source.get("selected") or {}
        for t in data.get("tasks", []):
            items.append({"ref": t.get("task_id", ""), "class": selected.get("class", "EXPOSED"),
                          "role": selected.get("role", source["role"]), "reason": selected.get("reason", "selected task")})
        rules = source.get("exclusions") or {}
        for e in data.get("exclusions", []):
            rule = rules.get(e.get("reason", ""))
            ref = e.get("task_id") or e.get("source") or ""
            if rule is None:
                items.append({"ref": ref, "class": "UNKNOWN", "role": source["role"],
                              "reason": f"undeclared exclusion reason {e.get('reason')!r} in source"})
            elif rule.get("class"):
                items.append({"ref": ref, "class": rule["class"], "role": rule.get("role", source["role"]),
                              "reason": rule.get("reason", e.get("reason"))})
            elif rule.get("note"):
                items.append({"ref": ref, "class": None, "role": rule.get("role", source["role"]), "note": rule["note"],
                              "reason": e.get("reason")})
    return items, record


def build_ledger(inventory: Mapping[str, Any], config: Mapping[str, Any], *, config_base: Path,
                 allow_provisional: bool = False) -> dict[str, Any]:
    definition = config.get("template_definition", "goal_template")
    entries = {row["task_id"]: row for row in inventory["entries"]}
    sources, blocked, uncertainty = [], [], []
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    notes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    referenced: dict[str, dict[str, TaskIdentity]] = defaultdict(dict)   # role -> task_id -> identity (exposure evidence only)
    explicit_templates: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))  # definition -> role -> keys
    unknown_scopes: dict[str, list[str]] = defaultdict(list)          # split -> reasons (fail closed)
    provisional_gaps: dict[str, list[str]] = defaultdict(list)        # split -> reasons (explicit provisional mode only)
    for source in config["sources"]:
        items, record = _extract(source, config_base)
        sources.append(record)
        if record["status"] == "unresolved":
            reason = f"exposure source {source['id']} is unresolved: {source['unresolved_reason']}"
            for split in source["scope_splits"]:
                (provisional_gaps if allow_provisional else unknown_scopes)[split].append(reason)
            continue
        if not record["available"]:
            reason = f"declared evidence source {source['id']} is unavailable"
            if source.get("required", True):
                blocked.append(reason)
            for split in source["scope_splits"]:
                unknown_scopes[split].append(reason)
            continue
        expected = source.get("expected_sha256")
        if expected:
            # raw bytes, or the same bytes with only CRLF->LF normalised (a Windows autocrlf checkout of a committed LF
            # file); any other byte difference blocks
            record["sha256_matched"] = ("raw" if expected == record["sha256"] else
                                        "lf_normalized" if expected == record["sha256_lf_normalized"] else None)
            if record["sha256_matched"] is None:
                blocked.append(f"source {source['id']} sha256 {record['sha256']} (LF-normalised "
                               f"{record['sha256_lf_normalized']}) != expected {expected}")
                continue
        if source["kind"] == "template_list":
            explicit_templates[source["template_definition"]][source["role"]].update(source["templates"])
            continue
        unresolved = 0
        for item in items:
            task_id = normalize_task_reference(item["ref"]) if isinstance(item["ref"], str) else None
            if task_id is None:
                unresolved += 1
                uncertainty.append({"source": source["id"], "reference": str(item["ref"])[:200],
                                    "problem": "reference cannot be resolved to a task ID"})
                continue
            identity = parse_task_id(task_id)
            if item["class"] in ("EXPOSED", "POSSIBLY_EXPOSED"):   # listed-but-excluded tasks never define overlap
                referenced[item["role"]][task_id] = identity
            entry = {"source": source["id"], "role": item["role"], "class": item["class"], "reason": item["reason"],
                     "source_sha256": record["sha256"]}
            if item["class"] is None:
                notes[task_id].append({**entry, "note": item.get("note")})
            else:
                evidence[task_id].append(entry)
            if task_id not in entries and identity.split in inventory["splits"]:
                uncertainty.append({"source": source["id"], "reference": task_id,
                                    "problem": "referenced task is not in this inventory (different data root?)"})
        if unresolved:
            for split in source["scope_splits"]:
                unknown_scopes[split].append(f"{unresolved} unresolvable references in source {source['id']}")
    # template keys per (definition, role); explicit template lists are kept per declared definition
    keys: dict[str, dict[str, set[str]]] = {
        d: {role: {template_key(i, d) for i in ids.values()} for role, ids in referenced.items()} for d in TEMPLATE_DEFINITIONS}
    for d, by_role in explicit_templates.items():
        for role, templates in by_role.items():
            keys[d].setdefault(role, set()).update(templates)
    trials: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))   # trial -> role -> task_ids
    for role, ids in referenced.items():
        for task_id, identity in ids.items():
            trials[identity.trial][role].add(task_id)
    ledger = []
    for task_id, row in sorted(entries.items()):
        ev = list(evidence.get(task_id, []))
        for reason in unknown_scopes.get(row["split"], []):
            ev.append({"source": None, "role": "scope", "class": "UNKNOWN", "reason": reason, "source_sha256": None})
        if row.get("problems") and any(not p.startswith("missing_game_file") for p in row["problems"]):
            ev.append({"source": "inventory", "role": "inventory", "class": "UNKNOWN",
                       "reason": "inventory problems: " + "; ".join(row["problems"]), "source_sha256": None})
        exposure = max((e["class"] for e in ev), key=lambda c: SEVERITY[c], default="NO_MODEL_OUTCOME_FOUND")
        templates = dict(row.get("templates") or {})
        overlap = {d: sorted(role for role, ks in keys[d].items() if templates.get(d) in ks) for d in sorted(TEMPLATE_DEFINITIONS)}
        reuse = sorted(role for role, ids in trials.get(row.get("trial") or "", {}).items() if ids - {task_id})
        kinds = []
        if any(e["class"] in ("EXPOSED", "POSSIBLY_EXPOSED") for e in ev):
            kinds.append("model_outcome_exposure")
        if reuse:
            kinds.append("exact_task_reuse")
        if any(overlap.values()):
            kinds.append("procedural_template_overlap")
        if exposure == "UNKNOWN":
            kinds.append("unknown_evidence")
        gaps = list(provisional_gaps.get(row["split"], []))
        ledger.append({"task_id": task_id, "split": row["split"], "family": row.get("family"),
                       "scene": row.get("scene"), "trial": row.get("trial"), "template": templates.get(definition),
                       "templates": templates, "installed": row["installed"], "exposure_class": exposure,
                       "screening_class": exposure if row["installed"] else "INELIGIBLE",
                       "exposure_kinds": kinds, "template_overlap_roles": overlap, "exact_trial_reuse_roles": reuse,
                       "evidence_complete": exposure != "UNKNOWN" and not gaps, "provisional_gaps": gaps,
                       "evidence": ev, "notes": notes.get(task_id, [])})
    return {"template_definition": definition, "sources": sources, "blocked": blocked, "ledger": ledger,
            "uncertainty": uncertainty, "provisional": bool(provisional_gaps),
            "provisional_gaps": {k: v for k, v in sorted(provisional_gaps.items())},
            "referenced_templates": {k: sorted(v) for k, v in sorted(keys[definition].items())},
            "referenced_templates_by_definition": {d: {k: sorted(v) for k, v in sorted(by.items())} for d, by in sorted(keys.items())}}


def _summary(rows: Sequence[Mapping[str, Any]], definition: str) -> dict[str, Any]:
    return {"tasks": len(rows), "by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
            "templates": len({(r.get("templates") or {}).get(definition) for r in rows}),
            "scenes": dict(sorted(Counter(r["scene"] for r in rows).items()))}


def roster_approval_blockers(result: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Reasons why NO screen from this ledger may support a scientific roster approval (empty = none found here;
    approval still needs oracle validation and research decisions)."""
    reasons = [f"provisional exposure gap in {split}: {gap}" for split, gaps in result.get("provisional_gaps", {}).items() for gap in gaps]
    unknown = Counter(r["split"] for r in result["ledger"] if r["exposure_class"] == "UNKNOWN" and r["installed"])
    reasons += [f"{n} installed {split} task(s) have UNKNOWN exposure" for split, n in sorted(unknown.items())]
    if result["uncertainty"]:
        reasons.append(f"{len(result['uncertainty'])} unresolved evidence reference(s) recorded")
    by_def = result["referenced_templates_by_definition"]
    for policy in config.get("screening_policies", []):
        referenced = set(by_def.get(policy.get("template_definition", result["template_definition"]), {}))
        roles = set(policy["exclude_template_roles"]) | set(policy.get("exclude_trial_reuse_roles", []))
        missing = sorted(roles - referenced)
        if missing:
            reasons.append(f"policy {policy['name']} names role(s) without any evidence: {missing}")
    return reasons


def screen_pools(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Confirmatory (valid_unseen) and pilot (valid_seen) candidate pools under each declared, named policy.
    Pools are candidate SCREENS, never rosters: eligibility still requires environment-validated oracles."""
    by_def = {d: {role: set(ks) for role, ks in by.items()} for d, by in result["referenced_templates_by_definition"].items()}
    base_definition = result["template_definition"]
    ledger = result["ledger"]

    def pool(split: str) -> dict[str, Any]:
        broad = [r for r in ledger if r["split"] == split and r["screening_class"] == "NO_MODEL_OUTCOME_FOUND"]
        out = {"policy_input": f"{split}: installed and NO_MODEL_OUTCOME_FOUND",
               "outcome_screened": {**_summary(broad, base_definition), "task_ids": [r["task_id"] for r in broad],
                                    "provisional_task_ids": [r["task_id"] for r in broad if r["provisional_gaps"]]},
               "policies": {}}
        for policy in config.get("screening_policies", []):
            d = policy.get("template_definition", base_definition)
            roles, reuse_roles = list(policy["exclude_template_roles"]), list(policy.get("exclude_trial_reuse_roles", []))
            excluded = {role: by_def[d].get(role, set()) for role in roles}
            kept, removed = [], []
            for r in broad:
                key = (r.get("templates") or {}).get(d)
                template_roles = sorted(role for role, ks in excluded.items() if key in ks)
                reuse = sorted(set(r["exact_trial_reuse_roles"]) & set(reuse_roles))
                if template_roles or reuse:
                    removed.append({"task_id": r["task_id"], "family": r["family"], "template": key,
                                    "matched_template_roles": template_roles, "matched_trial_reuse_roles": reuse})
                else:
                    kept.append(r)
            out["policies"][policy["name"]] = {
                "description": policy.get("description"), "template_definition": d,
                "exclude_template_roles": roles, "exclude_trial_reuse_roles": reuse_roles,
                "roles_without_any_evidence": sorted(role for role in set(roles) | set(reuse_roles) if role not in by_def[d]),
                **_summary(kept, d), "removed_by_template_overlap": sum(bool(x["matched_template_roles"]) for x in removed),
                "removed_by_exact_trial_reuse": sum(bool(x["matched_trial_reuse_roles"]) for x in removed),
                "removed": removed, "task_ids": [r["task_id"] for r in kept]}
        names = list(out["policies"])
        out["policy_set_differences"] = [
            {"policies": [a, b],
             "only_in_first": sorted(set(out["policies"][a]["task_ids"]) - set(out["policies"][b]["task_ids"])),
             "only_in_second": sorted(set(out["policies"][b]["task_ids"]) - set(out["policies"][a]["task_ids"]))}
            for a, b in combinations(names, 2)]
        return out

    confirmatory, pilot = pool("valid_unseen"), pool("valid_seen")
    exposed = {r["template"] for r in ledger if r["exposure_class"] in ("EXPOSED", "POSSIBLY_EXPOSED")}
    pilot_rows = [r for r in ledger if r["split"] == "valid_seen" and r["screening_class"] == "NO_MODEL_OUTCOME_FOUND"]
    conf_rows = [r for r in ledger if r["split"] == "valid_unseen" and r["screening_class"] == "NO_MODEL_OUTCOME_FOUND"]
    overlap = {"pilot_templates_shared_with_exposed_tasks": sorted({r["template"] for r in pilot_rows} & exposed),
               "pilot_confirmatory_shared_templates": sorted({r["template"] for r in pilot_rows} & {r["template"] for r in conf_rows}),
               "pilot_confirmatory_shared_scenes": sorted({r["scene"] for r in pilot_rows} & {r["scene"] for r in conf_rows}),
               "pilot_confirmatory_shared_task_ids": sorted({r["task_id"] for r in pilot_rows} & {r["task_id"] for r in conf_rows})}
    blockers = roster_approval_blockers(result, config)
    return {"confirmatory_valid_unseen": confirmatory, "pilot_valid_seen": pilot, "pilot_vs_confirmatory_and_exposure": overlap,
            "provisional": bool(result.get("provisional")), "evidence_complete": not blockers,
            "roster_approval_blockers": blockers,
            "warning": "screening pools are not eligibility; no roster is selected; oracle validation and research decisions follow"}

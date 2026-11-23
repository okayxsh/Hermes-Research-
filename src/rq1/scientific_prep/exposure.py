"""Exposure ledger, template-overlap screening and candidate pools (preparation evidence only).

Exposure is reconstructed ONLY from explicitly declared evidence sources (sources config, schema
``rq1-sci-prep-exposure-sources-v1``).  Each source is hashed; a missing/altered *required* source blocks the run.
Classes (most severe wins):  EXPOSED > POSSIBLY_EXPOSED > UNKNOWN > NO_MODEL_OUTCOME_FOUND.
``NO_MODEL_OUTCOME_FOUND`` means "no declared source shows a model outcome" -- never "unexposed".  Missing evidence
and unresolvable references push tasks towards UNKNOWN, never towards NO_MODEL_OUTCOME_FOUND.
``INELIGIBLE`` (not installed/runnable) is reported separately as the screening class; oracle eligibility is a
different, later artifact (oracle.py) and is never inferred here.
"""
from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq1.scientific_prep.identity import TaskIdentityError, normalize_task_reference, parse_task_id, template_key
from rq1.utils.hashing import sha256_file

SOURCES_SCHEMA = "rq1-sci-prep-exposure-sources-v1"
CLASSES = ("EXPOSED", "POSSIBLY_EXPOSED", "UNKNOWN", "NO_MODEL_OUTCOME_FOUND")
SEVERITY = {name: rank for rank, name in enumerate(reversed(CLASSES))}
SOURCE_KINDS = ("task_manifest", "controlled_failures", "task_list", "csv_task_column", "inline_task_ids")


class ExposureConfigError(ValueError):
    pass


def _get(obj: Any, pointer: str) -> Any:
    for part in [p for p in pointer.split("/") if p]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def load_sources(config_path: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema") != SOURCES_SCHEMA:
        raise ExposureConfigError(f"sources config must declare schema {SOURCES_SCHEMA}")
    ids = [s.get("id") for s in config.get("sources", [])]
    if not ids or len(ids) != len(set(ids)) or not all(ids):
        raise ExposureConfigError("sources need unique non-empty ids")
    for s in config["sources"]:
        if s.get("kind") not in SOURCE_KINDS:
            raise ExposureConfigError(f"source {s['id']}: kind must be one of {SOURCE_KINDS}")
        if s["kind"] != "inline_task_ids" and not s.get("path"):
            raise ExposureConfigError(f"source {s['id']}: path is required")
        for key in ("role", "scope_splits"):
            if not s.get(key):
                raise ExposureConfigError(f"source {s['id']}: {key} is required")
        cls = s.get("class")
        if s["kind"] in ("task_list", "csv_task_column", "inline_task_ids", "controlled_failures") and cls not in CLASSES[:2]:
            raise ExposureConfigError(f"source {s['id']}: class must be EXPOSED or POSSIBLY_EXPOSED")
        if s["kind"] == "inline_task_ids" and not s.get("provenance"):
            raise ExposureConfigError(f"source {s['id']}: inline task IDs need a provenance note")
        rules = [s.get("selected") or {}, s.get("replaced") or {}, *(s.get("exclusions") or {}).values()]
        for rule in rules:
            if rule.get("class") is not None and rule["class"] not in CLASSES[:3]:
                raise ExposureConfigError(f"source {s['id']}: rule class {rule['class']!r} must be EXPOSED, "
                                          "POSSIBLY_EXPOSED, UNKNOWN or null (INELIGIBLE comes only from the inventory)")
    for p in config.get("screening_policies", []):
        if not p.get("name") or not isinstance(p.get("exclude_template_roles"), list):
            raise ExposureConfigError("each screening policy needs a name and exclude_template_roles")
    return config


def _resolve(path_text: str, base: Path) -> Path:
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else (base / p)


def _extract(source: Mapping[str, Any], base: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns (evidence items with raw references, source record)."""
    record: dict[str, Any] = {k: source.get(k) for k in ("id", "kind", "role", "required", "scope_splits", "description",
                                                         "expected_sha256", "provenance")}
    items: list[dict[str, Any]] = []
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
    record.update(available=True, sha256=sha256_file(path))
    if source["kind"] == "csv_task_column":
        reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))
        for row in reader:
            items.append({"ref": row.get(source.get("column", "task_id"), ""), "class": source["class"],
                          "role": source["role"], "reason": source.get("reason", "listed in source")})
        return items, record
    data = json.loads(path.read_text(encoding="utf-8"))
    if source["kind"] == "task_list":
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


def build_ledger(inventory: Mapping[str, Any], config: Mapping[str, Any], *, config_base: Path) -> dict[str, Any]:
    definition = config.get("template_definition", "goal_template")
    entries = {row["task_id"]: row for row in inventory["entries"]}
    sources, blocked, uncertainty = [], [], []
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    notes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    referenced_templates: dict[str, set[str]] = defaultdict(set)     # role -> template keys of referenced tasks
    unknown_scopes: dict[str, list[str]] = defaultdict(list)          # split -> reasons
    for source in config["sources"]:
        items, record = _extract(source, config_base)
        sources.append(record)
        if not record["available"]:
            reason = f"declared evidence source {source['id']} is unavailable"
            if source.get("required", True):
                blocked.append(reason)
            for split in source["scope_splits"]:
                unknown_scopes[split].append(reason)
            continue
        if source.get("expected_sha256") and source["expected_sha256"] != record["sha256"]:
            blocked.append(f"source {source['id']} sha256 {record['sha256']} != expected {source['expected_sha256']}")
            continue
        unresolved = 0
        for item in items:
            task_id = normalize_task_reference(item["ref"])
            if task_id is None:
                unresolved += 1
                uncertainty.append({"source": source["id"], "reference": str(item["ref"])[:200],
                                    "problem": "reference cannot be resolved to a task ID"})
                continue
            identity = parse_task_id(task_id)
            if item["class"] in ("EXPOSED", "POSSIBLY_EXPOSED"):   # listed-but-excluded tasks never define overlap
                referenced_templates[item["role"]].add(template_key(identity, definition))
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
    ledger = []
    for task_id, row in sorted(entries.items()):
        ev = list(evidence.get(task_id, []))
        for reason in unknown_scopes.get(row["split"], []):
            ev.append({"source": None, "role": "scope", "class": "UNKNOWN", "reason": reason, "source_sha256": None})
        if row.get("problems") and any(not p.startswith("missing_game_file") for p in row["problems"]):
            ev.append({"source": "inventory", "role": "inventory", "class": "UNKNOWN",
                       "reason": "inventory problems: " + "; ".join(row["problems"]), "source_sha256": None})
        exposure = max((e["class"] for e in ev), key=lambda c: SEVERITY[c], default="NO_MODEL_OUTCOME_FOUND")
        ledger.append({"task_id": task_id, "split": row["split"], "family": row.get("family"),
                       "scene": row.get("scene"), "template": (row.get("templates") or {}).get(definition),
                       "installed": row["installed"], "exposure_class": exposure,
                       "screening_class": exposure if row["installed"] else "INELIGIBLE",
                       "evidence": ev, "notes": notes.get(task_id, [])})
    return {"template_definition": definition, "sources": sources, "blocked": blocked, "ledger": ledger,
            "uncertainty": uncertainty, "referenced_templates": {k: sorted(v) for k, v in referenced_templates.items()}}


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"tasks": len(rows), "by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
            "templates": len({r["template"] for r in rows}), "scenes": dict(sorted(Counter(r["scene"] for r in rows).items()))}


def screen_pools(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Confirmatory (valid_unseen) and pilot (valid_seen) candidate pools under each declared policy.
    Pools are candidate SCREENS, never rosters: eligibility still requires environment-validated oracles."""
    by_role = {role: set(keys) for role, keys in result["referenced_templates"].items()}
    ledger = result["ledger"]

    def pool(split: str) -> dict[str, Any]:
        broad = [r for r in ledger if r["split"] == split and r["screening_class"] == "NO_MODEL_OUTCOME_FOUND"]
        out = {"policy_input": f"{split}: installed and NO_MODEL_OUTCOME_FOUND", "outcome_screened": {**_summary(broad),
               "task_ids": [r["task_id"] for r in broad]}, "policies": {}}
        for policy in config.get("screening_policies", []):
            excluded = set().union(*(by_role.get(role, set()) for role in policy["exclude_template_roles"])) \
                if policy["exclude_template_roles"] else set()
            kept = [r for r in broad if r["template"] not in excluded]
            removed = [r for r in broad if r["template"] in excluded]
            missing_roles = [role for role in policy["exclude_template_roles"] if role not in by_role]
            out["policies"][policy["name"]] = {"exclude_template_roles": policy["exclude_template_roles"],
                                               "roles_without_any_evidence": missing_roles,
                                               **_summary(kept), "removed_by_template_overlap": len(removed),
                                               "task_ids": [r["task_id"] for r in kept]}
        return out

    confirmatory, pilot = pool("valid_unseen"), pool("valid_seen")
    exposed = {r["template"] for r in ledger if r["exposure_class"] in ("EXPOSED", "POSSIBLY_EXPOSED")}
    pilot_rows = [r for r in ledger if r["split"] == "valid_seen" and r["screening_class"] == "NO_MODEL_OUTCOME_FOUND"]
    conf_rows = [r for r in ledger if r["split"] == "valid_unseen" and r["screening_class"] == "NO_MODEL_OUTCOME_FOUND"]
    overlap = {"pilot_templates_shared_with_exposed_tasks": sorted({r["template"] for r in pilot_rows} & exposed),
               "pilot_confirmatory_shared_templates": sorted({r["template"] for r in pilot_rows} & {r["template"] for r in conf_rows}),
               "pilot_confirmatory_shared_scenes": sorted({r["scene"] for r in pilot_rows} & {r["scene"] for r in conf_rows}),
               "pilot_confirmatory_shared_task_ids": sorted({r["task_id"] for r in pilot_rows} & {r["task_id"] for r in conf_rows})}
    return {"confirmatory_valid_unseen": confirmatory, "pilot_valid_seen": pilot, "pilot_vs_confirmatory_and_exposure": overlap,
            "warning": "screening pools are not eligibility; no roster is selected; oracle validation and research decisions follow"}

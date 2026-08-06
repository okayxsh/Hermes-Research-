"""Fail-closed Phase 6 real runtime.

Phase 7 may add installed-version adapters after evidence is observed on the
university machine.  This module deliberately invents no Hermes or ALFWorld API.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

from rq1.bridge.environment import real_adapter_capability
from rq1.hermes.capabilities import probe_hermes_capabilities
from rq1.pilot.models import EvidenceLevel, PilotStatus, PilotTestSpec, RuntimeExecution
from rq1.pilot.gates import validate_task_manifest
from rq1.recovery.verification import real_recovery_capabilities


class RealPilotRuntime:
    simulated = False

    def __init__(self, root: Path) -> None:
        self.root = root

    def execute(self, spec: PilotTestSpec, *, run_id: str, attempt_id: str, output_dir: Path) -> RuntimeExecution:
        index = int(spec.test_id.split("_")[1])
        if index == 0:
            return RuntimeExecution(PilotStatus.PASSED, EvidenceLevel.STATIC, {
                "repository": "$REPO", "schemas_present": (self.root / "data" / "schemas").is_dir(),
                "external_software_modified": False,
            })
        if index == 1:
            supported = platform.system() == "Linux" and platform.machine().lower() in {"x86_64", "amd64"}
            details = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "supported_university_platform": supported}
            if supported:
                return RuntimeExecution(PilotStatus.PASSED, EvidenceLevel.INSTALLED, details)
            return RuntimeExecution(PilotStatus.BLOCKED, EvidenceLevel.STATIC, details, "Real pilot requires x86_64 Ubuntu 22.04/24.04 or WSL2.", "Run Phase 7 on the approved university machine.")
        if index == 2:
            path = self.root / "artifacts" / "stage_reports" / "installation.json"
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if payload.get("installation_ready") is True:
                return RuntimeExecution(PilotStatus.PASSED, EvidenceLevel.INSTALLED, {"installation_report": "artifacts/stage_reports/installation.json", "installation_ready": True})
            return RuntimeExecution(PilotStatus.BLOCKED, EvidenceLevel.STATIC, {"installation_report_present": path.exists()}, "A fresh passing installation report is unavailable.", "Run setup verification on the university machine; do not install from the pilot runner.")
        if index == 36:
            return RuntimeExecution(PilotStatus.PASSED, EvidenceLevel.STATIC, {"report_generated": True, "real_evidence_promoted_from_mock": False, "phase7_required": True})
        if 12 <= index <= 27 or index == 29:
            manifest_path = self.root / "data" / "task_lists" / ("acquisition_train.json" if index == 27 else "pilot_seen.json")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                return RuntimeExecution(PilotStatus.BLOCKED, EvidenceLevel.STATIC, {"task_manifest": str(manifest_path.relative_to(self.root))}, f"Pilot task manifest is unavailable: {exc}", "Supply an approved train/valid_seen pilot manifest; never use valid_unseen.")
            allowed = {"train"} if index == 27 else {"valid_seen"}
            errors = validate_task_manifest(manifest, allowed_splits=allowed)
            tasks = manifest.get("tasks", [])
            if errors or not tasks:
                return RuntimeExecution(PilotStatus.BLOCKED, EvidenceLevel.STATIC, {"task_manifest": str(manifest_path.relative_to(self.root)), "validation_errors": errors, "task_count": len(tasks) if isinstance(tasks, list) else 0}, "Approved pilot tasks are missing or invalid.", "Populate only the approved train/valid_seen pilot list during Phase 7.")
        hermes = probe_hermes_capabilities(project_root=self.root)
        alfworld = real_adapter_capability()
        recovery = real_recovery_capabilities()
        hermes_payload = hermes.to_dict()
        if hermes_payload.get("executable"):
            hermes_payload["executable"] = Path(str(hermes_payload["executable"])).name
        hermes_payload["project_plugin_locations"] = ["$REPO/.hermes/plugins"] if hermes_payload.get("project_plugin_locations") else []
        details = {
            "hermes": hermes_payload,
            "alfworld": {"available": alfworld.available, "version": alfworld.version, "details": alfworld.details},
            "recovery": recovery,
            "real_operation_executed": False,
        }
        return RuntimeExecution(
            PilotStatus.BLOCKED,
            EvidenceLevel.INSTALLED if hermes.installed else EvidenceLevel.STATIC,
            details,
            "Required real integration capability is unavailable or unverified.",
            "Phase 7 must capability-probe the installed versions and implement only observed adapter surfaces.",
        )

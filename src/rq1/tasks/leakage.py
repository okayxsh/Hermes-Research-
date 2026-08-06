from __future__ import annotations
from rq1.skills.leakage import find_leakage
from rq1.tasks.models import TaskManifest

def evaluation_leakage(manifest: TaskManifest, skill_texts: list[str]) -> list[str]:
    errors=[]
    if manifest.manifest_type != "evaluation": return ["evaluation leakage check requires evaluation manifest"]
    for text in skill_texts:
        if any(task.task_id in text for task in manifest.tasks): errors.append("evaluation task ID appears in skill content")
        if find_leakage(text): errors.append("task-specific skill leakage pattern")
    return errors

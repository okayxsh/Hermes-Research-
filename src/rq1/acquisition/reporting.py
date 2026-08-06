from __future__ import annotations
import json
from pathlib import Path
def write_report(path: Path, payload: dict) -> Path:
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n', encoding='utf-8'); return path

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from rq1.utils.time import utc_now

def run_dir(root: Path, run_id: str) -> Path: return root / "artifacts" / "autopilot" / run_id
def state_path(root: Path, run_id: str) -> Path: return run_dir(root, run_id) / "state.json"
def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp=path.with_suffix(".tmp"); temp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); temp.replace(path)
def load(root: Path, run_id: str) -> dict[str, Any]: return json.loads(state_path(root,run_id).read_text(encoding="utf-8"))
def initialize(root: Path, run_id: str, plan: dict[str, Any], stages: list[str]) -> dict[str, Any]:
    value={"schema_version":1,"run_id":run_id,"created_at":utc_now(),"plan":plan,"plan_hash":plan["content_hash"],"top_status":"RUNNING","stop_requested":False,"last_successful_stage":None,"heartbeat":None,"stages":{name:{"status":"pending","attempts":[]} for name in stages}}
    write_atomic(state_path(root,run_id),value); return value
def heartbeat(root: Path, state: dict[str, Any]) -> None: state["heartbeat"]=utc_now(); write_atomic(state_path(root,state["run_id"]),state)
def append_contingency(root: Path, run_id: str, payload: dict[str, Any]) -> None:
    path=run_dir(root,run_id)/"contingencies.jsonl"; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(payload,sort_keys=True)+"\n")

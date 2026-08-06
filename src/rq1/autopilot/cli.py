from __future__ import annotations
import json
from pathlib import Path
from rq1.autopilot.executor import Autopilot
from rq1.autopilot.forecast import forecast_from_pilot
from rq1.autopilot.models import RunMode

def command(root: Path, args) -> int:
    autopilot=Autopilot(root); action=args.autopilot_command
    if action=="plan":
        plan=autopilot.plan(RunMode(args.mode),args.output_dir); print(json.dumps({"dry_run":True,"plan":plan.to_dict()},indent=2,sort_keys=True)); return 0
    if action=="doctor":
        from rq1.autopilot.health import sample
        print(json.dumps({"health":sample(root),"final_gates":__import__("rq1.freeze.validation",fromlist=["validate_final_gates"]).validate_final_gates(root).to_dict(),"real_execution":"capability_gated"},indent=2)); return 0
    if action=="forecast":
        result=forecast_from_pilot(Path(args.pilot_report)); print(json.dumps(result.to_dict(),indent=2)); return 0 if result.evidence_available else 1
    if action in {"bootstrap","final"}:
        if not args.yes: raise RuntimeError("Autopilot execution requires --yes; use `rq1 autopilot plan` first")
        if args.detach:
            from rq1.autopilot.health import sample
            health=sample(root)
            if not health["systemd_user_available"] and not health["tmux_available"]:
                raise RuntimeError("Detached execution is blocked: neither systemd user services nor tmux is available")
            raise RuntimeError("Detached execution is capability-gated until the target machine validates the supervisor handoff; use foreground mode for observed pilot work")
        plan=autopilot.plan(RunMode(action),args.output_dir)
        if action=="final":
            if not args.approval or not Path(args.approval).is_file(): raise RuntimeError("Final autopilot requires an explicit approval manifest")
            values=plan.to_dict(); values["approval_references"]=(str(Path(args.approval)),); values["content_hash"]=""; from rq1.autopilot.models import RunPlan,canonical_hash; values["content_hash"]=canonical_hash({key:value for key,value in values.items() if key!="content_hash"}); plan=RunPlan(**values)
        data=autopilot.create(plan); data=autopilot.run(plan.run_plan_id); print(json.dumps(data,indent=2,sort_keys=True)); return 0 if data["top_status"] in {"PILOT_GO","SUCCESS"} else 1
    if action=="status": print(json.dumps(autopilot.status(args.run_id),indent=2,sort_keys=True)); return 0
    if action=="logs":
        path=root/"artifacts"/"autopilot"/args.run_id/"contingencies.jsonl"; print(path.read_text(encoding="utf-8") if path.is_file() else ""); return 0
    if action=="resume": print(json.dumps(autopilot.run(args.run_id),indent=2,sort_keys=True)); return 0
    if action=="stop": print(json.dumps(autopilot.stop(args.run_id),indent=2,sort_keys=True)); return 0
    raise RuntimeError("Unknown autopilot command")

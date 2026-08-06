from __future__ import annotations
import json, os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4
from rq1.autopilot.models import ContingencyRecord, ErrorClass, MutationClass, RunMode, RunPlan, StageDefinition, TopStatus, canonical_hash
from . import state
from rq1.autopilot.forecast import forecast_from_pilot
from rq1.freeze.validation import git_state, validate_final_gates
from rq1.utils.time import utc_now

BOOTSTRAP=("preflight","setup","installation_verification","real_pilot","runtime_forecast","pilot_decision")
FINAL=("revalidate_gates","acquisition","acquisition_validation","snapshots","profiles","evaluation_tasks","evaluation_activation","evaluation","evaluation_validation","analysis","figures","archive")
def definitions(mode: RunMode) -> tuple[StageDefinition,...]:
    names=BOOTSTRAP if mode==RunMode.BOOTSTRAP else FINAL; result=[]
    for index,name in enumerate(names):
        uncertain=name in {"acquisition","evaluation"}; result.append(StageDefinition(name,() if not index else (names[index-1],),MutationClass.UNCERTAIN_MUTATION if uncertain else MutationClass.READ_ONLY,not uncertain,"stage_boundary"))
    return tuple(result)

class Autopilot:
    def __init__(self, root: Path): self.root=root
    def plan(self, mode: RunMode, output_dir: str | None=None) -> RunPlan:
        commit,_,_=git_state(self.root); values={"schema_version":1,"run_plan_id":str(uuid4()),"mode":mode.value,"repository_commit":commit,"output_directory":output_dir or "results/final"}; values["content_hash"]=""; values["content_hash"]=canonical_hash({key:value for key,value in values.items() if key!="content_hash"}); return RunPlan(**values)
    def create(self, plan: RunPlan) -> dict:
        return state.initialize(self.root,plan.run_plan_id,plan.to_dict(),[item.name for item in definitions(RunMode(plan.mode))])
    def _block(self, data: dict, stage: str, code: str, message: str, kind: ErrorClass=ErrorClass.COMPATIBILITY) -> dict:
        data["stages"][stage]["status"]="blocked"; data["top_status"]=TopStatus.BLOCKED.value
        record=ContingencyRecord(1,data["run_id"],stage,None,utc_now(),code,kind.value,message,False,"no_retry",data.get("last_successful_stage"),(),(),(),(),"Review evidence and resolve the documented blocker.",True,True,"No scientific evidence was produced.")
        state.append_contingency(self.root,data["run_id"],record.to_dict()); state.heartbeat(self.root,data); return data
    def _pass(self,data:dict,stage:str,details:dict|None=None)->None:
        data["stages"][stage]["status"]="passed"; data["stages"][stage]["attempts"].append({"attempt_id":str(uuid4()),"completed_at":utc_now(),"details":details or {}}); data["last_successful_stage"]=stage; state.heartbeat(self.root,data)
    def run(self, run_id: str) -> dict:
        data=state.load(self.root,run_id); mode=RunMode(data["plan"]["mode"])
        for stage_def in definitions(mode):
            stage=stage_def.name
            if data["stages"][stage]["status"]=="passed": continue
            if data.get("stop_requested"): data["top_status"]=TopStatus.STOPPED.value; state.heartbeat(self.root,data); return data
            data["stages"][stage]["status"]="running"; state.heartbeat(self.root,data)
            if mode==RunMode.BOOTSTRAP and stage=="preflight": self._pass(data,stage); continue
            if mode==RunMode.BOOTSTRAP and stage=="runtime_forecast": self._pass(data,stage,{"evidence_available":False}); continue
            if mode==RunMode.FINAL and stage=="revalidate_gates":
                gates=validate_final_gates(self.root)
                if not gates.valid: return self._block(data,stage,"BLOCKED_FINAL_GATES","; ".join(gates.reasons),ErrorClass.SCIENTIFIC)
                self._pass(data,stage); continue
            # Final scientific operations and real bootstrap integration must be supplied by observed adapters.
            return self._block(data,stage,"BLOCKED_REAL_ADAPTER_UNAVAILABLE",f"{stage} requires a capability-confirmed real adapter; no fake fallback is permitted")
        if mode==RunMode.BOOTSTRAP: data["top_status"]=TopStatus.PILOT_GO.value
        else: data["top_status"]=TopStatus.SUCCESS.value
        state.heartbeat(self.root,data); return data
    def stop(self,run_id:str)->dict:
        data=state.load(self.root,run_id); data["stop_requested"]=True; state.heartbeat(self.root,data); return data
    def status(self,run_id:str)->dict: return state.load(self.root,run_id)

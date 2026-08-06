from __future__ import annotations
import json
from pathlib import Path
from rq1.autopilot.models import RuntimeForecast

def forecast_from_pilot(path: Path, *, acquisition_count: int | None=None, evaluation_count: int | None=None, snapshots: int | None=None, repetitions: int | None=None, perturbations: int=1, workers: int=1) -> RuntimeForecast:
    report=json.loads(path.read_text(encoding="utf-8")); measurements=report.get("runtime_benchmark", report.get("measurements", {}))
    acq=measurements.get("acquisition_episode_seconds"); recovery=measurements.get("recovery_episode_seconds"); overhead=measurements.get("validation_analysis_packaging_seconds",0)
    cells=None if None in (evaluation_count,snapshots,repetitions) else evaluation_count*snapshots*repetitions*perturbations
    if not isinstance(acq,(int,float)) or not isinstance(recovery,(int,float)) or cells is None or acquisition_count is None:
        return RuntimeForecast(1,str(path),cells,None,None,None,None,None,None,None,None,False)
    serial=(acquisition_count*acq+cells*recovery+overhead)/3600; parallel=(acquisition_count*acq+cells*recovery/workers+overhead)/3600
    return RuntimeForecast(1,str(path),cells,parallel*.8,parallel,parallel*1.25,serial,parallel,parallel*workers,float(measurements.get("disk_gib",0)),float(measurements.get("log_gib",0)),(acquisition_count*900+cells*900+overhead)/3600,True)

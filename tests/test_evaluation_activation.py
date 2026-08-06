from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from rq1.evaluation.activation import ActivationError, build_activation, invalidate, require_runtime_opt_in, validate_activation, write_activation
from rq1.freeze.models import FreezeValidation
from rq1.tasks.models import TaskManifest, TaskRecord
from rq1.tasks.validation import manifest_hash

class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.evidence={}
        task=TaskRecord('valid_unseen:one','valid_unseen','pick_and_place','one','source','game',1)
        value={'schema_version':1,'manifest_type':'evaluation','status':'frozen','split':'valid_unseen','alfworld_version':'0.4.2','data_root_identity':'data','repository_commit':'commit','selection_policy':{'seed':1},'requested_count':1,'actual_count':1,'family_counts':{'pick_and_place':1},'tasks':[task.to_dict()],'exclusions':[],'duplicate_resolution':[],'generated_at':'now','approved_at':'now','approval_reference':'r','manifest_sha256':''}; value['manifest_sha256']=manifest_hash(value)
        self._write('evaluation_task_manifest',value)
        self._write('pilot_report',{'mode':'real','experimental_ready':True,'go_no_go':{'decision':'go'},'pilot_run_id':'pilot'})
        for name in ('acquisition_validation','snapshot_validation','profile_validation','checkpoint_replay','perturbation','solvability','recovery_context','relevance_rules'): self._write(name,{'valid':True,'status':'validated'})
        inputs={'model_digest':'model','alfworld_data_sha256':'data','hermes_version':'hermes','prompt_hashes':{'p':'h'},'repetition_count':2}
        freeze={'repository_commit':'commit','inputs':inputs}; (self.root/'artifacts'/'freezes').mkdir(parents=True); (self.root/'artifacts'/'freezes'/'environment-freeze.json').write_text(json.dumps(freeze)); (self.root/'artifacts'/'freezes'/'protocol-freeze.json').write_text(json.dumps(freeze))
        self.approval={'approved_by':'reviewer','approved_at':'now','reference':'approval','repetition_count':2,'action_budget':12,'timeout_seconds':900,'queue_policy_version':'v1','evidence_paths':self.evidence}
    def tearDown(self): self.temp.cleanup()
    def _write(self,name,payload):
        path=self.root/(name+'.json'); path.write_text(json.dumps(payload)); self.evidence[name]=str(path)
    def _patches(self):
        return patch('rq1.evaluation.activation.validate_final_gates',return_value=FreezeValidation(True,(),None,None)), patch('rq1.evaluation.activation.git_state',return_value=('commit',True,None))
    def test_missing_prerequisite_blocks(self):
        self.approval['evidence_paths']=dict(self.evidence); self.approval['evidence_paths'].pop('pilot_report')
        a,b=self._patches()
        with a,b,self.assertRaises(ActivationError): build_activation(self.root,self.approval,self.approval['evidence_paths'])
    def test_activation_is_immutable_and_drift_invalidates(self):
        a,b=self._patches()
        with a,b:
            manifest=build_activation(self.root,self.approval,self.evidence); path=write_activation(self.root,manifest)
            self.assertEqual([],validate_activation(self.root,manifest))
            with self.assertRaises(FileExistsError): write_activation(self.root,manifest)
            Path(self.evidence['snapshot_validation']).write_text('{"valid":false}')
            self.assertTrue(validate_activation(self.root,manifest))
    def test_runtime_requires_opt_in_before_execution(self):
        a,b=self._patches()
        with a,b:
            manifest=build_activation(self.root,self.approval,self.evidence); path=write_activation(self.root,manifest)
            with self.assertRaises(ActivationError): require_runtime_opt_in(self.root,path)
    def test_invalidated_activation_cannot_run(self):
        a,b=self._patches()
        with a,b,patch.dict('os.environ', {'RQ1_RUN_FINAL_EVALUATION':'1'}):
            manifest=build_activation(self.root,self.approval,self.evidence); path=write_activation(self.root,manifest)
            invalidate(self.root,path,'review withdrawal')
            with self.assertRaises(ActivationError): require_runtime_opt_in(self.root,path)

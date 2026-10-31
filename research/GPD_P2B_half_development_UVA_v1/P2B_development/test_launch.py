"""Admission/retention guards in isolated fixtures; no scheduler calls."""
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from unittest.mock import patch
import launch
from p2b_policy import CASE,sha,BUDGETS

class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ,{},clear=False))
        self.stack.enter_context(patch.object(launch,'ROOT',self.root))
        manifests={}
        for label,name in [('capsule','CAPSULE'),('packet','PACKET'),('acceleration','ACCELERATION'),('driver','PACKAGE'),('transfer','BUNDLE')]:
            p=self.root/label;p.mkdir();(p/'fixture.txt').write_text('fixture')
            (p/'MANIFEST.json').write_text(json.dumps({'files':[{'path':'fixture.txt','bytes':7,'sha256':sha(p/'fixture.txt')}]}))
            manifests[label+'_manifest_sha256']=sha(p/'MANIFEST.json');self.stack.enter_context(patch.object(launch,name,p))
        for folder,name,value in [('exact_center','ASSESSMENT.json',dict(decision='EXACT_CENTER_PASS_REVIEW_REQUIRED',gates={'all':True})),('execution','AUDIT.json',dict(status='PASS_EXECUTION_EVIDENCE_GAP_RESOLVED'))]:
            d=self.root/'transfer/evidence'/folder;d.mkdir(parents=True);(d/name).write_text(json.dumps(value))
        review=self.root/'performance.json';review.write_text(json.dumps(dict(qualified_backends=['cpu_guarded'],fastest_measured_backend='cpu_guarded')))
        self.registry=self.root/'control/GPD_P2B_half_development_v1';self.registry.mkdir(parents=True)
        self.spec=dict(case=CASE,phase='half_noise_development',backend='cpu_guarded',threads=1,solver_blas_threads=8,
            scientific_budgets=dict(BUDGETS),resources=dict(account='spinquest_standard',nodes=1,ntasks=1,gpus=0,cpus_per_task=8,requeue=False,
            stop_after_seconds=36000,wall_seconds=43200,tree_rss_cap_GiB=56,reserve_GiB=4,memory_GiB=64,partition='standard-afton'),
            files={str(review):sha(review)},performance_review=str(review),development_released=True,independent_assessment_released=False,
            versions=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ('numpy','scipy','torch')}),
            python=sys.executable,output_root=str(self.root/'GPD_P2B_half_development_scratch'),persistent_root=str(self.root/'GPD_P2B_half_development_retained'),
            case_registry=str(self.registry),resume_parent=None,**manifests)
        self.path=self.root/'spec.json'
    def validate(self,allocated=False,permit_hash=None,stale=False):
        self.path.write_text(json.dumps(self.spec))
        (self.registry/'CLAIM.json').write_text(json.dumps(dict(case=CASE,transfer_manifest_sha256=self.spec['transfer_manifest_sha256'],
            latest_spec=str(self.path),latest_spec_sha256='wrong' if stale else sha(self.path))))
        permit=self.root/'permit.json';permit.write_text(json.dumps(dict(authorized=True,spec_sha256=permit_hash or sha(self.path),case=CASE)))
        env=dict(SLURM_JOB_ID='fixture',SLURM_JOB_ACCOUNT='spinquest_standard',SLURM_JOB_NUM_NODES='1',SLURM_NTASKS='1',SLURM_CPUS_PER_TASK='8',SLURM_MEM_PER_NODE='65536',SLURM_JOB_PARTITION='standard-afton')
        with patch.dict(os.environ,env),patch.object(launch.sys,'platform','linux'):
            return launch.validate(self.path,CASE,approval=permit,allocated=allocated)
    def test_valid_static_and_allocated(self):self.validate();self.validate(True)
    def test_wrong_account(self):
        self.spec['resources']['account']='wrong'
        with self.assertRaises(RuntimeError):self.validate()
    def test_budget_or_threads_drift(self):
        self.spec['scientific_budgets']['outer_iterations']=61
        with self.assertRaises(RuntimeError):self.validate()
        self.spec['scientific_budgets']=dict(BUDGETS);self.spec['threads']=8
        with self.assertRaises(RuntimeError):self.validate()
    def test_existing_output(self):
        Path(self.spec['output_root']).mkdir()
        with self.assertRaises(RuntimeError):self.validate()
    def test_changed_source(self):
        (self.root/'driver/fixture.txt').write_text('changed')
        with self.assertRaises(RuntimeError):self.validate()
    def test_unpinned_review_and_stale_claim(self):
        with self.assertRaises(RuntimeError):self.validate(stale=True)
        self.spec['files']={}
        with self.assertRaises(RuntimeError):self.validate()
    def test_wrong_permit(self):
        with self.assertRaises(RuntimeError):self.validate(True,'wrong')
    def test_wrong_case_and_automatic_assessment(self):
        self.spec['case']='GPD_P2B_exact_center_s31'
        with self.assertRaises(RuntimeError):self.validate()
        self.spec['case']=CASE;self.spec['independent_assessment_released']=True
        with self.assertRaises(RuntimeError):self.validate()
    def test_resume_requires_explicit_mode(self):
        self.spec['resume_parent']={'root':'fixture'}
        with self.assertRaises(RuntimeError):self.validate()
    def test_case_lock_excludes_second_supervisor(self):
        from supervise import claim_case
        one=claim_case(self.spec)
        try:
            with self.assertRaises(BlockingIOError):claim_case(self.spec)
        finally:one.close()

class ParentTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.lock={'fixture':True};(self.root/'RUN_LOCK.json').write_text(json.dumps(self.lock));h=sha(self.root/'RUN_LOCK.json')
        torch.save(dict(flat=torch.ones(3,dtype=torch.float64),completed_updates=137,remaining_updates=263,run_lock_sha256=h),self.root/'ADAM_RESUME.pt')
        (self.root/'INTERRUPTED.json').write_text(json.dumps(dict(run_lock_sha256=h)))
        (self.root/'ATTEMPT_ENTRY.json').write_text(json.dumps(dict(starting_progress=dict(Adam_updates=0,outer_iterations=0,solver_evaluations=0))))
        self.seal()
    def seal(self,code=75):
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in self.root.iterdir() if p.is_file() and p.name not in ['TERMINAL_RETENTION.json','SUPERVISOR_fixture.json']]
        (self.root/'TERMINAL_RETENTION.json').write_text(json.dumps(dict(verified=True,files=files)))
        (self.root/'SUPERVISOR_fixture.json').write_text(json.dumps(dict(child_exit=code,terminal_retention_verified=True)))
    def test_clean_pause_stages_exact_bytes(self):
        from resume_validation import verify_parent,stage_parent
        parent=verify_parent(self.root,self.lock)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'new';stage_parent(parent,out,self.lock)
            self.assertEqual(sha(out/'ADAM_RESUME.pt'),sha(self.root/'ADAM_RESUME.pt'))
    def test_failure_mutation_and_no_progress_rejected(self):
        from resume_validation import verify_parent
        self.seal(code=7)
        with self.assertRaises(RuntimeError):verify_parent(self.root,self.lock)
        self.seal();(self.root/'INTERRUPTED.json').write_text('{}')
        with self.assertRaises(RuntimeError):verify_parent(self.root,self.lock)
    def test_completed_result_not_restarted(self):
        from resume_validation import verify_parent
        (self.root/'RESULT.json').write_text('{}');self.seal()
        with self.assertRaises(RuntimeError):verify_parent(self.root,self.lock)
    def test_no_completed_update_progress_rejected(self):
        from resume_validation import verify_parent
        (self.root/'ATTEMPT_ENTRY.json').write_text(json.dumps(dict(starting_progress=dict(Adam_updates=137,outer_iterations=0,solver_evaluations=0))))
        self.seal()
        with self.assertRaises(RuntimeError):verify_parent(self.root,self.lock)

if __name__=='__main__':unittest.main(verbosity=2)

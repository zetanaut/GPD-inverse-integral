"""Development decisions and input invariants; no scientific fit or new noise."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from p2b_policy import BUNDLE,TRUE_TARGETS,assess_values,check_restart,read,tolerance

class PolicyTests(unittest.TestCase):
    def setUp(self):self.baseline=read(BUNDLE/'evidence/baseline/BASELINE.json')
    def values(self,observed=.10,errors=(.04,.01),rms=1.5,eligible=True):
        targets=[t*(1-e) for t,e in zip(TRUE_TARGETS,errors)]
        return assess_values(eligible,observed,targets,rms,self.baseline)
    def test_improvement_without_exact_control_cutoffs(self):
        d=self.values();self.assertEqual(d['decision'],'DEVELOPMENT_PASS_REVIEW_REQUIRED')
        self.assertFalse(d['assessment_released']);self.assertTrue(d['PDF_RMS_is_diagnostic_not_gate'])
    def test_target_worsening_stops_despite_better_field(self):
        self.assertEqual(self.values(errors=(.06,.001))['decision'],'STOP_DEVELOPMENT_FAILED')
        self.assertEqual(self.values(errors=(.001,.02))['decision'],'STOP_DEVELOPMENT_FAILED')
    def test_field_tie_and_worsening_stop(self):
        for v in [self.baseline['observed_Hplus_relative_L2'],.12]:
            self.assertEqual(self.values(observed=v)['decision'],'STOP_DEVELOPMENT_FAILED')
    def test_frozen_squared_error_tolerance(self):
        b=self.baseline['observed_Hplus_squared_relative_L2'];t=tolerance(b,b)
        self.assertEqual(self.values(observed=(b-.5*t)**.5)['decision'],'STOP_DEVELOPMENT_FAILED')
        self.assertEqual(self.values(observed=(b-2*t)**.5)['decision'],'DEVELOPMENT_PASS_REVIEW_REQUIRED')
    def test_target_tolerance_and_ineligible(self):
        e=self.baseline['target_absolute_relative_errors'];t=tolerance(e[0],e[0])
        self.assertEqual(self.values(errors=(e[0]+.5*t,e[1]))['decision'],'DEVELOPMENT_PASS_REVIEW_REQUIRED')
        self.assertEqual(self.values(errors=(e[0]+2*t,e[1]))['decision'],'STOP_DEVELOPMENT_FAILED')
        self.assertEqual(self.values(eligible=False)['decision'],'STOP_INELIGIBLE')
    def test_nonfinite_scores_rejected(self):
        for kw in [dict(observed=float('nan')),dict(rms=float('inf')),dict(errors=(float('nan'),.01))]:
            with self.assertRaises(ValueError):self.values(**kw)
    def test_restart_budgets(self):
        check_restart(dict(run_lock_sha256='x',completed_updates=137,remaining_updates=263),'x','adam')
        check_restart(dict(run_lock_sha256='x',next_iteration=60,remaining_outer_iterations=0,history=[{}]*60,oracle_counts={'evaluations':121}),'x','solver')
        for bad in [dict(run_lock_sha256='x',completed_updates=400,remaining_updates=400),dict(run_lock_sha256='wrong')]:
            with self.assertRaises(RuntimeError):check_restart(bad,'x','adam')
        with self.assertRaises(RuntimeError):check_restart(dict(run_lock_sha256='x',next_iteration=60,remaining_outer_iterations=0,history=[{}]*60,oracle_counts={'evaluations':12001}),'x','solver')
    def test_frozen_noise_bytes_and_covariance_formula(self):
        from development_information import validate_inputs
        import numpy as np
        x,y,c=validate_inputs();self.assertEqual(y.shape,(12,3));self.assertEqual(c.shape,(36,36))
        with np.load(BUNDLE/'inputs/CENTRAL_FORWARD.npz',allow_pickle=False) as f:
            self.assertTrue(np.array_equal(c,f['covariance']));self.assertFalse(np.array_equal(y,f['observed']))
    def test_truth_scoring_requires_sealed_selection(self):
        from score_forward import score
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):score(None,None,Path(td))
    def test_saved_noise_mutation_rejected(self):
        import development_information as module
        import shutil
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);shutil.copytree(BUNDLE/'inputs',root/'inputs')
            p=root/'inputs/PDF_NOISE_Z.npy';data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
            with patch.object(module,'BUNDLE',root),self.assertRaises(RuntimeError):module.validate_inputs()

if __name__=='__main__':unittest.main(verbosity=2)

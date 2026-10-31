"""Submission failure/duplicate guards; mocks every scheduler call."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import submit_attempt as submit

class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.control=self.root/'control';self.control.mkdir();self.registry=self.root/'registry';self.registry.mkdir()
        self.spec=self.control/'RUN_SPEC.json';self.spec.write_text(json.dumps(dict(case_registry=str(self.registry),resume_parent=None)))
        self.script=self.control/'RUN.sbatch';self.script.write_text('fixture never executed\n')
        (self.control/'PREPARATION.json').write_text(json.dumps(dict(submission_script_sha256=submit.sha(self.script))))
        (self.registry/'CLAIM.json').write_text(json.dumps(dict(submitted_job_id=None)))
        self.ready=self.root/'ready.txt';self.ready.write_text('mock live checks\n')
    def invoke(self,result=None,error=None):
        argv=['submit_attempt.py','--control',str(self.control),'--readiness',str(self.ready)]
        with patch.object(sys,'argv',argv),patch.object(submit,'validate'),patch.object(submit.subprocess,'run',return_value=result,side_effect=error) as call,contextlib.redirect_stdout(io.StringIO()):
            submit.main()
        return call
    def test_confirmed_once_only(self):
        r=subprocess.CompletedProcess([],0,'123;fixture\n','')
        self.assertEqual(self.invoke(r).call_count,1)
        self.assertEqual(json.loads((self.registry/'CLAIM.json').read_text())['submitted_job_id'],'123')
        with self.assertRaises(RuntimeError):self.invoke(r)
    def test_ambiguous_timeout_preserved_and_not_repeated(self):
        with self.assertRaises(subprocess.TimeoutExpired):self.invoke(error=subprocess.TimeoutExpired('mock sbatch',60))
        self.assertIn('Ambiguous',json.loads((self.control/'SUBMISSION.json').read_text())['action'])
        with self.assertRaises(RuntimeError):self.invoke(subprocess.CompletedProcess([],0,'999\n',''))
    def test_rejection_is_preserved(self):
        with self.assertRaises(RuntimeError):self.invoke(subprocess.CompletedProcess([],1,'','mock rejection'))
        self.assertEqual(json.loads((self.control/'SUBMISSION.json').read_text())['returncode'],1)
    def test_modified_batch_script_blocks_rpc(self):
        self.script.write_text('changed\n')
        with patch.object(submit.subprocess,'run') as rpc:
            with self.assertRaises(RuntimeError):self.invoke()
            rpc.assert_not_called()
        self.assertFalse((self.control/'SUBMISSION_STARTED.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)

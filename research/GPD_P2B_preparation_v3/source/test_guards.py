"""Fail-closed preparation/qualification guards; no scheduler or numerical fit."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import p2b_common as common
import qualify_joint
sys.path.insert(0, str(common.ROOT))
from verify_package import verify


class Guards(unittest.TestCase):
    valid = dict(SLURM_JOB_ID='TEST_ONLY', SLURM_JOB_NUM_NODES='1',
                 SLURM_CPUS_PER_TASK='1', SLURM_NTASKS='1', SLURM_MEM_PER_NODE='32768')

    def allocation(self, env, platform='linux'):
        with patch.dict(os.environ, env, clear=True), patch.object(sys, 'platform', platform):
            qualify_joint.allocation_guard()

    def test_nonlinux_rejected(self):
        with self.assertRaises(RuntimeError):
            self.allocation(self.valid, 'darwin')

    def test_no_allocation_rejected(self):
        with self.assertRaises(RuntimeError):
            self.allocation({})

    def test_extra_cpu_rejected(self):
        with self.assertRaises(RuntimeError):
            self.allocation(dict(self.valid, SLURM_CPUS_PER_TASK='2'))

    def test_extra_task_rejected(self):
        with self.assertRaises(RuntimeError):
            self.allocation(dict(self.valid, SLURM_NTASKS='2'))

    def test_low_memory_rejected(self):
        with self.assertRaises(RuntimeError):
            self.allocation(dict(self.valid, SLURM_MEM_PER_NODE='16384'))

    def test_declared_allocation_shape_accepted(self):
        self.allocation(self.valid)

    def test_payload_mutation_detected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root/'payload.txt').write_bytes(b'original')
            (root/'MANIFEST.json').write_text(json.dumps(dict(files=[dict(path='payload.txt',bytes=8,sha256=hashlib.sha256(b'original').hexdigest())])))
            self.assertEqual(verify(root), 1)
            (root/'payload.txt').write_bytes(b'changed!')
            with self.assertRaises(RuntimeError):
                verify(root)

    def test_prepared_information_binding(self):
        # The state interface is not called by the constructor.
        self.assertEqual(common.information(object()).observed.numel(), 36)

    def test_design_mutation_detected(self):
        import shutil
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            shutil.copytree(common.ROOT/'inputs', root/'inputs')
            (root/'DESIGN.json').write_text('{}\n')
            with patch.object(common, 'ROOT', root), self.assertRaises(RuntimeError):
                common.information(object())


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Standard-library tests of the supervisor using tiny nonscientific children."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import supervise
from p2b_policy import sha

class SupervisorTests(unittest.TestCase):
    def setUp(self):
        claim = patch.object(supervise, 'claim_case', lambda spec: None)
        claim.start(); self.addCleanup(claim.stop)

    def exercise(self, pressure=False, zero_exit=False, descendant=False, slow_copy=False):
        with tempfile.TemporaryDirectory(prefix='GPD_P2B_ops_') as td:
            root = Path(td)
            source = root/'scratch'/'case'
            source.mkdir(parents=True)
            target = root/'durable'/'case'
            (source/'checkpoint.pt').write_bytes(b'synthetic-checkpoint')
            worker = root/'launch.py'
            worker.write_text('import signal,time,sys,os\nfrom pathlib import Path\n'
                              'signal.signal(signal.SIGUSR1, lambda *a: sys.exit(75))\n'
                              'signal.signal(signal.SIGTERM, lambda *a: sys.exit(' + ('0' if zero_exit else '76') + '))\n'
                              + ('pid=os.fork()\nif pid==0:\n'
                                 ' signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                                 ' Path(' + repr(str(root/'descendant')) + ').write_text(str(os.getpid()))\n'
                                 ' time.sleep(10)\n sys.exit(0)\n'
                                 'while not Path(' + repr(str(root/'descendant')) + ').exists(): time.sleep(.01)\n'
                                 if descendant else '') +
                              'Path(' + repr(str(root/'ready')) + ').touch()\n'
                              'time.sleep(10)\n')
            copy_started = threading.Event()
            copy_release = threading.Event()
            calls = []
            def rss(pid):
                calls.append(pid)
                deadline = time.monotonic()+5
                while not (root/'ready').exists():
                    if time.monotonic()>deadline:
                        raise RuntimeError('Test child did not start')
                    time.sleep(.01)
                if slow_copy:
                    if len(calls) == 1:
                        return 1000
                    self.assertTrue(copy_started.wait(2))
                    copy_release.set()
                return 2**30 if pressure else 1000
            spec = {'output_root': str(root/'scratch'), 'persistent_root': str(root/'durable'),
                    'resources': {'reserve_GiB': .01, 'tree_rss_cap_GiB': .5,
                                  'stop_after_seconds': 100 if slow_copy else 0}}
            watch = types.SimpleNamespace(memory=lambda: {'available_bytes': 2**30, 'swap_used_bytes': 0}, tree_rss=rss)
            actual_retain = supervise.retain
            def retain(source, target, terminal=False):
                if slow_copy and not terminal:
                    copy_started.set()
                    if not copy_release.wait(3):
                        raise RuntimeError('Watchdog blocked behind provisional copy')
                if terminal and calls:
                    self.assertEqual(supervise.live_group_members(calls[0]), [])
                return actual_retain(source, target, terminal)
            with patch.object(supervise, 'validate', lambda *a, **kw: spec), \
                 patch.object(supervise, 'PACKAGE', root), patch.dict(sys.modules, {'resource_watch': watch}), \
                 patch.dict(os.environ, {'SLURM_JOB_ID': 'synthetic'}), \
                 patch.object(supervise, 'retain', retain), patch.object(supervise, 'POLL_SECONDS', .01), \
                 patch.object(supervise, 'COPY_INTERVAL', 0 if slow_copy else 300), \
                 patch.object(sys, 'argv', ['supervise', '--spec', 'synthetic', '--case', 'case', '--approval', 'synthetic']):
                result = supervise.main()
            self.assertEqual(result, 74 if pressure else 75)
            self.assertEqual(sha(source/'checkpoint.pt'), sha(target/'checkpoint.pt'))
            rec = json.loads((target/'SUPERVISOR_synthetic.json').read_text())
            self.assertEqual(rec['resource_stop'], 'TREE_RSS_LIMIT' if pressure else None)
            self.assertEqual(rec['child_exit'], (0 if zero_exit else 76) if pressure else 75)
            self.assertIsNone(rec['cleanup_error'])
            self.assertEqual(rec['provisional_copy_errors'], [])
            self.assertTrue(rec['terminal_retention_verified'])
            self.assertTrue(json.loads((target/'TERMINAL_RETENTION.json').read_text())['verified'])
            if slow_copy:
                self.assertGreaterEqual(len(calls), 2)
                self.assertFalse(json.loads((target/'provisional/PROVISIONAL.json').read_text())['coherent_restart_verified'])

    @unittest.skipUnless(sys.platform == 'linux', 'Linux /proc process-group test; required on receiver')
    def test_deadline_forwards_pause_and_retains(self):
        self.exercise()

    @unittest.skipUnless(sys.platform == 'linux', 'Linux /proc process-group test; required on receiver')
    def test_memory_stop_targets_owned_child_and_retains(self):
        self.exercise(pressure=True)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux /proc process-group test; required on receiver')
    def test_resource_stop_cannot_report_success(self):
        self.exercise(pressure=True, zero_exit=True)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux /proc process-group test; required on receiver')
    def test_descendant_cannot_write_during_terminal_retention(self):
        self.exercise(pressure=True, zero_exit=True, descendant=True)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux /proc process-group test; required on receiver')
    def test_live_copy_does_not_block_memory_watchdog(self):
        self.exercise(pressure=True, slow_copy=True)

    def test_exited_group_signal_race_is_harmless(self):
        with patch.object(supervise.os, 'killpg', side_effect=ProcessLookupError):
            supervise.signal_group(123456789, signal.SIGTERM)

    def test_signal_before_child_assignment_is_forwarded(self):
        with tempfile.TemporaryDirectory(prefix='GPD_P2B_signal_') as td:
            spec = {'output_root': td + '/scratch', 'persistent_root': td + '/durable',
                    'resources': {'reserve_GiB': .01, 'tree_rss_cap_GiB': .5, 'stop_after_seconds': 1}}
            watch = types.SimpleNamespace(memory=lambda: {'available_bytes': 2**30, 'swap_used_bytes': 0}, tree_rss=lambda p: 0)
            child = types.SimpleNamespace(pid=123456789, poll=lambda: 0)
            def spawn(*a, **kw):
                signal.getsignal(signal.SIGUSR1)(signal.SIGUSR1, None)
                return child
            with patch.object(supervise, 'validate', lambda *a, **kw: spec), \
                 patch.dict(sys.modules, {'resource_watch': watch}), \
                 patch.dict(os.environ, {'SLURM_JOB_ID': 'synthetic_signal'}), \
                 patch.object(supervise.subprocess, 'Popen', spawn), \
                 patch.object(supervise, 'signal_group') as forwarded, \
                 patch.object(supervise, 'stop_owned_group'), \
                 patch.object(sys, 'argv', ['supervise', '--spec', 'synthetic', '--case', 'case', '--approval', 'synthetic']):
                self.assertEqual(supervise.main(), 0)
            forwarded.assert_called_once_with(child.pid, signal.SIGUSR1)

if __name__ == '__main__':
    unittest.main(verbosity=2)

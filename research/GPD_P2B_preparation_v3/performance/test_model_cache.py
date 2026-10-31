"""Compare the cache to the actual legacy coordinate builder on small matrices."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from model_cache import ExactModelCache

parser = argparse.ArgumentParser()
parser.add_argument('--capsule', type=Path, required=True)
args, rest = parser.parse_known_args()
path = args.capsule/'project/completion_runs/NLO_constrained_solver/solver.py'
spec = importlib.util.spec_from_file_location('p2b_legacy_cache_reference', path)
legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.R = np.vstack((np.eye(5), np.sin(np.arange(15).reshape(3,5)+.3)))
        self.g = np.cos(np.arange(5)+.2)
        self.A = np.empty((0,5))
        self.settings = dict(legacy.SETTINGS)
        self.cache = ExactModelCache(legacy.model_coordinates)

    def check_original(self):
        result = self.cache(self.R,self.g,self.A,self.settings)
        original = legacy.model_coordinates(self.R,self.g,self.A,self.settings)
        for key in original:
            np.testing.assert_array_equal(result[key],original[key])
        return result

    def test_repeat_is_exact_hit(self):
        self.check_original(); self.check_original()
        self.assertEqual(self.cache.stats['hits'],1)

    def test_inplace_R_change_invalidates(self):
        self.check_original(); self.R[0,0] += .125; self.check_original()
        self.assertEqual(self.cache.stats['misses'],2)

    def test_gradient_change_invalidates(self):
        self.check_original(); self.g[0] += .1; self.check_original()
        self.assertEqual(self.cache.stats['misses'],2)

    def test_working_set_change_invalidates(self):
        self.check_original(); self.A = np.ones((1,5)); self.check_original()
        self.assertEqual(self.cache.stats['misses'],2)

    def test_rank_setting_change_invalidates(self):
        self.check_original(); self.settings['span_rank_tolerance']=1e-11; self.check_original()
        self.assertEqual(self.cache.stats['misses'],2)

    def test_caller_cannot_poison_cache(self):
        r=self.check_original()
        with self.assertRaises(ValueError): r['Q'][0,0]=42
        r['rank']=-1
        self.assertEqual(self.check_original()['rank'],5)


if __name__=='__main__':
    unittest.main(argv=[sys.argv[0]]+rest,verbosity=2)

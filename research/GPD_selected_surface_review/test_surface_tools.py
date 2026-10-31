import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from surface_tools import checked_relative, predict_numpy, residual_stats, signed_mean_arrays, replace_problem_targets, audit

ROOT = Path(__file__).resolve().parent

class MeanInputTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({'row_id':['a','b'], 'ReH_source':[2.,-3.], 'sigma_reported':[.5,2.]})
        self.arrays = {'row_id':np.array(['a','b']), 'map_id':np.array(['x','y']),
                       'y':np.array([-2.,3.]), 'sigma':np.array([.5,2.]), 't':np.array([-.1,-.2])}
    def test_internal_sign_once(self):
        out=signed_mean_arrays(self.arrays,self.rows,[4.,-5.]);np.testing.assert_array_equal(out['y'],[-4.,5.])
    def test_originals_unchanged(self):
        old={k:v.copy() for k,v in self.arrays.items()};out=signed_mean_arrays(self.arrays,self.rows,[4.,-5.]);out['t'][0]=999
        for k in old:np.testing.assert_array_equal(self.arrays[k],old[k])
    def test_nontarget_arrays_unchanged(self):
        out=signed_mean_arrays(self.arrays,self.rows,[4.,-5.])
        for k in self.arrays:
            if k!='y':np.testing.assert_array_equal(out[k],self.arrays[k])
    def test_row_order_rejected(self):
        a=copy.deepcopy(self.arrays);a['row_id']=np.array(['b','a'])
        with self.assertRaises(ValueError):signed_mean_arrays(a,self.rows,[4.,-5.])
    def test_bad_baseline_sign_rejected(self):
        a=copy.deepcopy(self.arrays);a['y']=-a['y']
        with self.assertRaises(ValueError):signed_mean_arrays(a,self.rows,[4.,-5.])
    def test_changed_sigma_rejected(self):
        a=copy.deepcopy(self.arrays);a['sigma']*=2
        with self.assertRaises(ValueError):signed_mean_arrays(a,self.rows,[4.,-5.])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):signed_mean_arrays(self.arrays,self.rows,[np.nan,1.])
    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):signed_mean_arrays(self.arrays,self.rows,[1.])
    def test_problem_adapter_changes_y_only(self):
        p=SimpleNamespace(y=torch.tensor([-2.,3.],dtype=torch.float64),sigma=torch.tensor([.5,2.],dtype=torch.float64),forward=SimpleNamespace(row_ids=('a','b')),other='unchanged')
        arr=signed_mean_arrays(self.arrays,self.rows,[4.,-5.]);replace_problem_targets(p,arr)
        np.testing.assert_array_equal(p.y.numpy(),[-4.,5.]);self.assertEqual(p.other,'unchanged')
    def test_problem_row_order_rejected(self):
        p=SimpleNamespace(y=torch.tensor([-2.,3.],dtype=torch.float64),sigma=torch.tensor([.5,2.],dtype=torch.float64),forward=SimpleNamespace(row_ids=('b','a')))
        with self.assertRaises(ValueError):replace_problem_targets(p,self.arrays)
    def test_problem_sigma_rejected(self):
        p=SimpleNamespace(y=torch.tensor([-2.,3.],dtype=torch.float64),sigma=torch.tensor([1.,2.],dtype=torch.float64),forward=SimpleNamespace(row_ids=('a','b')))
        with self.assertRaises(ValueError):replace_problem_targets(p,self.arrays)
    def test_residual_affine_identity(self):
        pred=np.array([.3,-.2]);arr=signed_mean_arrays(self.arrays,self.rows,[4.,-5.]);s=self.arrays['sigma']*np.sqrt(2)
        np.testing.assert_allclose((pred-arr['y'])/s-(pred-self.arrays['y'])/s,-(arr['y']-self.arrays['y'])/s)
    def test_jacobian_target_independence(self):
        A=torch.tensor([[1.,2.],[3.,-.5]],dtype=torch.float64);x=torch.tensor([.1,.2],dtype=torch.float64,requires_grad=True);sig=torch.tensor([.5,2.],dtype=torch.float64)
        def residual(y):return lambda q:(A@q-y)/sig/np.sqrt(2)
        j0=torch.autograd.functional.jacobian(residual(torch.tensor([1.,2.])),x)
        j1=torch.autograd.functional.jacobian(residual(torch.tensor([3.,4.])),x)
        self.assertTrue(torch.equal(j0,j1))
    def test_statistics(self):
        s=residual_stats([2.,-8.],[2.,2.]);self.assertEqual(s['count_abs_pull_gt_3'],1);self.assertAlmostEqual(s['rms_pull'],np.sqrt(8.5))
    def test_statistics_bad_sigma(self):
        with self.assertRaises(ValueError):residual_stats([1.,2.],[1.,0.])
    def test_statistics_bad_shape(self):
        with self.assertRaises(ValueError):residual_stats([1.,2.],[1.])
    def test_unsafe_path(self):
        with self.assertRaises(ValueError):checked_relative(ROOT,'../x')
    def test_missing_path(self):
        with self.assertRaises(ValueError):checked_relative(ROOT,'no_such_source')

if __name__=='__main__':unittest.main()

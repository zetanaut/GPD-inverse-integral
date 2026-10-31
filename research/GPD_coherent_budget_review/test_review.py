import unittest
import numpy as np
from replay_math import rms,response_metrics,drift_metrics,grid,scale,target_values,moments_check,poly_loss
from probe_active_constraints import directional_profile

class ResponseTests(unittest.TestCase):
    def test_identity(self):
        r=response_metrics([1.,2.,3.],[1.,2.,3.]);self.assertAlmostEqual(r['norm_ratio'],1);self.assertAlmostEqual(r['gain'],1)
    def test_reversal_is_not_recovery(self):
        r=response_metrics([1.,2.],[-1.,-2.]);self.assertAlmostEqual(r['norm_ratio'],1);self.assertAlmostEqual(r['cosine'],-1);self.assertLess(r['gain'],0)
    def test_orthogonal(self):
        r=response_metrics([1.,0.],[0.,1.]);self.assertEqual(r['gain'],0);self.assertAlmostEqual(r['norm_ratio'],1)
    def test_zero_input(self):self.assertIsNone(response_metrics([0.,0.],[1.,2.])['norm_ratio'])
    def test_zero_output(self):self.assertIsNone(response_metrics([1.,2.],[0.,0.])['cosine'])
    def test_matching_shape(self):
        with self.assertRaises(ValueError):response_metrics([1],[1,2])
    def test_rms_requires_finite(self):
        with self.assertRaises(ValueError):rms([1,np.nan])
    def test_rms_requires_nonempty(self):
        with self.assertRaises(ValueError):rms([])
    def test_not_difference_of_norms(self):self.assertAlmostEqual(drift_metrics([1,0],[0,1])['change_over_early_norm'],2**.5)
    def test_constant_budget(self):self.assertEqual(drift_metrics([1,2],[1,2])['change_rms'],0)
    def test_common_shift(self):
        a=np.array([2.,3.]);b=np.array([1.,1.]);c=np.array([4.,-1.]);np.testing.assert_allclose((a+c)-(b+c),a-b)
    def test_cohort_decomposition(self):
        x=np.arange(12,dtype=float).reshape(3,4);mu=x.mean(0);g=np.array([1.,-.5,2,7]);w=np.array([1,.5,2,3])
        self.assertAlmostEqual(np.mean(((g-x)*w)**2),np.mean(((g-mu)*w)**2)+np.mean(((x-mu)*w)**2))

class IntegralTests(unittest.TestCase):
    @staticmethod
    def field(x,xi,t):return np.stack([x*(1-x*x)**2,2*x*(1-x*x)**2,(1-x*x)**2],axis=1)
    def test_split_grid(self):
        x,w=grid(np.array([.1,.4,.8]),32);np.testing.assert_allclose(w.sum(1),1.,atol=1e-15)
    def test_noiseless_polynomial(self):self.assertLess(poly_loss(self.field,[-1.,-.3]),1e-24)
    def test_full_moments(self):
        r=moments_check(self.field,[-.3],32);self.assertEqual(len(r),16);self.assertLess(max(a['RMS'] for a in r),1e-11)
    def test_first_moment(self):
        a=target_values(self.field,.3,-.3,32);self.assertAlmostEqual(a[1],(2/3)*(1/3-2/5+1/7))
    def test_channel_scales(self):self.assertGreater(scale(2,2),scale(0,2))

class CurvatureTests(unittest.TestCase):
    def test_affine(self):
        p,d,c=directional_profile(lambda x:np.array([2*x-1.,-3*x-2.]))
        self.assertLess(d,1e-9);self.assertLess(max(np.max(abs(r['nonlinear_minus_affine'])) for r in p),1e-9)
    def test_norm_curvature(self):
        p,d,c=directional_profile(lambda x:np.array([np.hypot(1.,x)-1.]))
        self.assertAlmostEqual(p[-1]['affine'][0],0.);self.assertGreater(p[-1]['actual'][0],1.)
    def test_profile_cache(self):
        seen=[]
        def call(x):seen.append(x);return np.array([x])
        p,d,c=directional_profile(call);self.assertEqual(len(seen),len(set(seen)));self.assertEqual(len(seen),9)
    def test_nonfinite_diagnostic(self):
        with self.assertRaises(ValueError):directional_profile(lambda x:np.array([np.nan]))

if __name__=='__main__':unittest.main()

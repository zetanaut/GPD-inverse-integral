import unittest
import numpy as np
from first_order import tangent_direction,normal_residual
class Tests(unittest.TestCase):
 def test_empty(self):
  d,m=tangent_direction([3.,4.],np.empty((0,2)))
  np.testing.assert_allclose(d,[-.6,-.8]);self.assertEqual(m['rank'],0)
 def test_normal_removed(self):
  d,m=tangent_direction([3.,4.],[[1.,0.]])
  np.testing.assert_allclose(d,[0.,-1.]);self.assertLess(m['directional_derivative'],0)
 def test_dependent_rows(self):
  d,m=tangent_direction([3.,4.],[[1.,0.],[2.,0.]])
  self.assertEqual(m['rank'],1);np.testing.assert_allclose(d,[0.,-1.],atol=1e-14)
 def test_full_rank(self):
  d,m=tangent_direction([3.,4.],np.eye(2));np.testing.assert_array_equal(d,[0.,0.])
 def test_zero_rows(self):
  d,m=tangent_direction([0.,1.],[[0.,0.]])
  np.testing.assert_array_equal(d,[0.,-1.])
 def test_zero_gradient(self):
  d,m=tangent_direction([0.,0.],[[1.,0.]])
  np.testing.assert_array_equal(d,[0.,0.])
 def test_row_rescale(self):
  d,_=tangent_direction([3.,4.],[[1e9,0.]])
  np.testing.assert_allclose(d,[0.,-1.])
 def test_signed_multiplier(self):
  lam,res,ok=normal_residual([-2.,0.],[[1.,0.]])
  self.assertTrue(ok);np.testing.assert_allclose(lam,[2.]);np.testing.assert_allclose(res,[0.,0.])
 def test_wrong_sign_not_allowed(self):
  lam,res,ok=normal_residual([2.,0.],[[1.,0.]])
  self.assertTrue(ok);np.testing.assert_array_equal(lam,[0.]);np.testing.assert_array_equal(res,[2.,0.])
 def test_nan(self):
  with self.assertRaises(ValueError):tangent_direction([np.nan],[[1.]])
 def test_shapes(self):
  with self.assertRaises(ValueError):tangent_direction([1.,2.],[[1.]])
 def test_rank_tolerance(self):
  with self.assertRaises(ValueError):tangent_direction([1.],[[1.]],0)
if __name__=='__main__':unittest.main()

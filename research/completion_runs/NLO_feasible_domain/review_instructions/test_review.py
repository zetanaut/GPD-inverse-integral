import unittest
import numpy as np
from kkt_check import check_kkt

class KKTChecks(unittest.TestCase):
 def test_unconstrained_stationary(self):
  self.assertTrue(check_kkt([0,0],[],np.empty((0,2)))['pass'])
 def test_unconstrained_not_stationary(self):
  self.assertFalse(check_kkt([1,0],[],np.empty((0,2)))['pass'])
 def test_boundary_minimum_nonzero_raw_gradient(self):
  # min (x-2)^2 subject to x<=1, at x=1.
  d=check_kkt([-2],[0],[[1]])
  self.assertTrue(d['pass']);self.assertEqual(d['inequality_multipliers'],[2.])
 def test_wrong_sign_multiplier_does_not_pass(self):
  self.assertFalse(check_kkt([2],[0],[[1]])['pass'])
 def test_infeasible_even_if_stationary(self):
  self.assertFalse(check_kkt([0],[1e-12],[[1]])['pass'])
 def test_inactive_constraints_no_multiplier(self):
  d=check_kkt([-2],[-.1],[[1]])
  self.assertFalse(d['pass']);self.assertEqual(d['inequality_multipliers'],[0.])
 def test_equality_profile(self):
  # min (x-2)^2+(y+1)^2 at x=1, y=-1.
  d=check_kkt([-2,0],[],np.empty((0,2)),equality_values=[0],equality_jacobian=[[1,0]])
  self.assertTrue(d['pass']);self.assertEqual(d['equality_multipliers'],[2.])
 def test_bad_equality(self):
  self.assertFalse(check_kkt([0],[],np.empty((0,1)),equality_values=[1e-3],equality_jacobian=[[1]])['pass'])
 def test_redundant_constraints(self):
  d=check_kkt([-2],[0,0],[[1],[2]])
  self.assertTrue(d['pass']);self.assertLess(d['stationarity_L2'],1e-12)
 def test_constraint_rescaling(self):
  for scale in (1e-5,1,1e5):
   d=check_kkt([-2],[0],[[scale]])
   self.assertTrue(d['pass']);self.assertAlmostEqual(d['inequality_multipliers'][0]*scale,2.)
 def test_complementarity_not_only_active_threshold(self):
  d=check_kkt([-10],[-1e-7],[[1]],complementarity_tolerance=1e-8)
  self.assertFalse(d['pass']);self.assertFalse(d['gates']['complementarity'])
 def test_nonfinite(self):
  self.assertFalse(check_kkt([np.nan],[0],[[1]])['pass'])
 def test_malformed(self):
  with self.assertRaises(ValueError):check_kkt([1,2],[0],[[1]])
 def test_missing_equality_jacobian(self):
  with self.assertRaises(ValueError):check_kkt([0],[],np.empty((0,1)),equality_values=[0])
 def test_no_constraint_qualification_claim(self):
  d=check_kkt([0],[0],[[0]])
  self.assertIn('not sufficiency',d['qualification'])
 def test_no_mutation(self):
  g=np.array([-2.]);c=np.array([0.]);A=np.array([[1.]])
  old=[v.copy() for v in (g,c,A)];check_kkt(g,c,A)
  for a,b in zip((g,c,A),old):np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main(verbosity=2)

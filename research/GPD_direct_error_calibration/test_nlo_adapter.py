"""Small execution tests; only cached NLO arrays, no fitting or new data."""
import unittest, hashlib
from pathlib import Path
from unittest.mock import patch
import fit_nlo as f

class NLOTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with f.np.load(f.ROOT/'completion_runs/NLO_feasible_domain/inputs/d000.npz',allow_pickle=False) as d:cls.y=d['y'];cls.cov=d['covariance']
  cls.p=f.Problem(cls.y,cls.cov,31);cls.d=f.Domain()
 def test_original_network_size(self):self.assertEqual(self.p.x0.numel(),3867)
 def test_all_moment_columns_free(self):self.assertEqual(sum(N//2+1 for _,N,_ in self.p.p.spec),42)
 def test_constructor_reads_no_checkpoint(self):
  with patch.object(f.torch,'load',side_effect=AssertionError('checkpoint read forbidden')):
   p=f.Problem(self.y,self.cov,43)
  self.assertEqual(p.x0.numel(),3867)
 def test_initialization_independent_of_observations(self):
  p=f.Problem(self.y+0.001,self.cov,31)
  self.assertTrue(f.torch.equal(p.x0,self.p.x0))
 def test_repeated_seed(self):self.assertTrue(f.torch.equal(f.Problem(self.y,self.cov,31).x0,self.p.x0))
 def test_separate_seeds(self):self.assertFalse(f.torch.equal(f.Problem(self.y,self.cov,43).x0,self.p.x0))
 def test_same_objective(self):
  x=self.p.x0
  a=float(self.p.objective(x));b=float(self.p.residual(x).square().sum());c=float(self.p.independent_objective(x))
  self.assertAlmostEqual(a,b,places=10);self.assertAlmostEqual(a,c,places=10)
 def test_independent_gradient(self):
  x=self.p.x0.clone().requires_grad_(True);g=f.torch.autograd.grad(self.p.objective(x),x)[0]
  d=f.torch.sin(f.torch.arange(len(x),dtype=f.torch.float64));d/=f.torch.linalg.vector_norm(d)
  exact=float(g@d)
  for h in (1e-4,1e-5,1e-6):
   fd=float((self.p.objective(x.detach()+h*d)-self.p.objective(x.detach()-h*d))/(2*h))
   self.assertLess(abs(fd-exact)/max(1.,abs(exact)),1e-5)
 def test_parity(self):
  x=f.tensor([[-.7,-.2,0.,.2,.7]]);h=self.p.field(self.p.x0,x,f.tensor([.275])).detach().numpy()[0]
  f.np.testing.assert_allclose(h[:2],-h[:2,::-1],atol=1e-14);f.np.testing.assert_allclose(h[2],h[2,::-1],atol=1e-14)
 def test_target_quadrature(self):
  a=f.targets(self.p,self.p.x0,96);b=f.targets(self.p,self.p.x0,192)
  f.np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
 def test_head_restoration(self):
  a,log=f.restore_head(self.p,self.d,self.p.x0)
  self.assertIsNotNone(a);self.assertLessEqual(float(self.d.constraints(self.p,a).max()),0.)
  i=0
  for n,_,s in self.p.spec:
   if n not in ('output.weight','output.bias'):self.assertTrue(f.torch.equal(a[i:i+s],self.p.x0[i:i+s]))
   i+=s
 def test_bad_geometry_rejected(self):
  with self.assertRaises(ValueError):f.Problem(self.y[:2],self.cov[:2,:2],31)
 def test_bad_covariance_rejected(self):
  with self.assertRaises((ValueError,RuntimeError)):f.Problem(self.y,-self.cov,31)

if __name__=='__main__':unittest.main(verbosity=2)

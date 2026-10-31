import unittest
import numpy as np
from residual_subproblem import solve_model
from moment_linearization import norm_model

class LocalModels(unittest.TestCase):
    def base(self, target, e=None, E=None, radius=3, A=None,c=None,Q=None):
        target=np.asarray(target,float);n=len(target)
        return solve_model(-2*target,np.eye(n),np.eye(n) if Q is None else Q,
            np.zeros((0,n)) if A is None else np.asarray(A,float),
            np.zeros(0) if c is None else np.asarray(c,float),
            np.zeros(n) if e is None else np.asarray(e,float),
            np.eye(n) if E is None else np.asarray(E,float),radius)
    def test_norm_value(self):
        v,g=norm_model(np.array([.6,.8]),np.eye(2),np.zeros(2))
        self.assertEqual(v,0);np.testing.assert_allclose(g,[.6,.8])
    def test_norm_gradient(self):
        rng=np.random.default_rng(17);e=rng.normal(size=5);E=rng.normal(size=(5,3));s=rng.normal(size=3)
        v,g=norm_model(e,E,s);h=1e-5
        fd=np.array([(norm_model(e,E,s+h*np.eye(3)[i])[0]-norm_model(e,E,s-h*np.eye(3)[i])[0])/(2*h) for i in range(3)])
        np.testing.assert_allclose(g,fd,rtol=1e-8,atol=1e-9)
    def test_zero_subgradient(self):
        v,g=norm_model(np.zeros(2),np.eye(2),np.zeros(2));self.assertEqual(v,-1);np.testing.assert_array_equal(g,0)
    def test_shape(self):
        with self.assertRaises(ValueError):norm_model(np.zeros(2),np.eye(2),np.zeros(3))
    def test_nonfinite_norm(self):
        with self.assertRaises(ValueError):norm_model(np.array([np.nan]),np.ones((1,1)),np.ones(1))
    def test_curvature_retained(self):
        # Scalar linearization permits this tangent step; residual model rejects it.
        e=np.array([1.,0.]);s=np.array([0.,.1]);v,g=norm_model(e,np.eye(2),np.zeros(2))
        self.assertEqual(v+g@s,0);self.assertGreater(norm_model(e,np.eye(2),s)[0],0)
    def test_circle_solution(self):
        s,r=self.base([0.,1.],e=[1.,0.],E=np.eye(2),radius=2)
        expected=np.array([-1+1/np.sqrt(2),1/np.sqrt(2)])
        self.assertTrue(r['optimizer_success']);np.testing.assert_allclose(s,expected,atol=2e-6)
        self.assertLess(r['moment_norm_constraint'],1e-8)
    def test_one_dimensional_boundary(self):
        s,r=self.base([2.]);np.testing.assert_allclose(s,[1.],atol=1e-8)
    def test_additional_linear_bound(self):
        s,r=self.base([2.],A=[[1.]],c=[-.2]);np.testing.assert_allclose(s,[.2],atol=1e-8)
    def test_trust_radius(self):
        s,r=self.base([2.,1.],E=np.zeros((2,2)),radius=.1)
        np.testing.assert_allclose(s,.1*np.array([2,1])/np.sqrt(5),atol=1e-7)
    def test_interior(self):
        s,r=self.base([.2,.1]);np.testing.assert_allclose(s,[.2,.1],atol=1e-7)
    def test_coordinate_rotation(self):
        q=np.array([[0.,-1.],[1.,0.]])
        a,_=self.base([.5,.7],radius=.2);b,_=self.base([.5,.7],radius=.2,Q=q)
        np.testing.assert_allclose(a,b,atol=1e-8)
    def test_original_input_not_mutated(self):
        e=np.array([.1,.2]);E=np.eye(2);ec=e.copy();Ec=E.copy();self.base([.3,.1],e=e,E=E)
        np.testing.assert_array_equal(e,ec);np.testing.assert_array_equal(E,Ec)
    def test_bad_Q(self):
        with self.assertRaises(ValueError):self.base([0.,0.],Q=np.ones((2,2)))
    def test_bad_Q_shape(self):
        with self.assertRaises(ValueError):self.base([0.,0.],Q=np.ones(2))
    def test_infeasible_base(self):
        with self.assertRaises(ValueError):self.base([0.],e=[2.])
        with self.assertRaises(ValueError):self.base([0.],A=[[1.]],c=[.01])
    def test_bad_radius(self):
        with self.assertRaises(ValueError):self.base([0.],radius=0.)
    def test_nan(self):
        with self.assertRaises(ValueError):self.base([np.nan])
if __name__=='__main__':unittest.main(verbosity=2)

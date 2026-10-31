import unittest,json
import numpy as np
from pathlib import Path
from solver import solve,positive_qp,SETTINGS
from problem import kkt

class Analytic:
    def __init__(self,target,start,kind):
        self.target=np.array(target,float);self.x0=np.array(start,float);self.kind=kind
        self.names=['circle' if kind=='circle' else 'upper'];self.counts={'evaluations':0};self.timers={}
    def values(self,x):
        self.counts['evaluations']+=1;r=x-self.target
        c=np.array([x@x-1 if self.kind=='circle' else x[0]-1])
        return {'J':float(r@r),'components':np.array([r@r,0.]),'residual':r,'c':c}
    def derivatives(self,x,rows):
        A=np.array([2*x if self.kind=='circle' else np.eye(len(x))[0]])[rows]
        return 2*(x-self.target),np.eye(len(x)),A
    def constraint_rows(self,x,rows):return self.derivatives(x,rows)[2]

class Controls(unittest.TestCase):
    def run_problem(self,oracle):
        histories=[];x,result=solve(oracle,kkt,lambda x,r,t:histories.append(r))
        v=oracle.values(x);idx=np.flatnonzero(v['c']>=-.01);g,R,A=oracle.derivatives(x,idx)
        check=kkt(g,v['c'],A,idx,oracle.names)
        self.assertTrue(check['pass'],(x,result['status'],check))
        self.assertEqual(result['status'],'original_coordinate_KKT_pass')
        self.assertTrue(all(t['feasible'] for r in histories for t in r['trials'] if t['accepted']))
        return x,result,check
    def test_active_boundary_nonzero_raw_gradient(self):
        x,r,k=self.run_problem(Analytic([2.],[0.],'linear'))
        self.assertAlmostEqual(x[0],1.,places=8);self.assertGreater(k['raw_gradient_L2'],1.9)
    def test_curved_boundary(self):
        x,r,k=self.run_problem(Analytic([2.,0.],[.5,.2],'circle'))
        np.testing.assert_allclose(x,[1.,0.],atol=1e-6)
        self.assertGreater(r['counts']['SOC_linear_solves'],0)
    def test_interior_solution_and_multiplier_sign(self):
        x,r,k=self.run_problem(Analytic([-1.],[1.],'linear'))
        self.assertLess(abs(x[0]+1),1e-7)
    def test_infeasible_is_not_converged(self):
        p=Analytic([2.],[2.],'linear');x,r=solve(p,kkt,lambda *a:None)
        self.assertEqual(r['status'],'infeasible_initial')
    def test_wrong_sign_boundary_multiplier_rejected(self):
        d=kkt(np.array([2.]),np.array([0.]),np.array([[1.]]),np.array([0]),['c'])
        self.assertFalse(d['pass'])
    def test_primal_QP(self):
        x,lam,r=positive_qp(np.array([2.,2.]),np.array([-4.,-6.]),np.eye(2),np.array([1.,1.]))
        self.assertTrue(r['success']);np.testing.assert_allclose(x,[1.,1.]);np.testing.assert_allclose(lam,[2.,4.])
    def test_QP_redundant_rows(self):
        x,lam,r=positive_qp(np.array([2.]),np.array([-4.]),np.array([[1.],[1.]]),np.array([1.,1.]))
        self.assertTrue(r['success']);self.assertAlmostEqual(x[0],1.)
    def test_curved_directional_derivatives(self):
        p=Analytic([2.,0.],[.4,.3],'circle');x=p.x0;direction=np.array([.6,-.8]);eps=1e-5
        g,R,A=p.derivatives(x,np.array([0]));plus=p.values(x+eps*direction);minus=p.values(x-eps*direction)
        self.assertLess(abs((plus['J']-minus['J'])/(2*eps)-g@direction),1e-9)
        np.testing.assert_allclose((plus['c']-minus['c'])/(2*eps),A@direction,atol=1e-9)
    def test_unchanged_state_stop(self):
        # An inconsistent derivative oracle is only a termination control.
        p=Analytic([2.],[0.],'linear')
        orig=p.values
        def values(x):
            d=orig(x);d['J']=4.;return d
        p.values=values
        settings={**SETTINGS,'maximum_trials':1,'minimum_radius':1e-100}
        x,r=solve(p,kkt,lambda *a:None,settings)
        self.assertEqual(r['status'],'three_unchanged_state_terminations')
        self.assertEqual(len(r['history']),3);np.testing.assert_array_equal(x,p.x0)
    def test_no_penalty_or_historical_restarts(self):
        import ast
        code=ast.parse((Path(__file__).parent/'solver.py').read_text())
        for node in ast.walk(code):
            if isinstance(node,ast.Name):self.assertNotIn(node.id,('run_al','Adam','LBFGS','truth_values'))

if __name__=='__main__':unittest.main(verbosity=2)

import ast,unittest
import numpy as np
from pathlib import Path
from optimizer import run_al,SETTINGS,necessary_kkt
from kkt_check import check_kkt

class AnalyticOracle:
    n_constraints=1
    def set_augmented(self,lam,rho):self.lam=lam;self.rho=rho
    def __call__(self,x,weights):
        j=(x[0]-2)**2;c=np.array([x[0]-1]);g=np.array([2*(x[0]-2)])
        return j,c,g,g+weights
    def augmented(self,x):
        j,c,g,_=self(x,[0.]);v=np.maximum(0,self.lam+self.rho*c)
        return j+np.sum(v*v-self.lam*self.lam)/(2*self.rho),g+v

class Mechanics(unittest.TestCase):
    def test_analytic_algorithm_active_boundary(self):
        # Algorithm control only; it is not a third neural optimization.
        x,r=run_al([0.],AnalyticOracle(),lambda *a:None)
        self.assertLess(abs(x[0]-1),1e-7)
        self.assertGreater(abs(2*(x[0]-2)),1.9)
        d=check_kkt([-2.],[0.],[[1.]])
        self.assertTrue(d['pass']);self.assertEqual(d['inequality_multipliers'],[2.])
    def test_infeasible_stationary_rejected(self):
        d=necessary_kkt(np.array([0.]),np.array([1e-12]),np.array([[1.]]),np.array([0]),['c'])
        self.assertFalse(d['pass'])
    def test_inactive_constraints_not_fake_derivatives(self):
        d=necessary_kkt(np.zeros(2),np.array([-.1]),np.empty((0,2)),np.array([],int),['c'])
        self.assertTrue(d['pass']);self.assertEqual(d['evaluated_jacobian_indices'],[])
    def test_rank_failure_reported(self):
        d=necessary_kkt(np.array([-2.]),np.zeros(2),np.array([[1.],[2.]]),np.arange(2),['a','b'])
        self.assertFalse(d['pass']);self.assertFalse(d['constraint_qualification']['LICQ_for_active_filter_rows'])
    def test_old_scoring_wrapper_is_excluded(self):
        for name in ('domain.py','optimizer.py','run.py'):
            tree=ast.parse((Path(__file__).parent/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    strings=[a.name for a in node.names]+[getattr(node,'module','') or '']
                    self.assertFalse(any('profile_protocol' in s or s=='adapter' for s in strings))
                if isinstance(node,ast.Name):self.assertNotIn(node.id,('point_score','ActualPointScorer'))

if __name__=='__main__':unittest.main(verbosity=2)

"""Check portable PDF math and cache invalidation against independent calculations."""
import unittest
import numpy as np
import torch
from gpd_tools.pdf_information import ForwardInformation, map_xf_to_stored
from gpd_tools.model_cache import ExactModelCache
from gpd_tools.solver import model_coordinates, SETTINGS

class LinearField:
    def field(self, flat, x, xi, t):
        return flat.reshape(1, 3, 1) * (1 + x[:, None, :])

class PDFTests(unittest.TestCase):
    def test_xf_convention(self):
        x=np.array([.1,.3])
        out=map_xf_to_stored(x,2*x,x,3*x,2*x,4*x,3*x,5*x)
        np.testing.assert_allclose(out,np.stack([15+0*x,-3+0*x,5*x],axis=-1))

    def test_correlated_whitening_and_derivative(self):
        x=np.geomspace(.02,.8,12)
        A=np.eye(36)+.02*np.ones((36,36))
        cov=A@A.T
        info=ForwardInformation(LinearField(),x,np.zeros((12,3)),cov)
        flat=torch.tensor([.1,.2,-.3],dtype=torch.float64)
        delta=info.prediction(flat).numpy()
        expected=delta@np.linalg.solve(cov,delta)/160
        self.assertAlmostEqual(float(info.residual(flat).square().sum()),expected,places=13)
        direction=torch.tensor([.3,-.2,.7],dtype=torch.float64)
        h=1e-6
        fd=(info.residual(flat+h*direction)-info.residual(flat-h*direction))/(2*h)
        torch.testing.assert_close(info.jacobian(flat)@direction,fd,atol=1e-10,rtol=1e-8)

    def test_invalid_covariance_rejected(self):
        with self.assertRaises(ValueError):
            ForwardInformation(LinearField(),np.geomspace(.02,.8,12),np.zeros((12,3)),np.zeros((36,36)))

class CacheTests(unittest.TestCase):
    def test_exact_repeat_and_mutation_invalidation(self):
        R=np.vstack([np.eye(5),np.sin(np.arange(15).reshape(3,5)+.3)])
        g=np.cos(np.arange(5)+.2);A=np.empty((0,5));settings=dict(SETTINGS)
        cached=ExactModelCache(model_coordinates)
        for step in range(4):
            if step==2:R[0,0]+=.125
            if step==3:settings['span_rank_tolerance']=1e-11
            got=cached(R,g,A,settings);expected=model_coordinates(R,g,A,settings)
            for key in expected:np.testing.assert_array_equal(got[key],expected[key])
        self.assertEqual(cached.stats['hits'],1)
        self.assertEqual(cached.stats['misses'],3)
        with self.assertRaises(ValueError):got['Q'][0,0]=42

if __name__=='__main__':unittest.main()

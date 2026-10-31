"""Field-only integration: frozen noisy input, whitening and derivative consistency."""
import sys
import unittest
from p2b_policy import CAPSULE,PACKET,BUNDLE,INPUT_TASK,sha
sys.path.insert(0,str(CAPSULE/'runtime'))
sys.path.insert(0,str(PACKET/'source'))

class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import core
        core.setup()
        from p2b_common import field_only
        from development_information import information
        cls.base=field_only(core);cls.info=information(cls.base)
    def test_original_cff_bytes_unchanged(self):
        original=CAPSULE/'project/GPD_operating_range_plan/prepared'/f'{INPUT_TASK}.npz'
        self.assertEqual(sha(original),sha(BUNDLE/'inputs/CFF_HALF.npz'))
    def test_whitening_and_derivatives(self):
        import math
        import numpy as np
        import torch
        from torch.func import jvp
        from development_information import validate_inputs
        x,y,cov=validate_inputs();flat=self.base.x0.detach().clone().requires_grad_(True)
        prediction=self.info.prediction(flat).detach().numpy()
        expected=np.linalg.solve(np.linalg.cholesky(cov),prediction-y.reshape(-1))/math.sqrt(160)
        residual=self.info.residual(flat);jac=self.info.jacobian(flat)
        self.assertEqual(tuple(jac.shape),(36,3867));self.assertTrue(torch.isfinite(jac).all())
        np.testing.assert_allclose(residual.detach().numpy(),expected,rtol=2e-12,atol=2e-12)
        gradient=torch.autograd.grad(residual.square().sum(),flat)[0]
        torch.testing.assert_close(gradient,2*jac.T@residual.detach(),rtol=2e-11,atol=2e-10)
        direction=torch.sin(torch.arange(flat.numel(),dtype=torch.float64));direction/=direction.norm()
        automatic=jvp(self.info.residual,(flat.detach(),),(direction,))[1]
        torch.testing.assert_close(jac@direction,automatic,rtol=2e-11,atol=2e-11)
        eps=1e-5
        difference=(self.info.residual(flat.detach()+eps*direction)-self.info.residual(flat.detach()-eps*direction))/(2*eps)
        torch.testing.assert_close(automatic,difference,rtol=2e-6,atol=2e-7)
        print('Field-only noisy adapter verified: residual 36; Jacobian 36 x 3867; no NLO map or optimizer invoked.')

if __name__=='__main__':unittest.main(verbosity=2)

"""Mechanical checks of this export; not statistical experiments."""
import json, tempfile, unittest
from pathlib import Path
import numpy as np
import closeout as c

class CloseoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.weights,cls.meta=c.load_weights(c.ROOT)
    def test_source_members(self): self.assertEqual(c.source_check(c.ROOT),30)
    def test_source_identity(self):
        contract=json.loads((c.ROOT/'evidence/original_contract.json').read_text())
        self.assertEqual(c.digest(c.ROOT/'evidence/final_summary_by_DNN.csv'),contract['input']['sha256'])
    def test_parameter_count_and_original_checkpoints(self):
        for s in c.STATES:self.assertEqual(sum(a.size for a in self.weights[s].values()),3867)
    def test_nominated_central_is_not_reselected(self): self.assertEqual(c.CENTRAL,'central13')
    def test_parity_endpoints(self):
        x=np.array([[-1.,-.7,-.2,0.,.2,.7,1.]])
        for s in c.STATES:
            a=c.field(self.weights[s],x,np.array([.275]),np.array([-.363]),self.meta[s]['t_mean'],self.meta[s]['t_std'])
            np.testing.assert_allclose(a[:,:2,:],-a[:,:2,::-1],atol=5e-15)
            np.testing.assert_allclose(a[:,2:,:],a[:,2:,::-1],atol=5e-15)
            np.testing.assert_array_equal(a[:,:,[0,-1]],np.zeros((1,3,2)))
    def test_domain_and_shapes(self):
        s=c.CENTRAL;m=self.meta[s]
        with self.assertRaises(ValueError):c.field(self.weights[s],[[0.,1.1]],[.2],[-.3],m['t_mean'],m['t_std'])
        with self.assertRaises(ValueError):c.field(self.weights[s],[0.,.1],[.2],[-.3],m['t_mean'],m['t_std'])
    def test_weighted_residual_definition(self):
        r=c.residual_summary([2.,3.],[1.,5.],[.5,2.]);self.assertEqual(r['weighted_residual_sum'],5.)
        self.assertAlmostEqual(r['RMS_error_units'],np.sqrt(2.5))
    def test_no_sigma_inflation_or_zero_error(self):
        with self.assertRaises(ValueError):c.residual_summary([1.],[0.],[0.])
        with self.assertRaises(ValueError):c.residual_summary([1.],[0.],[-1.])
    def test_nonfinite_not_silently_dropped(self):
        with self.assertRaises(ValueError):c.residual_summary([1.,np.nan],[0.,0.],[1.,1.])
    def test_signed_response_identity(self):
        p=np.array([1.,2.]);z=np.array([2.,4.]);plus=np.array([5.,7.]);minus=np.array([3.,2.])
        d,o,e=c.decompose(p,z,plus,minus)
        np.testing.assert_array_equal(p+d+e+o,plus);np.testing.assert_array_equal(p+d+e-o,minus)
    def test_common_mode_not_labeled_noise(self):
        d,o,e=c.decompose([0.],[2.],[2.],[2.]);self.assertEqual(d[0],2.);self.assertEqual(o[0],0.);self.assertEqual(e[0],0.)
    def test_original_observations_and_coordinates(self):
        obs=c.read_csv(c.ROOT/'evidence/observations.csv');rows=c.normalize_predictions(c.ROOT,obs)
        self.assertEqual(len(rows),960)
        self.assertEqual(len({(r['state'],r['row_id']) for r in rows}),960)
    def test_gaussian_quadrature_linear_identity(self):
        x,w=c.gquad(.2,.4,.275,24)
        self.assertAlmostEqual(float(w.sum()),.2,places=14)
        self.assertAlmostEqual(float(w@x),.06,places=14)
    def test_frozen_results_are_not_default_destination(self):
        with self.assertRaises(ValueError):c.main(c.ROOT/'results')
    def test_schema_has_no_invented_confidence_band(self):
        obj=json.loads((c.ROOT/'results/UNCERTAINTY_STATUS.json').read_text())
        self.assertIsNone(obj['Hplus_confidence_band']);self.assertIsNone(obj['total_GPD_uncertainty'])
        self.assertFalse(obj['posterior'])

if __name__=='__main__':unittest.main()

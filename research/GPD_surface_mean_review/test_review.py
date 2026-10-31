import unittest
import numpy as np
from replica_tools import draw_table,summarize,comparison,source_to_internal
from audit import field,split_nodes,moment_scale,polynomial_matrix

class ReplicaHelpers(unittest.TestCase):
    def test_repeatable(self):
        a,z=draw_table([1,2],[.1,.4],4);b,w=draw_table([1,2],[.1,.4],4)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(z,w)
    def test_call_order_independent(self):
        a,_=draw_table([1,2],[.1,.4],1);draw_table([1,2],[.1,.4],9)
        b,_=draw_table([1,2],[.1,.4],1);np.testing.assert_array_equal(a,b)
    def test_scale_once(self):
        y=np.array([1.,-2.]);s=np.array([.3,.4]);a,z=draw_table(y,s,3)
        np.testing.assert_allclose(a-y,s*z,rtol=0,atol=5e-16)
    def test_bad_sigma(self):
        for s in [[0,.4],[-.1,.4],[np.nan,.2]]:
            with self.assertRaises(ValueError):draw_table([1,2],s,0)
    def test_unique_ids(self):
        a,_=draw_table([0]*8,[1]*8,1);b,_=draw_table([0]*8,[1]*8,2)
        self.assertFalse(np.array_equal(a,b))
    def test_sign(self):
        np.testing.assert_array_equal(source_to_internal([1,2]),[-1,-2])
    def test_coherent_covariance(self):
        m=np.array([[1,2,3],[2,4,6],[3,6,9.]]);s=summarize(m)
        np.testing.assert_allclose(s['covariance'],np.cov(m,rowvar=False))
        np.testing.assert_allclose(s['anomaly_factor']@s['anomaly_factor'].T,s['covariance'])
    def test_constant_member_coordinate(self):
        s=summarize([[2,1],[2,2],[2,3]]);self.assertEqual(s['std'][0],0)
    def test_no_dropped_nan(self):
        with self.assertRaises(ValueError):summarize([[1,2],[np.nan,3]])
    def test_no_one_replica_uncertainty(self):
        with self.assertRaises(ValueError):summarize([[1,2]])
    def test_weights_unchanged(self):
        d=comparison([2,6],[0,0],[2,3]);self.assertAlmostEqual(d['RMS_original_sigma'],np.sqrt(2.5))
    def test_invalid_comparison(self):
        with self.assertRaises(ValueError):comparison([1],[2,3],[1])

class InferenceControls(unittest.TestCase):
    @staticmethod
    def model():
        rng=np.random.default_rng(47);w={}
        sizes={'input':(24,1),'blocks.0':(24,24),'blocks.1':(24,24),'output':(3,24)}
        for j in range(2): sizes.update({f'conditioners.{j}.0':(24,2),f'conditioners.{j}.2':(48,24)})
        for k,(b,a) in sizes.items():w[k+'.weight']=rng.normal(0,.05,(b,a));w[k+'.bias']=rng.normal(0,.05,b)
        return w,{'t_mean':-.3,'t_std_population':.2}
    def test_parity_and_endpoints(self):
        w,m=self.model();x=np.linspace(-1,1,31)[None];h=field(w,x,[.2],[-.3],m)
        np.testing.assert_allclose(h[0,:2],-h[0,:2,::-1],atol=2e-16)
        np.testing.assert_allclose(h[0,2],h[0,2,::-1],atol=2e-16)
        np.testing.assert_array_equal(h[:,:,[0,-1]],0)
    def test_output_homogeneity(self):
        w,m=self.model();x=np.array([[-.8,-.3,.1,.7]]);h=field(w,x,[.2],[-.3],m)
        v={k:a.copy() for k,a in w.items()};v['output.weight']*=2;v['output.bias']*=2
        np.testing.assert_allclose(field(v,x,[.2],[-.3],m),2*h,atol=1e-16)
    def test_split_quadrature(self):
        x,w=split_nodes(np.array([.1,.5,.9]),16)
        np.testing.assert_allclose(w.sum(axis=1),1,atol=2e-15)
        np.testing.assert_allclose((w*x**4).sum(axis=1),.2,atol=2e-15)
    def test_free_polynomial_space(self):
        x=np.linspace(.02,.98,41);P=polynomial_matrix(x,8);a=np.array([1.,-.2,.3,-.1,.04])
        np.testing.assert_allclose(P@np.linalg.lstsq(P,P@a,rcond=None)[0],P@a,atol=2e-14)
    def test_gluon_moment_scale(self):
        self.assertNotEqual(moment_scale(0,2),moment_scale(2,2));self.assertEqual(moment_scale(0,2),moment_scale(1,2))

if __name__=='__main__':unittest.main()

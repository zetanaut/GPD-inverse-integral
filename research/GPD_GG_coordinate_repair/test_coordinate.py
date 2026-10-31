from pathlib import Path
import ast,ctypes as ct,json,tempfile,unittest
from unittest.mock import Mock
import numpy as np
import parent_policy as p
from mapped_local import Reference
from prepare_resume import transform,SOURCE_SHA,sha

HERE=Path(__file__).parent
class Certificate(unittest.TestCase):
    def ref(self,values):
        r=Reference.__new__(Reference);r.integral=Mock(side_effect=values);r.count_native=Mock(return_value=498);return r
    def test_nodes_unchanged(self):self.assertEqual(p.NODES,(24,48,96))
    def test_limit_unchanged(self):self.assertEqual(p.TOLERANCE,1e-9)
    def test_cert_success(self):
        r=self.ref([np.array([.15,.5]),np.array([.15+1e-11,.5]),np.array([.15+2e-11,.5])])
        self.assertEqual(r.certify(.50500000331,.505)['node_counts'],[24,48,96]);self.assertEqual(r.integral.call_count,3)
    def test_first_difference_must_pass(self):
        r=self.ref(np.array([[.15,.5],[.15+1e-7,.5],[.15+1e-7,.5]]))
        with self.assertRaises(FloatingPointError):r.certify(.50500000331,.505)
    def test_second_difference_must_pass(self):
        r=self.ref(np.array([[.15,.5],[.15,.5],[.15+1e-7,.5]]))
        with self.assertRaises(FloatingPointError):r.certify(.50500000331,.505)
    def test_nonfinite_rejected(self):
        r=self.ref(np.array([[.15,.5],[.15,.5],[float('nan'),.5]]))
        with self.assertRaises(FloatingPointError):r.certify(.51,.505)
    def test_explicit_term_counted_once(self):
        r=self.ref(np.array([[.1,.2]]*3));v=r.certify(.51,.505)
        self.assertAlmostEqual(v['local'],.1+3*(.2+p.explicit_per_flavor()))
    def test_exact_crossover_rejected(self):
        with self.assertRaises(ValueError):p.check_domain(.505,.505)
    def test_roundoff_crossover_rejected(self):
        with self.assertRaises(ValueError):p.check_domain(np.nextafter(.505,1),.505)
    def test_near_but_resolved_allowed(self):p.check_domain(.505+1e-10,.505)
    def test_zero_rejected(self):
        with self.assertRaises(ValueError):p.check_domain(0,.505)
    def test_endpoint_rejected(self):
        with self.assertRaises(ValueError):p.check_domain(1,.505)
    def test_invalid_panel_coordinates(self):
        r=Reference.__new__(Reference)
        with self.assertRaises(ValueError):r.mapped_panel(.51,.505,0,[0])
    def test_native_rejection_propagates(self):
        r=Reference.__new__(Reference);r.ptr=ct.POINTER(ct.c_double);r.panel_native=Mock(return_value=4)
        with self.assertRaises(FloatingPointError):r.mapped_panel(.51,.505,0,[.5])

class NestedFallback(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
    def wrap(self,source,ref,tag='new'):return p.FallbackSource(source,ref,{'tag':tag},Path(self.tmp.name)/tag)
    def test_success_unchanged(self):
        a=(np.array([1.]),np.array([2.]),3.);s=Mock();s.evaluate.return_value=a;r=Mock()
        self.assertIs(self.wrap(s,r).evaluate(1,'GG',.51,.505,[.8],True),a);r.certify.assert_not_called()
    def test_nonlocal_bypass(self):
        s=Mock();r=Mock();self.wrap(s,r).evaluate(1,'GG',.51,.505,[.8],False);r.certify.assert_not_called()
    def test_LO_bypass(self):
        s=Mock();r=Mock();self.wrap(s,r).evaluate(0,'GG',.51,.505,[.8],True);r.certify.assert_not_called()
    def test_nonGG_bypass(self):
        s=Mock();r=Mock();self.wrap(s,r).evaluate(1,'QG',.51,.505,[.8],True);r.certify.assert_not_called()
    def test_failure_chain_and_new_cache(self):
        s=Mock()
        def source(*args):
            if args[-1]:raise FloatingPointError('original local failure')
            return np.array([1.]),np.array([2.]),0.
        s.evaluate.side_effect=source;r0=Mock();r0.certify.side_effect=FloatingPointError('old refinement failure')
        r1=Mock();r1.certify.return_value={'local':4.}
        chain=self.wrap(self.wrap(s,r0,'old'),r1,'new')
        self.assertEqual(chain.evaluate(1,'GG',.51,.505,[.8],True)[2],4.)
        self.assertEqual(chain.evaluate(1,'GG',.51,.505,[.8],True)[2],4.)
        self.assertEqual(r1.certify.call_count,1)
        self.assertEqual(len(list((Path(self.tmp.name)/'old').glob('*.json'))),0)
        self.assertEqual(len(list((Path(self.tmp.name)/'new').glob('*.json'))),1)
    def test_regular_failure_never_replaced(self):
        s=Mock();s.evaluate.side_effect=FloatingPointError('regular failed');r=Mock()
        with self.assertRaises(FloatingPointError):self.wrap(s,r).evaluate(1,'GG',.51,.505,[.8],True)
        r.certify.assert_not_called()
    def test_source_returns_nonfinite_regular(self):
        s=Mock();s.evaluate.side_effect=[FloatingPointError('bad local'),(np.array([np.nan]),np.array([1.]),0.)];r=Mock()
        with self.assertRaises(FloatingPointError):self.wrap(s,r).evaluate(1,'GG',.51,.505,[.8],True)
        r.certify.assert_not_called()

class ResumeAdapter(unittest.TestCase):
    def test_frozen_input_hash(self):self.assertEqual(sha(HERE/'test_inputs/refine_output_GG_local.py'),SOURCE_SHA)
    def test_generation_and_cache_keys(self):
        s=transform((HERE/'test_inputs/refine_output_GG_local.py').read_text());ast.parse(s)
        for term in ('GG_coordinate_fallback','reused_output','saved_rows','wrap_source'):
            self.assertIn(term,s)
        self.assertIn("len(saved_rows)!=120",s)
    def test_changed_source_anchor_rejected(self):
        with self.assertRaises(ValueError):transform('')
    def test_native_source_keeps_kernel_external(self):
        s=(HERE/'local_reference.cpp').read_text();self.assertIn('#include "mp_kernel.cpp"',s)
        self.assertNotIn('R gg_half(',s)
    def test_source_precision_and_flags(self):
        s=(HERE/'build_addon.py').read_text()
        self.assertIn('-DGPD_MP_BITS=192',s);self.assertIn('-ffp-contract=off',s);self.assertIn('-Wl,-headerpad_max_install_names',s)

if __name__=='__main__':unittest.main()

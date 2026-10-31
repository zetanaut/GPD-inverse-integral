import unittest,tempfile
from pathlib import Path
import numpy as np
from gg_local import FallbackSource,check_domain,panels,explicit_per_flavor

class Original:
    def __init__(self,local_failure=False,regular_failure=False):
        self.local_failure=local_failure;self.regular_failure=regular_failure;self.calls=[]
    def evaluate(self,order,sector,x,xi,y,include_local=False):
        self.calls.append((order,sector,include_local))
        if self.regular_failure:raise FloatingPointError('bad regular term')
        if include_local and self.local_failure:raise FloatingPointError('bad local')
        return np.array([2.]),np.array([3.]),4. if include_local else None
class Ref:
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    def certify(self,x,xi):
        self.calls+=1
        if self.fail:raise FloatingPointError('quadrature fails')
        return {'local':5.,'x':x,'xi':xi}
class Tests(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
    def wrap(self,s,r=None,identity='test'):
        return FallbackSource(s,r or Ref(),identity,self.t.name)
    def test_success_bypasses(self):
        s=Original();r=Ref();f=self.wrap(s,r);a=f.evaluate(1,'GG',.51,.5,[.8],True)
        self.assertEqual(a[2],4.);self.assertEqual(r.calls,0)
    def test_fallback(self):
        r=Ref();f=self.wrap(Original(True),r);a=f.evaluate(1,'GG',.51,.5,[.8],True)
        self.assertEqual(a[2],5.);self.assertEqual(r.calls,1)
    def test_regular_failure_not_masked(self):
        r=Ref();f=self.wrap(Original(True,True),r)
        with self.assertRaises(FloatingPointError):f.evaluate(1,'GG',.51,.5,[.8],True)
        self.assertEqual(r.calls,0)
    def test_other_order_not_masked(self):
        f=self.wrap(Original(True))
        with self.assertRaises(FloatingPointError):f.evaluate(0,'GG',.51,.5,[.8],True)
    def test_other_sector_not_masked(self):
        f=self.wrap(Original(True))
        with self.assertRaises(FloatingPointError):f.evaluate(1,'QG',.51,.5,[.8],True)
    def test_regular_only_unchanged(self):
        r=Ref();f=self.wrap(Original(True),r);a=f.evaluate(1,'GG',.51,.5,[.8],False)
        self.assertEqual(a[2],None);self.assertEqual(r.calls,0)
    def test_cache_second_call(self):
        s=Original(True);r=Ref();f=self.wrap(s,r)
        f.evaluate(1,'GG',.51,.5,[.8],True);f.evaluate(1,'GG',.51,.5,[.8],True)
        self.assertEqual(r.calls,1);self.assertEqual(s.calls[-1],(1,'GG',False))
    def test_failed_reference_not_cached(self):
        f=self.wrap(Original(True),Ref(True))
        with self.assertRaises(FloatingPointError):f.evaluate(1,'GG',.51,.5,[.8],True)
        self.assertFalse(list(Path(self.t.name).iterdir()))
    def test_recipe_separates_cache(self):
        s=Original(True);r=Ref();f=self.wrap(s,r,'a');g=self.wrap(s,r,'b')
        f.evaluate(1,'GG',.51,.5,[.8],True);g.evaluate(1,'GG',.51,.5,[.8],True)
        self.assertEqual(r.calls,2)
    def test_cache_still_checks_regular(self):
        s=Original(True);f=self.wrap(s);f.evaluate(1,'GG',.51,.5,[.8],True);s.regular_failure=True
        with self.assertRaises(FloatingPointError):f.evaluate(1,'GG',.51,.5,[.8],True)
    def test_no_pole_excision(self):
        for x,xi in [(.5050607301889878,.505),(.19,.135),(.08,.135),(-.19,.135)]:
            regs,pairs=panels(x,xi)
            self.assertAlmostEqual(sum(b-a for a,b in regs)+sum(2*r for p,r in pairs),2.,places=13)
    def test_direct_panels_ordered(self):
        reg,_=panels(.5050607301889878,.505)
        self.assertTrue(all(-1<=a<b<=1 for a,b in reg))
    def test_singular_domain_rejected(self):
        for x,xi in [(.505,.505),(0.,.5),(1.,.5),(-1.,.5)]:
            with self.assertRaises(ValueError):check_domain(x,xi)
    def test_nonfinite_domain_rejected(self):
        for x,xi in [(np.nan,.5),(.2,np.nan),(.2,0),(.2,1)]:
            with self.assertRaises(ValueError):check_domain(x,xi)
    def test_negative_x_supported(self):check_domain(-.2,.1)
    def test_explicit_source_delta(self):
        self.assertEqual(explicit_per_flavor(),float(np.float32(-1.)/np.float32(108.))*(35.*3.+74.*(4./3.)))
        self.assertLess(abs(3*explicit_per_flavor()+611/108),1e-7)
if __name__=='__main__':unittest.main()

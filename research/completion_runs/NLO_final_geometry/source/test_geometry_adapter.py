import unittest,json,time
from pathlib import Path
import geometry_adapter as a
np,torch=a.np,a.torch

class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank=a.original_bank();cls.domain=a.original_domain()
        with np.load(a.ROOT/'GPD_direct_error_calibration/results/inputs/d002/input.npz',allow_pickle=False) as z:cls.y=z['y'];cls.cov=z['covariance']
        cls.p=a.Problem(cls.y,cls.cov,31,cls.bank,cls.bank.ids)
    def test_repeated_seeds_and_data_independence(self):
        p=a.Problem(self.y+.01,self.cov,31,self.bank,self.bank.ids)
        self.assertTrue(torch.equal(p.x0,self.p.x0))
        self.assertFalse(torch.equal(a.Problem(self.y,self.cov,43,self.bank,self.bank.ids).x0,self.p.x0))
    def test_covariance_rejection(self):
        for C in (-self.cov,self.cov*0,np.full_like(self.cov,np.nan),np.triu(self.cov)):
            with self.assertRaises((ValueError,RuntimeError)):a.Problem(self.y,C,31,self.bank,self.bank.ids)
    def test_explicit_row_permutation(self):
        ix=np.arange(len(self.y))[::-1]
        p=a.Problem(self.y[ix],self.cov[np.ix_(ix,ix)],31,self.bank,[self.bank.ids[i] for i in ix])
        self.assertTrue(torch.equal(p.y,self.p.y));self.assertTrue(torch.equal(p.cov,self.p.cov))
        self.assertEqual(float(p.objective(p.x0)),float(self.p.objective(self.p.x0)))
    def test_missing_duplicate_and_wrong_ids_rejected(self):
        for ids in (self.bank.ids[:-1],['wrong']+list(self.bank.ids[1:]),[self.bank.ids[0]]*len(self.bank.ids)):
            with self.assertRaises(ValueError):a.Problem(self.y,self.cov,31,self.bank,ids)
    def test_changing_n_scalar_keeps_moment_denominator(self):
        for count in (1,3,9):
            bank=self.bank.subset(range(count));n=bank.n_scalar
            p=a.Problem(self.y[:n],self.cov[:n,:n],31,bank,bank.ids)
            x=p.x0;r=p.residual(x);c=p.components(x)
            self.assertAlmostEqual(float(r[:n].square().sum()),float(c[0]),places=10)
            self.assertAlmostEqual(float(r[n:].square().sum()),float(c[1]),places=12)
            self.assertAlmostEqual(float(c[1]),float(self.p.components(x)[1]),places=12)
            self.assertAlmostEqual(float(p.objective(x)),float(p.independent_objective(x)),places=9)
    def test_dynamic_constraint_count(self):
        d=a.Domain(self.domain.points[:2],self.domain.spin2)
        self.assertEqual(len(d.names),23)
        self.assertEqual(len(d.constraints(self.p,self.p.x0)),23)
    def test_model_and_moments_unchanged(self):
        self.assertEqual(self.p.x0.numel(),3867)
        self.assertEqual(len(self.p.p.xi),41)
        self.assertEqual(sum(N//2+1 for _,N,_ in self.p.p.spec),42)
        self.assertEqual(len(self.domain.mom.xd),80)

def replay():
    records=[];begin=time.monotonic();bank=a.original_bank();domain=a.original_domain()
    for ident in ('d002','d003'):
        out=a.ROOT/'GPD_direct_error_calibration/results'/('development_'+ident)
        result=json.loads((out/'RESULT.json').read_text());selected=result['selected']
        x=torch.load(out/selected['checkpoint'],weights_only=True)['flat']
        with np.load(a.ROOT/'GPD_direct_error_calibration/results/inputs'/ident/'input.npz',allow_pickle=False) as z:y=z['y'];cov=z['covariance']
        old=a.smoke.Problem(y,cov,selected['seed']);new=a.Problem(y,cov,selected['seed'],bank,bank.ids)
        oldd=a.smoke.Domain();xx=x.clone().requires_grad_(True)
        jo=old.objective(xx);go=torch.autograd.grad(jo,xx)[0]
        jn=new.objective(xx);gn=torch.autograd.grad(jn,xx)[0]
        ro=old.residual(x);rn=new.residual(x);co=oldd.constraints(old,x);cn=domain.constraints(new,x)
        errors={'J':abs(float(jo)-float(jn)),'residual_max':float((ro-rn).abs().max()),
                'gradient_max':float((go-gn).abs().max()),'constraints_max':float((co-cn).abs().max()),
                'field_max':float((old.field(x,old.bank['x'],old.bank['xi'])-new.field(x,old.bank['x'],old.bank['xi'])).abs().max())}
        assert errors['J']<1e-10 and errors['residual_max']<1e-10 and errors['gradient_max']<1e-8 and errors['constraints_max']<1e-8 and errors['field_max']==0,errors
        R=new.residual_jacobian(x).detach();oldR=old.residual_jacobian(x).detach()
        errors['Jacobian_max']=float((R-oldR).abs().max());assert errors['Jacobian_max']<1e-8
        direction=torch.cos(torch.arange(len(x))*.173+.31);direction/=direction.norm()
        probes=[]
        for h in (1e-4,1e-5,1e-6):
            fd=(new.residual(x+h*direction)-new.residual(x-h*direction))/(2*h)
            err=float(((fd-R@direction).abs()/torch.maximum(torch.ones_like(fd),(R@direction).abs())).max())
            probes.append({'step':h,'maximum_scaled_residual_Jacobian_error':err,'pass':err<=1e-6})
        assert all(p['pass'] for p in probes),probes
        mom=a.smoke.moment_validation(new,x)
        tv=a.smoke.targets(new,x,384)
        eligibility=bool(torch.isfinite(cn).all() and (cn<=0).all() and mom['pass'] and selected['forward_pass'] and max(abs(np.asarray(tv)-selected['targets']))<=1e-7)
        assert eligibility==selected['pass'] and eligibility
        records.append({'id':ident,'pass':True,'errors':errors,'Jacobian_probes':probes,'eligibility':eligibility,'full_forward_reused':True,'n_scalar':new.n_scalar,'constraints':len(cn)})
    dest=Path(__file__).resolve().parents[1]/'results/ORIGINAL_GEOMETRY_REPLAY.json'
    dest.write_text(json.dumps({'pass':True,'records':records,'seconds':time.monotonic()-begin,'fitting_steps':0},indent=2))
    print(json.dumps(records,indent=2))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests)
    res=unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful():raise SystemExit(1)
    replay()

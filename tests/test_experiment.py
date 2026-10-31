from pathlib import Path
import csv,json,math,tempfile,unittest,sys
import numpy as np
import torch
from torch.func import functional_call,jacrev,vmap
ROOT=Path(__file__).resolve().parents[1]
from gpd_reh.inputs import (prepare,read_source,SOURCE_SHA256,digest,coordinate_key,coordinate_only_t_nodes,anchor_indices,atomic_json,write_csv)
from gpd_reh.model import ExperimentalFiLM
from gpd_reh.reference_model import ReferenceNetwork
from gpd_reh.moments import FixedTMoments
from gpd_reh.bank import ExperimentalForward,MissingMaps,preflight,read_index,PHYSICS
from gpd_reh.problem import ExperimentalProblem
from gpd_tools.import_rows import import_rows

torch.set_num_threads(1)
SOURCE=ROOT/'data/experimental/final_summary_by_DNN.csv'
ROWS=ROOT/'tests/data/accepted_development_rows.npz'

class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)
        cls.meta=prepare(SOURCE,cls.path)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_exact_source_hash(self):self.assertEqual(digest(SOURCE),SOURCE_SHA256)
    def test_source_counts(self):
        self.assertEqual((self.meta['n_original'],self.meta['n_selected'],self.meta['n_excluded']),(189,160,29))
    def test_pair_family_counts(self):
        self.assertEqual((self.meta['n_exact_evolution_families'],self.meta['n_exact_xi_Q2_pairs']),(67,102))
    def test_quoted_central_and_sigma_retained(self):
        a=read_source(SOURCE)
        with SOURCE.open(newline='') as f: raw=list(csv.DictReader(f))
        for r,s in zip(a,raw):
            self.assertEqual(r['ReH_source'],float(s['Pred ReH']))
            self.assertEqual(r['ReH_internal'],-float(s['Pred ReH']))
            self.assertEqual(r['sigma_reported'],float(s['Pred ReH Std']))
    def test_no_other_cff_in_observations(self):
        with np.load(self.path/'observations.npz',allow_pickle=False) as f:
            self.assertEqual(set(f.files),{'row_id','map_id','xi','Q2','t','y','sigma'})
    def test_no_value_dependent_t_grid(self):
        r=[x for x in read_source(SOURCE) if x['paper_cut_pass']]
        t=coordinate_only_t_nodes(r);anchors=anchor_indices(r)
        for x in r:x['ReH_source']=1e5;x['sigma_reported']=1e-8
        self.assertEqual(t,coordinate_only_t_nodes(r));self.assertEqual(anchors,anchor_indices(r))
    def test_backward_rows_retained(self):self.assertEqual(self.meta['n_Q2_below_Q02_rows'],49)
    def test_boundary_flags_not_cuts(self):self.assertEqual(self.meta['n_boundary_flags_retained'],9)
    def test_corrupt_source_rejected(self):
        p=self.path/'mutated.csv';p.write_bytes(SOURCE.read_bytes()+b'\n')
        with self.assertRaises(ValueError):read_source(p)
    def test_exact_coordinate_keys(self):
        self.assertNotEqual(coordinate_key(.2,4),coordinate_key(np.nextafter(.2,1),4))
    def test_missing_bank_explicit(self):
        result=preflight(self.path,self.path/'none');self.assertEqual(result['n_missing_pairs'],102)
        with self.assertRaises(MissingMaps):ExperimentalForward(self.path,self.path/'none')

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31);self.old=ReferenceNetwork('film').double()
        self.model=ExperimentalFiLM(-.35,.2).double();self.model.load_state_dict(self.old.state_dict())
        self.x=torch.tensor(np.tile(np.linspace(-1,1,33),(2,1)),dtype=torch.float64)
        self.xi=torch.tensor([.2,.4],dtype=torch.float64);self.t=torch.full((2,),-.35,dtype=torch.float64)
    def test_parameter_count(self):self.assertEqual(sum(p.numel() for p in self.model.parameters()),3867)
    def test_exact_old_limit(self):self.assertTrue(torch.equal(self.model(self.x,self.xi,self.t),self.old(self.x,self.xi)))
    def test_parity(self):
        a=self.model(self.x,self.xi,self.t);b=self.model(-self.x,self.xi,self.t)
        torch.testing.assert_close(a[:,:2],-b[:,:2],rtol=0,atol=0);torch.testing.assert_close(a[:,2],b[:,2],rtol=0,atol=0)
    def test_endpoints(self):self.assertEqual(self.model(self.x,self.xi,self.t)[...,[0,-1]].abs().max().item(),0)
    def test_t_used(self):self.assertGreater((self.model(self.x,self.xi,self.t+.1)-self.model(self.x,self.xi,self.t)).norm().item(),1e-6)
    def test_no_new_Q2_input(self):
        import inspect
        self.assertEqual(list(inspect.signature(self.model.forward).parameters),['x','xi','t'])
    def test_transform_safe(self):
        fn=lambda x,xi,t:self.model(x[None],xi[None],t[None])[0]
        torch.testing.assert_close(vmap(fn)(self.x,self.xi,self.t),self.model(self.x,self.xi,self.t))
    def test_t_derivative(self):
        t=self.t.clone().requires_grad_();v=self.model(self.x,self.xi,t).square().sum();g=torch.autograd.grad(v,t)[0]
        eps=1e-6;fd=(self.model(self.x,self.xi,self.t+eps).square().sum()-self.model(self.x,self.xi,self.t-eps).square().sum())/(2*eps)
        self.assertAlmostEqual(float(g.sum()),float(fd),places=8)
    def test_bad_t_scale(self):
        with self.assertRaises(ValueError):ExperimentalFiLM(-.3,0)

class MomentTests(unittest.TestCase):
    def setUp(self):self.p=FixedTMoments([-1.2,-.3,-.11],order=32)
    @staticmethod
    def true_field(x,xi,t):
        env=(1-x*x)**2
        a=x*env*(1+3*t[:,None]+(1-t[:,None])*xi[:,None]**2)
        b=x*env*(2+t[:,None]**2+(.4+t[:,None])*xi[:,None]**2)
        g=env*(2+t[:,None]+(.2+t[:,None]**2)*xi[:,None]**2)
        return torch.stack((a,b,g),dim=1)
    def test_all_allowed_spaces(self):
        self.assertLess(float(self.p(self.true_field)),1e-23)
        self.assertEqual(sum(N//2+1 for _,N,_ in self.p.spec),42)
    def test_coefficients_vary_with_t(self):
        a=self.p.coefficients(self.true_field)
        self.assertNotAlmostEqual(a[0]['coefficients'][0],a[12]['coefficients'][0])
    def test_gluon_spin_power(self):self.assertIn((2,2,0),self.p.spec)
    def test_denominator_is_fixed_t_average(self):
        a=FixedTMoments([-.3],order=32);b=FixedTMoments([-.3,-.5],order=32)
        fn=lambda x,xi,t:torch.stack((x*(1-x*x)**2*torch.sin(8*xi[:,None]),x*0,(1-x*x)**2),dim=1)
        self.assertAlmostEqual(float(a(fn)),float(b(fn)),places=11)
    def test_nonpolynomial_detected(self):
        fn=lambda x,xi,t:self.true_field(x,xi,t)*torch.sin(7*xi)[:,None,None]
        self.assertGreater(float(self.p(fn)),1e-4)
    def test_disjoint_xi_diagnostic(self):
        d=np.linspace(.02,.98,81);d=(d[:-1]+d[1:])/2
        self.assertEqual(set(self.p.xi.tolist()).intersection(d.tolist()),set())

class NLOInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.prepared=cls.root/'prepared';cls.bank=cls.root/'bank'
        cls.prepared.mkdir();cls.imported=import_rows(ROWS,cls.bank)
        idx=read_index(cls.bank);e=idx['entries'][:2];obs=[]
        for i,r in enumerate(e):
            obs.append({'row_id':f'control{i}','xi':r['xi'],'Q2_GeV2':r['Q2'],'t_GeV2':-.3-.1*i,'map_id':f'm{i}'})
        write_csv(cls.prepared/'observations.csv',obs,list(obs[0]))
        np.savez(cls.prepared/'observations.npz',row_id=np.array([r['row_id'] for r in obs]),
                 y=np.array([.2,-.5]),sigma=np.array([.1,.2]))
        atomic_json(cls.prepared/'INPUT_SUMMARY.json',{'t_mean':-.35,'t_std_population':.05,'moment_t_nodes':[-.4,-.3]})
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_accepted_row_import_idempotent(self):self.assertEqual(import_rows(ROWS,self.bank)['added'],0)
    def test_old_row_replay(self):
        f=ExperimentalForward(self.prepared,self.bank);model=ExperimentalFiLM(-.35,.05).double();pred=f(model)
        records=__import__('gpd_reh.bank',fromlist=['load_rows']).load_rows(self.prepared,self.bank,'nominal')
        for i,(r,x,w) in enumerate(records):
            v=model(torch.tensor(x)[None],torch.tensor([float(r['xi'])],dtype=torch.float64),torch.tensor([float(r['t_GeV2'])],dtype=torch.float64))[0]
            torch.testing.assert_close(pred[i],torch.einsum('ocn,cn->o',torch.tensor(w),v),rtol=1e-12,atol=1e-12)
    def test_actual_objective_residual_identity(self):
        p=ExperimentalProblem(self.prepared,self.bank,moment_order=16);x=p.x0.clone().requires_grad_()
        a=p.objective(x);b=p.independent_objective(x)
        self.assertLess(abs(float(a-b)),1e-12)
        ga=torch.autograd.grad(a,x)[0];gb=torch.autograd.grad(b,x)[0]
        torch.testing.assert_close(ga,gb,rtol=1e-12,atol=1e-12)
    def test_no_imaginary_influence(self):
        p=ExperimentalProblem(self.prepared,self.bank,moment_order=16);loss=p.objective(p.x0).detach()
        with torch.no_grad():p.forward.weights[:,1]*=7;p.forward.weights[:,1]+=4
        torch.testing.assert_close(loss,p.objective(p.x0),rtol=0,atol=0)
    def test_real_weighted_loss_not_double_counted(self):
        p=ExperimentalProblem(self.prepared,self.bank,moment_order=16);p.lam=0
        h=lambda x,xi,t:p.field(p.x0,x,xi,t);y=p.forward(h)[:,0]
        expected=((y-p.y)/p.sigma).square().mean()
        torch.testing.assert_close(p.objective(p.x0),expected,rtol=1e-14,atol=1e-14)
    def test_directional_derivative_actual_rows(self):
        p=ExperimentalProblem(self.prepared,self.bank,moment_order=16);x=p.x0.clone().requires_grad_()
        g=torch.autograd.grad(p.objective(x),x)[0];d=torch.sin(torch.arange(len(x),dtype=torch.float64));d/=d.norm()
        h=1e-5;fd=(p.objective(p.x0+h*d)-p.objective(p.x0-h*d))/(2*h)
        err=abs(float(fd-g@d))/max(1,abs(float(fd)),abs(float(g@d)))
        self.assertLess(err,1e-6)
    def test_integrated_jacobian(self):
        p=ExperimentalProblem(self.prepared,self.bank,moment_order=16)
        x=p.x0.clone().requires_grad_();J=p.jacobian(x);r=p.residual(x)
        g=torch.autograd.grad(r.square().sum(),x)[0]
        torch.testing.assert_close(2*J.T@r.detach(),g,rtol=1e-10,atol=1e-10)
        self.assertEqual(tuple(J.shape),(len(r),3867))
    def test_source_corruption_rejected(self):
        idx=read_index(self.bank);r=idx['entries'][0];p=self.bank/r['file'];raw=p.read_bytes()
        try:
            p.write_bytes(raw+b'bad')
            with self.assertRaises(ValueError):read_index(self.bank)
        finally:p.write_bytes(raw)

if __name__=='__main__':unittest.main(verbosity=2)

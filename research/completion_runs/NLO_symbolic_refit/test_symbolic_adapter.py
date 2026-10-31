"""Mechanical tests only; no refits or new physics rows."""
import os,sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(os.environ['NLO_PILOT_PACKET']).resolve()
sys.path.insert(0,str(ROOT/'GPD_NLO_resolution_review/pilot'))
import neural_pilot as n
from symbolic_adapter import install_supports,propose_supports

def problem(truth='smooth',seed=31):
 p=n.Problem(truth,seed)
 ck=torch.load(n.WORK/'fits'/f'{truth}_seed{seed}'/'final.pt',weights_only=True)
 p.model.load_state_dict(ck['state_dict']);p.x0=ck['flat'].clone()
 return p
@pytest.mark.parametrize('truth',['smooth','Dterm_added'])
@pytest.mark.parametrize('seed',[31,43])
def test_full_support_identity(truth,seed):
 p=problem(truth,seed);x=p.x0.detach().requires_grad_(True)
 f=p.original_objective(x);g=torch.autograd.grad(f,x)[0];r=p.residual(x).detach()
 y=p.y.clone();rows=p.forward.rows.clone();s=p.p.scales.clone()
 install_supports(p,[list(range(N//2+1)) for _,N,_ in p.p.spec])
 f2=p.original_objective(x);g2=torch.autograd.grad(f2,x)[0]
 assert torch.equal(y,p.y) and torch.equal(rows,p.forward.rows) and torch.equal(s,p.p.scales)
 torch.testing.assert_close(f,f2,rtol=1e-10,atol=1e-18)
 torch.testing.assert_close(g,g2,rtol=1e-9,atol=1e-14)
 torch.testing.assert_close(r,p.residual(x),rtol=1e-10,atol=1e-14)

def test_selected_subspace_loss_and_gradient():
 p=problem();ss=[[0] for _ in p.p.spec];install_supports(p,ss)
 x=p.x0.detach().requires_grad_(True);r=p.residual(x);f=p.original_objective(x)
 g=torch.autograd.grad(f,x,retain_graph=True)[0];g2=torch.autograd.grad(r.square().sum(),x)[0]
 torch.testing.assert_close(f,r.square().sum(),rtol=1e-10,atol=1e-14)
 torch.testing.assert_close(g,g2,rtol=1e-8,atol=1e-12)

def test_unknown_coefficients_not_frozen():
 p=problem();install_supports(p,[[0] for _ in p.p.spec])
 m=torch.ones((len(p.p.xi),len(p.p.spec)),dtype=torch.float64)
 assert p.p.residual(m).abs().max()<1e-12
 assert p.p.residual(7*m).abs().max()<1e-11

@pytest.mark.parametrize('bad',[[0,0],[-1],[0,9],[True]])
def test_invalid_support(bad):
 p=problem();ss=[list(range(N//2+1)) for _,N,_ in p.p.spec];ss[0]=bad
 with pytest.raises(ValueError):install_supports(p,ss)

def test_empty_support():
 p=problem();install_supports(p,[[] for _ in p.p.spec]);m=torch.ones((len(p.p.xi),len(p.p.spec)),dtype=torch.float64)
 torch.testing.assert_close(p.p.residual(m),m/p.p.scales)

def test_deterministic_proposals_and_coordinate_conversion():
 p=problem();r=propose_supports(p);rr=propose_supports(p)
 assert r==rr and len(r)==12
 x=np.array([.05,.25,.9])
 for e in r:
  z=(x/.6)**2;k=np.arange(len(e['coefficient_scaled']))
  np.testing.assert_allclose((z[:,None]**k)@e['coefficient_scaled'],(x[:,None]**(2*k))@e['coefficient_xi'],rtol=1e-11,atol=1e-13)

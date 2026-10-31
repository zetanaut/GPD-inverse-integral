"""Candidate coupled NLO forward interface. Numerical acceptance is separate.
State coefficients: [Sigma, T_rescaled, Hg], hard charge 2/9*(Sigma+T).
"""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'));sys.path.insert(0,str(ROOT/'frozen/python'))
from basis_factory import Elements
from src.replay import analytic_input
from src.ordered_evolution import evolve
from nlo_forward import alpha_two_loop
import nlo_hard

class HardGrid:
 def __init__(self,b,parity):self.b=b;self.xi=b.xi;self.edges=b.edges;self.parity=parity
 def interpolation(self,y):return self.b.basis(y,self.parity)

def initial(b,dterm=False):
 q=analytic_input(b.nodes[b.qids],b.xi);g=analytic_input(b.nodes[b.gids],b.xi)
 v=np.r_[q[0],q[1],g[2]]
 if dterm:
  z=b.nodes[b.qids]/b.xi;qD=np.where(abs(z)<1,z*(1-z*z)**2,0.)
  z=b.nodes[b.gids]/b.xi;gD=np.where(abs(z)<1,b.xi*(1-z*z)**2,0.)
  v+=np.r_[.09*qD,-.02*qD,.03*gD]
 return v

def evaluate_field(b,v,y):
 nq=len(b.qids);q=b.basis(y,'odd');g=b.basis(y,'even')
 return np.array([q@v[:nq],q@v[nq:2*nq],g@v[2*nq:]])

def hard_row(b,q2,order=96,log_q2_muf2=0.):
 qgrid=HardGrid(b,'odd');ggrid=HardGrid(b,'even');xi=b.xi;alpha=alpha_two_loop(q2)
 y,w=nlo_hard._subtracted_nodes(qgrid,order);B=qgrid.interpolation(y);Bc=qgrid.interpolation([xi])[0]
 lo=(w/(y-xi))@(B-Bc)+(w/(y+xi))@B+(np.log((1-xi)/xi)-1j*np.pi)*Bc
 cq,_,qd=nlo_hard.correction_rows(qgrid,alpha,order,log_q2_muf2,3)
 _,cg,gd=nlo_hard.correction_rows(ggrid,alpha,order,log_q2_muf2,3)
 return np.r_[2/9*(lo+cq),2/9*(lo+cq),cg],{'alpha':alpha,'quark':qd,'gluon':gd,'state':'Sigma,T_rescaled,Hg','gluon_charges_applied_once':True}

def load(directory):
 directory=Path(directory);key=json.loads((directory/'KEY.json').read_text());bs=key['basis'];b=Elements(bs['xi'],bs['degree'],bs['layout'])
 with np.load(directory/'operators.npz') as z:k0=z['K0'].copy();k1=z['K1'].copy()
 return b,k0,k1,key

def forward(directory,q2,field=None,order=192):
 b,k0,k1,key=load(directory);v=initial(b) if field is None else np.asarray(field)
 end,diag=evolve(v,k0,k1,q2,rtol=1e-11,atol=1e-13);row,hdiag=hard_row(b,q2,order)
 return row@end,end,{'evolution':diag,'hard':hdiag}

def jacobian(directory,q2,order=192):
 """Real 2-by-N derivative of [Re CFF, Im CFF] with respect to H(Q0²).
 The complex row is propagated with the bilinear transpose adjoint, preserving
 the ordering of the noncommuting generators. No fitted model is evaluated.
 """
 b,k0,k1,key=load(directory);row,_=hard_row(b,q2,order)
 adj,diag=evolve(row,k0,k1,q2,rtol=1e-12,atol=1e-14,adjoint=True)
 return np.stack([adj.real,adj.imag]),diag

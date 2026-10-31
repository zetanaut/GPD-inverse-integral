"""Endpoint logarithms in a continuous physical-x spectral element space.
Adds bubble s*(1-s)*log(s)^k on the two cells meeting xi. These are free
basis functions, not fitted counterterms or imposed conservation constraints.
Polynomial span and endpoint/parity conditions are preserved. Coefficients
remain values at an augmented nodal set; QR orthogonalization conditions the
basis before forming its interpolation map.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from spectral import Elements
from scipy.special import roots_jacobi

class LogElements:
 def __init__(self,xi,degree=8,layout='log3'):
  self.xi=xi;self.degree=degree;self.layout=layout;self.enrichment=int(layout[3:]);assert self.enrichment in [3,5]
  self.base=Elements(xi,degree,'crossover');self.edges=self.base.edges;self.lgl=self.base.lgl;self.cells=[];extras=[];self.transforms={};self.conditions={}
  centre=int(np.where(self.edges==xi)[0][0]);self.special=[centre-1,centre]
  z,w=leggauss(128);u=(z+1)/2;s=u*u;weight=w*u # integrates ds on[0,1]
  for cell in self.special:
   left,right=self.edges[cell:cell+2];width=right-left;direction=-1 if cell==centre-1 else 1
   y=xi+direction*width*s;ids=cell*degree+np.arange(1,degree)
   # Bubble cardinal functions vanish at the two cell ends; use even base.
   bubble=self.base.basis(y,'even')[:,ids]
   logs=np.column_stack([s*(1-s)*np.log(s)**k for k in range(1,self.enrichment+1)])
   qb,rb=np.linalg.qr(np.sqrt(weight)[:,None]*bubble,mode='reduced');projection=np.linalg.solve(rb,qb.T@(np.sqrt(weight)[:,None]*logs));residual=logs-bubble@projection
   qe,re=np.linalg.qr(np.sqrt(weight)[:,None]*residual,mode='reduced');whiten=np.linalg.solve(re,np.eye(self.enrichment))
   self.cells.append({'cell':cell,'direction':direction,'width':width,'ids':ids,'projection':projection,'whiten':whiten,'raw_qr_condition':float(np.linalg.cond(re))})
   distances=np.geomspace(5e-4,.08,self.enrichment) if self.enrichment==3 else np.geomspace(1e-5,.12,self.enrichment)
   extras.extend(xi+direction*width*distances)
  self.nodes=np.sort(np.r_[self.base.nodes,extras]);self.n=len(self.nodes);self.qids=np.arange(1,self.n-1);self.gids=np.arange(self.n-1)
  for parity,ids in [('odd',self.qids),('even',self.gids)]:
   A=self.raw(self.nodes[ids],parity);self.transforms[parity]=np.linalg.solve(A,np.eye(len(ids)));self.conditions[parity]=float(np.linalg.cond(A))
 def raw(self,y,parity):
  y=np.atleast_1d(y);v=abs(y);polys=self.base.basis(y,parity);extra=np.zeros((len(y),2*self.enrichment))
  for j,record in enumerate(self.cells):
   s=record['direction']*(v-self.xi)/record['width'];mask=(s>0)&(s<1)
   if np.any(mask):
    ss=s[mask];L=np.column_stack([ss*(1-ss)*np.log(ss)**k for k in range(1,self.enrichment+1)]);bubble=self.base.basis(v[mask],'even')[:,record['ids']];extra[mask,j*self.enrichment:(j+1)*self.enrichment]=(L-bubble@record['projection'])@record['whiten']
  if parity=='odd':extra*=np.sign(y)[:,None]
  return np.column_stack([polys,extra])
 def basis(self,y,parity):return self.raw(y,parity)@self.transforms[parity]
 def outputs(self,n):
  xs=[];ws=[]
  for i,(a,b) in enumerate(zip(self.edges[:-1],self.edges[1:])):
   nn=max(24,n) if i in self.special else n
   if i in self.special:
    # s=u^(3/2); Jacobi beta=1/2 accounts for the exact Jacobian.
    # Avoids resolving source coordinates below its excluded pole distance.
    z,w=roots_jacobi(nn,0.,.5);u=(z+1)/2
    if b==self.xi:x=b-(b-a)*u**1.5
    else:x=a+(b-a)*u**1.5
    ww=(b-a)*1.5*w/2**1.5
   else:
    z,w=leggauss(nn);x=(a+b)/2+(b-a)/2*z;ww=(b-a)/2*w
   xs.append(x);ws.append(ww)
  return np.concatenate(xs),np.concatenate(ws)
 def specification(self):
  return {'xi':self.xi,'degree':self.degree,'layout':self.layout,'enrichment':self.enrichment,'edges':self.edges.tolist(),'nodes':self.nodes.tolist(),'state':'Sigma,T_rescaled,Hg','endpoints':'odd quarks vanish at0; all fields vanish at1; even gluon free at0; shared crossover value','extra_basis':'s*(1-s)*log(s)^k, k=1..enrichment, on two crossover-adjacent cells; local bubble QR conditioning','interpolation_conditions':self.conditions,'raw_log_QR_conditions':[r['raw_qr_condition'] for r in self.cells],'continuity':'shared field at interfaces; independent slopes'}

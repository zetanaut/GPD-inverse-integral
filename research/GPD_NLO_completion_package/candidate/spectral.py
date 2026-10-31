"""Continuous physical-x spectral elements with independent one-sided slopes.
Basis uses LGL interpolation nodes; raw kernels use interior GL test quadrature.
No kernel collocation at xi, endpoints, or zero. Physical parity is explicit.
"""
from pathlib import Path
import sys,hashlib,json,time
import numpy as np
from numpy.polynomial.legendre import Legendre,leggauss
sys.path.insert(0,str(Path(__file__).resolve().parent))
from continuous_action import geometry,source_local
from action_probe import SECTORS

class Elements:
 def __init__(self,xi,degree=6,layout='graded'):
  self.xi=xi;self.degree=degree;self.layout=layout
  if layout=='graded':
   self.edges=np.r_[xi*np.array([0,.5,.85,1]),xi+(1-xi)*np.array([.02,.1,.35,1])]
  elif layout=='boundary':
   self.edges=np.r_[xi*np.array([0,.5,.9,.99,1]),xi+(1-xi)*np.array([.001,.01,.1,.4,1])]
  elif layout=='crossover':
   self.edges=np.r_[xi*np.array([0,.5,.9,.99,.998,1]),xi+(1-xi)*np.array([.0002,.002,.02,.1,.4,1])]
  else:raise ValueError(layout)
  self.lgl=np.r_[-1,Legendre.basis(degree).deriv().roots(),1]
  self.nodes=np.concatenate([((b+a)/2+(b-a)/2*self.lgl)[:-1] for a,b in zip(self.edges[:-1],self.edges[1:])] + [np.array([1.])])
  self.n=len(self.nodes)
  self.qids=np.arange(1,self.n-1);self.gids=np.arange(self.n-1)
  self.bw=np.array([1/np.prod(self.lgl[j]-np.delete(self.lgl,j)) for j in range(degree+1)])
 def basis(self,y,parity):
  y=np.atleast_1d(y);v=abs(y);ids=np.clip(np.searchsorted(self.edges,v,side='right')-1,0,len(self.edges)-2)
  a=self.edges[ids];b=self.edges[ids+1];z=(2*v-a-b)/(b-a)
  delta=z[:,None]-self.lgl;close=abs(delta)<1e-13
  raw=np.divide(self.bw,delta,out=np.zeros_like(delta),where=~close)
  good=~close.any(axis=1);raw[good]/=raw[good].sum(axis=1)[:,None]
  raw[~good]=close[~good].astype(float)
  full=np.zeros((len(y),self.n));full[np.arange(len(y))[:,None],ids[:,None]*self.degree+np.arange(self.degree+1)]=raw
  if parity=='odd':return np.sign(y)[:,None]*full[:,self.qids]
  if parity=='even':return full[:,self.gids]
  raise ValueError(parity)
 def outputs(self,n):
  z,w=leggauss(n);return np.concatenate([(a+b)/2+(b-a)/2*z for a,b in zip(self.edges[:-1],self.edges[1:])]),np.concatenate([(b-a)/2*w for a,b in zip(self.edges[:-1],self.edges[1:])])
 def specification(self):return {'xi':self.xi,'degree':self.degree,'layout':self.layout,'edges':self.edges.tolist(),'nodes':self.nodes.tolist(),'state':'Sigma,T_rescaled,Hg','endpoints':'odd quarks vanish at 0; all channels vanish at 1; even gluon free at 0','continuity':'shared field at interior interfaces; derivatives independent'}

def block(p,b,x,n,local_n=0):
 y,w=geometry(x,b.xi,n,extra=np.r_[-b.edges[::-1],b.edges]);output={}
 for parity in ['odd','even']:
  output[parity]=(b.basis(y,parity),b.basis([x],parity)[0])
 ans=[];locals_cache={}
 for order in [0,1]:
  row=[]
  for s in SECTORS:
   fy,fx=output['even' if s in ['QG','GG'] else 'odd']
   r,k,_=p.evaluate(order,s,x,b.xi,y)
   if local_n and order==1 and s=="GG":
    from fixed_local import gg_local
    local=gg_local(p,x,b.xi,local_n)
   else:
    local_key=(order,"NS_plus" if s=="QQ_singlet" else s)
    if local_key not in locals_cache:locals_cache[local_key]=source_local(p,order,local_key[1],x,b.xi)
    local=locals_cache[local_key]
   row.append((w*r)@fy+(w*k)@(fy-fx)+local*fx)
  ans.append(row)
 return ans,len(y)

def project(b,x,w,actions):
 Bq=b.basis(x,'odd');Bg=b.basis(x,'even');Bs=[Bq,Bq,Bq,Bq,Bg,Bg]
 masses={'odd':Bq.T@(w[:,None]*Bq),'even':Bg.T@(w[:,None]*Bg)}
 arrays={};secs=[]
 for order in [0,1]:
  sec=[]
  for j,s in enumerate(SECTORS):
   a=np.stack([v[order][j] for v in actions]);mat=np.linalg.solve(masses['even' if s in ['GQ','GG'] else 'odd'],Bs[j].T@(w[:,None]*a));sec.append(mat);arrays[f'K{order}_{s}']=mat
  nq=len(b.qids);ng=len(b.gids);z=np.zeros((nq,nq));qg=np.zeros((nq,ng));gq=np.zeros((ng,nq))
  arrays[f'K{order}']=np.block([[sec[2],z,sec[3]],[z,sec[0],qg],[sec[4],gq,sec[5]]])
 arrays['x']=x;arrays['weights']=w
 return arrays,{k:float(np.linalg.cond(v)) for k,v in masses.items()}

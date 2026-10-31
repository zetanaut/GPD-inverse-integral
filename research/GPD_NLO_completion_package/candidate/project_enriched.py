"""Diagonal equilibration of the same Galerkin mass equation, no constraints."""
import numpy as np
from action_probe import SECTORS

def project(b,x,w,actions):
 Bq=b.basis(x,'odd');Bg=b.basis(x,'even');Bs=[Bq,Bq,Bq,Bq,Bg,Bg];masses={'odd':Bq.T@(w[:,None]*Bq),'even':Bg.T@(w[:,None]*Bg)};scaled={};scale={};cond={}
 for p,M in masses.items():
  scale[p]=1/np.sqrt(np.diag(M));scaled[p]=M*scale[p][:,None]*scale[p];cond[p]={'unscaled':float(np.linalg.cond(M)),'equilibrated':float(np.linalg.cond(scaled[p]))}
 arrays={};residuals=[]
 for order in [0,1]:
  sec=[]
  for j,s in enumerate(SECTORS):
   p='even' if s in ['GQ','GG'] else 'odd';a=np.stack([v[order][j] for v in actions]);rhs=Bs[j].T@(w[:,None]*a);srhs=scale[p][:,None]*rhs;sol=np.linalg.solve(scaled[p],srhs);mat=scale[p][:,None]*sol;residuals.append(float(np.linalg.norm(scaled[p]@sol-srhs)/max(1,np.linalg.norm(srhs))));sec.append(mat);arrays[f'K{order}_{s}']=mat
  nq=len(b.qids);ng=len(b.gids);z=np.zeros((nq,nq));qg=np.zeros((nq,ng));gq=np.zeros((ng,nq));arrays[f'K{order}']=np.block([[sec[2],z,sec[3]],[z,sec[0],qg],[sec[4],gq,sec[5]]])
 arrays['x']=x;arrays['weights']=w;cond['max_scaled_solve_relative_residual']=max(residuals);return arrays,cond

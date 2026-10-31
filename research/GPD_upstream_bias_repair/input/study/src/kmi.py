"""Shared four-output CFF network with the same reduced-BKM loss in every arm.

Only (Q2,xB,t) condition the CFF model. Beam energy/angle appear solely in the
fixed observable arrays. The 100x100 tanh layout follows the captured source's
model layout; scaling, deterministic fixed-budget training and L2 are explicit
choices of THIS development comparison, not a reproduction of publication fits.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from torch import nn

class CFFNet(nn.Module):
    def __init__(self,width=100):
        super().__init__();self.net=nn.Sequential(nn.Linear(3,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,4)).double()
    def forward(self,x):return self.net(x)

def make_features(kin,mean=None,std=None):
    k,q2,xb,t,phi=np.asarray(kin,float).T
    x=np.column_stack([np.log(q2),np.log(xb/(1-xb)),np.log(-t)])
    if mean is None:mean=x.mean(0);std=x.std(0);std=np.where(std>1e-12,std,1.)
    return (x-mean)/std,np.asarray(mean),np.asarray(std)

def fit_kmi(kin,A,b,target,sigma,fitmask,seed,cfg):
    idx=np.flatnonzero(fitmask)
    _,mu,sd=make_features(np.asarray(kin)[idx]);x,_,_=make_features(kin,mu,sd)
    x=torch.tensor(x[idx]);AA=torch.tensor(A[idx]);bb=torch.tensor(b[idx]);yy=torch.tensor(target[idx]);ss=torch.tensor(sigma[idx])
    torch.manual_seed(int(seed)+10000);model=CFFNet(cfg['kmi_width'])
    opt=torch.optim.Adam(model.parameters(),lr=cfg['kmi_learning_rate'])
    h=[];tic=time.perf_counter()
    for step in range(1,cfg['kmi_epochs']+1):
        opt.zero_grad(set_to_none=True);theta=model(x);prediction=bb+torch.sum(AA*theta,dim=1)
        data=torch.mean(((prediction-yy)/ss)**2)
        ridge=cfg['kmi_l2']*sum(torch.sum(p*p) for p in model.parameters())
        loss=data+ridge
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite KMI objective')
        loss.backward();opt.step()
        if step==1 or step%50==0 or step==cfg['kmi_epochs']:h.append([step,float(data.detach()),float(ridge.detach())])
    return {'model':model,'mean':mu,'std':sd,'history':np.asarray(h),'seconds':time.perf_counter()-tic}

def predict_cffs(result,kin):
    x,_,_=make_features(kin,result['mean'],result['std'])
    with torch.no_grad():return result['model'](torch.tensor(x)).numpy()

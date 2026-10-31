"""Matched positive periodic surrogates; hard arm adds ONLY output reflection.

Paper Eq.(28) is applied to positive cross-section outputs, not to logits.
Both networks have identical parameter count, initialization, features, optimizer,
and epoch-selection rule. Projection costs two forward evaluations; time is logged.
"""
from __future__ import annotations
import copy,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CrossSectionNet(nn.Module):
    def __init__(self,width=48,depth=2,hard_even=False):
        super().__init__();self.hard_even=bool(hard_even)
        layers=[];n=6
        for _ in range(depth):layers.extend([nn.Linear(n,width),nn.Tanh()]);n=width
        layers.append(nn.Linear(n,1));self.net=nn.Sequential(*layers).double()
    def forward(self,x):
        p=F.softplus(self.net(x)).squeeze(-1)
        if not self.hard_even:return p
        reflected=torch.cat([x[:,:5],-x[:,5:6]],dim=1)
        return (p+F.softplus(self.net(reflected)).squeeze(-1))/2


def features(kin,mean,std):
    kin=np.asarray(kin,float)
    k,q2,xb,t,phi=kin.T
    if np.any((q2<=0)|(xb<=0)|(xb>=1)|(t>=0)):
        raise ValueError('Unsupported transformed coordinates')
    base=np.column_stack([k,np.log(q2),np.log(xb/(1-xb)),np.log(-t)])
    base=(base-mean)/std
    rad=np.deg2rad(phi)
    return np.column_stack([base,np.cos(rad),np.sin(rad)])


def transform_fit(kin):
    k,q2,xb,t,_=np.asarray(kin,float).T
    b=np.column_stack([k,np.log(q2),np.log(xb/(1-xb)),np.log(-t)])
    mu=b.mean(0);sd=b.std(0);sd=np.where(sd>1e-12,sd,1.)
    return mu,sd


def make_model(seed,config,arm):
    torch.manual_seed(int(seed))
    return CrossSectionNet(config['width'],config['depth'],hard_even=(arm=='hard_even'))


def _train(kin,y,sigma,idx,epochs,seed,cfg,arm,validation=None):
    mu,sd=transform_fit(kin[idx]);scale=float(np.median(np.abs(y[idx])))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid response scale')
    x=torch.tensor(features(kin,mu,sd),dtype=torch.float64)
    yy=torch.tensor(y/scale,dtype=torch.float64);ss=torch.tensor(sigma/scale,dtype=torch.float64)
    ii=torch.tensor(idx);vv=None if validation is None else torch.tensor(validation)
    model=make_model(seed,cfg,arm);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    history=[];best=float('inf');best_epoch=epochs;best_state=None;tic=time.perf_counter()
    for epoch in range(1,int(epochs)+1):
        opt.zero_grad(set_to_none=True)
        pred=model(x[ii]);data=torch.mean(((pred-yy[ii])/ss[ii])**2)
        ridge=cfg['l2']*sum(torch.sum(p*p) for p in model.parameters())
        loss=data+ridge
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite surrogate objective')
        loss.backward();opt.step()
        with torch.no_grad():
            val=float(torch.mean(((model(x[vv])-yy[vv])/ss[vv])**2)) if vv is not None else float(data)
        history.append([epoch,float(data.detach()),float(ridge.detach()),val])
        if vv is not None and val<best:
            best=val;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    if validation is not None:model.load_state_dict(best_state)
    return model,mu,sd,scale,best_epoch,np.asarray(history),time.perf_counter()-tic


def fit_surrogate(kin,y,sigma,fitmask,valmask,seed,config,arm):
    """Selection on angular orbit holdout, then same-data all-row refit.

    No truth, dense checking values, CFF estimate, or GPD estimate is accepted by
    this function. Zero-noise input uses identical prescribed numerical weights.
    """
    kin,y,sigma=[np.asarray(x) for x in (kin,y,sigma)]
    if arm not in ('free','hard_even'):raise ValueError('Unknown arm')
    if np.any(sigma<=0) or not all(np.isfinite(a).all() for a in (kin,y,sigma)):
        raise ValueError('Invalid training input')
    tr=np.flatnonzero(fitmask&~valmask);va=np.flatnonzero(fitmask&valmask)
    if not len(tr) or not len(va):raise ValueError('Empty split')
    out=_train(kin,y,sigma,tr,config['epochs'],seed,config,arm,va)
    selection,smu,ssd,sscale,epoch,hist,t1=out
    with torch.no_grad():
        split_pred=sscale*selection(torch.tensor(features(kin,smu,ssd),dtype=torch.float64)).numpy()
    final,mu,sd,scale,_,refhist,t2=_train(kin,y,sigma,np.flatnonzero(fitmask),epoch,seed,config,arm)
    return {'model':final,'mean':mu,'std':sd,'scale':scale,'selected_epoch':epoch,
            'history':hist,'refit_history':refhist,'split_prediction':split_pred,
            'selection_seconds':t1,'refit_seconds':t2}


def predict(result,kin):
    with torch.no_grad():
        return result['scale']*result['model'](torch.tensor(features(kin,result['mean'],result['std']),dtype=torch.float64)).numpy()

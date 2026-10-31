"""Coherent source-table replica helpers; not a GPD uncertainty estimator."""
from __future__ import annotations
import numpy as np


def draw_table(y, sigma, replica_id, root_seed=20261011):
    y=np.asarray(y,dtype=float);sigma=np.asarray(sigma,dtype=float)
    if y.ndim!=1 or sigma.shape!=y.shape or not len(y):raise ValueError('Need paired nonempty vectors')
    if not np.isfinite(y).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):raise ValueError('Invalid source data/error')
    if not isinstance(replica_id,(int,np.integer)) or replica_id<0:raise ValueError('Invalid replica ID')
    z=np.random.Generator(np.random.PCG64(np.random.SeedSequence([root_seed,391870,int(replica_id)]))).standard_normal(y.size)
    return y+sigma*z,z


def source_to_internal(values):
    v=np.asarray(values,dtype=float)
    if not np.isfinite(v).all():raise ValueError('Nonfinite surface')
    return -v


def summarize(members):
    r=np.asarray(members,dtype=float)
    if r.ndim!=2 or len(r)<2 or not r.shape[1] or not np.isfinite(r).all():raise ValueError('Need at least two finite complete surfaces')
    mean=r.mean(axis=0);a=r-mean;cov=a.T@a/(len(r)-1);std=np.sqrt(np.maximum(np.diag(cov),0))
    q16,q50,q84=np.quantile(r,[.16,.50,.84],axis=0)
    return dict(mean=mean,std=std,q16=q16,q50=q50,q84=q84,covariance=cov,anomaly_factor=a.T/np.sqrt(len(r)-1))


def comparison(prediction,center,sigma):
    p,c,s=map(lambda a:np.asarray(a,dtype=float),(prediction,center,sigma))
    if p.ndim!=1 or not(p.shape==c.shape==s.shape):raise ValueError('Shape mismatch')
    if not all(np.isfinite(a).all() for a in (p,c,s)) or np.any(s<=0):raise ValueError('Nonfinite values or invalid weights')
    r=p-c
    return dict(RMS_absolute=float(np.sqrt(np.mean(r*r))),RMS_original_sigma=float(np.sqrt(np.mean((r/s)**2))),
                mean_signed_original_sigma=float(np.mean(r/s)),n_abs_pull_gt3=int(np.sum(abs(r/s)>3)))

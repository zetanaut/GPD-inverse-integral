"""Truth-based diagnostics, imported ONLY by the post-fit scoring controller."""
from __future__ import annotations
import numpy as np
from .geometry import chart_t
from .truths import Truth

def split_grid(xi,order=160):
    z,w=np.polynomial.legendre.leggauss(order)
    x=np.r_[xi*(z+1)/2,xi+(1-xi)*(z+1)/2]
    weights=np.r_[xi*w/2,(1-xi)*w/2]
    return x,weights

def field_metrics(evaluator,truth,geometry,order=160,nxi=33,nu=9):
    rg=geometry['ranges'];xis=np.linspace(*rg['xi'],nxi);us=np.linspace(0,1,nu)
    numerator={'all':0.,'ERBL':0.,'DGLAP':0.};denominator={k:0. for k in numerator};volume={k:0. for k in numerator}
    max_slice=0.;channel_num=np.zeros(3);channel_den=np.zeros(3)
    for xi in xis:
        xx,ww=split_grid(xi,order)
        for u in us:
            t=chart_t(xi,u,*rg['t']);pred=np.asarray(evaluator(xx,xi,t));target=truth.values(xx,xi,t)
            if pred.shape!=target.shape or not np.isfinite(pred).all():raise ValueError('Invalid fitted field')
            cp=(2/9)*(pred[:,0]+pred[:,1]);ct=(2/9)*(target[:,0]+target[:,1]);d=cp-ct
            channel_num+=np.sum(ww[:,None]*(pred-target)**2,axis=0);channel_den+=np.sum(ww[:,None]*target**2,axis=0)
            max_slice=max(max_slice,float(np.sqrt(np.sum(ww*d*d)/max(1e-30,np.sum(ww*ct*ct)))))
            for k,sel in [('all',slice(None)),('ERBL',slice(0,order)),('DGLAP',slice(order,None))]:
                numerator[k]+=float(np.sum(ww[sel]*d[sel]**2));denominator[k]+=float(np.sum(ww[sel]*ct[sel]**2));volume[k]+=float(np.sum(ww[sel]))
    return {'charge_relative_L2':{k:float(np.sqrt(numerator[k]/max(denominator[k],1e-30))) for k in numerator},
        'charge_absolute_RMS':{k:float(np.sqrt(numerator[k]/volume[k])) for k in numerator},
        'charge_max_slice_relative_L2':max_slice,'internal_channel_relative_L2':np.sqrt(channel_num/np.maximum(channel_den,1e-30)).tolist(),
        'mesh':{'nxi':nxi,'nu':nu,'x_order_per_region':order},'not_an_uncertainty_interval':True}

def cff_metrics(pred,target,sigma):
    pred,target,sigma=map(lambda a:np.asarray(a,float),(pred,target,sigma))
    if pred.shape!=target.shape or pred.ndim!=2 or pred.shape[1]!=2:raise ValueError('B,2 CFFs required')
    err=pred-target
    return {'absolute_RMS':np.sqrt(np.mean(err*err,axis=0)).tolist(),
        'relative_L2':(np.linalg.norm(err,axis=0)/np.maximum(np.linalg.norm(target,axis=0),1e-30)).tolist(),
        'weighted_RMS':np.sqrt(np.mean((err/sigma)**2,axis=0)).tolist()}

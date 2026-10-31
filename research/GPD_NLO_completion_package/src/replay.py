"""Cached full-support CFF diagnostics: no compilation, kernel builds or fits.

Uses the original six-point PHYSICAL-X field interpolation and accepted hard
coefficient functions. Diagnostic decompositions do not constitute a continuum
reference or a repaired evolution operator.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frozen/python'))
import nlo_hard
from nlo_forward import alpha_two_loop
XI=.135

def push(eta,xi):
    e=np.asarray(eta,dtype=float)
    return np.where(e<-.5,-(xi*xi)**(1+e),np.where(e>.5,(xi*xi)**(1-e),2*xi*e))

def pull(y,xi):
    y=np.atleast_1d(np.asarray(y,dtype=float));e=y/(2*xi)
    lo=y < -xi;hi=y > xi
    e[lo]=.5*np.log(-y[lo]/xi**2)/np.log(xi)
    e[hi]=.5*np.log(xi**2/y[hi])/np.log(xi)
    return e

def interp(x,values,y,xi=XI,physical=True):
    x=np.asarray(x,float);values=np.asarray(values);y=np.atleast_1d(np.asarray(y,float))
    n=len(x)
    if n<6 or values.shape[0]!=n or not np.isfinite(values).all():
        raise ValueError('Expected >=6 finite nodal values along axis zero')
    if not 0<xi<1 or not np.isfinite(y).all() or np.any(np.abs(y)>1):
        raise ValueError('Invalid skewness/query support')
    eta=pull(y,xi);e=-1+(2*np.arange(n)+1)/n
    if not np.allclose(x,push(e,xi),rtol=1e-13,atol=1e-14):
        raise ValueError('This diagnostic supports the saved grid2 nodes only')
    loc=np.clip(np.floor(.5*(n*(eta+1)+1)).astype(int),1,n)
    start=np.clip(loc-3,0,n-6);ids=start[:,None]+np.arange(6)
    nodes=(x if physical else e)[ids];query=y if physical else eta
    w=np.ones_like(nodes)
    for j in range(6):
        for k in range(6):
            if j!=k:w[:,j]*=(query-nodes[:,k])/(nodes[:,j]-nodes[:,k])
    return np.einsum('ij,ij...->i...',w,values[ids])

class Grid:
    def __init__(self,x,xi=XI):
        self.x=np.asarray(x,float);self.xi=float(xi);self.n=len(x)
        e=-1+(2*np.arange(self.n)+1)/self.n
        self.edges=push(np.unique(np.r_[-1,e,-.5,0,.5,1,np.linspace(-1,1,self.n+1)]),xi)
    def interpolation(self,y):return interp(self.x,np.eye(self.n),y,self.xi)

def analytic_input(x,xi=XI):
    x=np.asarray(x);env=(1-x*x)**2
    return np.stack([2*x*env*(1+.2*xi*xi),.5*x*env*(1-.1*xi*xi),.8*env*(1+.1*xi*xi)])

def components(x,field,order=96,q2=2.3,xi=XI):
    field=np.asarray(field,float)
    if field.shape!=(3,len(x)):raise ValueError('Channel order is Sigma,T_rescaled,Hg')
    grid=Grid(x,xi);y,w=nlo_hard._subtracted_nodes(grid,order)
    if np.any(y==xi):raise FloatingPointError('Quadrature rounded onto crossover')
    f=interp(x,field.T,y,xi);fc=interp(x,field.T,[xi],xi)[0]
    hc=2/9*(f[:,0]+f[:,1]);hcc=2/9*(fc[0]+fc[1]);g=f[:,2];gc=fc[2]
    lo=np.sum(w*((hc-hcc)/(y-xi)+hc/(y+xi)))+hcc*np.log((1-xi)/xi)-1j*np.pi*hcc
    alpha=alpha_two_loop(q2);corrections=[]
    for channel,v,vc in [('quark',hc,hcc),('gluon',g,gc)]:
        c,e=nlo_hard._constant_integral(xi,alpha,channel,0.,3,min(.25*xi,.08))
        corrections.append(np.sum(w*nlo_hard.correction_kernel(y,xi,alpha,channel)*(v-vc))+c*vc)
    z=np.array([lo,*corrections]);total=z.sum()
    return {'lo':[float(z[0].real),float(z[0].imag)],'quark_nlo':[float(z[1].real),float(z[1].imag)],'gluon_nlo':[float(z[2].real),float(z[2].imag)],'total':[float(total.real),float(total.imag)],'crossover_charge_gluon':[float(hcc),float(gc)]}

def load_field(n):
    with np.load(ROOT/'frozen/fields/physical_x'/f'physical_x_field_n{n}.npz',allow_pickle=False) as f:
        return f['x'].copy(),f['initial'].copy(),f['evolved'].copy()

def diagnostics():
    rows=[]
    for n in [48,64,96,128]:
        x,initial,h=load_field(n)
        for order in [48,96,192]:rows.append(dict(nx=n,order=order,**components(x,h,order)))
    lookup={(r['nx'],r['order']):r for r in rows}
    saved=__import__('json').loads((ROOT/'evidence/previous_resolution/RESULT.json').read_text())
    replay=max(np.max(np.abs(np.array(lookup[(r['nx'],96)]['total'])-[r['Re'],r['Im']])) for r in saved['physical_x_evolution_probe'])
    quad=max(np.max(np.abs(np.array(lookup[n,192]['total'])-lookup[n,96]['total'])) for n in [48,64,96,128])
    delta={k:(np.array(lookup[128,192][k])-lookup[96,192][k]).tolist() for k in ['lo','quark_nlo','gluon_nlo','total']}
    transplant=[]
    for a,b in [(96,128),(128,96)]:
        xa,_,ha=load_field(a);xb,_,hb=load_field(b)
        transferred=interp(xb,hb.T,xa).T
        ca=np.array(components(xa,ha,192)['total']);cb=np.array(components(xb,hb,192)['total'])
        ct=np.array(components(xa,transferred,192)['total'])
        transplant.append({'destination_nx':a,'reference_nx':b,'observed_difference':(ca-cb).tolist(),'native_field_at_common_map':(ca-ct).tolist(),'resampling_and_reconstruction':(ct-cb).tolist(),'identity_error':float(np.max(np.abs((ca-ct)+(ct-cb)-(ca-cb))))})
    return {'scope':'cached-field diagnostics only; not NLO certification','replay_max_absolute_difference':float(replay),'quadrature_96_to_192_max_component_change':float(quad),'rows':rows,'128_minus_96_components':delta,'transplant':transplant}

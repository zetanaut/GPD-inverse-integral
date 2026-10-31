"""Read-only FiLM and integral replay, independently expressed in NumPy.
Uses the model's fixed coefficients/conventions; never trains or edits a checkpoint.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss

SPECS=tuple((c,n,n-1 if c<2 else n-2) for c in range(3) for n in (2,4,6,8))

def rms(a):
    a=np.asarray(a,dtype=float)
    if a.size==0 or not np.isfinite(a).all():raise ValueError('A finite nonempty array is required')
    return float(np.sqrt(np.mean(a*a)))

def numpy_field(cp,meta):
    s={k:v.detach().cpu().numpy() for k,v in cp['state_dict'].items()}
    tm=float(meta['t_mean']);ts=float(meta['t_std_population'])
    if not ts>0:raise ValueError('Invalid t scaler')
    def linear(x,key):return x@s[key+'.weight'].T+s[key+'.bias']
    def raw(x,xi,t):
        c=np.column_stack(((xi**2-.18)/.18,(t-tm)/ts))
        h=np.tanh(linear(x[...,None],'input'))
        for i in range(2):
            cond=linear(np.tanh(linear(c,f'conditioners.{i}.0')),f'conditioners.{i}.2')
            gamma,beta=np.split(cond,2,axis=-1)
            h=h+np.tanh((1+gamma[:,None,:])*linear(h,f'blocks.{i}')+beta[:,None,:])/2
        return linear(h,'output')
    def field(x,xi,t):
        x=np.asarray(x,float);xi=np.asarray(xi,float);t=np.asarray(t,float)
        if x.ndim!=2 or xi.shape!=(len(x),) or t.shape!=xi.shape:raise ValueError('Coordinate shape')
        p=raw(x,xi,t);m=raw(-x,xi,t)
        a=np.concatenate(((p[...,:2]-m[...,:2])/2,(p[...,2:]+m[...,2:])/2),axis=-1)
        return (a*np.maximum(1-x*x,0)[...,None]**2).transpose(0,2,1)
    return field

@lru_cache(None)
def gauss(n):
    x,w=leggauss(n)
    return x,w

def grid(xi,n):
    u,w=gauss(n);u=(u+1)/2;w=w/2;xi=np.asarray(xi,float)
    return (np.c_[xi[:,None]*u,xi[:,None]+(1-xi[:,None])*u],
            np.c_[xi[:,None]*w,(1-xi[:,None])*w])

def scale(ch,n):return 16/((n+1)*(n+3)*(n+5)) if ch<2 else 16/((n-1)*(n+1)*(n+3))

def design(xi,n):return np.column_stack([(np.asarray(xi)/.6)**(2*k) for k in range(n//2+1)])

def moment_values(field,xi,t,n):
    x,w=grid(xi,n);h=field(x,np.asarray(xi),np.full(len(xi),t))
    return np.column_stack([2*np.sum(w*x**m*h[:,c],axis=1) for c,N,m in SPECS])

def poly_loss(field,ts,n=48):
    xi=np.linspace(.02,.98,41);total=0.
    for t in ts:
        M=moment_values(field,xi,t,n)
        for i,(c,N,_) in enumerate(SPECS):
            P=design(xi,N);a=np.linalg.lstsq(P,M[:,i],rcond=None)[0]
            total+=float(np.sum(((M[:,i]-P@a)/scale(c,N))**2))
    return total/(len(ts)*len(xi)*len(SPECS))

def moments_check(field,ts,n=192):
    fit=np.linspace(.02,.98,41);check=.02+.96*(np.arange(80)+.5)/80;rows=[]
    for t in ts:
        mf=moment_values(field,fit,t,n);mc=moment_values(field,check,t,n);errors=[]
        for i,(c,N,_) in enumerate(SPECS):
            a=np.linalg.lstsq(design(fit,N),mf[:,i],rcond=None)[0]
            e=(mc[:,i]-design(check,N)@a)/scale(c,N);errors.append(e)
            rows.append(dict(t_GeV2=float(t),channel=('Sigma','T_rescaled','Hg')[c],spin=N,
                             order_per_side=n,RMS=rms(e),max_abs=float(np.max(abs(e)))))
        for j,N in enumerate((2,4,6,8)):
            e=(2/9)*(errors[j]+errors[j+4])
            rows.append(dict(t_GeV2=float(t),channel='Hcharge_plus_same_quark_scale',spin=N,
                             order_per_side=n,RMS=rms(e),max_abs=float(np.max(abs(e)))))
    return rows

def target_values(field,xi,t,n=192):
    u,w=gauss(n)
    def integrate(edges,power):
        val=0.
        for a,b in zip(edges[:-1],edges[1:]):
            x=(a+b)/2+(b-a)*u/2;h=field(x[None],np.array([xi]),np.array([t]))[0]
            val+=float(np.sum(w*(b-a)/2*x**power*(2/9)*(h[0]+h[1])))
        return val
    return np.array([integrate(sorted(set([.2,.4]+([xi] if .2<xi<.4 else []))),0)/.2,
                     integrate([0.,xi,1.],1)])

def response_metrics(input_delta,output_delta):
    a=np.asarray(input_delta,float);b=np.asarray(output_delta,float)
    if a.shape!=b.shape:raise ValueError('Matched coordinates required')
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    return dict(input_rms=rms(a),output_rms=rms(b),norm_ratio=None if na==0 else float(nb/na),
                difference_rms=rms(b-a),gain=None if na==0 else float(a@b/(na*na)),
                cosine=None if na*nb==0 else float(a@b/(na*nb)))

def drift_metrics(early,late):
    a=np.asarray(early,float);b=np.asarray(late,float)
    if a.shape!=b.shape:raise ValueError('Matched grids required')
    na=np.linalg.norm(a)
    return dict(early_rms=rms(a),late_rms=rms(b),change_rms=rms(b-a),
                norm_ratio=None if na==0 else float(np.linalg.norm(b)/na),
                change_over_early_norm=None if na==0 else float(np.linalg.norm(b-a)/na))

"""Coordinate-only nested sampling, exact float keys, and a disjoint surface check."""
from __future__ import annotations
import csv,itertools,json
from pathlib import Path
import numpy as np
from .io import coordinate_key

MASS=0.9382720813

def read_observations(path):
    with Path(path).open(newline='') as f: rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('xi','Q2_GeV2','t_GeV2','sigma_reported'): r[k]=float(r[k])
    return rows

def farthest_order(pool):
    pool=np.asarray(sorted(set(map(float,pool))))
    if len(pool)<2: raise ValueError('Too few skewness coordinates')
    chosen=[0,len(pool)-1]
    while len(chosen)<len(pool):
        distance=np.min(abs(pool[:,None]-pool[chosen][None,:]),axis=1)
        distance[chosen]=-1.; chosen.append(int(np.argmax(distance)))
    return pool[chosen]

def trapezoid_weights(x):
    x=np.asarray(x,float)
    if x.ndim!=1 or len(x)<2 or np.any(np.diff(x)<=0):raise ValueError('Increasing nodes required')
    d=np.diff(x);w=np.r_[d[0]/2,(d[:-1]+d[1:])/2,d[-1]/2]
    return w/w.sum()

def chart_t(xi,u,tmin,tmax):
    t0=-4*MASS*MASS*xi*xi/(1-xi*xi)
    hi=min(tmax,t0-.01) # A coordinate-only margin, not a constraint fitted to CFFs.
    if hi<=tmin:raise ValueError('Empty physical-t chart')
    return tmin+u*(hi-tmin)

def make_geometry(source):
    src=read_observations(source)
    xp=np.array(sorted({r['xi'] for r in src}));qmin=min(r['Q2_GeV2'] for r in src);qmax=max(r['Q2_GeV2'] for r in src)
    tmin=min(r['t_GeV2'] for r in src);tmax=max(r['t_GeV2'] for r in src)
    if len(xp)<27:raise ValueError('At least 27 distinct cached xi coordinates required')
    # Reserve eight xi values without observing ANY CFF/truth values.
    remaining=set(range(1,len(xp)-1));held=[]
    for target in np.linspace(xp[0],xp[-1],10)[1:-1]:
        idx=min(remaining,key=lambda i:(abs(xp[i]-target),i));held.append(idx);remaining.remove(idx)
    held_x=sorted(xp[held]);train_pool=[v for i,v in enumerate(xp) if i not in held]
    xo=farthest_order(train_pool)
    qfine=np.exp(np.linspace(np.log(qmin),np.log(qmax),9));qfine[0]=qmin;qfine[-1]=qmax
    ufine=np.linspace(0.,1.,9)
    grids={}
    for level,nxi,stride in [('ideal_L0',5,4),('ideal_L1',9,2),('ideal_L2',17,1)]:
        xis=np.sort(xo[:nxi]);qs=qfine[::stride];us=ufine[::stride]
        wx=trapezoid_weights(xis);wq=trapezoid_weights(np.log(qs));wu=trapezoid_weights(us)
        rows=[]
        for ix,xi in enumerate(xis):
            for iq,q in enumerate(qs):
                for iu,u in enumerate(us):
                    rows.append({'row_id':f'{level}:{len(rows):05d}', 'xi':float(xi),'Q2_GeV2':float(q),
                        't_GeV2':float(chart_t(xi,u,tmin,tmax)), 'u_t':float(u),
                        'mass':float(wx[ix]*wq[iq]*wu[iu]),'sigma_Re':.04,'sigma_Im':.04,
                        'role':'noise_free_surface','leading_power_boundary_flag':0})
        grids[level]=rows
    # All three coordinates held out relative to the finest lattice: no fit selection.
    qcheck=np.sqrt(qfine[:-1]*qfine[1:]);ucheck=(ufine[:-1]+ufine[1:])/2
    check=[]
    for xi,q,u in itertools.product(held_x,qcheck,ucheck):
        check.append({'row_id':f'surface_check:{len(check):05d}','xi':float(xi),'Q2_GeV2':float(q),
            't_GeV2':float(chart_t(xi,u,tmin,tmax)), 'u_t':float(u), 'mass':1./(len(held_x)*len(qcheck)*len(ucheck)),
            'sigma_Re':.04,'sigma_Im':.04,'role':'prediction_check_only','leading_power_boundary_flag':0})
    grids['surface_check']=check
    # All exact experimental rows are kept; no silent kinematic cleanup/averaging.
    exp=[]
    for r in src:
        exp.append({'row_id':r['row_id'],'xi':r['xi'],'Q2_GeV2':r['Q2_GeV2'],'t_GeV2':r['t_GeV2'],
            'u_t':None,'mass':1./len(src),'sigma_Re':r['sigma_reported'],
            'sigma_Im':r['sigma_reported'],'role':'experimental_geometry_pseudodata',
            'leading_power_boundary_flag':int(r['leading_power_boundary_flag'])})
    grids['experimental']=exp
    return {'ranges':{'xi':[float(xp[0]),float(xp[-1])],'Q2':[qmin,qmax],'t':[tmin,tmax]},
        'ideal_chart':'xi; uniform u mapping [tmin,min(tmax,t0(xi)-.01)]; log(Q2)',
        'ideal_weights':'normalized product trapezoid weights in xi,u,log(Q2), NOT independent noisy observations',
        'training_xi_order':list(map(float,xo[:17])),'heldout_xi':list(map(float,held_x)),
        'cached_xi_pool':list(map(float,xp)), 'grids':grids}

def map_requests(geometry):
    requests={}
    for role,rows in geometry['grids'].items():
        for r in rows:
            key=coordinate_key(r['xi'],r['Q2_GeV2'])
            if key not in requests:
                requests[key]={'xi':r['xi'],'Q2':r['Q2_GeV2'],'xi_hex':float(r['xi']).hex(),
                    'Q2_hex':float(r['Q2_GeV2']).hex(),'roles':[]}
            if role not in requests[key]['roles']:requests[key]['roles'].append(role)
    return [requests[k] for k in sorted(requests)]

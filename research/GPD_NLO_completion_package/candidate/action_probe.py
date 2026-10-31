"""Continuous analytic-input actions, source-shared but independent quadrature.
No matrix assembly. Both sides of y=+/-x are paired before summing.
"""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.source_parts import SourceParts,SECTORS
from src.replay import analytic_input

def bridge():
    records=list((ROOT/'work/point_bridge').glob('*/BUILD.json'))
    if len(records)!=1:raise RuntimeError('Expected one successful scalar build')
    m=json.loads(records[0].read_text());return SourceParts(records[0].parent/'parts.dylib'),m

def panels(x,xi):
    cuts=sorted(set([-1.,-xi,-abs(x),0.,abs(x),xi,1.]))
    poles=[-abs(x),abs(x)];radii=[]
    for p in poles:
        i=cuts.index(p);radii.append(.45*min(p-cuts[i-1],cuts[i+1]-p))
    allcuts=sorted(set(cuts+[p-r for p,r in zip(poles,radii)]+[p+r for p,r in zip(poles,radii)]))
    regular=[]
    for a,b in zip(allcuts[:-1],allcuts[1:]):
        mid=(a+b)/2
        if not any(abs(mid-p)<r for p,r in zip(poles,radii)):regular.append((a,b))
    return regular,list(zip(poles,radii))

def integrate(fn,x,xi,n):
    z,w=leggauss(n);values=[]
    reg,pairs=panels(x,xi)
    for a,b in reg:
        y=(a+b)/2+(b-a)/2*z
        values.append(np.einsum('i,i...->...',w,fn(y))*(b-a)/2)
    for p,r in pairs:
        t=r/2*(1+z)
        values.append(np.einsum('i,i...->...',w,fn(p+t)+fn(p-t))*r/2)
    return np.sum(values,axis=0)

def action(p,order,s,x,xi,n,f):
    probe=(x+1)/2
    while min(abs(probe-x),abs(probe+x),abs(probe-xi),abs(probe+xi))<1e-12:probe=(probe+1)/2
    _,_,local=p.evaluate(order,s,x,xi,[probe],include_local=True)
    fx=f(np.array([x]))[0]
    def fn(y):
        r,k,_=p.evaluate(order,s,x,xi,y)
        fy=f(y)
        return r*fy+k*(fy-fx)
    return float(integrate(fn,x,xi,n)+local*fx),local

def main():
    p,meta=bridge();start=time.monotonic();rows=[]
    for x in [.03,.09,.134,.136,.19,.67]:
      for order in [0,1]:
       for s in SECTORS:
        idx=2 if s in ['QG','GG'] else (1 if s.startswith('NS') else 0)
        f=lambda y:analytic_input(y)[idx]
        row={'x':x,'xi':.135,'order':order,'sector':s,'input_channel':idx,'values':[]}
        for n in [16,32,64]:
            try:
                val,local=action(p,order,s,x,.135,n,f)
                row['values'].append({'n':n,'action':val,'local':local})
            except (FloatingPointError,ValueError) as exc:
                row['values'].append({'n':n,'error':str(exc)})
        rows.append(row)
        (ROOT/'work/action/probe.json').write_text(json.dumps({'source_key':meta['key'],'candidate_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.monotonic()-start,'rows':rows},indent=2,allow_nan=False)+'\n')
    print('seconds',time.monotonic()-start)
    for row in rows:
        vals=row['values']
        if all('action' in v for v in vals):
            delta=abs(vals[-1]['action']-vals[-2]['action'])
            if delta>1e-4:print(row['x'],row['order'],row['sector'],[v['action'] for v in vals],flush=True)
        else:print(row,flush=True)
if __name__=='__main__':main()

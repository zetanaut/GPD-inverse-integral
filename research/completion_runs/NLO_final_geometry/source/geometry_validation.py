"""Unchanged scientific preflight with explicit geometry and immediate native caches."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]
sys.path[:0]=[str(ROOT/'GPD_NLO_resolution_review/pilot')]
import common
pf=common.module('final_geometry_preflight',ROOT/'GPD_NLO_resolution_review/pilot/preflight.py')
NLO=common.NLO
CACHE=PREP/'native_validation';CACHE.mkdir(exist_ok=True)
pf.WORK=CACHE
SOURCES={str(p.relative_to(ROOT)):common.digest(p) for p in [Path(__file__),ROOT/'GPD_NLO_resolution_review/pilot/preflight.py']+list((NLO/'candidate').glob('*.py'))+list((NLO/'src').glob('*.py'))+list((NLO/'frozen/python').glob('*.py'))}
def ah(a):
 a=np.ascontiguousarray(a);return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
def cached(kind,meta,fn):
 meta=dict(meta,kind=kind,sources=SOURCES,Q02=2.,nf=3,scheme='MSbar',runtime_record=common.digest(PREP/'runtime_repair/BUNDLE.json'))
 token=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();path=CACHE/kind/(token+'.npz');path.parent.mkdir(exist_ok=True)
 if path.exists():
  with np.load(path,allow_pickle=False) as z:
   assert json.loads(str(z['metadata_json']))==meta
   return z['value'].copy(),json.loads(str(z['diagnostics_json']))
 start=time.monotonic();value,diag=fn();assert np.isfinite(value).all()
 np.savez_compressed(path,value=value,diagnostics_json=json.dumps(diag),metadata_json=json.dumps(meta,sort_keys=True),seconds=time.monotonic()-start)
 return value,diag
_evolve=pf.evolve
def evolve(v,k0,k1,q2,**kwargs):
 return cached('ordered_evolution',{'initial':ah(v),'K0':ah(k0),'K1':ah(k1),'Q2':q2,'settings':kwargs},lambda:_evolve(v,k0,k1,q2,**kwargs))
pf.evolve=evolve
_rk=pf.evolve_field
def rk(v,k0,k1,q2,**kwargs):
 return cached('RK4',{'initial':ah(v),'K0':ah(k0),'K1':ah(k1),'Q2':q2,'settings':kwargs},lambda:(_rk(v,k0,k1,q2,**kwargs),{}))[0]
pf.evolve_field=rk
_hard=pf.hard_row
def hard(b,q2,order):
 return cached('hard',{'basis':b.specification(),'Q2':q2,'order':order},lambda:_hard(b,q2,order))[0]
pf.hard=hard
_reference=pf.reference
def reference(b,h,q2,eps):
 value,diag=cached('independent_hard',{'basis':b.specification(),'state':ah(h),'Q2':q2,'eps':eps},lambda:_reference(b,h,q2,eps))
 return complex(value),diag
pf.reference=reference
def validate(point,q2,field,out):
 out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
 expected={'point':point,'Q2':q2,'field':field,'sources':SOURCES,'operators':{d:{'KEY':common.digest(NLO/d/'KEY.json'),'matrix':common.digest(NLO/d/'operators.npz')} for d in point['directories'].values()}}
 if out.exists():
  record=json.loads(out.read_text());assert record['provenance']==expected
  return record
 result=pf.validate_point(point,q2,field)
 result['provenance']=expected
 common.write(out,result)
 return result

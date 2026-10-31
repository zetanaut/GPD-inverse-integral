"""Reuse the registered DEVELOPMENT streams only; never create final data.
Fitter input contains y and covariance only. Truth is retained only in controller
records and cannot select a start, restore a field or produce a cutoff here.
"""
from __future__ import annotations
import os,json,hashlib,math,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def make(root,ident,out):
 registry=json.loads((HERE/'DEVELOPMENT_REGISTRY.json').read_text())
 if ident not in ('d000','d001','d002','d003'):raise ValueError('Only the four bounded development IDs are enabled')
 entry=next(r for r in registry['datasets'] if r['opaque_id']==ident)
 center=[];sources=[]
 for xi in registry['geometry']['xi']:
  for q in registry['geometry']['fit_Q2']:
   p=root/'GPD_NLO_resolution_review/pilot/work/preflight'/f'{entry["development_truth_controller_only"]}_xi{xi}_Q2{q}.json'
   rec=json.loads(p.read_text())
   if not rec['pass']:raise ValueError('Unvalidated saved DEVELOPMENT mean')
   center.extend(rec['checks']['strong10']['CFF']);sources.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 center=np.array(center);s=.04*np.maximum(1.,abs(center));L=np.zeros((18,18))
 for k in range(0,18,2):L[k:k+2,k:k+2]=[[s[k],0],[.25*s[k+1],s[k+1]*math.sqrt(1-.25**2)]]
 C=L@L.T;rng=np.random.Generator(np.random.PCG64(entry['noise_seed']));normal=rng.normal(size=18);y=center+L@normal
 out.mkdir(parents=True,exist_ok=True);dest=out/'input.npz'
 if dest.exists():
  with np.load(dest,allow_pickle=False) as d:
   if not np.array_equal(d['y'],y) or not np.array_equal(d['covariance'],C):raise ValueError('Existing registered input differs')
  return dest
 np.savez_compressed(dest,y=y,covariance=C)
 (out/'CONTROLLER.json').write_text(json.dumps({'opaque_id':ident,'development_truth':entry['development_truth_controller_only'],
 'noise_seed':entry['noise_seed'],'sources':sources,'input_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
 'note':'not calibration/test data; original data noise stream retained, new estimator uses fixed31/43 seeds'},indent=2)+'\n')
 # Cross-check original d000 rather than merely trusting the seed recipe.
 if ident=='d000':
  with np.load(root/'completion_runs/NLO_feasible_domain/inputs/d000.npz',allow_pickle=False) as d:
   if not np.array_equal(d['y'].reshape(-1),y) or not np.array_equal(d['covariance'],C):raise ValueError('d000 replay disagreement')
 return dest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default=os.environ.get('GPD_REVIEW_ROOT'));p.add_argument('--id',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 if not a.root:raise ValueError('Existing project root required')
 print(make(Path(a.root),a.id,Path(a.out)))

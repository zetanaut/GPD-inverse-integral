"""Source-keyed, per-output cached weak/Galerkin construction for one pilot.
Run only after work/STRATEGY.json is present. Single-thread execution required.
"""
from pathlib import Path
import sys,json,time,hashlib,argparse,os
import numpy as np
import shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from cached_source import CachedSource
from src.source_parts import SourceParts,SECTORS
from spectral import Elements,block,project

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--degree',type=int,default=6);ap.add_argument('--layout',default='graded');ap.add_argument('--q',type=int,default=32);ap.add_argument('--outer',type=int,default=9);ap.add_argument('--xi',type=float,default=.135);ap.add_argument('--cap',type=float,default=900);ap.add_argument('--local-n',type=int,default=0);ap.add_argument('--bridge',choices=['gg','mp'],default='mp');a=ap.parse_args()
 if not (ROOT/'work/STRATEGY.json').is_file():raise RuntimeError('strategy first')
 meta=json.loads((ROOT/('work/action/MP_BUILD.json' if a.bridge=='mp' else 'work/action/GG_BUILD.json')).read_text());p=SourceParts(ROOT/('work/mp_guard_bridge' if a.bridge=='mp' else 'work/gg_guard_bridge')/meta['key']/'parts.dylib');p=CachedSource(p,meta['key']);b=Elements(a.xi,a.degree,a.layout);x,w=b.outputs(a.outer)
 sig={'source_build_key':meta['key'],'basis':b.specification(),'inner_n':a.q,'local_n':a.local_n,'outer_n':a.outer,'output_coordinates':x.tolist(),'weights':w.tolist(),'prescription':json.loads((ROOT/'PILOT.json').read_text()),'method_sources':{v:digest(ROOT/'candidate'/v) for v in ['spectral.py','continuous_action.py','action_probe.py','assemble.py','fixed_local.py','cached_source.py']},'cap_seconds':a.cap}
 key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();wd=ROOT/'work/operators'/key;wd.mkdir(parents=True,exist_ok=True)
 (wd/'KEY.json').write_text(json.dumps(sig,indent=2)+'\n');t=time.monotonic();timings=[];actions=[]
 for i,xx in enumerate(x):
  dest=wd/f'row_{i:04}.npz'
  bm=ROOT/'work/action/CROSSOVER_TIMING.json'
  if not dest.exists() and bm.exists():
   br=json.loads(bm.read_text())
   if br.get('source_build_key')==meta['key'] and br['x']==float(xx) and br['degree']==a.degree and br['layout']==a.layout and br['inner_n']==a.q and br['local_n']==a.local_n and all(digest(ROOT/'candidate'/v)==hv for v,hv in br['method_sources'].items()):
    shutil.copy2(ROOT/'work/action/crossover_benchmark_block.npz',dest)
  if dest.exists():
   with np.load(dest) as z:ar=[[z[f'K{o}_{s}'].copy() for s in SECTORS] for o in [0,1]]
   timings.append({'i':i,'cache_hit':True})
  else:
   if time.monotonic()-t>a.cap:raise TimeoutError('finite construction cap; saved rows retained')
   t0=time.monotonic();ar,ny=block(p,b,float(xx),a.q,local_n=a.local_n)
   np.savez_compressed(dest,**{f'K{o}_{s}':ar[o][j] for o in [0,1] for j,s in enumerate(SECTORS)})
   timings.append({'i':i,'x':float(xx),'seconds':time.monotonic()-t0,'nodes':ny,'cache_hit':False})
  actions.append(ar)
  (wd/'PROGRESS.json').write_text(json.dumps({'rows':timings,'seconds':time.monotonic()-t,'key':key},indent=2)+'\n')
  print('ROW',i+1,len(x),'seconds',time.monotonic()-t,flush=True)
 mats,cond=project(b,x,w,actions)
 np.savez_compressed(wd/'operators.npz',**mats)
 (wd/'RESULT.json').write_text(json.dumps({'key':key,'seconds':time.monotonic()-t,'mass_condition_numbers':cond,'finite':all(np.isfinite(v).all() for v in mats.values()),'operator_sha256':digest(wd/'operators.npz'),'basis':b.specification()},indent=2)+'\n')
 print('COMPLETE',wd,flush=True)
if __name__=='__main__':main()

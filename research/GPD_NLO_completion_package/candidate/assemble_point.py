"""Source-keyed, per-output cached weak/Galerkin construction for one pilot.
Run only after work/STRATEGY.json is present. Single-thread execution required.
"""
from pathlib import Path
import sys,json,time,hashlib,argparse,os
import numpy as np
import shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from cached_source import CachedSource
from fast_qg import FastQG
from src.source_parts import SourceParts,SECTORS
from basis_factory import Elements
from spectral import block
from project_enriched import project

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--degree',type=int,default=6);ap.add_argument('--layout',default='log3');ap.add_argument('--q',type=int,default=32);ap.add_argument('--outer',type=int,default=9);ap.add_argument('--xi',type=float,default=.135);ap.add_argument('--cap',type=float,default=900);ap.add_argument('--local-n',type=int,default=0);ap.add_argument('--bridge',choices=['gg','mp'],default='mp');ap.add_argument('--reuse-parent',type=Path);a=ap.parse_args()
 if not (ROOT/'work/STRATEGY.json').is_file():raise RuntimeError('strategy first')
 meta=json.loads((ROOT/('work/action/MP_BUILD.json' if a.bridge=='mp' else 'work/action/GG_BUILD.json')).read_text());p=SourceParts(ROOT/('work/mp_guard_bridge' if a.bridge=='mp' else 'work/gg_guard_bridge')/meta['key']/'parts.dylib');p=CachedSource(p,meta['key']);qmeta=json.loads((ROOT/'work/action/FAST_QG_BUILD.json').read_text());p=FastQG(p,ROOT/'work/fast_qg_bridge'/qmeta['key']/'qg.dylib');b=Elements(a.xi,a.degree,a.layout);x,w=b.outputs(a.outer)
 sig={'source_build_key':meta['key'],'fast_qg_key':qmeta['key'],'basis':b.specification(),'inner_n':a.q,'local_n':a.local_n,'outer_n':a.outer,'output_coordinates':x.tolist(),'weights':w.tolist(),'prescription':json.loads((ROOT/'PILOT.json').read_text()),'method_sources':{v:digest(ROOT/'candidate'/v) for v in ['spectral.py','continuous_action.py','action_probe.py','assemble_point.py','fixed_local.py','cached_source.py','fast_qg.py','enriched.py','basis_factory.py','project_enriched.py']},'cap_seconds':a.cap}
 reused={}
 if a.reuse_parent:
  a.reuse_parent=a.reuse_parent.resolve();parent=json.loads((a.reuse_parent/'KEY.json').read_text())
  for k in ['source_build_key','basis','inner_n','local_n','outer_n','output_coordinates','weights','prescription']:
   if parent[k]!=sig[k]:raise ValueError('Incompatible parent '+k)
  for k in ['spectral.py','continuous_action.py','action_probe.py','fixed_local.py','cached_source.py']:
   if parent['method_sources'][k]!=digest(ROOT/'candidate'/k):raise ValueError('Changed parent method '+k)
  guard=json.loads((ROOT/'work/action/FAST_BLOCK_CHECK.json').read_text())
  if guard['source_key']!=meta['key'] or guard['qg_guard_key']!=qmeta['key'] or guard['maximum_row_L1_difference']!=0:raise ValueError('Arithmetic compatibility not demonstrated')
  for f in sorted(a.reuse_parent.glob('row_????.npz')):
   with np.load(f) as check:
    if not all(np.isfinite(check[k]).all() for k in check.files):raise ValueError('Nonfinite parent block')
   reused[f.name]=digest(f)
  sig['reused_parent']={'path':str(a.reuse_parent.relative_to(ROOT)),'key':a.reuse_parent.name,'row_hashes':reused,'compatibility':'Only QG guard arithmetic changes; 360 points and two complete source blocks identical. All other methods, geometry, prescriptions and local callbacks are the same.'}
 key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();wd=ROOT/'work/operators'/key;wd.mkdir(parents=True,exist_ok=True)
 (wd/'KEY.json').write_text(json.dumps(sig,indent=2)+'\n');t=time.monotonic();timings=[];actions=[]
 for i,xx in enumerate(x):
  dest=wd/f'row_{i:04}.npz'
  if not dest.exists() and dest.name in reused:shutil.copy2(a.reuse_parent/dest.name,dest)
  bm=ROOT/f'work/action/benchmark_xi_{a.xi:g}.json'
  if not dest.exists() and bm.exists():
   br=json.loads(bm.read_text())
   if br['source_key']==meta['key'] and br['qg_key']==qmeta['key'] and br['x']==float(xx) and br['basis']==b.specification() and br['inner_n']==a.q and br['outer_n']==a.outer and a.local_n==0 and all(digest(ROOT/'candidate'/name)==h for name,h in br['method_sources'].items()):
    shutil.copy2(ROOT/f'work/action/benchmark_xi_{a.xi:g}.npz',dest)
  if dest.exists():
   with np.load(dest) as z:ar=[[z[f'K{o}_{s}'].copy() for s in SECTORS] for o in [0,1]]
   timings.append({'i':i,'cache_hit':True})
  else:
   if time.monotonic()-t>a.cap:raise TimeoutError('finite construction cap; saved rows retained')
   t0=time.monotonic()
   try:ar,ny=block(p,b,float(xx),a.q,local_n=a.local_n)
   except Exception as ex:
    (wd/'FAILURE.json').write_text(json.dumps({'i':i,'x':float(xx),'xi':a.xi,'error':repr(ex),'key':key,'seconds':time.monotonic()-t},indent=2)+'\n');raise
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

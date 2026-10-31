"""Isolate crossover output quadrature, preserving all other saved source rows."""
from pathlib import Path
import sys,json,time,hashlib,argparse,shutil
import numpy as np
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]/'GPD_NLO_completion_package';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from forward import load
from src.source_parts import SourceParts,SECTORS
from cached_source import CachedSource
from fast_qg import FastQG
from spectral import block
from project_enriched import project
REPAIR=PREP.parents[1]/'GPD_final_geometry_GG_local_repair'
sys.path.insert(0,str(REPAIR))
from activation import wrap_source,identity

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);ap.add_argument('--special',type=int,required=True);ap.add_argument('--cap',type=float,default=1200);ap.add_argument('--benchmark',action='store_true');a=ap.parse_args();a.parent=a.parent.resolve();b,_,_,old=load(a.parent)
 xx,ww=b.outputs(a.special);low,lw=b.outputs(old['outer_n']);mask=np.zeros(len(xx),bool);lm=np.zeros(len(low),bool)
 for cell in b.special:
  left,right=b.edges[cell:cell+2];mask|=(xx>left)&(xx<right);lm|=(low>left)&(low<right)
 x=np.r_[low[~lm],xx[mask]];w=np.r_[lw[~lm],ww[mask]];ix=np.argsort(x);x=x[ix];w=w[ix]
 assert old.get('GG_local_fallback')==identity(), 'Parent fallback signature differs'
 sig=dict(old);sig.update({'output_coordinates':x.tolist(),'weights':w.tolist(),'special_output_n':a.special,'parent':str(a.parent.relative_to(ROOT)),'parent_key':a.parent.name,'output_refinement_source':digest(Path(__file__)),'cap_seconds':a.cap})
 key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();wd=ROOT/'work/operators'/key;wd.mkdir(exist_ok=True);(wd/'KEY.json').write_text(json.dumps(sig,indent=2)+'\n')
 p=FastQG(wrap_source(CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/old['source_build_key']/'parts.dylib'),old['source_build_key'])),ROOT/'work/fast_qg_bridge'/old['fast_qg_key']/'qg.dylib')
 previous={float(z).hex():i for i,z in enumerate(old['output_coordinates'])};t=time.monotonic();timings=[];actions=[]
 order=[int(np.argmin(abs(x-b.xi)))] if a.benchmark else range(len(x))
 for i in order:
  z=float(x[i]);dest=wd/f'row_{i:04}.npz';entry={'i':i,'x':z};st=time.monotonic()
  if not dest.exists() and z.hex() in previous:
   src=a.parent/f'row_{previous[z.hex()]:04}.npz';shutil.copy2(src,dest);entry['parent_row_sha256']=digest(src)
  if dest.exists():
   with np.load(dest) as f:ar=[[f[f'K{o}_{s}'].copy() for s in SECTORS] for o in [0,1]]
   entry['cache_hit']=True
  else:
   if time.monotonic()-t>a.cap:raise TimeoutError('bounded output refinement; rows retained')
   try:ar,ny=block(p,b,z,old['inner_n'],old['local_n'])
   except Exception as ex:
    (wd/'FAILURE.json').write_text(json.dumps({'i':i,'x':z,'xi':b.xi,'error':repr(ex),'seconds':time.monotonic()-t},indent=2)+'\n');raise
   np.savez_compressed(dest,**{f'K{o}_{s}':ar[o][j] for o in [0,1] for j,s in enumerate(SECTORS)});entry.update(nodes=ny,cache_hit=False)
  entry['seconds']=time.monotonic()-st;timings.append(entry);actions.append(ar);print('ROW',i+1,len(x),entry,flush=True)
  (wd/('BENCHMARK.json' if a.benchmark else 'PROGRESS.json')).write_text(json.dumps({'rows':timings,'seconds':time.monotonic()-t,'key':key},indent=2)+'\n')
 if not a.benchmark:
  mats,cond=project(b,x,w,actions);np.savez_compressed(wd/'operators.npz',**mats);(wd/'RESULT.json').write_text(json.dumps({'key':key,'seconds':time.monotonic()-t,'mass_condition_numbers':cond,'operator_sha256':digest(wd/'operators.npz'),'finite':all(np.isfinite(v).all() for v in mats.values())},indent=2)+'\n')
 print('COMPLETE',wd,flush=True)
if __name__=='__main__':main()

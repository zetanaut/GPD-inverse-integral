"""Apply an exact add/subtract quadrature correction to saved source rows.
No kernel evaluation. The quadrature of the analytic log-squared subtraction
is replaced by its exact integral times the input value f(x).
"""
from pathlib import Path
import sys,json,hashlib,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from forward import load
from continuous_action import geometry
from qg_subtraction import subtraction
from project_enriched import project
from action_probe import SECTORS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);a=ap.parse_args();a.parent=a.parent.resolve();b,_,_,old=load(a.parent);x=np.array(old['output_coordinates']);w=np.array(old['weights']);sig=dict(old);sig.update(parent=str(a.parent.relative_to(ROOT)),parent_key=a.parent.name,qg_endpoint_subtraction=True,correction_method={s:sha(ROOT/'candidate'/s) for s in ['correct_qg_rows.py','qg_subtraction.py','project_enriched.py']},parent_row_hashes={p.name:sha(p) for p in sorted(a.parent.glob('row_????.npz'))})
 key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();wd=ROOT/'work/operators'/key;wd.mkdir(exist_ok=True);(wd/'KEY.json').write_text(json.dumps(sig,indent=2)+'\n');actions=[];records=[];start=time.monotonic()
 for i,xx in enumerate(x):
  with np.load(a.parent/f'row_{i:04}.npz') as f:ar=[[f[f'K{o}_{s}'].copy() for s in SECTORS] for o in [0,1]]
  y,ww=geometry(float(xx),b.xi,old['inner_n'],extra=np.r_[-b.edges[::-1],b.edges]);s,exact=subtraction(y,float(xx),b.xi);correction=exact-float(ww@s);ar[1][3]+=correction*b.basis([xx],'even')[0];actions.append(ar);records.append({'i':i,'x':float(xx),'scalar_quadrature_correction':correction});np.savez_compressed(wd/f'row_{i:04}.npz',**{f'K{o}_{s}':ar[o][j] for o in [0,1] for j,s in enumerate(SECTORS)})
 mats,cond=project(b,x,w,actions);np.savez_compressed(wd/'operators.npz',**mats);(wd/'RESULT.json').write_text(json.dumps({'operator_sha256':sha(wd/'operators.npz'),'finite':all(np.isfinite(v).all() for v in mats.values()),'mass_condition_numbers':cond,'seconds':time.monotonic()-start,'no_kernel_evaluations':True,'correction':records},indent=2)+'\n');print(wd)
if __name__=='__main__':main()

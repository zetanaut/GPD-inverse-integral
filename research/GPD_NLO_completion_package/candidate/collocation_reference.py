"""Strong-form reference from saved continuous source actions, no new kernels.
Pivoted QR selects interior interpolation locations; no mass matrix or weak
projection is used. Kernel formulas/quadrature are SHARED and explicitly keyed.
This is an independent discretization control, not an independent QCD oracle.
"""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
from scipy.linalg import qr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from forward import load
from basis_factory import Elements
from action_probe import SECTORS

def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);ap.add_argument('--degree',type=int,required=True);a=ap.parse_args();a.parent=a.parent.resolve();hi,k0,k1,key=load(a.parent);lo=Elements(hi.xi,a.degree,hi.layout)
 with np.load(a.parent/'operators.npz') as z:x=z['x'].copy();w=z['weights'].copy()
 B={p:lo.basis(x,p) for p in ['odd','even']};selected={};cond={};R={}
 for p,ids in [('odd',hi.qids),('even',hi.gids)]:
  _,_,piv=qr(B[p].T,pivoting=True,mode='economic');selected[p]=np.sort(piv[:B[p].shape[1]]);cond[p]=float(np.linalg.cond(B[p][selected[p]]));R[p]=lo.basis(hi.nodes[ids],p)
 actions=[]
 for i in range(len(x)):
  with np.load(a.parent/f'row_{i:04}.npz') as z:actions.append({k:z[k].copy() for k in z.files})
 mats={}
 for order in [0,1]:
  sec=[]
  for s in SECTORS:
   inp='even' if s in ['QG','GG'] else 'odd';outp='even' if s in ['GQ','GG'] else 'odd';idx=selected[outp];ar=np.stack([actions[i][f'K{order}_{s}'] for i in idx]);mat=np.linalg.solve(B[outp][idx],ar@R[inp]);mats[f'K{order}_{s}']=mat;sec.append(mat)
  nq=len(lo.qids);ng=len(lo.gids);z=np.zeros((nq,nq));qg=np.zeros((nq,ng));gq=np.zeros((ng,nq));mats[f'K{order}']=np.block([[sec[2],z,sec[3]],[z,sec[0],qg],[sec[4],gq,sec[5]]])
 sig={'parent':str(a.parent.relative_to(ROOT)),'parent_key':a.parent.name,'basis':lo.specification(),'method':'QR_selected_strong_collocation','selected_indices':{p:ii.tolist() for p,ii in selected.items()},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parent_operator_sha256':hashlib.sha256((a.parent/'operators.npz').read_bytes()).hexdigest(),'independence':'distinct strong discretization; shared kernel point functions, local prescription and source quadrature'}
 hkey=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();out=ROOT/'work/operators'/hkey;out.mkdir(exist_ok=True);np.savez_compressed(out/'operators.npz',**mats,x=x,weights=w);(out/'KEY.json').write_text(json.dumps(sig,indent=2)+'\n');(out/'RESULT.json').write_text(json.dumps({'interpolation_condition_numbers':cond,'finite':all(np.isfinite(v).all() for v in mats.values()),'no_conservation_projection':True},indent=2)+'\n');print(out)
if __name__=='__main__':main()

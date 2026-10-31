"""Restrict a refined weak operator to an exactly nested lower degree space.
With V6 subset V8, P6 P8 = P6, hence P6 K8 R8<-6 equals direct lower-order
Galerkin assembly at the SAME source quadrature; no new kernels are evaluated.
"""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
from scipy.linalg import block_diag
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from forward import load
from basis_factory import Elements

def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);ap.add_argument('--degree',type=int,default=6);a=ap.parse_args();hi,k0,k1,key=load(a.parent);lo=Elements(hi.xi,a.degree,hi.layout)
 with np.load(a.parent/'operators.npz') as f:x=f['x'].copy();w=f['weights'].copy()
 P=[];R=[];errors=[]
 for parity,hiids,loids in [('odd',hi.qids,lo.qids),('even',hi.gids,lo.gids)]:
  h=hi.basis(x,parity);l=lo.basis(x,parity)
  mass=l.T@(w[:,None]*l);rhs=l.T@(w[:,None]*h);scale=1/np.sqrt(np.diag(mass));proj=scale[:,None]*np.linalg.solve(mass*scale[:,None]*scale,scale[:,None]*rhs)
  embed=lo.basis(hi.nodes[hiids],parity)
  P.append(proj);R.append(embed);errors.append(float(np.max(abs(h@embed-l))))
 pp=block_diag(P[0],P[0],P[1]);rr=block_diag(R[0],R[0],R[1]);sig={'parent':str(a.parent.relative_to(ROOT)),'parent_key':a.parent.name,'basis':lo.specification(),'method':'exact_nested_mass_projection','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parent_operator_sha256':hashlib.sha256((a.parent/'operators.npz').read_bytes()).hexdigest()};hkey=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();out=ROOT/'work/operators'/hkey;out.mkdir(exist_ok=True)
 np.savez_compressed(out/'operators.npz',K0=pp@k0@rr,K1=pp@k1@rr,x=x,weights=w)
 (out/'KEY.json').write_text(json.dumps(sig,indent=2)+'\n');(out/'RESULT.json').write_text(json.dumps({'nested_basis_max_abs_error':errors,'left_inverse_error':float(np.max(abs(pp@rr-np.eye(len(pp))))),'finite':True},indent=2)+'\n');print(out)
if __name__=='__main__':main()

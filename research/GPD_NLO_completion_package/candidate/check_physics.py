from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from forward import load,initial,evaluate_field

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();b,k0,k1,key=load(a.directory);v=initial(b);x,w=b.outputs(32);nq=len(b.qids);mom=np.r_[(w*x)@b.basis(x,'odd'),np.zeros(nq),w@b.basis(x,'even')];records=[];m0=np.array([[-16/9,1.],[16/9,-1.]]);m1=np.array([[-2000/243,611/108],[2000/243,-611/108]]);expected=[m0,m1]
 for source in ['Sigma','Hg','T_rescaled']:
  u=np.zeros_like(v)
  if source=='Sigma':u[:nq]=v[:nq]
  elif source=='Hg':u[2*nq:]=v[2*nq:]
  else:u[nq:2*nq]=v[nq:2*nq]
  for order,k in [(0,k0),(1,k1)]:
   act=k@u;f=evaluate_field(b,act,x);q=float(w@(x*f[0]));g=float(w@f[2]);mt=float((w*x)@b.basis(x,'odd')@u[nq:2*nq]);tgot=float(w@(x*f[1]));ref=expected[order]@np.array([float((w*x)@b.basis(x,'odd')@u[:nq]),float(w@b.basis(x,'even')@u[2*nq:])]);tref=(-16/9 if order==0 else -2360/243)*mt;records.append({'reference_quark':float(ref[0]),'reference_gluon':float(ref[1]),'reference_T':float(tref),'T_moment_derivative':tgot,'component_errors':[abs(q-ref[0]),abs(g-ref[1]),abs(tgot-tref)],'input':source,'order':order,'quark_momentum_derivative':q,'gluon_momentum_derivative':g,'conservation_residual':float(mom@act),'normalization':float(mom@u)})
 out={'operator_key':a.directory.name,'momentum_columns':records,'no_enforcement':True,'independent_reference':'Altenbuchinger et al., arXiv:1012.4409v2 Eqs(7),(8),Table1; spin2 MSbar nf3 anomalous dimensions, converted alpha/(4pi) to alpha/(2pi).','reference_url':'https://arxiv.org/pdf/1012.4409','spin2_matrices':{'K0':m0.tolist(),'K1':m1.tolist()},'scope':'Full-support polynomial input, pure quark/gluon/T columns. Numerical moment diagnostics; no conservation projection.'}
 (a.directory/'PHYSICS_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

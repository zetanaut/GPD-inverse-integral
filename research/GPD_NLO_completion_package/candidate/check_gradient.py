"""Tangent-linear derivative reference, plus a resolvable centered difference.
The map is exactly linear. Larger differences avoid subtractive cancellation
seen in the separately retained 1e-5/1e-6 diagnostics; no physics is tuned.
"""
from pathlib import Path
import sys,json,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from forward import load,initial,hard_row,jacobian
from src.ordered_evolution import evolve
def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);ap.add_argument('--q2',type=float,required=True);a=ap.parse_args();a.parent=a.parent.resolve();p=json.loads((a.parent/f'POINT_Q2_{a.q2:g}.json').read_text());records=[]
 for label in ['weak10','strong10']:
  directory=ROOT/p['directories'][label];b,k0,k1,_=load(directory);v=initial(b);row,_=hard_row(b,a.q2,192);j,_=jacobian(directory,a.q2);nq=len(b.qids);rng=np.random.default_rng(731)
  for ch,sl in [('Sigma',slice(0,nq)),('T_rescaled',slice(nq,2*nq)),('Hg',slice(2*nq,len(v)))]:
   d=np.zeros_like(v);d[sl]=rng.normal(size=len(d[sl]));d/=np.linalg.norm(d);tangent,_=evolve(d,k0,k1,a.q2,rtol=1e-12,atol=1e-14);analytic=complex(*(j@d));direct=row@tangent;eps=1e-3;hp,_=evolve(v+eps*d,k0,k1,a.q2,rtol=1e-12,atol=1e-14);hm,_=evolve(v-eps*d,k0,k1,a.q2,rtol=1e-12,atol=1e-14);fd=row@(hp-hm)/(2*eps);records.append({'method':label,'channel':ch,'adjoint':[analytic.real,analytic.imag],'tangent_linear_absolute_error':float(abs(direct-analytic)),'centered_step':eps,'centered_absolute_error':float(abs(fd-analytic)),'preserved_smaller_step_diagnostics':str((a.parent/f'POINT_Q2_{a.q2:g}.json').relative_to(ROOT))})
 result={'xi':b.xi,'Q2':a.q2,'checks':records,'independence':'Forward tangent ODE versus backward transpose adjoint, with centered differences as a separate arithmetic check; exact linearity avoids a truncation-error tradeoff.'};(a.parent/'GRADIENT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

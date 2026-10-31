"""Independent continuum coefficient integration on saved evolved fields.
Subtract a smooth full-support polynomial matching the crossover value, then
integrate its full source coefficient on a deformed contour. Integrate the
remaining real-axis residual adaptively using the independent source formulas.
No nominal hard rows, constant-mode integrals, or Gauss rules are used here.
"""
from pathlib import Path
import sys,json,argparse,warnings,time
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'));sys.path.insert(0,str(ROOT/'evidence/hard_verified/corrected'))
from forward import load
from frozen.python.nlo_forward import alpha_two_loop
from reference_hard_contour import direct_contour,_base_quark,_base_gluon
def reference(b,h,q2,eps):
 xi=b.xi;nq=len(b.qids);qcoef=2/9*(h[:nq]+h[nq:2*nq]);gcoef=h[2*nq:];bq=b.basis([xi],'odd')[0];bg=b.basis([xi],'even')[0];qc=float(bq@qcoef);gc=float(bg@gcoef);fq=lambda z:qc*z*(1-z*z)**2/(xi*(1-xi*xi)**2);fg=lambda z:gc*(1-z*z)**2/(1-xi*xi)**2;alpha=alpha_two_loop(q2);smooth,serr=direct_contour(xi,alpha,fq,fg);total=0j;errs=[];messages=[]
 def fn(x):
  # Infinitesimal chooses the branch only; all queried denominators are nonzero.
  z=complex(x,1e-300);cq=-(_base_quark(z,xi,alpha,0.)-_base_quark(-z,xi,alpha,0.));cg=-4/3*(_base_gluon(z,xi,alpha,0.)+_base_gluon(-z,xi,alpha,0.));dq=float((b.basis([x],'odd')[0]-bq)@qcoef)-(fq(x)-qc);dg=float((b.basis([x],'even')[0]-bg)@gcoef)-(fg(x)-gc);return cq*dq+cg*dg
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  for l,r in zip(b.edges[:-1],b.edges[1:]):
   re=quad(lambda x:fn(x).real,l,r,epsabs=eps/len(b.edges),epsrel=eps,limit=250);im=quad(lambda x:fn(x).imag,l,r,epsabs=eps/len(b.edges),epsrel=eps,limit=250);total+=complex(re[0],im[0]);errs.append([re[1],im[1]])
  messages=[str(w.message) for w in caught]
 return smooth+total,{'eps':eps,'reference_quad_error_estimate':(np.sum(errs,axis=0)+serr).tolist(),'warnings':messages,'subtraction':'smooth full-support polynomial, matched crossover value; separate source contour'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);ap.add_argument('--q2',type=float,required=True);a=ap.parse_args();p=json.loads((a.parent/f'POINT_Q2_{a.q2:g}.json').read_text());out={};t0=time.monotonic()
 for label in ['weak10','strong10']:
  b,*_=load(ROOT/p['directories'][label]);out[label]={}
  for truth in ['smooth','Dterm_added']:
   r=p['results'][label][truth]
   with np.load(ROOT/r['native_field']) as f:h=f['evolved'].copy()
   vals=[]
   for eps in [1e-7,1e-9]:
    ref,diag=reference(b,h,a.q2,eps);nom=complex(*r['CFF']);vals.append({'reference':[ref.real,ref.imag],'candidate':[nom.real,nom.imag],'component_difference':[float((nom-ref).real),float((nom-ref).imag)],**diag})
   out[label][truth]=vals;print(label,truth,vals[-1],flush=True)
 result={'xi':b.xi,'Q2':a.q2,'checks':out,'seconds':time.monotonic()-t0,'independence':'Independent hard source formulas, analytic smooth-mode contour, adaptive real residual; shared represented fields, separate from nominal row and constant-mode quadrature.'};(a.parent/'HARD_REFERENCE.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

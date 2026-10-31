from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'));sys.path.insert(0,str(ROOT/'evidence/hard_verified/corrected'))
from forward import load,initial,hard_row,evaluate_field,HardGrid
from src.ordered_evolution import evolve
from reference_hard_contour import direct_contour
from nlo_forward import alpha_two_loop
import nlo_hard

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--q2',type=float,default=2.3);a=ap.parse_args();b,k0,k1,key=load(a.directory);v=initial(b);q2=a.q2;alpha=alpha_two_loop(q2);xi=b.xi
 row,_=hard_row(b,q2,96);nq=len(b.qids);checks=[]
 for kind in ['quark','gluon','both']:
  f=v.copy()
  if kind=='quark':f[2*nq:]=0
  if kind=='gluon':f[:2*nq]=0
  q=lambda z:2/9*(2*(1+.2*xi*xi)+.5*(1-.1*xi*xi))*z*(1-z*z)**2 if kind!='gluon' else 0*z
  g=lambda z:.8*(1+.1*xi*xi)*(1-z*z)**2 if kind!='quark' else 0*z
  ref,err=direct_contour(xi,alpha,q,g);nom=row@f
  checks.append({'channel':kind,'candidate':[nom.real,nom.imag],'reference':[ref.real,ref.imag],'abs_components':[abs((nom-ref).real),abs((nom-ref).imag)],'reference_error_estimate':err})
 h,_=evolve(v,k0,k1,q2);adj,_=evolve(row,k0,k1,q2,adjoint=True);rng=np.random.default_rng(731);d=rng.normal(size=len(v));d/=np.linalg.norm(d);der=[]
 for step in [1e-4,1e-5,1e-6]:
  hp,_=evolve(v+step*d,k0,k1,q2);hm,_=evolve(v-step*d,k0,k1,q2);fd=row@(hp-hm)/(2*step);exact=adj@d;der.append({'step':step,'absolute_error':float(abs(fd-exact))})
 grid=HardGrid(b,'odd');y,w=nlo_hard._subtracted_nodes(grid,96);B=grid.interpolation(y);Bc=grid.interpolation([xi])[0];lo=(w/(y-xi))@(B-Bc)+(w/(y+xi))@B+(np.log((1-xi)/xi)-1j*np.pi)*Bc
 lorow=np.r_[2/9*lo,2/9*lo,np.zeros(len(b.gids))];logrow=hard_row(b,q2,96,1.)[0]-row
 cancel=alpha/(2*np.pi)*(lorow@(k0@v))-logrow@v
 result={'hard_adapter':checks,'adjoint_absolute_error':float(abs(adj@v-row@h)),'directional_derivatives':der,'factorization_log_retained_order_residual':[float(cancel.real),float(cancel.imag)],'factorization_evolution_term':[float((alpha/(2*np.pi)*(lorow@(k0@v))).real),float((alpha/(2*np.pi)*(lorow@(k0@v))).imag)],'factorization_hard_term':[float((logrow@v).real),float((logrow@v).imag)],'status':'candidate controls only'}
 (a.directory/'API_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

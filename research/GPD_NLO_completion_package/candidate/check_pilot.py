from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from forward import load,initial,evaluate_field,hard_row
from src.ordered_evolution import evolve
from frozen.python.nlo_forward import evolve_field

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--q2',type=float,default=2.3);a=ap.parse_args();b,k0,k1,key=load(a.directory);v=initial(b);t=time.monotonic()
 h,diag=evolve(v,k0,k1,a.q2,rtol=1e-11,atol=1e-13)
 np.savez_compressed(a.directory/'smooth_field.npz',initial=v,evolved=h,nodes=b.nodes)
 rows={};out={}
 for order in [48,96]:
  row,hd=hard_row(b,a.q2,order);c=row@h;rows[order]=row;out[str(order)]={'Re':float(c.real),'Im':float(c.imag),'hard':hd}
 href,diagr=evolve(v,k0,k1,a.q2,rtol=1e-13,atol=1e-15)
 hrk=evolve_field(v,k0,k1,a.q2,steps=64)
 x,w=b.outputs(24);hvals=evaluate_field(b,h,x);initvals=evaluate_field(b,v,x)
 momentum0=float(w@(x*initvals[0]+initvals[2]));momentum=float(w@(x*hvals[0]+hvals[2]));nq=len(b.qids)
 actions=[]
 for o,k in [(0,k0),(1,k1)]:
  av=evaluate_field(b,k@v,x);actions.append({'order':o,'momentum_derivative':float(w@(x*av[0]+av[2]))})
 result={'basis':b.specification(),'q2':a.q2,'cff':out,'time_field_absolute':float(max(np.max(abs(h-href)),np.max(abs(h-hrk)))),'time_CFF_component_absolute':np.abs(np.array([((rows[96]@(h-href)).real),((rows[96]@(h-href)).imag)])).tolist(),'hard_quadrature_component_change':[float(abs(((rows[96]-rows[48])@h).real)),float(abs(((rows[96]-rows[48])@h).imag))],'momentum_initial':momentum0,'momentum_evolved':momentum,'momentum_generator':actions,'eigenvalue_max_real':float(np.linalg.eigvals(k0*.05+k1*.05**2).real.max()),'seconds':time.monotonic()-t,'status':'diagnostic only; no independent-reference acceptance yet'}
 (a.directory/'PILOT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

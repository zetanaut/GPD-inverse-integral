"""Source reflection and ordered-composition controls at a saved frozen point."""
from pathlib import Path
import sys,json,argparse,numpy as np
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]/'GPD_NLO_completion_package';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from forward import load,initial,hard_row,evaluate_field
from src.ordered_evolution import evolve,Prescription
from frozen.python.nlo_forward import alpha_two_loop
from src.source_parts import SourceParts,SECTORS
from cached_source import CachedSource
from fast_qg import FastQG
def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);ap.add_argument('--q2',type=float,required=True);a=ap.parse_args();b,k0,k1,key=load(a.parent);v=initial(b);mid=np.sqrt(2*a.q2);hm,_=evolve(v,k0,k1,mid,rtol=1e-12,atol=1e-14);h2,_=evolve(hm,k0,k1,a.q2,Prescription(q02=mid,alpha0=alpha_two_loop(mid)),rtol=1e-12,atol=1e-14);h,_=evolve(v,k0,k1,a.q2,rtol=1e-12,atol=1e-14);row,_=hard_row(b,a.q2,192)
 x=np.array([.13,.59,.93]);f=evaluate_field(b,h,x);fneg=evaluate_field(b,h,-x);field=float(np.max(abs(fneg-np.array([-1,-1,1])[:,None]*f)))
 m=json.loads((PREP/'runtime_repair/MP_BUILD.json').read_text());q=json.loads((PREP/'runtime_repair/FAST_QG_BUILD.json').read_text());p=FastQG(CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/m['key']/'parts.dylib'),m['key']),ROOT/'work/fast_qg_bridge'/q['key']/'qg.dylib');parity=[]
 for xx in [.55*b.xi,b.xi+.4*(1-b.xi)]:
  y=np.array([-.78,-.7*b.xi,.2*b.xi,.66,.95])
  for order in [0,1]:
   for s in SECTORS:
    r,k,_=p.evaluate(order,s,xx,b.xi,y);rr,kk,_=p.evaluate(order,s,-xx,b.xi,-y);sign=-1 if s in ['QG','GQ'] else 1;parity.append({'x':xx,'order':order,'sector':s,'max_scaled_error':float(max(np.max(abs(rr-sign*r)/np.maximum(1,abs(r))),np.max(abs(kk-sign*k)/np.maximum(1,abs(k)))))})
 result={'xi':b.xi,'Q2':a.q2,'composition_midpoint_Q2':float(mid),'composition_field_max_abs':float(np.max(abs(h-h2))),'composition_CFF_abs':float(abs(row@(h-h2))),'generator_commutator_relative_norm':float(np.linalg.norm(k0@k1-k1@k0)/max(1,np.linalg.norm(k0)*np.linalg.norm(k1))),'represented_field_parity_error':field,'source_parity':parity,'source_key':m['key'],'scope':'source reflection, represented parity and noncommuting time-ordered composition; no new operators'};(a.parent/'STRUCTURE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

"""Full-support physical even NS-minus source charge conservation.
Uses independent callable polynomial input, no odd-grid valence surrogate.
All local values are already source-keyed; no evolution is performed.
"""
from pathlib import Path
import sys,json,argparse,time
import numpy as np
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]/'GPD_NLO_completion_package';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from cached_source import CachedSource
from continuous_action import geometry,source_local

def main():
 ap=argparse.ArgumentParser();ap.add_argument('parent',type=Path);a=ap.parse_args();key=json.loads((a.parent/'KEY.json').read_text());xi=key['basis']['xi'];x=np.array(key['output_coordinates']);w=np.array(key['weights']);meta=json.loads((PREP/'runtime_repair/MP_BUILD.json').read_text());assert key['source_build_key']==meta['key'];p=CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/meta['key']/'parts.dylib'),meta['key']);path=a.parent/'ns_charge_rows.json';rows=json.loads(path.read_text()) if path.exists() else [];t=time.monotonic()
 for i,xx in enumerate(x):
  if i<len(rows):continue
  values=[]
  for n in [32,48]:
   y,wt=geometry(float(xx),xi,n);r,k,_=p.evaluate(1,'NS_minus',float(xx),xi,y);fy=(1-y*y)**2;fx=(1-xx*xx)**2;values.append(float(wt@(r*fy+k*(fy-fx))+source_local(p,1,'NS_minus',float(xx),xi)*fx))
  rows.append({'i':i,'x':float(xx),'actions':values});path.write_text(json.dumps(rows,indent=2)+'\n')
 values=np.array([r['actions'] for r in rows]);mom=2*w@values;result={'full_even_input':'(1-x^2)^2 on [-1,1]','charge':16/15,'nlo_charge_derivative':mom.tolist(),'inner_quadrature_change':float(abs(mom[1]-mom[0])),'source_key':meta['key'],'seconds':time.monotonic()-t,'physical_identity':'NS_minus zeroth moment is conserved; no explicit delta beyond plus prescription','scope':'independent even analytic input; no valence field evolution or odd-grid substitution'};(a.parent/'NS_CHARGE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

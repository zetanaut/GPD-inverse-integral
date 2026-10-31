from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from continuous_action import geometry
m=json.loads((ROOT/'work/action/GG_BUILD.json').read_text());p=SourceParts(ROOT/'work/gg_guard_bridge'/m['key']/'parts.dylib');rows=[];t=time.monotonic()
refs={.13499:13.04543494207196,.13481531067484:12.984819724595514,.09:17.58631476944388}
for x in [.09,.13481531067484,.13499,.134998,.135002]:
 row={'x':x,'reference_source_local':refs.get(x),'quadratures':[]}
 for n in [24,48,72]:
  t0=time.monotonic();y,w=geometry(x,.135,n,pair_power=1);_,k,_=p.evaluate(1,'GG',x,.135,y)
  kr=np.array([p.evaluate(1,'GG',float(z),.135,[x])[1][0] for z in y]);v=float(w@(k-kr)-3/108*(35*3+74*4/3))
  rec={'n':n,'value':v,'seconds':time.monotonic()-t0,'nodes':len(y)};row['quadratures'].append(rec);print(x,rec,flush=True)
  (ROOT/'work/action/FIXED_LOCAL.json').write_text(json.dumps({'source_key':m['key'],'rows':rows+[row],'seconds':time.monotonic()-t},indent=2)+'\n')
 rows.append(row)

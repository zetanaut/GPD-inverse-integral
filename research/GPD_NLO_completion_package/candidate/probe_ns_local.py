from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from continuous_action import geometry
m=json.loads((ROOT/'work/action/GG_BUILD.json').read_text());p=SourceParts(ROOT/'work/gg_guard_bridge'/m['key']/'parts.dylib');old=json.loads((ROOT/'work/action/CROSSOVER_LOCAL_FAILURE.json').read_text());rows=[];t=time.monotonic()
for x in [old['x'],2*.135-old['x'],.09]:
 for sector in ['NS_plus','NS_minus']:
  row={'x':x,'sector':sector,'values':[]}
  for n in [24,48,72,96]:
   y,w=geometry(x,.135,n,pair_power=1);_,k,_=p.evaluate(1,sector,x,.135,y);kr=np.array([p.evaluate(1,sector,float(z),.135,[x])[1][0] for z in y]);delta=(4/3)*(4/3-1.5)*(6.5-np.pi**2+4*1.202056903159594) if sector=='NS_plus' else 0
   val=float(w@(k-kr)+delta);row['values'].append({'n':n,'value':val});print(x,sector,n,val,flush=True)
  rows.append(row);(ROOT/'work/action/NS_LOCAL_DIRECT.json').write_text(json.dumps({'source_key':m['key'],'rows':rows,'seconds':time.monotonic()-t},indent=2)+'\n')

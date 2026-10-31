from pathlib import Path
import sys,json,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from continuous_action import source_local,geometry
m=json.loads((ROOT/'work/action/MP_BUILD.json').read_text());p=SourceParts(ROOT/'work/mp_guard_bridge'/m['key']/'parts.dylib');xc=json.loads((ROOT/'work/action/CROSSOVER_LOCAL_FAILURE.json').read_text())['x'];t=time.monotonic();rows=[]
for x in [xc,2*.135-xc,.09]:
 for s in ['NS_plus','NS_minus','GG']:
  start=time.monotonic();row={'x':x,'sector':s,'direct':[]}
  try:row['source_local']=source_local(p,1,s,x,.135)
  except Exception as ex:row['error']=repr(ex)
  row['source_seconds']=time.monotonic()-start
  for n in [48,72]:
   y,w=geometry(x,.135,n,pair_power=1);_,k,_=p.evaluate(1,s,x,.135,y);kr=np.array([p.evaluate(1,s,float(z),.135,[x])[1][0] for z in y]);delta=(4/3)*(4/3-1.5)*(6.5-np.pi**2+4*1.202056903159594) if s=='NS_plus' else (-3/108*(35*3+74*4/3) if s=='GG' else 0.);v=float(w@(k-kr)+delta);row['direct'].append({'n':n,'value':v});print(x,s,n,v,flush=True)
  rows.append(row);(ROOT/'work/action/MP_LOCAL_CHECKS.json').write_text(json.dumps({'source_key':m['key'],'rows':rows,'seconds':time.monotonic()-t},indent=2)+'\n');print(row,flush=True)

from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from action_probe import bridge
from continuous_action import geometry,source_local
p,meta=bridge();rows=[];t=time.monotonic()
for x in [.03,.09,.134,.136,.19,.67]:
 for s in ['NS_plus','NS_minus','GG']:
  delta=(4/3)*(4/3-1.5)*(6.5-np.pi**2+4*1.202056903159594) if s=='NS_plus' else (-3/108*(35*3+74*4/3) if s=='GG' else 0.)
  row={'x':x,'sector':s,'source_local':source_local(p,1,s,x,.135),'explicit_delta':delta,'independent_local':[]}
  for n in [24,48]:
   y,w=geometry(x,.135,n)
   _,k,_=p.evaluate(1,s,x,.135,y)
   kr=np.array([p.evaluate(1,s,float(z),.135,[x])[1][0] for z in y])
   row['independent_local'].append({'n':n,'value':float(w@(k-kr)+delta)})
  rows.append(row)
  (ROOT/'work/action/local.json').write_text(json.dumps({'source_key':meta['key'],'seconds':time.monotonic()-t,'rows':rows},indent=2)+'\n')
  print(row,flush=True)

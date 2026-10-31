from pathlib import Path
import sys,time,json,warnings,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from action_probe import bridge,SECTORS
from continuous_action import action,adaptive_action
from src.replay import analytic_input
p,meta=bridge();t=time.monotonic();rows=[]
for x in [.03,.09,.134,.136,.19,.67]:
 for order in [0,1]:
  for s in SECTORS:
   idx=2 if s in ['QG','GG'] else (1 if s.startswith('NS') else 0)
   f=lambda y:analytic_input(y)[idx]
   row={'x':x,'order':order,'sector':s,'quadratures':[]}
   for n in [16,32,48]:
    try:row['quadratures'].append({'n':n,'action':float(action(p,order,s,x,.135,n,f))})
    except Exception as exc:row['quadratures'].append({'n':n,'error':repr(exc)})
   with warnings.catch_warnings(record=True) as ww:
    warnings.simplefilter('always')
    try:row['adaptive'],row['error_estimate']=adaptive_action(p,order,s,x,.135,f)
    except Exception as exc:row['adaptive_error']=repr(exc)
    row['warnings']=[str(w.message) for w in ww]
   rows.append(row)
   (ROOT/'work/action/refined.json').write_text(json.dumps({'source_key':meta['key'],'seconds':time.monotonic()-t,'rows':rows},indent=2,allow_nan=False)+'\n')
   v=row['quadratures'];d=abs(v[-1]['action']-v[-2]['action']) if all('action' in z for z in v) else None
   print(x,order,s,'fixed',v,'adaptive',row.get('adaptive'), 'warning_count',len(row['warnings']),flush=True)
print('seconds',time.monotonic()-t)

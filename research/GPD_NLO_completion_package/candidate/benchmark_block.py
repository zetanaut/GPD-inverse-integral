from pathlib import Path
import sys,time,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from action_probe import bridge,SECTORS
from spectral import Elements,block
from continuous_action import action
from src.replay import analytic_input
p,meta=bridge();b=Elements(.135,6,'graded');rows=[]
for x in [.09,.134,.136,.67]:
 t=time.monotonic();a,ny=block(p,b,x,32);seconds=time.monotonic()-t
 # Save the whole block before contraction/error measurement.
 np.savez_compressed(ROOT/f'work/action/block_{x}.npz',**{f'K{o}_{s}':a[o][j] for o in [0,1] for j,s in enumerate(SECTORS)})
 errs=[]
 for o in [0,1]:
  for j,s in enumerate(SECTORS):
   idx=2 if s in ['QG','GG'] else (1 if s.startswith('NS') else 0)
   ids=b.gids if idx==2 else b.qids
   c=analytic_input(b.nodes[ids])[idx]
   ref=action(p,o,s,x,.135,32,lambda y:analytic_input(y)[idx])
   errs.append(float(a[o][j]@c-ref))
 rows.append({'x':x,'seconds':seconds,'nodes':ny,'analytic_contraction_max_abs_error':max(map(abs,errs))})
 print(rows[-1],flush=True)
(ROOT/'work/action/BLOCK_TIMING.json').write_text(json.dumps({'source_key':meta['key'],'basis':b.specification(),'rows':rows},indent=2)+'\n')

from pathlib import Path
import sys,json,time
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from action_probe import bridge
from continuous_action import action
from src.replay import analytic_input
p,meta=bridge();t=time.monotonic();rows=[]
for n in [16,32]:
 z,w=leggauss(n);xi=.135
 nodes=np.r_[xi/2*(z+1),xi+(1-xi)/2*(z+1)];weights=np.r_[xi/2*w,(1-xi)/2*w]
 vals=[]
 for x in nodes:
  v=[]
  for order in [0,1]:
   parts=[]
   for s in ['QQ_singlet','QG','GQ','GG']:
    idx=2 if s in ['QG','GG'] else 0
    f=lambda y:analytic_input(y)[idx]
    try:parts.append(float(action(p,order,s,float(x),xi,48,f)))
    except Exception as exc:raise RuntimeError((n,x,order,s,repr(exc)))
   v.append(parts)
  vals.append(v)
  np.savez_compressed(ROOT/f'work/action/momentum_n{n}.npz',x=nodes[:len(vals)],weights=weights[:len(vals)],actions=np.array(vals))
  print('n',n,'x',x,'completed',len(vals),'seconds',time.monotonic()-t,flush=True)
 a=np.array(vals)
 for order in [0,1]:
  terms=[float(weights@(nodes*a[:,order,0])),float(weights@(nodes*a[:,order,1])),float(weights@a[:,order,2]),float(weights@a[:,order,3])]
  row={'n':n,'order':order,'terms_xQQ_xQG_GQ_GG':terms,'quark_column':terms[0]+terms[2],'gluon_column':terms[1]+terms[3]}
  rows.append(row);print('RESULT',row,flush=True)
 (ROOT/'work/action/momentum.json').write_text(json.dumps({'source_key':meta['key'],'seconds':time.monotonic()-t,'rows':rows},indent=2)+'\n')

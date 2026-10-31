from pathlib import Path
import sys,json,time
import mpmath as mp
import ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candidate'))
from action_probe import bridge
from gg_reference import kernel
p,meta=bridge();rows=[ast.literal_eval(l) for l in (ROOT/'work/action/gg_arithmetic_interrupted.log').read_text().splitlines() if l.startswith("{'x':")];t=time.monotonic();xi=.135
for x in [.09,.13481531067484,.136,.67]:
 for j in [3,5,7,9]:
  for sign in [-1,1]:
   y=xi+sign*10.**(-j)
   if min(abs(y-x),abs(y+x))<1e-14 or any(r['x']==x and r['y']==y for r in rows):continue
   r,k,_=p.evaluate(1,'GG',x,xi,[y]);v=float(k[0]);refs=[]
   for prec in [50,80]:
    with mp.workdps(prec):refs.append(str(kernel(x,y,xi)))
   ref=float(refs[-1]);row={'x':x,'y':y,'xi':xi,'double':v,'mp50':refs[0],'mp80':refs[1],'absolute_error':abs(v-ref),'scaled_error':abs(v-ref)/max(1,abs(ref))}
   rows.append(row);print(row,flush=True)
(ROOT/'work/action/GG_ARITHMETIC.json').write_text(json.dumps({'source_key':meta['key'],'seconds':time.monotonic()-t,'rows':rows},indent=2)+'\n')

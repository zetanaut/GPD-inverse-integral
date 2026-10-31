from pathlib import Path
import sys,json,time
import numpy as np,mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from ns_reference import kernel
m=json.loads((ROOT/'work/action/GG_BUILD.json').read_text());p=SourceParts(ROOT/'work/gg_guard_bridge'/m['key']/'parts.dylib');xc=json.loads((ROOT/'work/action/CROSSOVER_LOCAL_FAILURE.json').read_text())['x'];rows=[];t=time.monotonic()
for x,y in [(.09,xc),(.19,xc),(.67,xc),(xc,.09),(xc,xc+1e-8),(xc,xc-1e-8),(xc,xc+1e-10),(xc,xc-1e-10),(.13499,.134999999),(.03,.21)]:
 for s,sgn in [('NS_plus',1),('NS_minus',-1)]:
  _,v,_=p.evaluate(1,s,x,.135,[y]);mp.mp.dps=45;ref45=kernel(x,y,.135,sgn);mp.mp.dps=75;ref75=kernel(x,y,.135,sgn);diff=float(v[0]-ref75);r={'x':x,'y':y,'sector':s,'value':float(v[0]),'mp45':str(ref45),'mp75':str(ref75),'mp_refinement':float(abs(ref45-ref75)),'absolute_error':abs(diff),'scaled_error':abs(diff)/max(1.,abs(float(ref75)))};rows.append(r);print(r,flush=True)
  (ROOT/'work/action/NS_ARITHMETIC.json').write_text(json.dumps({'source_key':m['key'],'rows':rows,'seconds':time.monotonic()-t},indent=2)+'\n')

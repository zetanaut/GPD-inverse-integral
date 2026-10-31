"""Resolve a suspected source-action error using an analytic full-support input."""
from pathlib import Path
import sys,json,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from cached_source import CachedSource
from fast_qg import FastQG
from continuous_action import geometry
m=json.loads((ROOT/'work/action/MP_BUILD.json').read_text());q=json.loads((ROOT/'work/action/FAST_QG_BUILD.json').read_text());p=FastQG(CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/m['key']/'parts.dylib'),m['key']),ROOT/'work/fast_qg_bridge'/q['key']/'qg.dylib')
xi=.135;records=[];t=time.monotonic()
for delta in [-6.887419712331422e-8,4.413050408436092e-8,5.7841465500581e-9]:
 for n in [32,48,72,96]:
  x=xi+delta;y,w=geometry(x,xi,n);r,_,_=p.evaluate(1,'QG',x,xi,y);g=.8*(1+.1*xi*xi)*(1-y*y)**2;v=float(w@(r*g));record={'delta':delta,'n':n,'QG_action':v,'nodes':len(y),'seconds':time.monotonic()-t};records.append(record);print(record,flush=True);(ROOT/'work/action/CROSSOVER_QG_QUADRATURE.json').write_text(json.dumps({'source_key':m['key'],'qg_key':q['key'],'input':'0.8*(1+0.1*xi^2)*(1-y^2)^2, full support','rows':records},indent=2)+'\n')

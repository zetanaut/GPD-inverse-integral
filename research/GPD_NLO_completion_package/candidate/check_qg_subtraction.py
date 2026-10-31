from pathlib import Path
import sys,json,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from cached_source import CachedSource
from fast_qg import FastQG
from continuous_action import geometry
from qg_subtraction import subtraction
m=json.loads((ROOT/'work/action/MP_BUILD.json').read_text());q=json.loads((ROOT/'work/action/FAST_QG_BUILD.json').read_text());p=FastQG(CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/m['key']/'parts.dylib'),m['key']),ROOT/'work/fast_qg_bridge'/q['key']/'qg.dylib');xi=.135;records=[];limits=[];t=time.monotonic()
for x in [.09,.19,xi-6.887419712331422e-8,xi+4.413050408436092e-8,xi+5.7841465500581e-9]:
 fx=.8*(1+.1*xi*xi)*(1-x*x)**2
 for n in [32,48,72,96]:
  y,w=geometry(x,xi,n);r,_,_=p.evaluate(1,'QG',x,xi,y);g=.8*(1+.1*xi*xi)*(1-y*y)**2;s,integ=subtraction(y,x,xi);v=float(w@(r*g-s*fx)+integ*fx);record={'x':x,'n':n,'subtracted_QG_action':v,'original_action':float(w@(r*g)),'seconds':time.monotonic()-t};records.append(record);print(record,flush=True)
 for sign in [-1,1] if x<xi else [1]:
  ds=sign*min(abs(x-xi),x,1-x)*np.array([1e-2,1e-3,1e-4,1e-5]);yy=x+ds;r=p.evaluate(1,'QG',x,xi,yy)[0];s,_=subtraction(yy,x,xi);limits.append({'x':x,'side':sign,'distance':ds.tolist(),'kernel_minus_analytic_log_squared':(r-s).tolist()})
 (ROOT/'work/action/QG_SUBTRACTION_CHECK.json').write_text(json.dumps({'source_key':m['key'],'qg_key':q['key'],'rows':records,'limits':limits,'method':'exact analytically derived leading singularity, no fitted coefficients'},indent=2)+'\n')

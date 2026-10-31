from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from continuous_action import source_local,geometry
m=json.loads((ROOT/'work/action/GG_BUILD.json').read_text());p=SourceParts(ROOT/'work/gg_guard_bridge'/m['key']/'parts.dylib')
old=json.loads((ROOT/'work/action/GG_ARITHMETIC.json').read_text());rows=[];t=time.monotonic()
for r in old['rows']:
 _,k,_=p.evaluate(1,'GG',r['x'],r['xi'],[r['y']]);ref=float(r['mp80']);v=float(k[0]);rows.append({'x':r['x'],'y':r['y'],'value':v,'ref':ref,'absolute_error':abs(v-ref),'scaled_error':abs(v-ref)/max(1,abs(ref)),'guarded':abs(r['y']-r['xi'])<1e-4*r['xi']})
local=[]
for x in [.09,.13481531067484,.135-1e-5,.135+1e-5,.136]:
 t0=time.monotonic()
 try:v=source_local(p,1,'GG',x,.135);rec={'x':x,'local':v,'seconds':time.monotonic()-t0}
 except Exception as e:rec={'x':x,'error':repr(e),'seconds':time.monotonic()-t0}
 local.append(rec);print(rec,flush=True)
 if 'local' in rec:
  for n in [24,48]:
   y,w=geometry(x,.135,n,pair_power=1);_,k,_=p.evaluate(1,'GG',x,.135,y)
   kr=np.array([p.evaluate(1,'GG',float(z),.135,[x])[1][0] for z in y])
   rec[f'independent_{n}']=float(w@(k-kr)-3/108*(35*3+74*4/3))
 (ROOT/'work/action/GG_GUARD_CHECK.json').write_text(json.dumps({'source_key':m['key'],'point_rows':rows,'local':local,'seconds':time.monotonic()-t},indent=2)+'\n')
print('max guarded error',max(r['scaled_error'] for r in rows if r['guarded']),flush=True)

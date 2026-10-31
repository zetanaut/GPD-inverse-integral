from pathlib import Path
import sys,json,time,ctypes,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.source_parts import SourceParts
m=json.loads((ROOT/'work/action/MP_BUILD.json').read_text());p=SourceParts(ROOT/'work/mp_guard_bridge'/m['key']/'parts.dylib');lib=ctypes.CDLL(str(ROOT/'work/mpfr_probe/qg.dylib'));fn=lib.mp_qg_guard;fn.argtypes=[ctypes.c_double]*3;fn.restype=ctypes.c_double;rows=[];skipped=[];t=time.monotonic();tf=0;ts=0
for xi in [.135,.275,.455]:
 for ratio in [.2,.67,.99999,1.00001,1.9]:
  x=xi*ratio
  for sg in [-1,1]:
   for eps in [-1e-3,1e-3,-1e-7,1e-7,-1e-11,1e-11]:
    y=sg*xi*(1+eps)
    if min(abs(y-x),abs(y+x),abs(y-xi),abs(y+xi))<1e-14:skipped.append([x,y,xi]);continue
    t0=time.monotonic();ref=p.evaluate(1,'QG',x,xi,[y])[0][0];ts+=time.monotonic()-t0;t0=time.monotonic();v=3*fn(x,y,xi);tf+=time.monotonic()-t0;rows.append({'x':x,'y':y,'xi':xi,'nf3_MPFR':v,'captured_software_wide':float(ref),'absolute_error':abs(v-ref),'scaled_error':abs(v-ref)/max(1,abs(ref))})
out={'rows':rows,'skipped_exact_singular_inputs':skipped,'max_scaled_error':max(r['scaled_error'] for r in rows),'seconds_original':ts,'seconds_MPFR':tf,'total_seconds':time.monotonic()-t,'source_sha256':{s:hashlib.sha256((ROOT/'candidate'/s).read_bytes()).hexdigest() for s in ['mp_kernel.cpp','mp_qg.cpp','make_fast_qg.py']},'independence':'arithmetic implementation comparison; same captured QG expressions, not independent physics'};(ROOT/'work/action/FAST_QG_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='rows'});assert out['max_scaled_error']<1e-12

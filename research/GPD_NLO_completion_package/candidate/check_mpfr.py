from pathlib import Path
import ctypes,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
libs=[]
for name in ['guard128.dylib','guard.dylib']:
 lib=ctypes.CDLL(str(ROOT/'work/mpfr_probe'/name));lib.mp_gg_guard.argtypes=[ctypes.c_double]*3+[ctypes.c_int];lib.mp_ns_guard.argtypes=[ctypes.c_double]*3+[ctypes.c_int]*2;lib.mp_gg_guard.restype=lib.mp_ns_guard.restype=ctypes.c_double;libs.append(lib)
t=time.monotonic();out=[]
for name in ['GG_ARITHMETIC','NS_ARITHMETIC']:
 m=json.loads((ROOT/f'work/action/{name}.json').read_text())
 for rec in m['rows']:
  x,y,xi=rec['x'],rec['y'],rec.get('xi',.135);ref=float(rec.get('mp80',rec.get('mp75')));vals=[]
  for lib in libs:
   if name=='GG_ARITHMETIC':v=lib.mp_gg_guard(x,y,xi,0)+3*lib.mp_gg_guard(x,y,xi,1)
   else:
    sg=1 if rec['sector']=='NS_plus' else -1;v=lib.mp_ns_guard(x,y,xi,sg,0)+3*lib.mp_ns_guard(x,y,xi,sg,1)
   vals.append(v)
  out.append({'source_record':name,'x':x,'y':y,'reference':ref,'precision128':vals[0],'precision192':vals[1],'absolute_error':abs(vals[1]-ref),'scaled_error':abs(vals[1]-ref)/max(1,abs(ref)),'precision_refinement_scaled':abs(vals[1]-vals[0])/max(1,abs(ref))})
r={'rows':out,'seconds':time.monotonic()-t,'max_scaled_error':max(r['scaled_error'] for r in out),'max_precision_refinement_scaled':max(r['precision_refinement_scaled'] for r in out),'source_sha256':hashlib.sha256((ROOT/'candidate/mp_kernel.cpp').read_bytes()).hexdigest(),'library_sha256':{name:hashlib.sha256((ROOT/'work/mpfr_probe'/name).read_bytes()).hexdigest() for name in ['guard128.dylib','guard.dylib']}}
(ROOT/'work/action/MPFR_CHECKS.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='rows'})
assert r['max_scaled_error']<1e-12
assert r['max_precision_refinement_scaled']<1e-12

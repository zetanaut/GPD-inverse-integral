from pathlib import Path
import re,difflib
ROOT=Path(__file__).resolve().parents[1]
nlo=(ROOT/'frozen/fortran/kernels/kernels_nlo.f90').read_text()
patched=nlo.replace('  use qg_wide, only: qg_guard_value','  use qg_wide, only: qg_guard_value\n  use mp_guard, only: mp_gg_guard,mp_ns_guard')
for name,fn,sg,part in [('KV1_NSp_pls','ns',1,0),('KV1_NSm_pls','ns',-1,0),('KV1_NSp_pls_nfl','ns',1,1),('KV1_NSm_pls_nfl','ns',-1,1),('KV1_GG_pls','gg',0,0),('KV1_GG_pls_nfl','gg',0,1)]:
 old=re.search(r'^    function '+name+r'\(.*?^    end function '+name+r'\s*$',patched,re.M|re.S|re.I).group(0)
 # Wide input-coordinate evaluation also avoids cancellation in Y-X near poles.
 condition='min(abs(y-xi),abs(y+xi),abs(y-x),abs(y+x)) < 0.0001_dp*xi'
 args=f'x,y,xi,{sg},{part}' if fn=='ns' else f'x,y,xi,{part}'
 new=old.replace('        X1 =',f'''        if({condition}) then
          K=mp_{fn}_guard({args})
          return
        endif
        X1 =''',1)
 patched=patched.replace(old,new)
(ROOT/'candidate/kernels_nlo_mp.f90').write_text(patched)
(ROOT/'candidate/mp_kernel_guard.diff').write_text(''.join(difflib.unified_diff(nlo.splitlines(True),patched.splitlines(True),fromfile='frozen/fortran/kernels/kernels_nlo.f90',tofile='candidate/kernels_nlo_mp.f90')))

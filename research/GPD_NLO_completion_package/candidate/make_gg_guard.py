"""Arithmetic-only GG guard; exact Eq.185 source expressions promoted to wide.
Run only to regenerate candidate source. Frozen files are never written.
"""
from pathlib import Path
import re,difflib
ROOT=Path(__file__).resolve().parents[1]
def extract(text,name):
 return re.search(r'^    function '+name+r'\(.*?^    end function '+name+r'\s*$',text,re.M|re.S|re.I).group(0)
def promote(text):
 lines=[]
 for line in text.splitlines():
  code,sep,comment=line.partition('!')
  code=code.replace('real(dp)','type(wide_real)')
  code=re.sub(r'(?<![A-Za-z0-9_])((?:\d+\.\d*|\.\d+)(?:[eEdD][+-]?\d+)?)(?:_dp)?',lambda m:'wide('+m.group(1)+'_dp)',code)
  lines.append(code+(sep+comment if sep else ''))
 return '\n'.join(lines)
q=(ROOT/'frozen/fortran/kernels/qg_wide.f90').read_text();common=(ROOT/'frozen/fortran/kernels/kernels_common.f90').read_text();nlo=(ROOT/'frozen/fortran/kernels/kernels_nlo.f90').read_text()
start=q[:q.index('  function qg_guard_value')].replace('qg_wide','gg_wide').replace('public::qg_guard_value','public::gg_guard_value')
start=start.replace('! Same QG expressions','! Same GG expressions')
bridge='''  function gg_guard_value(x,y,xi,nfl_part) result(v)
    real(dp),intent(in)::x,y,xi
    integer,intent(in)::nfl_part
    real(dp)::v
    type(wide_real)::a,b,c,X1,Y1,X2,Y2
    a=wide(x);b=wide(y);c=wide(xi)
    X1=(wide(1._dp)+a/c)/wide(2._dp);X2=(wide(1._dp)-a/c)/wide(2._dp)
    Y1=(wide(1._dp)+b/c)/wide(2._dp);Y2=(wide(1._dp)-b/c)/wide(2._dp)
    if(nfl_part==0)then
      v=to_double((KV1_GG_half(X1,Y1)+KV1_GG_half(X2,Y2))/(wide(2._dp)*c))
    else
      v=to_double((KV1_GG_half_nfl(X1,Y1)+KV1_GG_half_nfl(X2,Y2))/(wide(2._dp)*c))
    endif
  end function
'''
helpers=q[q.index('  recursive function dilog'):q.lower().index('    function qg_f_a')]
parts=[promote(extract(common,k)) for k in ['GG_f_a','GG_f_b','GG_f_c','GG_h_V','GG_hbar_V']]
parts += [promote(extract(nlo,k)) for k in ['KV1_GG_half','KV1_GG_half_nfl']]
(ROOT/'candidate/gg_wide.f90').write_text(start+bridge+helpers+'\n'.join(parts)+'\nend module gg_wide\n')
patched=nlo.replace('  use qg_wide, only: qg_guard_value','  use qg_wide, only: qg_guard_value\n  use gg_wide, only: gg_guard_value')
for name,i in [('KV1_GG_pls',0),('KV1_GG_pls_nfl',1)]:
 old=extract(patched,name);new=old.replace('        X1 =',f'''        if(min(abs(y-xi),abs(y+xi)) < 0.0001_dp*xi) then
          K=gg_guard_value(x,y,xi,{i})
          return
        endif
        X1 =''',1);patched=patched.replace(old,new)
(ROOT/'candidate/kernels_nlo.f90').write_text(patched)
(ROOT/'candidate/kernel_guard.diff').write_text(''.join(difflib.unified_diff(nlo.splitlines(True),patched.splitlines(True),fromfile='frozen/fortran/kernels/kernels_nlo.f90',tofile='candidate/kernels_nlo.f90')))

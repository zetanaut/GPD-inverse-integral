"""Minimal first-row source diagnostic; no matrix construction or truth access."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json
ROOT=Path(os.environ.get('GPD_REVIEW_ROOT',Path(__file__).resolve().parents[3]));nlo=ROOT/'GPD_NLO_completion_package'
sys.path.insert(0,str(nlo));sys.path.insert(0,str(nlo/'candidate'))
from src.source_parts import SourceParts
from fast_qg import FastQG
from cached_source import CachedSource
from basis_factory import Elements
from spectral import block
m=json.loads((nlo/'work/action/MP_BUILD.json').read_text());q=json.loads((nlo/'work/action/FAST_QG_BUILD.json').read_text())
s=SourceParts(nlo/'work/mp_guard_bridge'/m['key']/'parts.dylib');fn=s.evaluate
if '--local-only' in sys.argv:
 print('CALL order=1 sector=NS_minus x=0.00045337219737495715 xi=0.215 y=0.813579 include_local=True',flush=True)
 print(fn(1,'NS_minus',0.00045337219737495715,.215,[.813579],True),flush=True)
 sys.exit(0)
def trace(order,sector,x,xi,y,include_local=False):
 print('CALL',order,sector,'n',len(y),'local',include_local,flush=True)
 value=fn(order,sector,x,xi,y,include_local)
 print('RETURN',order,sector,flush=True)
 return value
s.evaluate=trace
p=FastQG(CachedSource(s,m['key']),nlo/'work/fast_qg_bridge'/q['key']/'qg.dylib')
b=Elements(.215,10,'log3');xx,w=b.outputs(18)
block(p,b,float(xx[0]),48,local_n=0)
print('COMPLETE',flush=True)

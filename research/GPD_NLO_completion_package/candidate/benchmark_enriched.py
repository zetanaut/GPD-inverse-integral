from pathlib import Path
import sys,time,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from cached_source import CachedSource
from fast_qg import FastQG
from enriched import LogElements
from spectral import block
from forward import initial
from analytic_scale_control import action
m=json.loads((ROOT/'work/action/MP_BUILD.json').read_text());q=json.loads((ROOT/'work/action/FAST_QG_BUILD.json').read_text());p=FastQG(CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/m['key']/'parts.dylib'),m['key']),ROOT/'work/fast_qg_bridge'/q['key']/'qg.dylib');b=LogElements(.135,10,'log3');x,w=b.outputs(18);i=int(np.argmin(abs(x-.135)));xx=float(x[i]);t=time.monotonic();a,ny=block(p,b,xx,48,0)
np.savez_compressed(ROOT/'work/action/enriched_benchmark.npz',**{f'K{o}_{s}':a[o][j] for o in [0,1] for j,s in enumerate(['NS_plus','NS_minus','QQ_singlet','QG','GQ','GG'])})
v=initial(b);nq=len(b.qids);got=a[0][2]@v[:nq]+a[0][0]@v[nq:2*nq]+a[0][3]@v[2*nq:];ref=float(action(xx,.135));result={'x':xx,'xi':.135,'i':i,'nodes':ny,'seconds':time.monotonic()-t,'basis':b.specification(),'inner_n':48,'outer_n':18,'source_key':m['key'],'qg_key':q['key'],'independent_analytic_K0_action_error':float(abs(got-ref)),'finite':all(np.isfinite(v).all() for aa in a for v in aa)};(ROOT/'work/action/ENRICHED_TIMING.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

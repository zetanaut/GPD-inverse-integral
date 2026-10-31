from pathlib import Path
import sys,time,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from cached_source import CachedSource
from src.source_parts import SourceParts
from spectral import Elements,block
from forward import initial
from src.replay import analytic_input
m=json.loads((ROOT/'work/action/MP_BUILD.json').read_text());p=SourceParts(ROOT/'work/mp_guard_bridge'/m['key']/'parts.dylib');p=CachedSource(p,m['key']);b=Elements(.135,10,'crossover');x,w=b.outputs(18);z=float(x[np.argmin(abs(x-.135))]);t=time.monotonic()
a,ny=block(p,b,z,48,0)
np.savez_compressed(ROOT/'work/action/crossover_benchmark_block.npz',**{f'K{o}_{s}':a[o][j] for o in [0,1] for j,s in enumerate(['NS_plus','NS_minus','QQ_singlet','QG','GQ','GG'])})
record={'x':z,'nodes':ny,'seconds':time.monotonic()-t,'degree':10,'layout':'crossover','inner_n':48,'outer_n':18,'local_n':0,'source_build_key':m['key'],'method_sources':{name:hashlib.sha256((ROOT/'candidate'/name).read_bytes()).hexdigest() for name in ['spectral.py','continuous_action.py','action_probe.py','fixed_local.py','cached_source.py']},'finite':all(np.isfinite(v).all() for vs in a for v in vs)}
(ROOT/'work/action/CROSSOVER_TIMING.json').write_text(json.dumps(record,indent=2)+'\n');print(record)

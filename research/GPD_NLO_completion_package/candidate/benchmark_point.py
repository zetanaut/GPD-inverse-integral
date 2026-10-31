"""Measured nearest-crossover block before constructing another frozen point."""
from pathlib import Path
import sys,argparse,json,time,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts,SECTORS
from cached_source import CachedSource
from fast_qg import FastQG
from basis_factory import Elements
from spectral import block
ap=argparse.ArgumentParser();ap.add_argument('--xi',type=float,required=True);a=ap.parse_args();m=json.loads((ROOT/'work/action/MP_BUILD.json').read_text());q=json.loads((ROOT/'work/action/FAST_QG_BUILD.json').read_text());p=FastQG(CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/m['key']/'parts.dylib'),m['key']),ROOT/'work/fast_qg_bridge'/q['key']/'qg.dylib');b=Elements(a.xi,10,'log3');x,w=b.outputs(18);i=int(np.argmin(abs(x-a.xi)));t=time.monotonic();ar,ny=block(p,b,float(x[i]),48,0);stem=ROOT/f'work/action/benchmark_xi_{a.xi:g}';np.savez_compressed(str(stem)+'.npz',**{f'K{o}_{s}':ar[o][j] for o in [0,1] for j,s in enumerate(SECTORS)});r={'xi':a.xi,'x':float(x[i]),'i':i,'seconds':time.monotonic()-t,'nodes':ny,'finite':all(np.isfinite(v).all() for aa in ar for v in aa),'source_key':m['key'],'qg_key':q['key'],'basis':b.specification(),'inner_n':48,'outer_n':18,'method_sources':{n:hashlib.sha256((ROOT/'candidate'/n).read_bytes()).hexdigest() for n in ['spectral.py','continuous_action.py','action_probe.py','fixed_local.py','cached_source.py','fast_qg.py','enriched.py','basis_factory.py']}};Path(str(stem)+'.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

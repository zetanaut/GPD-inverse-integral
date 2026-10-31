from pathlib import Path
import sys,json,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'candidate'))
from src.source_parts import SourceParts
from cached_source import CachedSource
from fast_qg import FastQG
from spectral import Elements,block
m=json.loads((ROOT/'work/action/MP_BUILD.json').read_text());q=json.loads((ROOT/'work/action/FAST_QG_BUILD.json').read_text());p=FastQG(CachedSource(SourceParts(ROOT/'work/mp_guard_bridge'/m['key']/'parts.dylib'),m['key']),ROOT/'work/fast_qg_bridge'/q['key']/'qg.dylib');b=Elements(.135,10,'crossover');old=ROOT/'work/operators/7fb953185def547b8f34ce68b70943d218a4f3485091f07f3f20b3fd22ad37c2';k=json.loads((old/'KEY.json').read_text());rows=[]
for i in [48,90]:
 x=k['output_coordinates'][i];t=time.monotonic();a,ny=block(p,b,x,48,0);src=old/f'row_{i:04}.npz';path=ROOT/f'work/action/fast_block_{i}.npz';sectors=['NS_plus','NS_minus','QQ_singlet','QG','GQ','GG'];np.savez_compressed(path,**{f'K{o}_{s}':a[o][j] for o in [0,1] for j,s in enumerate(sectors)});diff={}
 with np.load(src) as z:
  for o in [0,1]:
   for j,s in enumerate(sectors):diff[f'K{o}_{s}']=float(np.sum(abs(z[f'K{o}_{s}']-a[o][j])))
 rows.append({'i':i,'x':x,'nodes':ny,'seconds':time.monotonic()-t,'row_L1_difference':diff})
out={'rows':rows,'source_key':m['key'],'qg_guard_key':q['key'],'maximum_row_L1_difference':max(v for r in rows for v in r['row_L1_difference'].values()),'independence':'arithmetic substitution check only'};(ROOT/'work/action/FAST_BLOCK_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['maximum_row_L1_difference']<1e-9

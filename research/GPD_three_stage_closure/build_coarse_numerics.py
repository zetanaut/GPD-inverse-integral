"""Cache only the exact numerical contractions used by coarse binding tests."""
import json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from closure.numerical_cache import build_one
with np.load(ROOT/'prepared/observed/ideal_L0_Z1.npz',allow_pickle=False) as z:
 pairs=list(dict.fromkeys((float(x),float(q)) for x,q in zip(z['xi'],z['Q2'])))
out=[]
for xi,q2 in pairs:
 t=time.monotonic();rec=build_one(xi,q2);rec['seconds']=time.monotonic()-t
 out.append(rec);print(json.dumps(rec),flush=True)
report={'case':'ideal_L0_Z1_s31','pairs':out,'complete':len(out)==len(pairs)}
(ROOT/'numerical_contractions/COARSE_INDEX.json').write_text(json.dumps(report,indent=2)+'\n')

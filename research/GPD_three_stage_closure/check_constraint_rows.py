"""Selected exact-geometry numerical derivative check; no solver step."""
import json,sys,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from closure.solver_binding import problem_and_domain,oracle_at
p,d=problem_and_domain('ideal_L0_Z1_s31');o=oracle_at(p,d,p.x0)
a=p.x0.numpy();ids=np.array([0,1,2,3,6,11],dtype=int)
t=time.monotonic();g,R,A=o.derivatives(a,ids);sec=time.monotonic()-t
rng=np.random.default_rng(992);direction=rng.normal(size=len(a));direction/=np.linalg.norm(direction)
ad=A@direction
checks=[]
for h in (1e-3,1e-4,1e-5,1e-6):
 vp=o.values(a+h*direction)['c'][ids];vm=o.values(a-h*direction)['c'][ids];fd=(vp-vm)/(2*h);rel=np.abs(fd-ad)/np.maximum(1,np.abs(ad));checks.append({'h':h,'FD':fd.tolist(),'max_scaled_abs_difference':float(max(rel))})
fd=np.asarray(checks[1]['FD']);rel=np.abs(fd-ad)/np.maximum(1,np.abs(ad))
record={'status':'CONSTRAINT_DERIVATIVES_CHECKED','indices':ids.tolist(),'name':[d.names[i] for i in ids],'AD':ad.tolist(),'FD':fd.tolist(),'scaled_abs_difference':rel.tolist(),'step_sweep':checks,'max_scaled_abs_difference':float(max(rel)),'seconds':sec,'oracle_timers':o.timers,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scientific_fit':False}
assert np.isfinite(rel).all()
(ROOT/'readiness/CONSTRAINT_ROWS.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)

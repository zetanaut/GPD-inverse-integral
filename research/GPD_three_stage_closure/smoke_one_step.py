"""At most one isolated outer step from the original fresh seed; no fit result."""
import json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from closure.solver_binding import problem_and_domain,oracle_at,isolated_solver,accepted
p,d=problem_and_domain('ideal_L0_Z1_s31')
o=oracle_at(p,d,p.x0);solver,o,stats,settings=isolated_solver(o)
settings=dict(settings,max_iterations=1)
records=[];start=time.monotonic()
def record(x,rec,trial):
 records.append({'iteration':rec['iteration'],'J':rec['J'],'accepted':rec.get('accepted'),'trials':len(rec['trials'])})
 print(json.dumps(records[-1]),flush=True)
last,diag=solver.solve(o,accepted.old.kkt,record,settings)
result={'status':'ONE_OUTER_STEP_SMOKE_NOT_FIT','solver_status':diag['status'],'seconds':time.monotonic()-start,'history_count':len(diag['history']),'records':records,'oracle_counts':o.counts,'oracle_timers':o.timers,'binding_stats':stats,'solver_counts':diag['counts'],'solver_linear_seconds':diag.get('linear_seconds'),'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'changed_initial':bool(not np.array_equal(last,p.x0.numpy())),'scientific_fit':False}
out=ROOT/'readiness';out.mkdir(exist_ok=True);(out/'ONE_STEP_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)

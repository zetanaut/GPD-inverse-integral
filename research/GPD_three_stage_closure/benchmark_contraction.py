import os,sys,time,json,hashlib
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'GPD_ReH_experimental_run'))
import build_experimental_contractions as bc
xi=.25;q2=8.51
folder=ROOT/'GPD_ReH_experimental_run/families'/hashlib.sha256(float(xi).hex().encode()).hexdigest()[:16]
state=json.loads((folder/'PIPELINE.json').read_text())
t=time.monotonic();out={}
for label,directory in state['directories'].items():
 basis,k0,k1,_=bc.gv.pf.op(directory)
 U,nfev=bc.ordered_map(k0,k1,q2)
 h192=bc.gv.pf.hard(basis,q2,192);h384=bc.gv.pf.hard(basis,q2,384)
 out[label]={'shape':list(U.shape),'nfev':nfev,'seconds_total':time.monotonic()-t,'hard_delta_max':float(abs(h384-h192).max())}
 print(json.dumps({'label':label,**out[label]}),flush=True)
rec={'xi':xi,'Q2':q2,'total_seconds':time.monotonic()-t,'labels':out}
p=ROOT/'GPD_three_stage_closure/runs/benchmark/CONTRACTION_COST.json';p.write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps({'total_seconds':rec['total_seconds']}),flush=True)

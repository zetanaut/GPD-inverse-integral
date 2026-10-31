import os,time,json,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from closure.bank import RowBank
from closure.problem import ClosureProblem
study=json.loads((ROOT/'prepared/STUDY.json').read_text())
bank=RowBank(ROOT/'inputs/accepted_experimental_banks',ROOT/'scale_rows')
p=ClosureProblem(ROOT/'prepared/observed/ideal_L0_Z1.npz',bank,study,seed=31)
x=torch.load(ROOT/'runs/benchmark/adam400.pt',weights_only=True)['flat']
torch.set_num_threads(1)
t=time.monotonic();j=p.jacobian(x);seconds=time.monotonic()-t
rec={'branch':'ideal_L0_Z1','shape':list(j.shape),'seconds':seconds,'finite':bool(torch.isfinite(j).all())}
(ROOT/'runs/benchmark/JACOBIAN_COST.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec),flush=True)

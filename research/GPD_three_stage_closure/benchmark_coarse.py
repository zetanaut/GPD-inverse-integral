"""One coarse data-plus-polynomiality Adam timing; not a selected closure fit."""
import json,time,os,sys,platform
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from closure.bank import RowBank
from closure.problem import ClosureProblem
study=json.loads((ROOT/'prepared/STUDY.json').read_text())
bank=RowBank(ROOT/'inputs/accepted_experimental_banks',ROOT/'scale_rows')
p=ClosureProblem(ROOT/'prepared/observed/ideal_L0_Z1.npz',bank,study,seed=31)
torch.set_num_threads(1)
x=p.x0.clone().requires_grad_(True)
opt=torch.optim.Adam([x],lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0)
start=time.monotonic();history=[]
for step in range(400):
 opt.zero_grad(set_to_none=True);loss=p.objective(x)
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite')
 loss.backward();opt.step()
 if step+1 in (1,10,100,400):
  rec={'step':step+1,'J':float(loss.detach()),'seconds':time.monotonic()-start}
  history.append(rec);print(json.dumps(rec),flush=True)
out=ROOT/'runs/benchmark';out.mkdir(parents=True,exist_ok=True)
tmp=out/'adam400.pt.partial';torch.save({'flat':x.detach(),'seed':31,'stage':'Adam400_TIMING_ONLY'},tmp);tmp.replace(out/'adam400.pt')
rec={'branch':'ideal_L0_Z1','seed':31,'adam_steps':400,'seconds':time.monotonic()-start,'history':history,'platform':platform.platform(),'torch':torch.__version__,'status':'timing_only_not_finite_solver'}
(out/'COST.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec),flush=True)

"""Real FiLM/PDF device probe. This is NOT a full GPU fitting implementation."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from p2b_common import setup, field_only, information, write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capsule',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()): raise RuntimeError('Use a new empty probe output')
    a.out.mkdir(parents=True,exist_ok=True)
    if a.threads<1 or a.threads>8:raise ValueError('Probe threads must be 1..8')
    if a.device=='cuda':
        if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('CUDA probe requires an allocated Linux GPU node')
        if int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<a.threads:
            raise RuntimeError('Thread request exceeds allocation')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    core=setup(a.capsule)
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from torch.func import jacrev,jvp
    if a.device=='cuda' and (not torch.cuda.is_available() or torch.version.cuda is None):
        write(a.out/'PROBE_STATUS.json',dict(status='CUDA_UNAVAILABLE',torch=str(torch.__version__),cuda_build=torch.version.cuda,CPU_fallback=False))
        raise RuntimeError('A CUDA-enabled Torch environment and allocated GPU are required; no CPU fallback')
    if a.device=='cuda' and torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one allocated GPU for this probe')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(a.threads)  # after legacy setup, which sets it to one
    limiter=threadpool_limits(limits=a.threads)
    base=field_only(core)
    candidate=copy.copy(base)
    candidate.model=copy.deepcopy(base.model).to(device=a.device,dtype=torch.float64)
    candidate.x0=base.x0.to(a.device)
    cpu_pdf=information(base)
    device_pdf=copy.copy(cpu_pdf);device_pdf.base=candidate
    for name in ('x','zero','observed','cholesky'):
        setattr(device_pdf,name,getattr(cpu_pdf,name).to(a.device))
    with np.load(ROOT/'evidence/DIAGNOSTIC_STATES.npz',allow_pickle=False) as z:
        flat_cpu=torch.tensor(z['half_baseline'].copy(),dtype=torch.float64)
    flat=flat_cpu.to(a.device)
    # Fixed benchmark coordinates from the archived observation geometry.
    with np.load(core.STUDY/'prepared/observed/exp_Re_exact_Z1.npz',allow_pickle=False) as z:
        rows=np.linspace(0,159,12,dtype=int)
        xi=torch.tensor(z['xi'][rows]); t=torch.tensor(z['t'][rows])
    u=(np.polynomial.legendre.leggauss(32)[0]+1)/2
    x=torch.tensor(np.concatenate((xi.numpy()[:,None]*u,xi.numpy()[:,None]+(1-xi.numpy()[:,None])*u),axis=1))
    xd,skew,td=x.to(a.device),xi.to(a.device),t.to(a.device)
    checks={};timings={}
    def sync():
        if a.device=='cuda':torch.cuda.synchronize()
    def compare(name,left,right,derivative=False):
        left=left.detach().cpu().numpy();right=right.detach().cpu().numpy()
        rtol,atol=(2e-7,1e-8) if derivative else (1e-9,1e-10)
        checks[name]=dict(pass_=bool(np.allclose(left,right,rtol=rtol,atol=atol)),max_abs=float(abs(left-right).max()),rtol=rtol,atol=atol)
    def timed(name,fn):
        sync();first=time.perf_counter();fn();sync();cold=time.perf_counter()-first
        samples=[]
        for _ in range(3):
            sync();start=time.perf_counter();result=fn();sync();samples.append(time.perf_counter()-start)
        timings[name]=dict(first_seconds=cold,steady_seconds=samples,median_seconds=statistics.median(samples))
        return result
    with torch.no_grad():
        compare('original_FiLM_field',base.field(flat_cpu,x,xi,t),candidate.field(flat,xd,skew,td))
        compare('PDF_prediction',cpu_pdf.prediction(flat_cpu),device_pdf.prediction(flat))
        compare('PDF_residual',cpu_pdf.residual(flat_cpu),device_pdf.residual(flat))
    cpu_J=cpu_pdf.jacobian(flat_cpu)
    J=timed('PDF_36_by_3867_Jacobian',lambda:jacrev(device_pdf.residual,chunk_size=8)(flat))
    compare('PDF_Jacobian',cpu_J,J,derivative=True)
    direction=torch.sin(torch.arange(1,3868,dtype=torch.float64));direction/=torch.linalg.norm(direction)
    direction=direction.to(a.device)
    directional=jvp(device_pdf.residual,(flat,),(direction,))[1]
    compare('PDF_JVP',directional,cpu_J@direction.cpu(),derivative=True)
    theta=flat.clone().requires_grad_(True)
    grad=torch.autograd.grad(device_pdf.residual(theta).square().sum(),theta)[0]
    compare('PDF_gradient',grad,2*cpu_J.T@cpu_pdf.residual(flat_cpu),derivative=True)
    def field_backward():
        z=flat.detach().requires_grad_(True)
        f=candidate.field(z,xd,skew,td)
        return torch.autograd.grad(f.square().mean(),z)[0]
    timed('FiLM_forward_backward_fixed_12_slices',field_backward)
    if a.device=='cuda':
        if not (flat.is_cuda and J.is_cuda and all(p.is_cuda for p in candidate.model.parameters())):
            raise RuntimeError('Probe did not actually execute on CUDA')
        gpu=dict(name=torch.cuda.get_device_name(0),total_memory=torch.cuda.get_device_properties(0).total_memory,
                 peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    else:gpu=None
    ok=all(c['pass_'] for c in checks.values())
    write(a.out/'DEVICE_PROBE.json',dict(status='PASS' if ok else 'FAIL',device=a.device,checks=checks,
        timings=timings,gpu=gpu,torch=str(torch.__version__),cuda_build=torch.version.cuda,
        threads=torch.get_num_threads(),threadpools=threadpool_info(),precision='float64',TF32=False,
        full_pipeline_GPU_qualified=False,scope='Actual FiLM/PDF smoke and kernel timings only; excludes CFF/moment/constraints, full Jacobian, QR/eigh, SLSQP, transfers and scientific finalization. No full-fit speedup inferred.'))
    limiter.restore_original_limits()
    print(json.dumps(dict(status='PASS' if ok else 'FAIL',device=a.device,checks=len(checks),full_pipeline_GPU_qualified=False)))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())

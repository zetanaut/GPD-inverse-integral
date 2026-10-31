#!/usr/bin/env python3
"""Cheap source/geometry/truth/NLO-input checks. NEVER invokes an optimizer."""
from __future__ import annotations
import json,platform,sys,time
from pathlib import Path
import numpy as np,scipy,torch
from closure.io import atomic_json,sha256
from closure.bank import RowBank
from closure.truths import Truth
from closure.metrics import split_grid
from closure.problem import ClosureProblem
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1);beg=time.perf_counter()
    study=json.loads((ROOT/'prepared/STUDY.json').read_text());bank=RowBank(ROOT/'inputs/accepted_experimental_banks')
    checks=[];max_m=0.
    for name in ('Z1','Z2'):
        truth=Truth(name)
        for N in [2,4,6,8,10,12,14,16,18]:
            for xi in [.02,.135,.275,.455,.72,.98]:
                x,w=split_grid(xi,384)
                for t in [-1.3732,-.65465,-.363,-.11]:
                    h=truth.values(x,xi,t)
                    for c in range(3):
                        p=N-1 if c<2 else N-2
                        v=2*np.dot(w*x**p,h[:,c]);ref=float(truth.moment(c,N,xi,t))
                        max_m=max(max_m,abs(v-ref))
    # Complete production-sized residual and scalar gradient; no full giant Jacobian.
    p=ClosureProblem(ROOT/'prepared/observed/exp_Re_exact_Z1.npz',bank,study,31)
    theta=p.x0.clone().requires_grad_(True);r=p.residual(theta);value=r.square().sum();g=torch.autograd.grad(value,theta)[0]
    rng=np.random.default_rng(9021);d=torch.tensor(rng.standard_normal(len(theta)),dtype=torch.float64);d/=d.norm()
    ad=float(g@d)
    for eps in [1e-4,1e-5,1e-6]:
        fd=float((p.objective((theta+eps*d).detach())-p.objective((theta-eps*d).detach()))/(2*eps))
        checks.append({'step':eps,'AD':ad,'FD':fd,'scaled_difference':abs(fd-ad)/max(1,abs(ad))})
    with np.load(ROOT/'prepared/observed/exp_Re_noisy_Z1.npz',allow_pickle=False) as noisy, np.load(ROOT/'prepared/observed/exp_Re_exact_Z1.npz',allow_pickle=False) as exact:
        noise_rms=float(np.sqrt(np.mean((noisy['y']-exact['y'])**2)));clean_rms=float(np.sqrt(np.mean(exact['y']**2)))
    result={'status':'PREPARATION_AND_ADAPTER_CHECKED_NOT_FITTED',
        'exact_existing_map_pairs_verified':len(bank.records),'moment_comparisons':2*9*6*4*3,
        'max_absolute_truth_moment_discrepancy':max_m,
        'full_160_Re_13_t_adapter':{'parameters':len(theta),'residual_count':len(r),
            'initial_objective':float(value.detach()),'finite_gradient':bool(torch.isfinite(g).all()),'directional_checks':checks},
        'first_truth_input_RMS':clean_rms,'first_truth_added_noise_RMS':noise_rms,
        'software_tests':'see TESTS.log; separate from numerical comparisons',
        'new_evolution_rows_run':0,'new_inverse_fits_run':0,
        'unexecuted':'ideal 217 new scale-row exports, full worker optimizer binding, all closure fits and symbolic refits',
        'seconds':time.perf_counter()-beg,
        'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'platform':platform.platform()}}
    atomic_json(ROOT/'results/PREPARATION_AUDIT.json',result);return result
if __name__=='__main__':print(json.dumps(main(),indent=2))

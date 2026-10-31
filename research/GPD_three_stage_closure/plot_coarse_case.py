"""Post-selection truth/reconstruction and CFF comparisons; never selects a fit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from closure.solver_binding import problem_and_domain
from closure.truths import Truth
from closure.bank import RowBank
from closure.problem import GroupedForward
from closure.geometry import chart_t
from run_coarse_case import OUT,TASK


def main():
    result=json.loads((OUT/"RESULT.json").read_text())
    ck=torch.load(OUT/result["scored_checkpoint"],map_location="cpu",weights_only=True)
    p,_=problem_and_domain(TASK)
    flat=ck["flat"].detach().double()
    model=p.set_model(flat)
    truth=Truth("Z1")
    out=OUT/"plots";out.mkdir(exist_ok=True)
    def charge(x,xi,t):
        with torch.no_grad():
            h=model(torch.tensor(x[None],dtype=torch.float64),torch.tensor([xi],dtype=torch.float64),
                    torch.tensor([t],dtype=torch.float64))[0].numpy().T
        return (2/9)*(h[:,0]+h[:,1])
    def true_charge(x,xi,t):
        h=truth.values(x,xi,t)
        return (2/9)*(h[:,0]+h[:,1])
    x=np.linspace(-.999,.999,401)
    xi_slice=np.linspace(.12,.45,121)
    t_slice=np.linspace(-1.35,-.38,121)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3))
    for t in (-.38,-.8,-1.2):
        axes[0].plot(x,true_charge(x,.275,t),label=f"truth t={t}")
        axes[0].plot(x,charge(x,.275,t),"--",label=f"fit t={t}")
    axes[0].set(xlabel="x",ylabel="charge H+",title="x slices; xi=0.275")
    pred_xi=np.array([charge(np.array([.5]),v,-.8)[0] for v in xi_slice])
    true_xi=np.array([true_charge(np.array([.5]),v,-.8)[0] for v in xi_slice])
    axes[1].plot(xi_slice,true_xi,label="truth");axes[1].plot(xi_slice,pred_xi,"--",label="fit")
    axes[1].set(xlabel="xi",ylabel="charge H+",title="xi slice; x=0.5, t=-0.8")
    pred_t=np.array([charge(np.array([.5]),.275,v)[0] for v in t_slice])
    true_t=np.array([true_charge(np.array([.5]),.275,v)[0] for v in t_slice])
    axes[2].plot(t_slice,true_t,label="truth");axes[2].plot(t_slice,pred_t,"--",label="fit")
    axes[2].set(xlabel="t / GeV²",ylabel="charge H+",title="t slice; x=0.5, xi=0.275")
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,ncol=2);axes[1].legend();axes[2].legend()
    fig.suptitle("Z1 / ideal L0 / seed31 — " + result["status"])
    fig.tight_layout();fig.savefig(out/"truth_reconstruction.png",dpi=160);plt.close(fig)
    bank=RowBank(HERE/"inputs/accepted_experimental_banks",HERE/"scale_rows")
    geom=json.loads((HERE/"prepared/GEOMETRY.json").read_text())
    cff={}
    for role in ("ideal_L0","surface_check"):
        rows=geom["grids"][role]
        with torch.no_grad():pred=GroupedForward(rows,bank,"reference")(model).numpy()
        target=bank.field_cff(rows,truth,"reference")
        cff[role]={"pred":pred,"truth":target}
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    for i,role in enumerate(("ideal_L0","surface_check")):
        for j,comp in enumerate(("Re","Im")):
            ax=axes[i,j];target=cff[role]["truth"][:,j];pred=cff[role]["pred"][:,j]
            ax.scatter(target,pred,s=8,alpha=.55)
            lo=min(target.min(),pred.min());hi=max(target.max(),pred.max())
            ax.plot([lo,hi],[lo,hi],"k--",lw=1)
            ax.set(xlabel=f"truth {comp}",ylabel=f"fit {comp}",title=f"{role}: {comp}")
            ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"cff_fit_and_holdout.png",dpi=160);plt.close(fig)
    journal=[json.loads(line) for line in (OUT/"ITERATIONS.jsonl").read_text().splitlines()]
    steps=[r["iteration"] for r in journal]
    J=[r["J"] for r in journal]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot(steps,J,marker="o",ms=3);axes[0].set(xlabel="outer iteration",ylabel="supplied-data J",title="finite-budget objective")
    axes[1].plot(steps,[r["radius_end"] for r in journal],marker="o",ms=3)
    axes[1].set(xlabel="outer iteration",ylabel="trust radius",title="solver radius")
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"iteration_history.png",dpi=160);plt.close(fig)
    np.savez_compressed(out/"curve_arrays.npz",x=x,xi_slice=xi_slice,t_slice=t_slice,
        x_truth=np.stack([true_charge(x,.275,t) for t in (-.38,-.8,-1.2)]),
        x_fit=np.stack([charge(x,.275,t) for t in (-.38,-.8,-1.2)]),
        xi_truth=true_xi,xi_fit=pred_xi,t_truth=true_t,t_fit=pred_t,
        cff_fit_truth=cff["ideal_L0"]["truth"],cff_fit_pred=cff["ideal_L0"]["pred"],
        cff_holdout_truth=cff["surface_check"]["truth"],cff_holdout_pred=cff["surface_check"]["pred"])
    print(json.dumps({"plots":[str(q) for q in sorted(out.glob("*.png"))]}),flush=True)


if __name__=="__main__":main()

"""Truth-blind candidate validation, then scoring of one real-only closure."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from closure.io import sha256
from closure.solver_binding import problem_and_domain
from gpd_reh.moments import FixedTMoments
from gpd_reh.polynomiality_base import design
from run_real_only_case import ALLOWED, atomic_json, atomic_torch

OUT = None
TASK = None


def targets(problem, flat, order):
    z, w = np.polynomial.legendre.leggauss(order)
    values = []
    for lo, hi, mode in ((.2, .4, "interval"), (0., 1., "xmoment")):
        x = (lo+hi)/2 + (hi-lo)*z/2
        weights = w*(hi-lo)/2
        with torch.no_grad():
            field = problem.field(flat, torch.tensor(x[None],dtype=torch.float64),
                                  torch.tensor([.275],dtype=torch.float64),
                                  torch.tensor([-.363],dtype=torch.float64))[0]
            charge = (2/9)*(field[0]+field[1])
            result = (charge*torch.tensor(weights*(5. if mode=="interval" else x),
                                          dtype=torch.float64)).sum()
        values.append(float(result))
    return values


def independent_moments(problem, flat, order):
    fit_xi = np.asarray(problem.p.xi)
    new_xi = .02+.96*(np.arange(160)+.25)/160
    fit = FixedTMoments(problem.p.t_nodes.numpy(), fit_xi, order=order,
                        spins=tuple(problem.study["polynomiality"]["spins"]))
    new = FixedTMoments(problem.p.t_nodes.numpy(), new_xi, order=order,
                        spins=tuple(problem.study["polynomiality"]["spins"]))
    maps = [torch.tensor(design(new_xi, spin) @ np.linalg.pinv(design(fit_xi, spin)),dtype=torch.float64)
            for _, spin, _ in fit.spec]
    model = lambda x, xi, t: problem.field(flat, x, xi, t)
    results = []
    with torch.no_grad():
        for t in fit.t_nodes:
            mf = fit.raw_moments(model, t)
            mn = new.raw_moments(model, t)
            results.append(torch.stack([(mn[:,i]-maps[i]@mf[:,i])/fit.scales[i]
                                        for i in range(len(fit.spec))], dim=1).numpy())
    raw = np.stack(results)
    return raw


def moment_check(problem, flat):
    by_order = {n: independent_moments(problem, flat, n) for n in (96,192,384)}
    entries = []
    for i,(channel,spin,_) in enumerate(problem.p.spec):
        values=by_order[384][:,:,i]
        entries.append({"channel":channel,"spin":spin,"RMS":float(np.sqrt(np.mean(values**2))),
                        "max_abs":float(np.max(abs(values)))})
    rms={str(n):float(np.sqrt(np.mean(v*v))) for n,v in by_order.items()}
    refinement=float(np.sqrt(np.mean((by_order[384]-by_order[192])**2)))
    return {"quadrature_RMS":rms,"residual_refinement_RMS_192_384":refinement,
            "entries":entries,"pass":bool(rms["384"]<=1e-3 and refinement<=1e-7),
            "scope":"13 fixed t nodes, 160 disjoint xi; finite checks only"}


def full_operator_checks(problem, flat, candidate_dir):
    # Three coordinate-selected corners/midpoint; independent hard/time/field
    # checks use the original Python sources and retained generator arrays.
    from closure.numerical_cache import family_path
    sys.path.insert(0, str(ROOT / "GPD_NLO_resolution_review/pilot"))
    import common
    # The accepted optimizer patches its shared `preflight.cached_row` to a
    # historical fixed-Q2 cache. Isolate the original source validator for
    # these new exact scales; retain its source and numerical settings.
    pf=common.module("closure_postfit_preflight",ROOT/"GPD_NLO_resolution_review/pilot/preflight.py")
    rows=problem.geometry
    choices=[(0,rows[0]),(len(rows)//2,rows[len(rows)//2]),(len(rows)-1,rows[-1])]
    pf.WORK=candidate_dir/"operator_work"
    pf.WORK.mkdir(parents=True,exist_ok=True)
    reports=[]
    for index,row in choices:
        xi,q2,t=row["xi"],row["Q2_GeV2"],row["t_GeV2"]
        _,state=family_path(xi)
        point={"xi":xi,"directories":state["directories"]}
        def fixed_t_model(x,skewness):
            return problem.field(flat,x,skewness,torch.full_like(skewness,t))
        result=pf.validate_point(point,q2,f"fitted_candidate_row{index}",fixed_t_model)
        path=candidate_dir/f"FULL_OPERATOR_ROW{index}.json"
        atomic_json(path,result)
        reports.append({"row":index,"xi_hex":xi.hex(),"Q2_hex":q2.hex(),"t_hex":t.hex(),
                        "pass":result["pass"],"gates":result["gates"],
                        "CFF_estimated_total":result["CFF_estimated_total"],
                        "CFF_limits":result["CFF_limits"],
                        "file":str(path.relative_to(OUT)),"sha256":sha256(path)})
    return {"pass":all(item["pass"] for item in reports),"coordinate_selected_rows":reports,
            "scope":"three first/mid/last geometry rows, source Python full-operator checks"}


def validate_candidate(problem, domain, candidate, rank):
    candidate_dir=OUT/f"candidate{rank}"
    candidate_dir.mkdir(exist_ok=True)
    path=OUT/candidate["path"]
    if sha256(path)!=candidate["sha256"]:
        raise RuntimeError("Candidate checkpoint changed")
    ck=torch.load(path,map_location="cpu",weights_only=True)
    flat=ck["flat"].detach().double()
    start=time.monotonic()
    with torch.no_grad():
        c=domain.constraints(problem,flat).numpy()
        J=float(problem.objective(flat))
        components=problem.components(flat).numpy().tolist()
    if abs(J-candidate["J"])>1e-10*max(1,abs(J)):
        raise RuntimeError("Candidate objective changed")
    t192=targets(problem,flat,192);t384=targets(problem,flat,384)
    target_refinement=max(abs(np.asarray(t192)-t384))
    finite=bool(np.isfinite(c).all() and np.isfinite(J))
    precheck=bool(finite and np.max(c)<=0 and target_refinement<=1e-7)
    result={"rank":rank,"stage":candidate["stage"],"J":J,"components":components,
            "constraint_max":float(np.max(c)),"constraint_exceedances":int(np.sum(c>0)),
            "constraints_finite":bool(np.isfinite(c).all()),
            "moment_predictive_constraint":float(c[0]),
            "target_192":t192,"target_384":t384,
            "target_refinement_max_abs":float(target_refinement),"precheck_pass":precheck,
            "source_checkpoint":candidate}
    if finite:
        result["moments"]=moment_check(problem,flat)
        # Retain source-operator diagnostics even for a finite candidate that
        # misses a legacy eligibility guide. Such a result may still be scored
        # as an explicitly labelled diagnostic field.
        result["full_operator"]=full_operator_checks(problem,flat,candidate_dir)
    else:
        result["moments"]={"pass":False,"not_run":"nonfinite_candidate"}
        result["full_operator"]={"pass":False,"not_run":"nonfinite_candidate"}
    result["pass"]=bool(precheck and result["moments"]["pass"] and result["full_operator"]["pass"])
    result["seconds"]=time.monotonic()-start
    atomic_json(candidate_dir/"CHECK.json",result)
    return result


def main(task_id):
    global OUT, TASK
    if task_id not in ALLOWED:
        raise ValueError("Task is not in the real-only sidecar allowlist")
    TASK=task_id
    OUT=HERE/"runs"/task_id
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out=OUT/"RESULT.json"
    if out.exists():
        print(out.read_text(),flush=True)
        return
    complete_path=OUT/"OPTIMIZER_COMPLETE.json"
    if not complete_path.exists():
        raise RuntimeError("Finite-budget optimizer has not completed")
    complete=json.loads(complete_path.read_text())
    problem,domain=problem_and_domain(TASK)
    validations=[];selected=None
    start=time.monotonic()
    for rank,candidate in enumerate(complete["candidates_by_supplied_J"][:4]):
        prior=OUT/f"candidate{rank}/CHECK.json"
        result=json.loads(prior.read_text()) if prior.exists() else validate_candidate(problem,domain,candidate,rank)
        validations.append(result)
        print(json.dumps({"rank":rank,"stage":result["stage"],"J":result["J"],
                          "pass":result["pass"],"seconds":result["seconds"]}),flush=True)
        if result["pass"]:
            selected=result
            break
    # The best finite state remains available for an explicitly labelled
    # diagnostic comparison even if original quality gates abstain.
    scored=selected if selected is not None else next((r for r in validations if math.isfinite(r["J"])),None)
    if scored is None:
        raise RuntimeError("No finite candidate to compare")
    source=OUT/scored["source_checkpoint"]["path"]
    ck=torch.load(source,map_location="cpu",weights_only=True)
    destination=OUT/("final.pt" if selected is not None else "diagnostic_best.pt")
    if not destination.exists():
        atomic_torch(destination,ck)
    from score_checkpoint import score
    scores_path=OUT/"SCORES.json"
    if scores_path.exists():
        scores=json.loads(scores_path.read_text())
    else:
        scores=score(TASK,destination,[HERE/"scale_rows"],scores_path)
    with torch.no_grad():
        flat=ck["flat"].detach().double()
        coefficients=problem.p.coefficients(lambda x,xi,t:problem.field(flat,x,xi,t))
    atomic_json(OUT/"FIXED_T_COEFFICIENTS.json",coefficients)
    result={"status":"FINITE_BUDGET_FEASIBLE" if selected is not None else "ABSTAIN_DIAGNOSTIC_COMPARISON_ONLY",
            "task":TASK,"selected":selected,"validations":validations,
            "scored_checkpoint":str(destination.relative_to(OUT)),
            "scored_checkpoint_sha256":sha256(destination),
            "scores_sha256":sha256(scores_path),
            "validation_and_scoring_seconds":time.monotonic()-start,
            "optimization":complete,"peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "interpretation":"one finite-budget seed and coarse surface; no continuum or uncertainty claim"}
    atomic_json(out,result)
    print(json.dumps({"status":result["status"],"stage":scored["stage"],
                      "J":scored["J"],"seconds":result["validation_and_scoring_seconds"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",choices=ALLOWED,required=True)
    main(parser.parse_args().task)

"""Three fixed five-step input-response continuations using the existing solver."""
from __future__ import annotations
import importlib.util, json, math, os, platform, sys, time
from pathlib import Path
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import scipy
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "completion_runs/ReH_central_t_comparison"
EXP = ROOT / "GPD_ReH_experimental_run"
REVIEW = ROOT / "GPD_experimental_t_enriched_review"
PARENT = ROOT / "completion_runs/ReH_t_enriched_restart/results/enriched/TERMINAL.pt"
sys.path[:0] = [str(OLD), str(EXP), str(ROOT / "GPD_direct_error_calibration"),
                str(ROOT / "GPD_NLO_constrained_solver_review")]
from adapter import BranchProblem, BranchDomain, NODES, atomic, sha, solver_binding
import fit_nlo as accepted

CASES = ("zero", "plus", "minus")
PARENT_SHA = "86babd3689dd4d81054a75f5893709495ad6b713041e925f166a5b66d0e7d11a"

def load_solver():
    path = ROOT / "completion_runs/NLO_constrained_solver/solver.py"
    spec = importlib.util.spec_from_file_location("reh_input_response_solver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def jsonable(v):
    if isinstance(v, dict): return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray): return jsonable(v.tolist())
    if isinstance(v, np.generic): return jsonable(v.item())
    if isinstance(v, float) and not math.isfinite(v): return {"nonfinite":repr(v)}
    return v

def save_tensor(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.pt"); torch.save(value,temp); temp.replace(path)

def checkpoint(problem, case, flat, stage, **extra):
    vector = torch.as_tensor(flat,dtype=torch.float64).detach().clone()
    return {"case":case,"branch":"enriched","stage":stage,"flat":vector,
            "state_dict":{k:v.detach().clone() for k,v in problem.parameters(vector).items()},
            "parent_sha256":PARENT_SHA,"t_nodes":NODES["enriched"],**extra}

def input_arrays():
    original = EXP / "prepared/observations.npz"
    with np.load(original,allow_pickle=False) as f: base={k:f[k].copy() for k in f.files}
    cases={}
    for name in CASES:
        path=HERE/"inputs"/(name+".npz")
        with np.load(path,allow_pickle=False) as f: data={k:f[k].copy() for k in f.files}
        for k,v in base.items():
            if k!="y" and not np.array_equal(data[k],v):
                raise RuntimeError("Changed coordinate, row identity, or uncertainty: "+name+" "+k)
        if not np.array_equal(data["sigma"],base["sigma"]): raise RuntimeError("Changed quoted errors")
        cases[name]=data
    if not np.array_equal(cases["zero"]["y"],base["y"]):raise RuntimeError("Zero input changed")
    if not np.allclose(cases["plus"]["y"]+cases["minus"]["y"],2*base["y"],rtol=0,atol=1e-14):
        raise RuntimeError("Antithetic inputs changed")
    return base,cases

def freeze():
    if sha(PARENT)!=PARENT_SHA:raise RuntimeError("Enriched parent changed")
    manifest=json.loads((HERE/"inputs/INPUT_MANIFEST.json").read_text())
    for item in manifest["entries"]:
        if sha(HERE/"inputs"/item["file"])!=item["sha256"]:raise RuntimeError("Input changed")
    files=[HERE/"pilot_run.py",REVIEW/"input_response.py",REVIEW/"DECISION.md",
           REVIEW/"WORKER_PROMPT.md",OLD/"adapter.py",OLD/"run.py",
           EXP/"gpd_reh/problem.py",EXP/"gpd_reh/model.py",EXP/"gpd_reh/moments.py",
           EXP/"experimental_domain.py",EXP/"experimental_optimizer.py",
           EXP/"inputs/final_summary_by_DNN.csv",EXP/"banks/index.json",
           ROOT/"completion_runs/NLO_constrained_solver/solver.py",
           ROOT/"GPD_NLO_constrained_solver_review/bind_solver.py",
           ROOT/"GPD_NLO_constrained_solver_review/residual_subproblem.py",PARENT]
    lock={"id":"ReH_input_response_pilot_v1","cases":list(CASES),
          "parent_sha256":PARENT_SHA,"input_manifest_sha256":sha(HERE/"inputs/INPUT_MANIFEST.json"),
          "sources":{str(p.relative_to(ROOT)):sha(p) for p in files},
          "settings":{"max_iterations":5,"initial_radius":.5,"max_evaluations":12000,
                      "other_settings":"unchanged inherited solver"},
          "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
                         "scipy":scipy.__version__,"torch":torch.__version__,"threads":torch.get_num_threads()}}
    path=HERE/"RUN_LOCK.json"
    if path.exists():
        if json.loads(path.read_text())!=lock:raise RuntimeError("Frozen pilot lock changed")
    else:atomic(path,lock)
    return lock

def make_problem(name, arrays):
    p=BranchProblem("enriched")
    baseline=p.y.detach().numpy()
    if not np.array_equal(baseline,arrays["zero"]["y"]):raise RuntimeError("Prepared observations changed")
    p.y=torch.as_tensor(arrays[name]["y"].copy(),dtype=torch.float64)
    if not np.array_equal(p.sigma.detach().numpy(),arrays[name]["sigma"]):raise RuntimeError("Sigma mismatch")
    if p.x0.numel()!=3867 or p.lam!=1 or len(p.p.t_nodes)!=13:raise RuntimeError("Model changed")
    return p

def preflight(parent, arrays):
    path=HERE/"PREFLIGHT.json"
    if path.exists():return json.loads(path.read_text())
    flat=parent["flat"].detach().clone()
    p0=make_problem("zero",arrays);d=BranchDomain("enriched")
    with torch.no_grad():
        baseline=p0.residual(flat).numpy();j0=float(p0.objective(flat));
        prediction=p0.forward(lambda x,xi,t:p0.field(flat,x,xi,t)).numpy()
        c0=d.constraints(p0,flat).numpy()
    prior=json.loads((ROOT/"completion_runs/ReH_t_enriched_restart/results/enriched/RESULT.json").read_text())
    if j0!=prior["J"] or not np.isfinite(c0).all() or np.max(c0)>0:raise RuntimeError("Parent objective/feasibility mismatch")
    if not np.allclose(prediction[:,0],baseline[:160]*p0.sigma.numpy()*math.sqrt(160)+p0.y.numpy(),rtol=0,atol=1e-13):
        raise RuntimeError("Zero prediction reproduction mismatch")
    cases={}
    for name in CASES:
        p=make_problem(name,arrays)
        with torch.no_grad():
            residual=p.residual(flat).numpy();c=d.constraints(p,flat).numpy()
        expected=-(arrays[name]["y"]-arrays["zero"]["y"])/arrays["zero"]["sigma"]/math.sqrt(160)
        if not np.allclose(residual[:160]-baseline[:160],expected,rtol=0,atol=2e-15):
            raise RuntimeError("Data residual not affine: "+name)
        if not np.array_equal(residual[160:],baseline[160:]):raise RuntimeError("Poly residual changed: "+name)
        if not np.array_equal(c,c0):raise RuntimeError("Data-independent domain changed: "+name)
        cases[name]={"initial_J":float(np.sum(residual*residual)),
                     "data_residual_affine_max_abs":float(np.max(np.abs(residual[:160]-baseline[:160]-expected))),
                     "polynomial_residual_bit_identical":True,"constraints_bit_identical":True}
    # One small original-objective directional check; no Jacobian design study.
    direction=torch.linspace(-1,1,flat.numel());direction/=torch.linalg.vector_norm(direction)
    x=flat.detach().requires_grad_(True);grad=torch.autograd.grad(p0.objective(x),x)[0]
    h=1e-5;fd=(float(p0.objective(flat+h*direction))-float(p0.objective(flat-h*direction)))/(2*h)
    ad=float(grad@direction)
    if abs(fd-ad)>1e-6*max(1.,abs(ad)):raise RuntimeError("Directional objective derivative mismatch")
    np.savez_compressed(HERE/"CONSTRAINT_VECTOR.npz",constraints=c0)
    result={"parent_sha256":PARENT_SHA,"J_zero":j0,"prior_J":prior["J"],
            "prediction_rows":len(prediction),"initial_constraint_count":len(c0),
            "initial_constraint_max":float(np.max(c0)),"initial_moment_constraint":float(c0[0]),
            "domain_data_independence":"ExperimentalDomain.constraints uses problem.field and frozen contractions; equality checked for all three case tensors",
            "directional_derivative_ad":ad,"directional_derivative_fd":fd,"cases":cases}
    atomic(path,result);return result

def run_case(name,parent,arrays,solver):
    out=HERE/"cases"/name;out.mkdir(parents=True,exist_ok=True)
    terminal=out/"TERMINAL.pt"
    if terminal.exists():
        result=json.loads((out/"RESULT.json").read_text())
        if result["run_lock_sha256"]!=sha(HERE/"RUN_LOCK.json"):raise RuntimeError("Completed source binding changed")
        return result
    if (out/"STARTED.json").exists():raise RuntimeError("Interrupted case preserved; no automatic restart: "+name)
    p=make_problem(name,arrays);d=BranchDomain("enriched");initial=parent["flat"].detach().clone()
    with torch.no_grad():j0=float(p.objective(initial))
    save_tensor(out/"STAGE_00.pt",checkpoint(p,name,initial,0,J=j0))
    atomic(out/"STARTED.json",{"case":name,"parent_sha256":PARENT_SHA,
                                "run_lock_sha256":sha(HERE/"RUN_LOCK.json"),
                                "stage0_sha256":sha(out/"STAGE_00.pt")})
    oracle=accepted.Oracle(p,d,initial.numpy().copy());oracle,binding_stats=solver_binding().bind(solver,oracle)
    settings=dict(solver.SETTINGS,max_iterations=5,max_evaluations=12000,
                  optimization_seconds=float("inf"),initial_radius=.5)
    atomic(out/"SETTINGS.json",jsonable(settings));started=time.monotonic()
    def record(flat,rec,trial):
        stage=rec["iteration"]+1;atomic(out/f"ITERATION_{stage:02}.json",jsonable(rec))
        vector=torch.as_tensor(flat,dtype=torch.float64)
        with torch.no_grad():value=float(p.objective(vector))
        save_tensor(out/f"STAGE_{stage:02}.pt",checkpoint(p,name,vector,stage,J=value))
        if trial is not None:save_tensor(out/f"TRIAL_{stage:02}.pt",{"flat":torch.as_tensor(trial,dtype=torch.float64),
                                                                    "finite":bool(np.isfinite(trial).all())})
        atomic(out/"PROGRESS.json",{"case":name,"completed_iterations":stage,"J":value,
                                    "accepted":rec.get("accepted"),"elapsed_seconds":time.monotonic()-started,
                                    "oracle_counts":jsonable(oracle.counts)})
        print(json.dumps({"case":name,"stage":stage,"J":value,"accepted":rec.get("accepted"),
                          "seconds":time.monotonic()-started}),flush=True)
    final,diag=solver.solve(oracle,accepted.old.kkt,record,settings)
    vector=torch.as_tensor(final,dtype=torch.float64)
    if vector.numel()!=3867 or not torch.isfinite(vector).all():raise RuntimeError("Nonfinite terminal vector")
    with torch.no_grad():components=p.components(vector).numpy();constraints=d.constraints(p,vector).numpy()
    save_tensor(terminal,checkpoint(p,name,vector,len(diag["history"]),J=float(sum(components)),
                                    solver_status=diag["status"]))
    atomic(out/"SOLVER.json",jsonable(diag))
    stages=[]
    for stage in range(len(diag["history"])+1):
        file=out/f"STAGE_{stage:02}.pt";cp=torch.load(file,weights_only=True,map_location="cpu")
        if np.isfinite(cp["J"]):stages.append({"stage":stage,"J":cp["J"],"path":file.name,"sha256":sha(file)})
    winner=min(stages,key=lambda x:(x["J"],x["stage"]))
    result={"case":name,"status":diag["status"],"iterations_recorded":len(diag["history"]),
            "terminal_J":float(sum(components)),"terminal_data_loss":float(components[0]),
            "terminal_poly_loss":float(components[1]),"maximum_constraint":float(np.max(constraints)),
            "moment_constraint":float(constraints[0]),"top_finite_candidate":winner,
            "all_finite_stages":stages,"solver_seconds":diag["seconds"],
            "total_seconds":time.monotonic()-started,"terminal_sha256":sha(terminal),
            "run_lock_sha256":sha(HERE/"RUN_LOCK.json"),"binding_stats":jsonable(binding_stats),
            "oracle_counts":jsonable(oracle.counts)}
    atomic(out/"RESULT.json",result);return result

def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if sha(PARENT)!=PARENT_SHA:raise RuntimeError("Parent changed")
    parent=torch.load(PARENT,weights_only=True,map_location="cpu")
    if parent["flat"].numel()!=3867:raise RuntimeError("Wrong parent")
    base,arrays=input_arrays();lock=freeze();preflight(parent,arrays)
    solver=load_solver();results=[]
    for name in CASES:
        result=run_case(name,parent,arrays,solver);results.append(result)
        atomic(HERE/"PROGRESS.json",{"completed_cases":[r["case"] for r in results],
                                     "run_lock_sha256":sha(HERE/"RUN_LOCK.json")})
    atomic(HERE/"THREE_CASES_COMPLETE.json",{"cases":results,"run_lock_sha256":sha(HERE/"RUN_LOCK.json")})
    print("THREE_CASES_COMPLETE",flush=True)

if __name__=="__main__":main()

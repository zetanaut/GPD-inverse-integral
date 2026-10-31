"""Four source-bound, five-step FiLM continuations from a common saved parent."""
from __future__ import annotations
import json, math, os, platform, resource, sys, time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
import scipy
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXP=ROOT/'GPD_ReH_experimental_run'
REVIEW=ROOT/'GPD_coherent_surface_review'
PILOT=ROOT/'completion_runs/ReH_input_response_pilot'
OLD=ROOT/'completion_runs/ReH_surface_mean_pilot'
TOOLS=ROOT/'GPD_selected_surface_review'
PARENT=OLD/'STAGE_20.pt'
CASES=('central_control','ensemble_mean','replica_000','replica_001')
PARENT_SHA='b18c6c963c109e1249f6c3d30bdf7cdfa429f7da31631aea81b8e13cafa07dbb'
sys.path[:0]=[str(PILOT),str(TOOLS)]
from pilot_run import BranchProblem,BranchDomain,accepted,atomic,checkpoint,jsonable,load_solver,save_tensor,sha,solver_binding
from surface_tools import replace_problem_targets


def arrays():
    with np.load(EXP/'prepared/observations.npz',allow_pickle=False) as z: old={k:z[k].copy() for k in z.files}
    new={}
    for name in CASES:
        with np.load(HERE/'prepared'/f'{name}.npz',allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
        if a.keys()!=old.keys():raise RuntimeError('Prepared field inventory changed')
        for k in old:
            if k!='y' and not np.array_equal(old[k],a[k]):raise RuntimeError(f'Prepared {k} changed: {name}')
        new[name]=a
    return old,new


def problem(a=None):
    p=BranchProblem('enriched')
    if p.x0.numel()!=3867 or p.lam!=1 or len(p.p.t_nodes)!=13:raise RuntimeError('FiLM/moment problem changed')
    if a is not None:replace_problem_targets(p,a)
    return p


def freeze():
    lock_input=json.loads((HERE/'prepared/INPUT_LOCK.json').read_text())
    if lock_input['cases']!=list(CASES) or lock_input['parent_sha256']!=PARENT_SHA or sha(PARENT)!=PARENT_SHA:
        raise RuntimeError('Input case/parent lock changed')
    for name,digest in lock_input['source_bindings'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Scientific input changed: '+name)
    for name,digest in lock_input['outputs'].items():
        if sha(HERE/'prepared'/name)!=digest:raise RuntimeError('Prepared input changed: '+name)
    files=[HERE/'dispatch.py',REVIEW/'prepare_inputs.py',REVIEW/'DECISION.md',REVIEW/'WORKER_PROMPT.md',
           PILOT/'pilot_run.py',OLD/'run_surface.py',TOOLS/'surface_tools.py',
           ROOT/'completion_runs/ReH_central_t_comparison/adapter.py',
           ROOT/'completion_runs/NLO_constrained_solver/solver.py',
           ROOT/'GPD_NLO_constrained_solver_review/bind_solver.py',
           ROOT/'GPD_NLO_constrained_solver_review/residual_subproblem.py',
           EXP/'gpd_reh/problem.py',EXP/'gpd_reh/model.py',EXP/'gpd_reh/moments.py',EXP/'gpd_reh/bank.py',
           EXP/'experimental_domain.py',EXP/'experimental_optimizer.py',
           EXP/'prepared/observations.npz',EXP/'prepared/observations.csv',EXP/'banks/index.json',PARENT]
    solver=load_solver()
    settings=dict(solver.SETTINGS,max_iterations=5,max_evaluations=12000,
                  optimization_seconds=float('inf'),initial_radius=.5)
    lock={'id':'ReH_coherent_GPD_pilot_v1','cases':list(CASES),'parent_sha256':PARENT_SHA,
          'prepared_input_lock_sha256':sha(HERE/'prepared/INPUT_LOCK.json'),
          'source_modules':{str(p.relative_to(ROOT)):sha(p) for p in files},
          'imported_module_files':{name:str(sys.modules[name].__file__) for name in ('pilot_run','adapter','fit_nlo','surface_tools')},
          'settings':jsonable(settings),
          'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
                         'scipy':scipy.__version__,'torch':torch.__version__,'torch_threads':torch.get_num_threads()},
          'unchanged_large_maps':{'index':'GPD_ReH_experimental_run/banks/index.json',
                                  'index_sha256':sha(EXP/'banks/index.json'),
                                  'family_root':'GPD_ReH_experimental_run/families'}}
    file=HERE/'RUN_LOCK.json'
    if file.exists():
        if json.loads(file.read_text())!=lock:raise RuntimeError('Frozen dispatcher/source lock changed')
    else:atomic(file,lock)
    atomic(HERE/'SETTINGS.json',jsonable(settings))
    return solver,settings


def preflight(flat,old,targets):
    file=HERE/'PREFLIGHT.json'
    if file.exists():return json.loads(file.read_text())
    p0=problem();d=BranchDomain('enriched')
    if not np.array_equal(p0.y.detach().numpy(),old['y']) or not np.array_equal(p0.sigma.detach().numpy(),old['sigma']):
        raise RuntimeError('Original problem data changed')
    with torch.no_grad():
        r0=p0.residual(flat).numpy();c0=d.constraints(p0,flat).numpy()
        pred=p0.forward(lambda x,xi,t:p0.field(flat,x,xi,t)).numpy()[:,0]
    if np.max(abs(r0[:160]-(pred-old['y'])/old['sigma']/math.sqrt(160)))>2e-12:
        raise RuntimeError('Original residual/sign formula changed')
    j0=p0.residual_jacobian(flat).numpy()
    result={'parent_sha256':PARENT_SHA,'original_data_loss':float(np.dot(r0[:160],r0[:160])),
            'original_poly_loss':float(np.dot(r0[160:],r0[160:])),
            'original_constraint_max':float(np.max(c0)),'constraint_count':len(c0),'cases':{}}
    for name in CASES:
        p=problem(targets[name])
        with torch.no_grad():r=p.residual(flat).numpy();c=d.constraints(p,flat).numpy()
        expected=-(targets[name]['y']-old['y'])/old['sigma']/math.sqrt(160)
        diff=float(np.max(abs(r[:160]-r0[:160]-expected)))
        if diff>2e-12 or not np.array_equal(r[160:],r0[160:]) or not np.array_equal(c,c0):
            raise RuntimeError('Target-only residual/domain preflight failed: '+name)
        result['cases'][name]={'initial_J':float(np.dot(r,r)),
                                'initial_data_loss':float(np.dot(r[:160],r[:160])),
                                'data_residual_affine_max_abs':diff,
                                'poly_residual_bit_identical':True,'constraints_bit_identical':True}
    # One bounded Jacobian comparison suffices: y is the only replaced field.
    j1=problem(targets['ensemble_mean']).residual_jacobian(flat).numpy()
    result['target_jacobian_max_abs_difference']=float(np.max(abs(j1-j0)))
    if result['target_jacobian_max_abs_difference']>1e-12:raise RuntimeError('Target changed residual Jacobian')
    atomic(file,result);return result


def run_case(name,parent,targets,solver,settings):
    out=HERE/'cases'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():
        result=json.loads((out/'RESULT.json').read_text())
        if result['run_lock_sha256']!=sha(HERE/'RUN_LOCK.json'):raise RuntimeError('Completed source lock changed')
        for row in result['all_finite_stages']:
            if sha(out/row['path'])!=row['sha256']:raise RuntimeError('Completed stage changed')
        return result
    if (out/'STARTED.json').exists():raise RuntimeError('Interrupted case preserved: '+name)
    p=problem(targets[name]);d=BranchDomain('enriched');initial=parent['flat'].detach().clone()
    with torch.no_grad():comp=p.components(initial).numpy()
    save_tensor(out/'STAGE_00.pt',checkpoint(p,name,initial,0,J=float(sum(comp)),
                                             data_loss=float(comp[0]),poly_loss=float(comp[1]),
                                             common_parent_sha256=PARENT_SHA))
    atomic(out/'STARTED.json',{'case':name,'parent_sha256':PARENT_SHA,
                               'run_lock_sha256':sha(HERE/'RUN_LOCK.json'),
                               'stage0_sha256':sha(out/'STAGE_00.pt')})
    oracle=accepted.Oracle(p,d,initial.numpy().copy())
    oracle,binding_stats=solver_binding().bind(solver,oracle)
    started=time.monotonic()
    def record(current,rec,trial):
        stage=rec['iteration']+1
        atomic(out/f'ITERATION_{stage:02}.json',jsonable(rec))
        vector=torch.as_tensor(current,dtype=torch.float64)
        with torch.no_grad():c=p.components(vector).numpy()
        save_tensor(out/f'STAGE_{stage:02}.pt',checkpoint(p,name,vector,stage,J=float(sum(c)),
                    data_loss=float(c[0]),poly_loss=float(c[1]),common_parent_sha256=PARENT_SHA))
        if trial is not None:save_tensor(out/f'TRIAL_{stage:02}.pt',
                    {'flat':torch.as_tensor(trial,dtype=torch.float64),'finite':bool(np.isfinite(trial).all())})
        progress={'case':name,'stage':stage,'J':float(sum(c)),'data_loss':float(c[0]),
                  'poly_loss':float(c[1]),'accepted':rec.get('accepted'),
                  'elapsed_seconds':time.monotonic()-started,'oracle_counts':jsonable(oracle.counts),
                  'peak_memory_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        atomic(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    final,diag=solver.solve(oracle,accepted.old.kkt,record,settings)
    vector=torch.as_tensor(final,dtype=torch.float64)
    if vector.numel()!=3867 or not torch.isfinite(vector).all():raise RuntimeError('Nonfinite terminal vector')
    with torch.no_grad():c=p.components(vector).numpy();constraints=d.constraints(p,vector).numpy()
    save_tensor(out/'TERMINAL.pt',checkpoint(p,name,vector,len(diag['history']),J=float(sum(c)),
                         data_loss=float(c[0]),poly_loss=float(c[1]),solver_status=diag['status'],
                         common_parent_sha256=PARENT_SHA))
    atomic(out/'SOLVER.json',jsonable(diag))
    stages=[]
    for stage in range(len(diag['history'])+1):
        path=out/f'STAGE_{stage:02}.pt';cp=torch.load(path,weights_only=True,map_location='cpu')
        if np.isfinite(cp['J']):stages.append({'stage':stage,'J':cp['J'],'data_loss':cp['data_loss'],
                             'poly_loss':cp['poly_loss'],'path':path.name,'sha256':sha(path)})
    winner=min(stages,key=lambda row:(row['J'],row['stage']))
    result={'case':name,'status':diag['status'],'iterations_recorded':len(diag['history']),
            'terminal_J':float(sum(c)),'terminal_data_loss':float(c[0]),'terminal_poly_loss':float(c[1]),
            'maximum_constraint':float(np.max(constraints)),'moment_constraint':float(constraints[0]),
            'selected_finite_candidate':winner,'all_finite_stages':stages,
            'solver_seconds':diag['seconds'],'elapsed_seconds':time.monotonic()-started,
            'terminal_sha256':sha(out/'TERMINAL.pt'),'run_lock_sha256':sha(HERE/'RUN_LOCK.json'),
            'binding_stats':jsonable(binding_stats),'oracle_counts':jsonable(oracle.counts),
            'peak_memory_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'storage_bytes':sum(f.stat().st_size for f in out.rglob('*') if f.is_file())}
    atomic(out/'RESULT.json',result)
    print('CASE_COMPLETE',json.dumps({'case':name,'status':diag['status'],'selected':winner}),flush=True)
    return result


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old,targets=arrays();solver,settings=freeze()
    parent=torch.load(PARENT,weights_only=True,map_location='cpu')
    if parent['flat'].numel()!=3867:raise RuntimeError('Parent parameter count changed')
    print('PREFLIGHT_START',flush=True);print(json.dumps(preflight(parent['flat'].detach().clone(),old,targets)),flush=True)
    for name in CASES:
        result=run_case(name,parent,targets,solver,settings)
        if name=='central_control':
            # Actual completed control is the cost measurement, not a benchmark re-run.
            atomic(HERE/'COST_PROJECTION.json',{'control_solver_seconds':result['solver_seconds'],
                  'control_elapsed_seconds':result['elapsed_seconds'],
                  'control_peak_memory_bytes_macos':result['peak_memory_bytes_macos'],
                  'control_storage_bytes':result['storage_bytes'],
                  'remaining_three_seconds_projection_from_control':3*result['elapsed_seconds'],
                  'projection_is_not_measurement':True})
    print('ALL_FOUR_COMPLETE',flush=True)

if __name__=='__main__':main()

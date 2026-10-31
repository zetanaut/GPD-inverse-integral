"""Bounded cumulative 5/10/20 continuation of three saved coherent-input GPD fits."""
from __future__ import annotations
import json,math,os,platform,resource,sys,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import scipy
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'completion_runs/ReH_coherent_GPD_pilot'
REVIEW=ROOT/'GPD_coherent_GPD_review'
PILOT=ROOT/'completion_runs/ReH_input_response_pilot'
TOOLS=ROOT/'GPD_selected_surface_review'
sys.path[:0]=[str(PRIOR),str(PILOT),str(TOOLS)]
from dispatch import problem,BranchDomain,accepted,atomic,checkpoint,jsonable,load_solver,save_tensor,sha,solver_binding
CASES=('ensemble_mean','replica_000','replica_001')
PARENT_SHA='b18c6c963c109e1249f6c3d30bdf7cdfa429f7da31631aea81b8e13cafa07dbb'


def freeze(plan,settings_by_case):
    if plan['id']!='ReH_coherent_budget_check_v1' or tuple(j['case'] for j in plan['jobs'])!=CASES:
        raise RuntimeError('Plan identity changed')
    files=[HERE/'run_budget.py',HERE/'PLAN.json',REVIEW/'DECISION.md',REVIEW/'WORKER_PROMPT.md',
           REVIEW/'prepare_budget_check.py',PRIOR/'dispatch.py',PRIOR/'RUN_LOCK.json',
           ROOT/'completion_runs/NLO_constrained_solver/solver.py',
           ROOT/'GPD_NLO_constrained_solver_review/bind_solver.py',
           ROOT/'GPD_NLO_constrained_solver_review/residual_subproblem.py',
           ROOT/'completion_runs/ReH_central_t_comparison/adapter.py',
           ROOT/'GPD_ReH_experimental_run/gpd_reh/problem.py',
           ROOT/'GPD_ReH_experimental_run/gpd_reh/model.py',
           ROOT/'GPD_ReH_experimental_run/gpd_reh/moments.py',
           ROOT/'GPD_ReH_experimental_run/gpd_reh/bank.py',
           ROOT/'GPD_ReH_experimental_run/experimental_domain.py',
           ROOT/'GPD_ReH_experimental_run/experimental_optimizer.py',
           ROOT/'GPD_ReH_experimental_run/prepared/observations.npz',
           ROOT/'GPD_ReH_experimental_run/prepared/observations.csv',
           ROOT/'GPD_ReH_experimental_run/inputs/final_summary_by_DNN.csv',
           ROOT/'GPD_ReH_experimental_run/banks/index.json']
    for j in plan['jobs']:
        files += [Path(j['start_checkpoint']),Path(j['observations']),
                  PRIOR/'cases'/j['case']/'SOLVER.json',PRIOR/'cases'/j['case']/'RESULT.json']
    lock={'id':'ReH_coherent_budget_check_v1','plan_sha256':sha(HERE/'PLAN.json'),
          'original_common_parent_sha256':PARENT_SHA,
          'sources':{str(p.relative_to(ROOT)):sha(p) for p in files},
          'settings_by_case':{k:jsonable(v) for k,v in settings_by_case.items()},
          'environment':{'python':sys.version,'platform':platform.platform(),
                         'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,
                         'torch_threads':torch.get_num_threads()},
          'continuation_state':'All original steps accepted; saved final radius; stagnation=0 and extra working set empty. Recreate oracle caches at saved x.',
          'maps':{'index':'GPD_ReH_experimental_run/banks/index.json',
                  'index_sha256':sha(ROOT/'GPD_ReH_experimental_run/banks/index.json'),
                  'families':'GPD_ReH_experimental_run/families'}}
    path=HERE/'RUN_LOCK.json'
    if path.exists():
        if json.loads(path.read_text())!=lock:raise RuntimeError('Frozen source/settings lock changed')
    else:atomic(path,lock)
    return lock


def load_case(j):
    name=j['case'];prior=PRIOR/'cases'/name
    res=json.loads((prior/'RESULT.json').read_text());sol=json.loads((prior/'SOLVER.json').read_text())
    if res['status']!='iteration_budget' or res['iterations_recorded']!=5 or len(sol['history'])!=5:
        raise RuntimeError('Expected completed five-step prefix')
    if any(h['accepted'] is not True or h['unchanged_state_terminations']!=0 for h in sol['history']):
        raise RuntimeError('Saved continuation contains nontrivial hidden stagnation/working-set state')
    if not math.isclose(sol['final_radius'],j['initial_trust_radius_from_saved_solver'],rel_tol=0,abs_tol=0):
        raise RuntimeError('Saved final radius changed')
    if sha(j['start_checkpoint'])!=j['start_sha256'] or sha(j['observations'])!=j['observations_sha256']:
        raise RuntimeError('Saved checkpoint or target changed')
    cp=torch.load(j['start_checkpoint'],weights_only=True,map_location='cpu')
    if cp['stage']!=5 or cp['case']!=name or cp['common_parent_sha256']!=PARENT_SHA:
        raise RuntimeError('Incorrect stage-five state')
    with np.load(j['observations'],allow_pickle=False) as z:target={k:z[k].copy() for k in z.files}
    p=problem(target)
    if cp['flat'].numel()!=3867 or p.x0.numel()!=3867 or p.lam!=1 or len(p.p.t_nodes)!=13:
        raise RuntimeError('Original FiLM/moment problem changed')
    if not np.array_equal(p.y.detach().numpy(),target['y']) or not np.array_equal(p.sigma.detach().numpy(),target['sigma']):
        raise RuntimeError('Target/sigma mismatch')
    d=BranchDomain('enriched')
    with torch.no_grad():c=p.components(cp['flat']).numpy();con=d.constraints(p,cp['flat']).numpy()
    if abs(float(sum(c))-cp['J'])>1e-11 or abs(float(sum(c))-res['selected_finite_candidate']['J'])>1e-11:
        raise RuntimeError('Stage-five objective replay mismatch')
    if not np.isfinite(con).all() or np.max(con)>0:raise RuntimeError('Saved stage-five state infeasible')
    return cp,target,p,d,res,sol


def new_checkpoint(p,name,flat,stage,immediate_sha,**extra):
    cp=checkpoint(p,name,flat,stage,**extra)
    cp.pop('parent_sha256',None) # inherited helper's legacy central13 label is inapplicable here
    cp['original_common_parent_sha256']=PARENT_SHA
    cp['immediate_resume_sha256']=immediate_sha
    cp['cumulative_outer_steps']=stage
    return cp


def run_case(j,solver,settings):
    name=j['case'];out=HERE/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():
        r=json.loads((out/'RESULT.json').read_text())
        if r['run_lock_sha256']!=sha(HERE/'RUN_LOCK.json'):raise RuntimeError('Completed case changed')
        for row in r['all_finite_stages']:
            if sha(out/row['path'])!=row['sha256']:raise RuntimeError('Saved stage changed')
        return r
    if (out/'STARTED.json').exists():
        raise RuntimeError('Interrupted case preserved; resume only after state audit: '+name)
    cp,target,p,d,prior,sol=load_case(j)
    original=Path(j['start_checkpoint']);copied=out/'STAGE_05.pt'
    copied.write_bytes(original.read_bytes())
    if sha(copied)!=j['start_sha256']:raise RuntimeError('Stage-five copy mismatch')
    start=cp['flat'].detach().clone()
    oracle=accepted.Oracle(p,d,start.numpy().copy())
    oracle,binding_stats=solver_binding().bind(solver,oracle)
    case_settings=settings[name]
    atomic(out/'SETTINGS.json',jsonable(case_settings))
    atomic(out/'STARTED.json',{'case':name,'original_common_parent_sha256':PARENT_SHA,
         'immediate_resume_sha256':j['start_sha256'],'run_lock_sha256':sha(HERE/'RUN_LOCK.json'),
         'initial_radius':case_settings['initial_radius'],'remaining_function_evaluations':case_settings['max_evaluations'],
         'completed_prefix_steps':5,'new_steps_maximum':15})
    began=time.monotonic()
    def record(current,rec,trial):
        stage=5+rec['iteration']+1
        if stage>20:raise RuntimeError('Budget exceeded')
        record={**rec,'cumulative_stage':stage,'prefix_steps':5}
        atomic(out/f'ITERATION_{stage:02}.json',jsonable(record))
        vector=torch.as_tensor(current,dtype=torch.float64)
        with torch.no_grad():components=p.components(vector).numpy()
        save_tensor(out/f'STAGE_{stage:02}.pt',new_checkpoint(p,name,vector,stage,j['start_sha256'],
                    J=float(sum(components)),data_loss=float(components[0]),poly_loss=float(components[1])))
        if trial is not None:
            save_tensor(out/f'TRIAL_{stage:02}.pt',{'flat':torch.as_tensor(trial,dtype=torch.float64),
                                                  'finite':bool(np.isfinite(trial).all())})
        progress={'case':name,'cumulative_stage':stage,'J':float(sum(components)),
                  'data_loss':float(components[0]),'poly_loss':float(components[1]),
                  'accepted':rec.get('accepted'),'radius_end':rec.get('radius_end'),
                  'stagnation':rec.get('unchanged_state_terminations'),
                  'elapsed_seconds':time.monotonic()-began,
                  'new_oracle_counts':jsonable(oracle.counts),
                  'total_evaluations':prior['oracle_counts']['evaluations']+oracle.counts['evaluations'],
                  'peak_memory_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'checkpoint_sha256':sha(out/f'STAGE_{stage:02}.pt')}
        atomic(out/'PROGRESS.json',progress)
        print(json.dumps(progress),flush=True)
    final,diag=solver.solve(oracle,accepted.old.kkt,record,case_settings)
    vector=torch.as_tensor(final,dtype=torch.float64)
    if vector.numel()!=3867 or not torch.isfinite(vector).all():raise RuntimeError('Nonfinite terminal vector')
    with torch.no_grad():c=p.components(vector).numpy();constraints=d.constraints(p,vector).numpy()
    stage=5+len(diag['history'])
    save_tensor(out/'TERMINAL.pt',new_checkpoint(p,name,vector,stage,j['start_sha256'],
                J=float(sum(c)),data_loss=float(c[0]),poly_loss=float(c[1]),solver_status=diag['status']))
    atomic(out/'SOLVER.json',jsonable(diag))
    stages=[]
    for s in [5]+list(range(6,stage+1)):
        path=out/f'STAGE_{s:02}.pt';item=torch.load(path,weights_only=True,map_location='cpu')
        if np.isfinite(item['J']):stages.append({'stage':s,'J':item['J'],'data_loss':item.get('data_loss'),
             'poly_loss':item.get('poly_loss'),'path':path.name,'sha256':sha(path)})
    by_budget={}
    for budget in (5,10,20):
        if budget<=stage:
            exact=next(row for row in stages if row['stage']==budget)
            best=min((row for row in stages if row['stage']<=budget),key=lambda row:(row['J'],row['stage']))
            by_budget[str(budget)]={'exact_stage':exact,'best_finite_original_objective':best}
        else:by_budget[str(budget)]={'status':'NOT_REACHED_EARLY_TERMINATION'}
    result={'case':name,'status':diag['status'],'original_common_parent_sha256':PARENT_SHA,
        'immediate_resume_sha256':j['start_sha256'],'new_iterations_recorded':len(diag['history']),
        'maximum_cumulative_steps':20,'terminal_cumulative_stage':stage,
        'terminal_J':float(sum(c)),'terminal_data_loss':float(c[0]),'terminal_poly_loss':float(c[1]),
        'maximum_constraint':float(np.max(constraints)),'moment_constraint':float(constraints[0]),
        'all_finite_stages':stages,'by_budget':by_budget,'solver_seconds':diag['seconds'],
        'elapsed_seconds':time.monotonic()-began,'terminal_sha256':sha(out/'TERMINAL.pt'),
        'run_lock_sha256':sha(HERE/'RUN_LOCK.json'),'binding_stats':jsonable(binding_stats),
        'new_oracle_counts':jsonable(oracle.counts),
        'total_function_evaluations':prior['oracle_counts']['evaluations']+oracle.counts['evaluations'],
        'peak_memory_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'storage_bytes':sum(f.stat().st_size for f in out.rglob('*') if f.is_file())}
    atomic(out/'RESULT.json',result)
    print('CASE_COMPLETE',json.dumps({'case':name,'status':diag['status'],'terminal_stage':stage,
                                     'J':float(sum(c))}),flush=True)
    return result


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    plan=json.loads((HERE/'PLAN.json').read_text());solver=load_solver()
    settings={}
    for j in plan['jobs']:
        _,_,_,_,_,sol=load_case(j)
        settings[j['case']]=dict(solver.SETTINGS,max_iterations=15,
            max_evaluations=j['remaining_function_evaluations'],optimization_seconds=float('inf'),
            initial_radius=sol['final_radius'])
    freeze(plan,settings)
    for j in plan['jobs']:run_case(j,solver,settings)
    print('ALL_THREE_COMPLETE',flush=True)
if __name__=='__main__':main()

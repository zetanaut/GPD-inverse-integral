#!/usr/bin/env python3
"""Create a non-overwriting, source-bound plan; never invoke an optimizer.

The worker still implements the bounded continuation using the existing solver.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
CASES=('ensemble_mean','replica_000','replica_001')
EXPECTED_PARENT='b18c6c963c109e1249f6c3d30bdf7cdfa429f7da31631aea81b8e13cafa07dbb'
EXPECTED_SOURCE='8d7274db9dd7e4fcc3ab84c0230701242274e42868f9c3e66e3d61a1a6838696'
EXPECTED_DISPATCH='b35ce8921508b61258d918b869b85f48932476fab29d64c15c1892a45009a9af'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def build(project):
    root=Path(project).resolve();base=root/'completion_runs/ReH_coherent_GPD_pilot'
    lock=read(base/'RUN_LOCK.json');ilock=read(base/'prepared/INPUT_LOCK.json')
    source=root/'GPD_ReH_experimental_run/inputs/final_summary_by_DNN.csv'
    if digest(source)!=EXPECTED_SOURCE:raise ValueError('Selected CDNN/FQDNN source identity mismatch')
    if lock['parent_sha256']!=EXPECTED_PARENT or digest(root/'completion_runs/ReH_surface_mean_pilot/STAGE_20.pt')!=EXPECTED_PARENT:raise ValueError('Common original parent mismatch')
    if digest(base/'dispatch.py')!=EXPECTED_DISPATCH:raise ValueError('Original dispatcher changed')
    index=root/'GPD_ReH_experimental_run/banks/index.json'
    if digest(index)!=lock['unchanged_large_maps']['index_sha256']:raise ValueError('Bank index changed')
    jobs=[]
    for case in CASES:
        folder=base/'cases'/case;r=read(folder/'RESULT.json');s=read(folder/'SOLVER.json');winner=r['selected_finite_candidate']
        if winner['stage']!=5 or r['iterations_recorded']!=5 or len(s['history'])!=5:raise ValueError('Expected preserved complete five-step case')
        cp=folder/winner['path'];obs=base/'prepared'/f'{case}.npz'
        if digest(cp)!=winner['sha256'] or digest(obs)!=ilock['outputs'][obs.name]:raise ValueError('Case checkpoint/input hash mismatch')
        jobs.append({'case':case,'original_same_parent_sha256':EXPECTED_PARENT,
            'start_checkpoint':str(cp),'start_sha256':digest(cp),'observations':str(obs),'observations_sha256':digest(obs),
            'completed_prefix_steps':5,'new_steps_maximum':15,'total_steps_maximum':20,
            'save_cumulative_stages':[10,20],'initial_trust_radius_from_saved_solver':s['final_radius'],
            'remaining_function_evaluations':max(0,12000-r['oracle_counts']['evaluations']),
            'output_directory':str(root/'completion_runs/ReH_coherent_budget_check'/case)})
    return {'id':'ReH_coherent_budget_check_v1','status':'PLAN_ONLY_NO_FITS',
        'purpose':'Compare continuation-budget effects on mean reproduction and two coherent input responses; not a GPD uncertainty ensemble.',
        'jobs':jobs,'prior_run_lock_sha256':digest(base/'RUN_LOCK.json'),'source_sha256':EXPECTED_SOURCE,
        'source_path':str(source),'map_index':str(index),'map_index_sha256':digest(index),
        'future_more_replicas_authorized':False,'automatic_extension_after_twenty':False,
        'quality_guides':'Record continuously; no new thresholds or conditional retries.',
        'unchanged':['FiLM','all moment coefficients free','NLO maps','Re-only inputs','sigmas','13 t slices','lambda','solver step model'],
        'no_gpd_truth_used':True}
def write_once(path,data):
    path=Path(path)
    if path.exists():
        if read(path)!=data:raise ValueError('Existing nonidentical plan; refusing overwrite')
        return
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();data=build(a.project)
    out=a.out or a.project/'completion_runs/ReH_coherent_budget_check/PLAN.json';write_once(out,data)
    print(f'Prepared plan only: {out}; no fitting or NLO construction.')

#!/usr/bin/env python3
"""Freeze the entire comparison BEFORE generating or fitting its targets."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from closure.geometry import make_geometry,map_requests
from closure.io import atomic_json,sha256
from closure.truths import specifications

ROOT=Path(__file__).resolve().parent

def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    geom=make_geometry(ROOT/'inputs/observations.csv')
    meta=json.loads((ROOT/'inputs/INPUT_SUMMARY.json').read_text())
    t=np.sort(np.array(meta['moment_t_nodes'],float))
    t13=np.sort(np.r_[t,(t[:-1]+t[1:])/2])
    tasks=[]
    # Finish the ideal experiment first, then matched sparse complex/Re/noise.
    for branch in ('ideal_L0','ideal_L1','ideal_L2','exp_complex_exact','exp_Re_exact','exp_Re_noisy'):
        for truth in ('Z1','Z2'):
            for seed in (31,43):
                tasks.append({'id':f'{branch}_{truth}_s{seed}','branch':branch,'truth':truth,'seed':seed,
                    'geometry':branch if branch.startswith('ideal_') else 'experimental',
                    'components':[0] if branch in ('exp_Re_exact','exp_Re_noisy') else [0,1],
                    'noise':'one frozen Re perturbation per truth' if branch=='exp_Re_noisy' else 'none'})
    spec={'id':'three_stage_zero_closure_v1','date':'2026-10-07','status':'PREPARED_NOT_FITTED',
        'target':'entire Hcharge_plus=(2/9)*(Sigma+T_rescaled) at Q0^2=2; also report target functionals',
        'source_observations_sha256':sha256(ROOT/'inputs/observations.csv'),
        'physics':{'order':'NLO','nf':3,'scheme':'MSbar','Q02':2.,'alpha0':.35691594851514724,
                   'muF_over_Q':1.,'muR_over_Q':1.,'channels':['Sigma','T_rescaled','Hg=xg']},
        'truths':specifications(),
        'network':{'parameters':3867,'architecture':'original reference-scale FiLM','t_mean':meta['t_mean'],
                   't_std':meta['t_std_population'],'no_Q2_feature':True,'seeds':[31,43]},
        'polynomiality':{'spins':[2,4,6,8],'t_nodes':t13.tolist(),'xi_fit':np.linspace(.02,.98,41).tolist(),
                         'lambda':1.,'all_allowed_coefficients_free':True,'finite_moment_checks_not_global_proof':True},
        'observation_rule':{'ideal_sigma':.04,'ideal_sigma_role':'numerical component scale, no random error',
              'experimental_sigma':'unchanged supplied combined statistical/systematic marginal errors',
              'experimental_covariance':'explicit diagonal working simulation law; no claim of full experimental covariance',
              'synthetic_Im_weight':'same numerical row scale as Re for the sparse exact complex bridge; no experimental Im input',
              'noise_seeds':{'Z1':2026100701,'Z2':2026100702},
              'loss':'sum_i mass_i * sum_supplied_components ((C_pred-y)/sigma)^2 + Lpoly',
              'normalization':'mass sums to one; removing Im removes only that term, not a factor-two rescaling'},
        'solver':{'implementation':'existing finite-budget residual-norm constrained solver; no replacement architecture',
              'adam_steps':400,'adam_lr':.003,'residual_norm_iterations':60,'starts':'fresh fixed seed every task',
              'candidate_stages':[0,20,40,60],'ranking':['J','seed','stage'],
              'diagnostics_first':True,'do_not_select_by_truth_or_unfitted_Im':True,
              'rule_freeze':'write exact source hash binding before any fit; same code and rules all branches',
              'numerical_domain':'same existing source-derived numerical and moment filters mapped to this geometry; not use Im DATA in Re-only cases'},
        'core_tasks':tasks,'symbolic_follow_on':{'parents':'lowest-J finite seed of ideal_L2 and exp_Re_exact per truth',
              'count_parents':4,'dictionary':'same allowed physical-xi powers; all coefficients refitted',
              'proposal_tolerance':.001,'per_parent':'one candidate plus 20-step all-term matched continuation',
              'total_extra_branches_at_most':8,'nominal':'always all-term; compare errors only after decisions'},
        'sampling_convergence':'three nested density levels; record achieved changes. No finite level declared a perfect continuum.',
        'scope':'development zero closure then matched experimental-information closure; no new experimental fit or calibration',
        'execution':'source/array preflight and one measured run before projected heavy batch; no automatic kernel construction'}
    atomic_json(out/'STUDY.json',spec);atomic_json(out/'GEOMETRY.json',geom)
    atomic_json(out/'FIT_CONFIG.json',{k:spec[k] for k in ('network','polynomiality','solver','physics')})
    requests=map_requests(geom);atomic_json(out/'MAP_REQUESTS.json',{'requests':requests})
    index=json.loads((ROOT/'inputs/accepted_experimental_banks/index.json').read_text())
    keys={(float(e['xi']).hex(),float(e['Q2']).hex()) for e in index['entries']}
    missing=[r for r in requests if (r['xi_hex'],r['Q2_hex']) not in keys]
    summary={'tasks':len(tasks),'geometry_rows':{k:len(v) for k,v in geom['grids'].items()},
        'all_unique_map_pairs':len(requests),'existing_exact_map_pairs':len(requests)-len(missing),
        'missing_Q2_rows':len(missing),'new_xi_families':sorted({r['xi_hex'] for r in missing}-{float(e['xi']).hex() for e in index['entries']}),
        'status':'MISSING_SCALE_ROWS_BEFORE_IDEAL_FIT','note':'Every requested xi already has a captured local generator family; generators themselves remain on worker machine.'}
    atomic_json(out/'PREFLIGHT.json',summary)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'prepared');a=p.parse_args()
    print(json.dumps(prepare(a.out),indent=2))

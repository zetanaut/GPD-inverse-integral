from pathlib import Path
import json,os,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path(os.environ.get('GPD_REVIEW_ROOT',str(HERE.parent)))
T=ROOT/'completion_runs/NLO_constrained_solver';D=HERE/'results'
def load(p):return json.loads(Path(p).read_text())
records=[];histories=[]
for label in (31,43):
 replay=load(D/'replay.json');r=next(r for r in replay['records'] if r['label']==label)
 f=load(D/f'FULL_{label}_saved.json');t=load(D/f'FULL_{label}_trial.json')
 hist=[load(p) for p in sorted((T/f'start{label}').glob('iteration*/ITERATION.json'))]
 steps=[v for h in hist for v in h['trials'] if v['accepted']]
 rejections=[v for h in hist for v in h['trials'] if not v['accepted']]
 phase={'label':label,'iterations':len(hist),'accepted':len(steps),'rejected_trials':len(rejections),
  'every_accepted_has_strict_feasibility_and_decrease':all(max(v['constraints'])<=0 and v['actual_decrease']>0 for v in steps),
  'last100_median_step':float(np.median([v['step_norm'] for v in steps[-100:]])),
  'last100_median_trust_radius':float(np.median([h['radius_start'] for h in hist[-100:]])),
  'scope':'Saved history audit, not fresh evaluations of all 1200 intermediate fields'}
 histories.append(phase)
 records.append({'label':label,'J':r['J'],'KKT_L2':r['KKT']['stationarity_L2'],'checkpoint_replay_pass':r['pass'],
  'full_forward_pass':f['full_forward']['pass'],'independent_moment_RMS':f['moments']['grids']['independent160']['quadrature_RMS']['384'],
  'moment_pass':f['moments']['pass'],'full_forward_points':len(f['full_forward']['records']),
  'full_forward_saved_budget_discrepancy':f['saved_CFF_budget_max_diff'],
  'local_model_check':load(D/f'CURVATURE_{label}.json'),
  'probe':load(D/f'LOCAL_STEPS_{label}.json'),
  'larger_probe_full_forward_pass':t['full_forward']['pass'],'larger_probe_moment_pass':t['moments']['pass'],
  'larger_probe_moment_RMS':t['moments']['grids']['independent160']['quadrature_RMS']['384'],
  'bound_outer_step':load(D/f'BOUND_STEP_{label}.json')})
result={'decision':'NOT_ACCEPTED_NOISY_ESTIMATOR','accepted_count':0,'integrity':load(D/'INTEGRITY.json'),
 'fresh_checkpoint_count':2,'fresh_full_forward_checks':60,'original_mechanical_tests':38,'new_model_tests':18,
 'one_step_integration_controls':2,'records':records,'histories':histories,
 'scope':'Packet audit plus local derivative/step diagnosis. No full replacement neural continuation, kernels, new noise, profiles, calibration, or experimental data.'}
(D/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'accepted_count':0,'hashes':result['integrity']['worker_manifest_files'],'replayed_final_states':2,'full_forward_checks':60,'tests':56,'one_step_checks':2},indent=2))

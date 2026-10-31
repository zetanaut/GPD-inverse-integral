"""File-only, matched-baseline assessment after final selection and retention."""
import argparse
import math
import os
from pathlib import Path
from p2b_policy import BUNDLE,CASE,TRUE_TARGETS,read,sha,binding,assess_values

def collect(path):
    path=Path(path);r,s,lock=read(path/'RESULT.json'),read(path/'SCORES.json'),read(path/'RUN_LOCK.json')
    if lock!=binding(CASE) or r['run_lock_sha256']!=sha(path/'RUN_LOCK.json'):raise RuntimeError('Result identity mismatch')
    for name,key in [('SCORES.json','scores_sha256'),('FORWARD_SCORES.json','forward_scores_sha256'),(r['scored_checkpoint'],'scored_checkpoint_sha256')]:
        if sha(path/name)!=r[key]:raise RuntimeError('Result artifact mismatch: '+name)
    forward=read(path/'FORWARD_SCORES.json')
    if sha(path/'SELECTION.json')!=s['selection_sha256'] or forward['selection_sha256']!=s['selection_sha256']:raise RuntimeError('Selection mismatch')
    supervisor=read(path/('SUPERVISOR_'+r['execution_job_id']+'.json'))
    if (supervisor['child_exit']!=0 or not supervisor['terminal_retention_verified'] or any(supervisor.get(k) for k in
        ['resource_stop','monitor_error','cleanup_error','terminal_retention_error','provisional_copy_errors'])):raise RuntimeError('Execution/retention not clean')
    receipt=read(path/'TERMINAL_RETENTION.json')
    if receipt.get('verified') is not True:raise RuntimeError('Unverified retention')
    for row in receipt['files']:
        rel=Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or (path/rel).stat().st_size!=row['bytes'] or sha(path/rel)!=row['sha256']:raise RuntimeError('Retained artifact changed')
    baseline=read(BUNDLE/'evidence/baseline/BASELINE.json')
    for name,digest in baseline['files'].items():
        if sha(BUNDLE/'evidence/baseline'/name)!=digest:raise RuntimeError('Baseline bytes changed')
    base_result=read(BUNDLE/'evidence/baseline/RESULT.json');base_scores=read(BUNDLE/'evidence/baseline/SCORES.json')
    if (base_result['status']!='FINITE_BUDGET_FEASIBLE' or not base_result['selected']['pass']
            or baseline['observed_Hplus_relative_L2']!=base_scores['observed_slices']['relative_L2']['all']
            or baseline['target_384']!=base_scores['target_384']):raise RuntimeError('Inconsistent baseline record')
    selected=r.get('selected');eligible=r['status']=='FINITE_BUDGET_FEASIBLE' and selected is not None and selected['pass']
    if eligible and (selected['constraint_exceedances']!=0 or not selected['constraints_finite']):raise RuntimeError('Inconsistent eligibility')
    rms=forward['observed_samples']['whitened_RMS']
    if not math.isclose(rms,r['PDF_whitened_RMS'],rel_tol=1e-12,abs_tol=1e-14):raise RuntimeError('PDF score mismatch')
    answer=assess_values(eligible,s['observed_slices']['relative_L2']['all'],s['target_384'],rms,baseline)
    answer.update(case=CASE,selected_stage=selected['stage'] if selected else None,result_sha256=sha(path/'RESULT.json'),
        baseline_sha256=sha(BUNDLE/'evidence/baseline/BASELINE.json'),truth_targets=TRUE_TARGETS,retained_files_verified=len(receipt['files']))
    return answer

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();os.environ['GPD_P2B_RUN_SPEC']=str(a.spec.resolve());result=collect(a.result)
    with a.out.open('x') as f:__import__('json').dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(result['decision'])

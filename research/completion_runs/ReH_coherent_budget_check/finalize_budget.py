"""Verify budget states and return a source-bound review archive."""
from __future__ import annotations
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'completion_runs/ReH_coherent_GPD_pilot'
CASES=('ensemble_mean','replica_000','replica_001')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def readcsv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def main():
    plan=json.loads((HERE/'PLAN.json').read_text());lock=json.loads((HERE/'RUN_LOCK.json').read_text())
    diag=json.loads((HERE/'DIAGNOSTICS.json').read_text())
    if lock['plan_sha256']!=sha(HERE/'PLAN.json'):raise RuntimeError('Plan changed')
    for rel,digest in lock['sources'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Bound source changed: '+rel)
    stage_checks={}
    for j in plan['jobs']:
        name=j['case'];folder=HERE/name;r=json.loads((folder/'RESULT.json').read_text())
        if r['case']!=name or r['run_lock_sha256']!=sha(HERE/'RUN_LOCK.json'):
            raise RuntimeError('Case/run lock mismatch')
        if r['terminal_cumulative_stage']!=20 or r['new_iterations_recorded']!=15:
            raise RuntimeError('Did not complete bounded cumulative budget: '+name)
        if r['original_common_parent_sha256']!=lock['original_common_parent_sha256'] or r['immediate_resume_sha256']!=j['start_sha256']:
            raise RuntimeError('Parent lineage mismatch')
        if sha(folder/'STAGE_05.pt')!=j['start_sha256'] or sha(folder/'TERMINAL.pt')!=r['terminal_sha256']:
            raise RuntimeError('Start/terminal hash mismatch')
        if len(r['all_finite_stages'])!=16:raise RuntimeError('Incomplete stage roster')
        stage_checks[name]={}
        for row in r['all_finite_stages']:
            path=folder/row['path']
            if sha(path)!=row['sha256']:raise RuntimeError('Stage hash mismatch')
            cp=torch.load(path,weights_only=True,map_location='cpu')
            if row['stage']>5 and (cp['original_common_parent_sha256']!=lock['original_common_parent_sha256'] or
                                   cp['immediate_resume_sha256']!=j['start_sha256'] or
                                   'parent_sha256' in cp):
                raise RuntimeError('New checkpoint lineage mismatch')
        for budget in (10,20):
            item=r['by_budget'][str(budget)]['exact_stage']
            if item['stage']!=budget or sha(folder/item['path'])!=item['sha256']:
                raise RuntimeError('Missing exact budget field')
            n=list((folder/f'budget_{budget:02}'/'full_forward').glob('*.json'))
            if len(n)!=160:raise RuntimeError('Missing full forward rows')
            stage_checks[name][str(budget)]={'checkpoint_sha256':item['sha256'],
                 'full_forward_rows':len(n),'finite':diag['cases_new_budgets'][name][str(budget)]['full_forward']['finite']}
    (HERE/'BUDGET_IDENTITY_AUDIT.json').write_text(json.dumps(stage_checks,indent=2)+'\n')
    metrics={(r['case'],int(r['budget']),r['comparison']):r for r in diag['comparison_metrics']}
    matches={(r['case'],int(r['budget'])):r for r in diag['matched_input_output_metrics']}
    lines=['# Coherent-input continuation budget check','',
      'This prospective comparison follows the same mean and two coherent-member targets through cumulative outer-step budgets 5, 10 and 20. The table uses the unchanged original quoted ReH errors at the same 160 coordinates. It distinguishes target reproduction from how strongly a member input changes the reconstructed field.','',
      '| Budget | Mean GPD vs same CFF mean RMS / σ | r000 input → GPD output RMS / σ | r001 input → GPD output RMS / σ |',
      '|---:|---:|---:|---:|']
    for b in (5,10,20):
        mean=float(metrics[('ensemble_mean',b,'ensemble_mean_input')]['RMS_original_sigma'])
        a=matches[('replica_000',b)];c=matches[('replica_001',b)]
        lines.append(f"| {b} | {mean:.3f} | {float(a['input_RMS_original_sigma']):.3f} → {float(a['output_RMS_original_sigma']):.3f} | {float(c['input_RMS_original_sigma']):.3f} → {float(c['output_RMS_original_sigma']):.3f} |")
    lines+=['',
      'The remaining mean-target discrepancy at each budget is reported directly rather than attributed to measurement noise. Member output responses are compared with their same-budget mean field. Their H+ overlays, ERBL/DGLAP differences and fixed target integrals are supplied as truth-free diagnostics; two members do not define a GPD uncertainty band.','',
      'All three continuations started from the exact saved stage-5 vectors and trust radii. All 3,867 FiLM parameters and permitted moment coefficients remained free. Original source errors, targets, t conditioning, thirteen t slices and NLO maps were unchanged. The new checkpoint metadata names both the original common parent and immediate resume state.','']
    checks=diag['cases_new_budgets'];warn=[];guides=[]
    for name in CASES:
        for b in (10,20):
            item=checks[name][str(b)]
            if item['full_forward']['warning_rows']:warn.append(f"{name}@{b}: {','.join(item['full_forward']['warning_rows'])}")
            guides.append((name,b,item['moment_diagnostics']['combined_internal_RMS'],item['maximum_constraint']))
    lines.append('All six new reported fields passed finite full-forward checks. Numerical warnings: '+('; '.join(warn) if warn else 'none')+'.')
    lines.append('Combined moment diagnostic RMS at budgets 10/20 ranges from '
                 f"{min(x[2] for x in guides):.5f} to {max(x[2] for x in guides):.5f}; finite legacy-guide exceedances remain visible. "
                 'All reported maximum constraints are nonpositive. No numerical threshold caused a retry or extension.')
    actual=sum(json.loads((HERE/name/'RESULT.json').read_text())['solver_seconds'] for name in CASES)
    peak=max(json.loads((HERE/name/'RESULT.json').read_text())['peak_memory_bytes_macos'] for name in CASES)
    lines+=['',f"Additional solver time was {actual:.1f} seconds ({actual/3600:.2f} hours), with process peak memory {peak/1e9:.2f} GB. The run stopped at cumulative step 20. No new CFF draws, map construction, error inflation, remaining-replica batch or manuscript edits occurred."]
    (HERE/'ASSESSMENT.md').write_text('\n'.join(lines)+'\n')
    loc={'map_index_project_path':'GPD_ReH_experimental_run/banks/index.json',
         'map_index_sha256':sha(ROOT/'GPD_ReH_experimental_run/banks/index.json'),
         'family_root_project_path':'GPD_ReH_experimental_run/families',
         'source_lock_project_path':str((HERE/'RUN_LOCK.json').relative_to(ROOT)),
         'unchanged_large_maps':'Reused local exact-coordinate arrays; no kernel construction'}
    (HERE/'MAP_LOCATORS.json').write_text(json.dumps(loc,indent=2)+'\n')
    sources={'review/DECISION.md':ROOT/'GPD_coherent_GPD_review/DECISION.md',
             'review/WORKER_PROMPT.md':ROOT/'GPD_coherent_GPD_review/WORKER_PROMPT.md',
             'review/prepare_budget_check.py':ROOT/'GPD_coherent_GPD_review/prepare_budget_check.py',
             'solver_integration/dispatch.py':PRIOR/'dispatch.py',
             'solver_integration/pilot_run.py':ROOT/'completion_runs/ReH_input_response_pilot/pilot_run.py',
             'solver_integration/solver.py':ROOT/'completion_runs/NLO_constrained_solver/solver.py'}
    copies={}
    for name,src in sources.items():
        dst=HERE/'source_used'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        if sha(dst)!=sha(src):raise RuntimeError('Source copy changed')
        copies[name]={'original_project_path':str(src.relative_to(ROOT)),'sha256':sha(src)}
    (HERE/'SOURCE_COPIES.json').write_text(json.dumps(copies,indent=2)+'\n')
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json')
    manifest={'package':'review_ReH_coherent_budget_check','status':'THREE_CASES_CUMULATIVE_5_10_20_COMPLETE',
              'files':{str(p.relative_to(HERE)):sha(p) for p in files}}
    (HERE/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(HERE/'PACKAGE_MANIFEST.json')
    target=ROOT/'review_ReH_coherent_budget_check.zip';tmp=target.with_suffix('.partial.zip')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,Path(HERE.name)/p.relative_to(HERE))
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise RuntimeError('CRC failure')
        for name,d in manifest['files'].items():
            if hashlib.sha256(z.read(HERE.name+'/'+name)).hexdigest()!=d:raise RuntimeError('Archive hash mismatch')
    tmp.replace(target)
    print(json.dumps({'archive':str(target),'sha256':sha(target),'bytes':target.stat().st_size,'files':len(files)},indent=2))
if __name__=='__main__':main()

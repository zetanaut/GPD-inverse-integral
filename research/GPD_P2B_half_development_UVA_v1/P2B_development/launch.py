"""Host/allocated admission for one released development fit and clean continuations."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from p2b_policy import ROOT,CAPSULE,PACKET,ACCELERATION,PACKAGE,BUNDLE,CASE,BUDGETS,binding,sha,read

def verify_manifest(root,expected):
    root=Path(root).resolve()
    if sha(root/'MANIFEST.json')!=expected:raise RuntimeError('Manifest pin mismatch: '+str(root))
    for row in read(root/'MANIFEST.json')['files']:
        rel=Path(row['path']);p=root/rel
        if (rel.is_absolute() or '..' in rel.parts or p.is_symlink() or not p.is_file()
                or root not in p.resolve().parents or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']):
            raise RuntimeError('Manifest member mismatch: '+str(p))

def validate(spec_path,case=None,resume=False,approval=None,allocated=False):
    spec_path=Path(spec_path).resolve();spec=read(spec_path)
    if spec['case']!=CASE or case not in (None,CASE) or spec['phase']!='half_noise_development':raise RuntimeError('Wrong case/phase')
    if bool(spec.get('resume_parent'))!=resume:raise RuntimeError('Explicit continuation mode mismatch')
    if (spec['backend'],spec['threads'],spec['solver_blas_threads'])!=('cpu_guarded',1,8):raise RuntimeError('Unqualified backend/thread policy')
    if spec['scientific_budgets']!=BUDGETS:raise RuntimeError('Scientific budget drift')
    r=spec['resources']
    if (r['account'],r['nodes'],r['ntasks'],r['gpus'],r['cpus_per_task'],r['requeue'])!=('spinquest_standard',1,1,0,8,False):
        raise RuntimeError('Unreviewed resource topology')
    if (not 0<r['stop_after_seconds']<r['wall_seconds']<=43200 or r['memory_GiB']!=64
            or r['tree_rss_cap_GiB']+r['reserve_GiB']>r['memory_GiB']):raise RuntimeError('Invalid watchdog margins')
    for label,root in [('capsule',CAPSULE),('packet',PACKET),('acceleration',ACCELERATION),('driver',PACKAGE),('transfer',BUNDLE)]:
        verify_manifest(root,spec[label+'_manifest_sha256'])
    for path,digest in spec['files'].items():
        if not Path(path).is_absolute() or sha(path)!=digest:raise RuntimeError('Pinned input/evidence changed: '+path)
    if spec['performance_review'] not in spec['files']:raise RuntimeError('Unpinned performance review')
    review=read(spec['performance_review'])
    if ('cpu_guarded' not in review['qualified_backends'] or review['fastest_measured_backend']!='cpu_guarded'
            or spec.get('development_released') is not True or spec.get('independent_assessment_released') is not False):
        raise RuntimeError('Unreviewed backend or scientific scope')
    gate=read(BUNDLE/'evidence/exact_center/ASSESSMENT.json')
    if gate['decision']!='EXACT_CENTER_PASS_REVIEW_REQUIRED' or not all(gate['gates'].values()):raise RuntimeError('Exact prerequisite failed')
    if read(BUNDLE/'evidence/execution/AUDIT.json')['status']!='PASS_EXECUTION_EVIDENCE_GAP_RESOLVED':raise RuntimeError('Execution review missing')
    versions=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ('numpy','scipy','torch')})
    if versions!=spec['versions'] or Path(sys.executable).resolve()!=Path(spec['python']).resolve():raise RuntimeError('Pinned environment mismatch')
    for key in ['output_root','persistent_root']:
        p=Path(spec[key])
        if not p.is_absolute() or not p.name.startswith('GPD_P2B_half_development_') or p.exists():raise RuntimeError('New distinct attempt root required: '+str(p))
    if Path(spec['output_root'])==Path(spec['persistent_root']):raise RuntimeError('Scratch/persistent roots must differ')
    registry=ROOT/'control/GPD_P2B_half_development_v1'
    if Path(spec['case_registry'])!=registry:raise RuntimeError('Case registry changed')
    claim=read(registry/'CLAIM.json')
    if (claim['case']!=CASE or claim['transfer_manifest_sha256']!=spec['transfer_manifest_sha256']
            or claim['latest_spec']!=str(spec_path) or claim['latest_spec_sha256']!=sha(spec_path)):
        raise RuntimeError('Case already claimed by another/stale attempt')
    os.environ['GPD_P2B_RUN_SPEC']=str(spec_path)
    if resume:
        from resume_validation import verify_parent
        if verify_parent(spec['resume_parent']['root'],binding(CASE))!=spec['resume_parent']:raise RuntimeError('Parent identity changed')
    if allocated:
        if sys.platform!='linux' or not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT')!='spinquest_standard':raise RuntimeError('Allocated Linux execution required')
        for key,value in [('SLURM_JOB_NUM_NODES',1),('SLURM_NTASKS',1),('SLURM_CPUS_PER_TASK',8),('SLURM_MEM_PER_NODE',65536)]:
            if int(os.environ.get(key,'0'))!=value:raise RuntimeError('Allocation mismatch: '+key)
        if os.environ.get('SLURM_JOB_PARTITION')!=r['partition']:raise RuntimeError('Partition mismatch')
        permit=read(approval) if approval else {}
        if permit!={'authorized':True,'case':CASE,'spec_sha256':sha(spec_path)}:raise RuntimeError('Spec-bound permit missing/mismatched')
    return spec

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--case',default=CASE)
    parser.add_argument('--approval',type=Path);parser.add_argument('--resume',action='store_true');parser.add_argument('--validate-only',action='store_true')
    a=parser.parse_args();spec=validate(a.spec,a.case,a.resume,a.approval,not a.validate_only)
    if a.validate_only:print('PASS static admission',sha(a.spec));return 0
    from development_information import validate_inputs
    validate_inputs()
    output=Path(spec['output_root']);output.mkdir(parents=False,exist_ok=False)
    q=output/'preflight';q.mkdir();groups=[]
    for script,extra in [('test_policy.py',[]),('test_forward_adapter.py',[]),('test_continuation.py',['--fixtures']),('test_operations.py',[]),('test_launch.py',[])]:
        with (q/(script+'.stdout')).open('x') as stdout,(q/(script+'.stderr')).open('x') as stderr:
            try:code=subprocess.run([sys.executable,'-B',str(PACKAGE/script)]+extra,stdout=stdout,stderr=stderr,timeout=180).returncode
            except subprocess.TimeoutExpired:code=124
        groups.append(dict(script=script,source_sha256=sha(PACKAGE/script),exit_code=code,stdout_sha256=sha(q/(script+'.stdout')),stderr_sha256=sha(q/(script+'.stderr'))))
        if code:
            dest=output/CASE;dest.mkdir();shutil.copytree(q,dest/'preflight')
            (dest/'QUALIFICATION_FAILED.json').write_text(__import__('json').dumps(dict(groups=groups,scientific_fit_started=False),indent=2)+'\n')
            return 76
    (q/'QUALIFICATION.json').write_text(__import__('json').dumps(dict(status='PASS',groups=groups,scope='Software fixtures; no new scientific fit in qualification'),indent=2)+'\n')
    dest=output/CASE
    if a.resume:
        from resume_validation import stage_parent
        stage_parent(spec['resume_parent'],dest,binding(CASE));shutil.copytree(q,dest/'preflight')
    entry=dict(spec_sha256=sha(a.spec),case=CASE,job_id=os.environ['SLURM_JOB_ID'],resume=a.resume,
               starting_progress=spec['resume_parent']['progress'] if a.resume else dict(Adam_updates=0,outer_iterations=0,solver_evaluations=0))
    # Driver owns first case-directory creation on initial entry.
    (output/'ATTEMPT_ENTRY.json').write_text(__import__('json').dumps(entry,indent=2)+'\n')
    from run_fit import run
    return run(CASE,dest,a.resume)

if __name__=='__main__':raise SystemExit(main())

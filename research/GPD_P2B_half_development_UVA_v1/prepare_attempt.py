"""Create one spec-bound cluster attempt; never submit or redraw scientific noise."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import sys

BUNDLE=Path(__file__).resolve().parent
sys.path.insert(0,str(BUNDLE/'P2B_development'))
from p2b_policy import ROOT,CAPSULE,PACKET,ACCELERATION,PACKAGE,CASE,BUDGETS,read,sha,binding
from launch import verify_manifest,validate

def save(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--partition',default='standard-afton');p.add_argument('--resume-from',type=Path)
    a=p.parse_args()
    if not re.fullmatch(r'GPD_P2B_half_development_[A-Za-z0-9_]+',a.tag):raise RuntimeError('Distinct development attempt tag required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.partition):raise RuntimeError('Invalid partition')
    deps=read(BUNDLE/'DEPENDENCIES.json')
    verify_manifest(BUNDLE,sha(BUNDLE/'MANIFEST.json'))
    for label,root in [('capsule',CAPSULE),('packet',PACKET),('acceleration',ACCELERATION)]:verify_manifest(root,deps[label+'_manifest_sha256'])
    for path,digest in deps['files'].items():
        if sha(path)!=digest:raise RuntimeError('Pinned dependency changed: '+path)
    from development_information import validate_inputs
    validate_inputs()
    registry=ROOT/'control/GPD_P2B_half_development_v1';registry.mkdir(parents=True,exist_ok=True)
    with (registry/'PREPARE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Do not supersede a currently executing attempt.
        with (registry/'RUNNING.lock').open('a') as active:
            fcntl.flock(active,fcntl.LOCK_EX|fcntl.LOCK_NB)
            claim_path=registry/'CLAIM.json';old=read(claim_path) if claim_path.exists() else None
            if old and not a.resume_from:raise RuntimeError('This scientific case already has an initial attempt; inspect CLAIM.json')
            if a.resume_from and not old:raise RuntimeError('No original case claim for continuation')
            if old and old['transfer_manifest_sha256']!=sha(BUNDLE/'MANIFEST.json'):raise RuntimeError('Case package changed')
            control=ROOT/'control'/a.tag
            if control.exists():raise RuntimeError('Attempt tag already exists; inspect it instead of repeating preparation/submission')
            output=Path('/sfs/weka/scratch/dmk9m/GPD_outputs')/a.tag
            persistent=ROOT/'results'/a.tag
            if output.exists() or persistent.exists():raise RuntimeError('Attempt output exists')
            spec=dict(schema=1,case=CASE,phase='half_noise_development',backend='cpu_guarded',threads=1,solver_blas_threads=8,
                scientific_budgets=BUDGETS,python=deps['python'],versions=deps['versions'],files=deps['files'],
                performance_review=deps['performance_review'],development_released=True,independent_assessment_released=False,
                scientific_noise='One frozen draw already included; no RNG at execution',output_root=str(output),persistent_root=str(persistent),
                case_registry=str(registry),resume_parent=None,resources=dict(account='spinquest_standard',partition=a.partition,
                nodes=1,ntasks=1,cpus_per_task=8,memory_GiB=64,gpus=0,wall_seconds=43200,stop_after_seconds=36000,
                tree_rss_cap_GiB=56,reserve_GiB=4,requeue=False),
                authorization='User 2026-10-29 authorized one half-noise development case/package; clean same-fit progress continuations keep original budgets; no new case or failed-attempt retry')
            for label,root in [('capsule',CAPSULE),('packet',PACKET),('acceleration',ACCELERATION),('driver',PACKAGE),('transfer',BUNDLE)]:
                spec[label+'_manifest_sha256']=sha(root/'MANIFEST.json')
            # A temporary in-memory identity config is made concrete before parent validation.
            control.mkdir()
            specpath=control/'RUN_SPEC.json'
            try:
                if a.resume_from:
                    previous=read(old['latest_spec'])
                    expected=Path(previous['persistent_root'])/CASE
                    if a.resume_from.resolve()!=expected.resolve():raise RuntimeError('Continue only the latest claimed terminal attempt')
                    from resume_validation import verify_parent
                    parent=verify_parent(a.resume_from)
                    spec['resume_parent']=parent
                save(specpath,spec)
                os.environ['GPD_P2B_RUN_SPEC']=str(specpath)
                if a.resume_from and read(a.resume_from/'RUN_LOCK.json')!=binding(CASE):raise RuntimeError('Scientific continuation identity mismatch')
                claim=dict(case=CASE,transfer_manifest_sha256=spec['transfer_manifest_sha256'],latest_spec=str(specpath),
                    latest_spec_sha256=sha(specpath),previous=old,submitted_job_id=None)
                temporary=registry/'CLAIM.json.partial';temporary.write_text(json.dumps(claim,indent=2)+'\n');os.replace(temporary,claim_path)
                permit=control/'RUN_PERMIT.json';save(permit,dict(authorized=True,case=CASE,spec_sha256=sha(specpath)))
                validate(specpath,CASE,bool(a.resume_from),permit,allocated=False)
                command=[spec['python'],'-B',str(PACKAGE/'supervise.py'),'--spec',str(specpath),'--case',CASE,'--approval',str(permit)]
                if a.resume_from:command.append('--resume')
                lines=['#!/usr/bin/env bash',f'#SBATCH --job-name=P2B_half_development','#SBATCH --account=spinquest_standard',
                    '#SBATCH --partition='+a.partition,'#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=8',
                    '#SBATCH --mem=64G','#SBATCH --time=12:00:00','#SBATCH --no-requeue','#SBATCH --signal=B:USR1@7200',
                    '#SBATCH --export=NIL',f'#SBATCH --output={ROOT}/logs/{a.tag}-%j.out',f'#SBATCH --error={ROOT}/logs/{a.tag}-%j.err',
                    'set -Eeuo pipefail','umask 077','export PATH=/opt/slurm/current/bin:/usr/bin:/bin','unset PYTHONPATH PYTHONHOME',
                    'export PYTHONPATH='+shlex.quote(str(ROOT/'envs/GPD_P2B_common_vendor')),
                    'export GPD_P2B_PROJECT_ROOT='+shlex.quote(str(ROOT)),
                    'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1',
                    'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1',
                    'export MPLBACKEND=Agg', 'exec '+' '.join(shlex.quote(v) for v in command)]
                script=control/'RUN.sbatch';script.write_text('\n'.join(lines)+'\n')
                save(control/'PREPARATION.json',dict(status='READY_FOR_LIVE_SCHEDULER_CHECK',spec_sha256=sha(specpath),
                     submission_script_sha256=sha(script),submitted=False,noise_drawn=False,fit_started=False))
                print(json.dumps(dict(control=str(control),spec=str(specpath),script=str(script),submitted=False)))
            except Exception as error:
                save(control/'PREPARATION_FAILED.json',dict(error=str(error),submitted=False,
                    action='Inspect this attempt; do not create another fresh scientific case or silently change pins.'))
                raise

if __name__=='__main__':main()

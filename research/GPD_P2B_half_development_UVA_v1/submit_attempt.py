"""Submit an already prepared, live-reviewed attempt once; ambiguous RPCs stop."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
BUNDLE=Path(__file__).resolve().parent
sys.path.insert(0,str(BUNDLE/'P2B_development'))
from p2b_policy import CASE,sha,read
from launch import validate

def write_new(path,data):
    with path.open('x') as f:
        json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--readiness',type=Path,required=True)
    a=p.parse_args();control=a.control.resolve();specpath=control/'RUN_SPEC.json';spec=read(specpath)
    registry=Path(spec['case_registry'])
    with (registry/'PREPARE.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (registry/'RUNNING.lock').open('a') as running:
            fcntl.flock(running,fcntl.LOCK_EX|fcntl.LOCK_NB)
            validate(specpath,CASE,bool(spec['resume_parent']),control/'RUN_PERMIT.json',allocated=False)
            prep=read(control/'PREPARATION.json');script=control/'RUN.sbatch'
            if prep['submission_script_sha256']!=sha(script):raise RuntimeError('Prepared batch script changed')
            if not a.readiness.is_file() or a.readiness.stat().st_size==0:raise RuntimeError('Save live scheduler/account/environment checks before submitting')
            if (control/'SUBMISSION_STARTED.json').exists():raise RuntimeError('Submission already attempted; inspect saved response and scheduler; never repeat an ambiguous RPC')
            shutil.copy2(a.readiness,control/'LIVE_READINESS.txt')
            stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
            write_new(control/'SUBMISSION_STARTED.json',dict(utc=stamp,spec_sha256=sha(specpath),script_sha256=sha(script),readiness_sha256=sha(control/'LIVE_READINESS.txt')))
            # The batch job may start before sbatch returns; let its supervisor own this lock.
            fcntl.flock(running,fcntl.LOCK_UN)
            try:
                result=subprocess.run(['sbatch','--parsable',str(script)],capture_output=True,text=True,timeout=60)
                receipt=dict(utc=stamp,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            except Exception as error:
                write_new(control/'SUBMISSION.json',dict(utc=stamp,error=repr(error),action='Ambiguous submission; query scheduler and preserve evidence. Do not repeat sbatch.'))
                raise
            job=result.stdout.strip().split(';')[0]
            receipt['job_id']=job if result.returncode==0 and job.isdigit() else None
            write_new(control/'SUBMISSION.json',receipt)
            if not receipt['job_id']:raise RuntimeError('No confirmed job ID; inspect scheduler before any further action')
            claim=read(registry/'CLAIM.json');claim['submitted_job_id']=job
            temporary=registry/'CLAIM.json.partial';temporary.write_text(json.dumps(claim,indent=2)+'\n');os.replace(temporary,registry/'CLAIM.json')
            print(json.dumps(receipt))

if __name__=='__main__':main()

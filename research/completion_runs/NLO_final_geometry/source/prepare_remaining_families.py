"""Disjoint missing-family construction after the first complete family passes.

Each child keeps the same one-thread native settings and per-stage caps. A child
failure stops all other children; completed rows and all failures are preserved.
No fits, final truths, statistical streams, or accepted banks are generated here.
"""
from pathlib import Path
import os,sys,json,time,subprocess,signal
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]
assert json.loads((PREP/'geometry/xi0.215/PIPELINE.json').read_text())['status']=='FIRST_FAMILY_VALIDATED'
dest=PREP/'geometry/REMAINING_EXECUTION.json'
assert not dest.exists(),'Preserve prior execution; no automatic repeat'
families=[.155,.305,.365,.505];concurrency=min(2,max(1,os.cpu_count() or 1))
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
state={'concurrency':concurrency,'per_process_threads':1,'first_family_pass_required':True,'families':families,'records':[],'status':'RUNNING'}
running=[];pending=list(families);start=time.monotonic()
def save():
 state['seconds']=time.monotonic()-start;tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(dest)
try:
 while pending or running:
  while pending and len(running)<concurrency:
   xi=pending.pop(0);log=PREP/'results'/f'family_{xi}_pipeline.log';handle=log.open('w')
   cmd=['/usr/bin/python3',str(PREP/'source/prepare_family.py'),'--xi',str(xi)]
   process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
   entry={'xi':xi,'pid':process.pid,'command':cmd,'log':str(log.relative_to(PREP)),'status':'RUNNING','started_seconds':time.monotonic()-start};state['records'].append(entry);running.append((process,handle,entry));save()
  for process,handle,entry in list(running):
   rc=process.poll()
   if rc is None:continue
   handle.close();entry.update(returncode=rc,status='PASS' if rc==0 else 'FAILED',finished_seconds=time.monotonic()-start);running.remove((process,handle,entry));save()
   if rc:raise RuntimeError('Missing-family construction/validation failed: '+str(entry['xi']))
  if running:time.sleep(1)
 state['status']='ALL_REMAINING_FAMILIES_VALIDATED';save()
except BaseException as e:
 state['status']='BLOCKED';state['error']=repr(e)
 for process,handle,entry in running:
  try:os.killpg(process.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
  handle.close();entry.update(status='INTERRUPTED_AFTER_OTHER_FAILURE',returncode=process.returncode,finished_seconds=time.monotonic()-start)
 state['not_started']=pending;save();raise

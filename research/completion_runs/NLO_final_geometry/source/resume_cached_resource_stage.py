"""Bounded completion of cached xi=.505 assembly after a wall-clock chunk cap.

Each affected assembly stage gets at most one additional invocation, with its
original 1200-second cap and exact original command. No saved row is recomputed.
No numerical failure, validation failure, signal, or fit abstention is retried.
"""
from pathlib import Path
import os,json,time,hashlib,subprocess,shutil,resource,zipfile
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1];NLO=ROOT/'GPD_NLO_completion_package'
OUT=PREP/'geometry/xi0.505';STATE=OUT/'PIPELINE.json'
DEST=PREP/'geometry/CACHED_RESOURCE_CONTINUATION.json'
assert not DEST.exists(),'Preserve earlier continuation; no automatic repeat'
record={'status':'RUNNING','xi':.505,'scientific_changes':False,'original_stage_cap_seconds':1200,
        'additional_invocations_per_capped_stage':1,'fits_or_noise':0,'attempts':[]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def save():write(DEST,record)
def locate(stage):
 matches=[];name=stage['name']
 for path in (NLO/'work/operators').glob('*/KEY.json'):
  k=read(path)
  if k.get('basis',{}).get('xi')!=.505:continue
  if k.get('assembly_adapter_sha256')!=sha(PREP/'source/assemble_missing.py'):continue
  if k.get('cap_seconds')!=1200:continue
  if name in ('base','inner72'):
   ok='parent' not in k and k.get('inner_n')==(48 if name=='base' else 72)
  else:
   ok=k.get('parent')==str(Path(stage['command'][2]).relative_to(NLO)) and k.get('special_output_n')==48 and 'qg_endpoint_subtraction' not in k
  if ok:
   assert hashlib.sha256(json.dumps(k,sort_keys=True).encode()).hexdigest()==path.parent.name
   matches.append(path.parent)
 assert len(matches)==1,('Ambiguous or missing exact cache',name,matches)
 return matches[0]
try:
 while True:
  state=read(STATE)
  if state['status']=='FAMILY_VALIDATED':break
  failed=[s for s in state['steps'] if s['returncode']!=0]
  assert len(failed)==1,'Only a recorded assembly chunk cap can be resumed'
  stage=failed[0];name=stage['name'];assert name in ('base','inner72','output48')
  assert name not in [a['stage'] for a in record['attempts']],'Second capped invocation is a resource stop'
  log=(OUT/stage['log']).read_text()
  marker='TimeoutError: bounded output refinement; rows retained' if name=='output48' else 'TimeoutError: finite construction cap; saved rows retained'
  assert stage['returncode']==1 and log.rstrip().endswith(marker),'Not a resumable cache chunk cap'
  assert stage['command'][stage['command'].index('--cap')+1]=='1200'
  bank=locate(stage);history=OUT/'resource_attempts'/name;history.mkdir(parents=True,exist_ok=False)
  shutil.copy2(STATE,history/'PIPELINE_before.json');shutil.copy2(OUT/stage['log'],history/'original.log')
  for filename in ('KEY.json','PROGRESS.json'):shutil.copy2(bank/filename,history/filename)
  rows={}
  for p in sorted(bank.glob('row_*.npz')):
   with zipfile.ZipFile(p) as z:assert z.testzip() is None
   with np.load(p,allow_pickle=False) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
   rows[p.name]=sha(p)
  write(history/'REUSED_ROW_HASHES.json',rows)
  entry={'stage':name,'bank':str(bank.relative_to(NLO)),'rows_reused':len(rows),'command':stage['command'],
         'original_failure':stage,'source_and_KEY_sha256':sha(bank/'KEY.json'),'status':'RUNNING'}
  record['attempts'].append(entry);save();start=time.monotonic();newlog=OUT/(name+'_resume_1.log')
  with newlog.open('w') as f:
   ret=subprocess.run(stage['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1260).returncode
  entry.update(returncode=ret,seconds=time.monotonic()-start,peak_child_RSS_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
  assert sha(bank/'KEY.json')==entry['source_and_KEY_sha256']
  assert all(sha(bank/name)==h for name,h in rows.items()),'Existing cached row changed'
  entry['saved_rows_bitwise_preserved']=True;entry['status']='PASS' if ret==0 else 'BLOCKED';save()
  if ret:raise RuntimeError('Bounded cached continuation failed; no further invocation')
  assert (bank/'operators.npz').is_file() and ('COMPLETE '+str(bank)) in newlog.read_text()
  progress=read(bank/'PROGRESS.json')['rows']
  assert all(progress[int(name[4:8])]['cache_hit'] for name in rows),'A saved row was recomputed'
  state.setdefault('preserved_resource_failures',[]).append(dict(stage,history=str(history.relative_to(PREP))))
  state['steps']=[s for s in state['steps'] if s['name']!=name]
  state['steps'].append(dict(stage,log=newlog.name,returncode=0,seconds=entry['seconds'],
      original_capped_attempt_seconds=stage['seconds'],cached_rows_reused=len(rows),
      peak_child_RSS_bytes=entry['peak_child_RSS_bytes']))
  state['status']='IN_PROGRESS';state.pop('error',None);write(STATE,state)
  # Continue only uncompleted stages through the unchanged family controller.
  command=['/usr/bin/python3',str(PREP/'source/prepare_family.py'),'--xi','0.505']
  controller_log=PREP/'results'/('family_0.505_after_'+name+'_resume.log');start=time.monotonic()
  with controller_log.open('w') as f:ret=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
  entry['remaining_controller']={'command':command,'log':str(controller_log.relative_to(PREP)),'returncode':ret,'seconds':time.monotonic()-start};save()
  if ret and read(STATE).get('error')=="RuntimeError('Genuine numerical gate failure; no retuning')":
   raise RuntimeError('Genuine numerical validation failure retained; stop')
 record['status']='CACHED_FAMILY_COMPLETED';save()
except BaseException as e:
 record['status']='BLOCKED';record['error']=repr(e);save();raise

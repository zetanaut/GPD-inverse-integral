"""Exactly fgdev000 then fgdev001, two fixed starts each; stop on abstention."""
from pathlib import Path
import os,json,time,subprocess,resource
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GPD_REVIEW_ROOT=str(ROOT))
logdir=PREP/'results';logdir.mkdir(exist_ok=True)
record={'status':'RUNNING','stages':[],'sequential':True,'allowed_ids':['fgdev000','fgdev001'],'initializations':[31,43],'final_truths_evaluated':False}
dest=logdir/'DEVELOPMENT_EXECUTION.json';assert not dest.exists(),'Do not automatically retrain/repeat'
def save():dest.write_text(json.dumps(record,indent=2)+'\n')
def run(name,args):
 command=['/usr/bin/python3',*map(str,args)];start=time.monotonic()
 with (logdir/(name+'.log')).open('w') as f:ret=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 record['stages'].append({'name':name,'command':command,'returncode':ret,'seconds':time.monotonic()-start,'peak_child_RSS_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss});save()
 if ret:raise RuntimeError('Preserved execution failure at '+name)
try:
 run('constructed_geometry_check',[PREP/'source/verify_constructed_geometry.py'])
 run('freeze',[PREP/'source/development_controller.py','--freeze'])
 for ident in record['allowed_ids']:
  run('prepare_'+ident,[PREP/'source/development_controller.py','--id',ident])
  run('fit_'+ident,[PREP/'source/fit_final_geometry.py','--input',PREP/'inputs'/ident/'input.npz','--out',PREP/'results'/ident])
  run('replay_'+ident,[PREP/'source/replay_development.py',ident])
  result=json.loads((PREP/'results'/ident/'RESULT.json').read_text())
  if result['status']!='finite_budget_feasible':raise RuntimeError('Preserved abstention for '+ident+'; no next dataset')
 record['status']='TWO_TECHNICAL_CONTROLS_COMPLETED';save()
except BaseException as e:
 record['status']='BLOCKED';record['error']=repr(e);save();raise

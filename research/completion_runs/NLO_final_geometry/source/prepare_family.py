"""Continue one registered remaining family after the first family passes; stop on any unresolved failure."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,subprocess,resource,argparse
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1];NLO=ROOT/'GPD_NLO_completion_package'
parser=argparse.ArgumentParser();parser.add_argument('--xi',type=float,required=True);XI=parser.parse_args().xi
assert XI in (.155,.305,.365,.505)
assert json.loads((PREP/'geometry/xi0.215/PIPELINE.json').read_text())['status']=='FIRST_FAMILY_VALIDATED'
QS=(2.3,4.7,9.3) if XI==.365 else (3.4,7.1,15.3)
OUT=PREP/f'geometry/xi{XI}';OUT.mkdir(parents=True,exist_ok=True)
STATE=OUT/'PIPELINE.json'
state=json.loads(STATE.read_text()) if STATE.exists() else {'xi':XI,'status':'IN_PROGRESS','steps':[]}
def save():STATE.write_text(json.dumps(state,indent=2)+'\n')
def run(name,script,args,cap=1200):
 done=next((s for s in state['steps'] if s['name']==name),None)
 if done:
  assert done['returncode']==0,'Preserved failed stage cannot be silently repeated'
  return (OUT/done['log']).read_text()
 cmd=['/usr/bin/python3',str(script),*map(str,args)];log=OUT/(name+'.log');start=time.monotonic()
 with log.open('w') as f:
  try:ret=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=cap+60).returncode
  except subprocess.TimeoutExpired:ret='RESOURCE_TIMEOUT'
 state['steps'].append({'name':name,'command':cmd,'log':log.name,'returncode':ret,'seconds':time.monotonic()-start,'peak_child_RSS_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})
 save()
 if ret!=0:raise RuntimeError('Preserved stage failure: '+name)
 return log.read_text()
def assembly_path(log):return Path([l[9:] for l in log.splitlines() if l.startswith('COMPLETE ')][-1])
def derive(script,parent,degree):
 return Path(run(parent.name[:12]+'_'+script+'_'+str(degree),NLO/'candidate'/script,[parent,'--degree',degree],60).strip())
def correction(name,parent):
 return Path(run(name,NLO/'candidate/correct_qg_rows.py',[parent],60).strip())
def rel(p):return str(p.relative_to(NLO))
try:
 # The initial base invocation is retained exactly; never rerun it for metadata.
 raw=assembly_path(run('base',PREP/'source/assemble_missing.py',['--xi',str(XI),'--degree','10','--layout','log3','--q','48','--outer','18','--cap','1200','--bridge','mp']));assert (raw/'operators.npz').exists()
 state['raw']=rel(raw);save()
 base=correction('correct_base',raw);state['base']=rel(base);save()
 innerraw=assembly_path(run('inner72',PREP/'source/assemble_missing.py',['--xi',str(XI),'--degree','10','--layout','log3','--q','72','--outer','18','--cap','1200','--bridge','mp']))
 inner=correction('correct_inner72',innerraw);state['inner72']=rel(inner);save()
 outputraw=assembly_path(run('output48',NLO/'candidate/refine_output.py',[raw,'--special','48','--cap','1200']))
 output=correction('correct_output48',outputraw);state['output48']=rel(output);save()
 dirs={'weak10':rel(base)}
 for d in (6,8):dirs['weak'+str(d)]=rel(derive('restrict_nested.py',base,d))
 for d in (6,8,10):dirs['strong'+str(d)]=rel(derive('collocation_reference.py',base,d))
 for name,parent in [('inner72',inner),('output48',output)]:
  dirs[name+'_weak10']=rel(parent);dirs[name+'_strong10']=rel(derive('collocation_reference.py',parent,10))
 state['directories']=dirs;save()
 for script in ('check_raw_moments.py','check_physics.py','check_saved_action.py'):
  run('source_'+script,NLO/'candidate'/script,[base],120)
 from geometry_validation import validate
 point={'xi':XI,'directories':dirs};state['validation']=[]
 for field in ('smooth','Dterm_added'):
  for q in QS:
   dest=OUT/(field+'_Q2_'+str(q)+'.json');result=validate(point,q,field,dest)
   state['validation'].append({'field':field,'Q2':q,'pass':result['pass'],'gates':result['gates'],'file':str(dest.relative_to(PREP))});save()
   print(json.dumps(state['validation'][-1]),flush=True)
   if not result['pass']:raise RuntimeError('Genuine numerical gate failure; no retuning')
 state['status']='FAMILY_VALIDATED';save()
except Exception as e:
 state['status']='BLOCKED';state['error']=repr(e);save();raise

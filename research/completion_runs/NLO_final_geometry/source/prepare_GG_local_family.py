"""Only xi=.505, with explicit repair lineage and retained append-only events."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,time,subprocess,resource,zipfile
import numpy as np
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1];NLO=ROOT/'GPD_NLO_completion_package'
ADDON=ROOT/'GPD_final_geometry_GG_local_repair';OUT=PREP/'geometry/xi0.505'
STATE=OUT/'GG_LOCAL_PIPELINE.json';PARENT=NLO/'work/operators/24cf6f1984fdb8ba8db479bc8d4bc46bb8d21d8da2043498aec9819f2595e2c3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def rel(p):return str(Path(p).relative_to(NLO))
assert read(ADDON/'LOCAL_VERIFICATION.json')['pass']
binding={'assembler':sha(PREP/'source/assemble_GG_local.py'),'refiner':sha(PREP/'source/refine_output_GG_local.py'),
 'addon_build':sha(ADDON/'ACTIVE_BUILD.json'),'verification':sha(ADDON/'LOCAL_VERIFICATION.json'),
 'original_pipeline':sha(OUT/'PIPELINE.json'),'original_FAILURE':sha(PARENT/'FAILURE.json'),
 'original_KEY':sha(PARENT/'KEY.json')}
original_rows={p.name:sha(p) for p in sorted(PARENT.glob('row_*.npz'))};assert len(original_rows)==111
if STATE.exists():
 state=read(STATE);assert state['binding']==binding and state['parent_row_hashes']==original_rows
 assert state['status'] not in ('BLOCKED','FAMILY_VALIDATED'),'Do not repeat a terminal pipeline'
else:
 state={'xi':.505,'status':'IN_PROGRESS','binding':binding,'parent':rel(PARENT),
  'parent_row_hashes':original_rows,'steps':[],'events':[],'validation':[],
  'scope':'Only missing .505 family; no final truths, fits or noise'}
def save(event,**details):
 state['events'].append({'index':len(state['events']),'event':event,**details});write(STATE,state)
def verify_parent():
 assert sha(OUT/'PIPELINE.json')==binding['original_pipeline']
 assert sha(PARENT/'KEY.json')==binding['original_KEY'] and sha(PARENT/'FAILURE.json')==binding['original_FAILURE']
 assert {p.name:sha(p) for p in PARENT.glob('row_*.npz')}==original_rows
def stage_cache(name):
 result=[]
 for p in (NLO/'work/operators').glob('*/KEY.json'):
  k=read(p)
  if k.get('basis',{}).get('xi')!=.505 or not k.get('GG_local_fallback'):continue
  if name in ('base','inner72'):
   ok='parent' not in k and k.get('inner_n')==(48 if name=='base' else 72) and k.get('assembly_adapter_sha256')==binding['assembler']
  else:ok=k.get('special_output_n')==48 and k.get('parent')==state.get('raw') and 'qg_endpoint_subtraction' not in k
  if ok:result.append(p.parent)
 assert len(result)==1,(name,result)
 return result[0]
def row_hashes(bank):
 out={}
 for f in sorted(bank.glob('row_*.npz')):
  with zipfile.ZipFile(f) as z:assert z.testzip() is None
  with np.load(f,allow_pickle=False) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
  out[f.name]=sha(f)
 return out
def run(name,script,args,cap=1200,assembly=False):
 command=['/usr/bin/python3',str(script),*map(str,args)]
 prior=[s for s in state['steps'] if s['name']==name]
 if any(s['returncode']==0 for s in prior):
  step=next(s for s in prior if s['returncode']==0);assert step['command']==command
  return (OUT/step['log']).read_text()
 for attempt in range(len(prior),2 if assembly else 1):
  before={};bank=None
  if attempt:
   old=prior[-1];expected='TimeoutError: bounded output refinement; rows retained' if name=='output48' else 'TimeoutError: finite construction cap; saved rows retained'
   assert old['returncode']==1 and (OUT/old['log']).read_text().rstrip().endswith(expected),'Only a wall cap may resume once'
   bank=stage_cache(name);before=row_hashes(bank)
   snapshot=OUT/f'GG_{name}_resume_snapshot.json';assert not snapshot.exists()
   write(snapshot,{'bank':rel(bank),'KEY_sha256':sha(bank/'KEY.json'),'rows':before,'previous_step':old})
  log=OUT/f'GG_{name}_attempt{attempt+1}.log';assert not log.exists()
  save('stage_started',name=name,attempt=attempt+1,command=command,cap_seconds=cap)
  start=time.monotonic()
  with log.open('w') as f:
   try:ret=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=cap+60).returncode
   except subprocess.TimeoutExpired:ret='RESOURCE_TIMEOUT'
  step={'name':name,'attempt':attempt+1,'command':command,'log':log.name,'returncode':ret,
        'seconds':time.monotonic()-start,'peak_child_RSS_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
  if before:
   assert all(sha(bank/n)==h for n,h in before.items()),'Saved row changed'
   step.update(saved_rows_reused=len(before),saved_rows_bitwise_preserved=True)
  state['steps'].append(step);prior.append(step);verify_parent();save('stage_finished',name=name,attempt=attempt+1,returncode=ret)
  if ret==0:return log.read_text()
  expected='TimeoutError: bounded output refinement; rows retained' if name=='output48' else 'TimeoutError: finite construction cap; saved rows retained'
  if not(assembly and attempt==0 and ret==1 and log.read_text().rstrip().endswith(expected)):
   raise RuntimeError('Preserved stage failure: '+name)
 raise RuntimeError('Additional resource exhaustion: '+name)
def assembly_path(log):return Path([l[9:] for l in log.splitlines() if l.startswith('COMPLETE ')][-1])
def correction(name,parent):return Path(run(name,NLO/'candidate/correct_qg_rows.py',[parent],60).strip())
def derive(script,parent,degree):return Path(run(parent.name[:12]+'_'+script+'_'+str(degree),NLO/'candidate'/script,[parent,'--degree',degree],60).strip())
try:
 save('pipeline_started',parent_rows=111)
 raw=assembly_path(run('base',PREP/'source/assemble_GG_local.py',['--xi','.505','--degree','10','--layout','log3','--q','48','--outer','18','--cap','1200','--bridge','mp','--reuse-parent',PARENT],assembly=True))
 assert all(sha(raw/n)==h for n,h in original_rows.items());state['raw']=rel(raw);save('base_complete',original_111_rows_bitwise_reused=True)
 base=correction('correct_base',raw);state['base']=rel(base);save('base_corrected')
 innerraw=assembly_path(run('inner72',PREP/'source/assemble_GG_local.py',['--xi','.505','--degree','10','--layout','log3','--q','72','--outer','18','--cap','1200','--bridge','mp'],assembly=True))
 inner=correction('correct_inner72',innerraw);state['inner72']=rel(inner);save('inner_complete')
 outputraw=assembly_path(run('output48',PREP/'source/refine_output_GG_local.py',[raw,'--special','48','--cap','1200'],assembly=True))
 output=correction('correct_output48',outputraw);state['output48']=rel(output);save('output_complete')
 dirs={'weak10':rel(base)}
 for degree in (6,8):dirs['weak'+str(degree)]=rel(derive('restrict_nested.py',base,degree))
 for degree in (6,8,10):dirs['strong'+str(degree)]=rel(derive('collocation_reference.py',base,degree))
 for name,parent in [('inner72',inner),('output48',output)]:
  dirs[name+'_weak10']=rel(parent);dirs[name+'_strong10']=rel(derive('collocation_reference.py',parent,10))
 state['directories']=dirs;save('projections_complete')
 for script in ('check_raw_moments.py','check_physics.py','check_saved_action.py'):
  run('source_'+script,NLO/'candidate'/script,[base],120)
 run('source_NS_charge',PREP/'source/check_ns_charge_missing.py',[base],120)
 run('source_structure',PREP/'source/check_structure_missing.py',[base,'--q2','15.3'],120)
 from geometry_validation import validate
 point={'xi':.505,'directories':dirs}
 for field in ('smooth','Dterm_added'):
  for q in (3.4,7.1,15.3):
   if any(v['field']==field and v['Q2']==q for v in state['validation']):continue
   path=OUT/f'GG_{field}_Q2_{q}.json';result=validate(point,q,field,path)
   row={'field':field,'Q2':q,'pass':result['pass'],'gates':result['gates'],'file':str(path.relative_to(PREP))}
   state['validation'].append(row);save('point_validated',**row);print(json.dumps(row),flush=True)
   if not result['pass']:raise RuntimeError('Genuine numerical gate failure; no retuning')
 verify_parent();state['status']='FAMILY_VALIDATED';save('pipeline_completed',original_failures_preserved=True)
except BaseException as e:
 state['status']='BLOCKED';state['error']=repr(e);save('pipeline_stopped',error=repr(e));raise

"""Continue only output48, then the unchanged remaining xi=.505 stages."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,time,subprocess,resource
import numpy as np
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1];NLO=ROOT/'GPD_NLO_completion_package'
ADDON=ROOT/'GPD_GG_coordinate_repair';OUT=PREP/'geometry/xi0.505'
STATE=OUT/'GG_COORDINATE_PIPELINE.json';OLD=OUT/'GG_LOCAL_PIPELINE.json'
PARTIAL=NLO/'work/operators/3469941c41e7591df44efe185683c5039e6ce759f832b125bdb02a56f29ff21d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def rel(p):return str(Path(p).relative_to(NLO))
def rows(bank):
    out={}
    for p in sorted(bank.glob('row_*.npz')):
        with np.load(p,allow_pickle=False) as z:assert all(np.isfinite(z[k]).all() for k in z.files)
        out[p.name]=sha(p)
    return out
old=read(OLD);assert old['status']=='BLOCKED'
assert read(ADDON/'LOCAL_VERIFICATION.json')['pass']
inherited=rows(PARTIAL);assert len(inherited)==120
protected={str(p.relative_to(ROOT)):sha(p) for p in (OLD,OUT/'PIPELINE.json',ADDON/'ACTIVE_BUILD.json',ADDON/'LOCAL_VERIFICATION.json',PREP/'source/refine_output_GG_coordinates.py')}
for bank in (PARTIAL,NLO/old['parent'],NLO/old['raw'],NLO/old['base'],NLO/old['inner72']):
    for p in bank.iterdir():
        if p.is_file() and (p.suffix=='.json' or p.name=='operators.npz' or p.name.startswith('row_')):protected[str(p.relative_to(ROOT))]=sha(p)
assert not STATE.exists(),'Preserve terminal/partial pipeline; no automatic rerun'
state={'xi':.505,'status':'IN_PROGRESS','prior_pipeline':str(OLD.relative_to(PREP)),
       'binding':protected,'parent':old['parent'],'parent_row_hashes':old['parent_row_hashes'],
       'raw':old['raw'],'base':old['base'],'inner72':old['inner72'],
       'reused_output':rel(PARTIAL),'inherited_output_rows':inherited,'steps':[],'events':[],'validation':[],
       'scope':'Only remaining output48 and .505 downstream; prior base/inner and other families reused'}
def save(event,**details):
    state['events'].append({'index':len(state['events']),'event':event,**details});write(STATE,state)
def preserve():
    for name,value in protected.items():assert sha(ROOT/name)==value,'Preserved input changed: '+name
def output_bank():
    matches=[]
    for p in (NLO/'work/operators').glob('*/KEY.json'):
        k=read(p)
        if k.get('GG_coordinate_fallback') and k.get('reused_output',{}).get('key')==PARTIAL.name and 'qg_endpoint_subtraction' not in k:
            matches.append(p.parent)
    assert len(matches)==1,matches
    return matches[0]
def run(name,script,args,cap=120,output=False):
    command=['/usr/bin/python3',str(script),*map(str,args)]
    for attempt in range(1,3 if output else 2):
        before={};bank=None
        if attempt==2:
            bank=output_bank();before=rows(bank)
            write(OUT/'GG_COORDINATE_output_resume_snapshot.json',{'bank':rel(bank),'KEY_sha256':sha(bank/'KEY.json'),'rows':before})
        log=OUT/f'GG_COORDINATE_{name}_attempt{attempt}.log';assert not log.exists()
        save('stage_started',name=name,attempt=attempt,command=command,outer_cap_seconds=cap)
        start=time.monotonic()
        with log.open('x') as f:
            try:ret=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=cap).returncode
            except subprocess.TimeoutExpired:ret='RESOURCE_TIMEOUT'
        record={'name':name,'attempt':attempt,'command':command,'log':log.name,'returncode':ret,
                'seconds':time.monotonic()-start,'outer_cap_seconds':cap,'peak_child_RSS_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
        if output:
            bank=output_bank()
            assert all(sha(bank/n)==h for n,h in inherited.items()),'120 inherited rows changed'
            assert all(sha(bank/n)==h for n,h in before.items()),'Completed resumed rows changed'
            record.update(inherited_output_rows_bitwise_preserved=120,resume_saved_rows_preserved=len(before),bank=rel(bank))
        state['steps'].append(record);preserve();save('stage_finished',name=name,attempt=attempt,returncode=ret)
        if ret==0:return log.read_text()
        resource_only=(ret=='RESOURCE_TIMEOUT' or (ret==1 and log.read_text().rstrip().endswith('TimeoutError: bounded output refinement; rows retained')))
        if not(output and attempt==1 and resource_only):raise RuntimeError('Preserved stage failure: '+name)
    raise RuntimeError('Additional output48 resource exhaustion')
def assembly_path(log):return Path([line[9:] for line in log.splitlines() if line.startswith('COMPLETE ')][-1])
def derive(script,parent,degree):
    method={'restrict_nested.py':'exact_nested_mass_projection','collocation_reference.py':'QR_selected_strong_collocation'}[script]
    matches=[]
    for p in (NLO/'work/operators').glob('*/KEY.json'):
        k=read(p)
        if k.get('parent')==rel(parent) and k.get('method')==method and k.get('basis',{}).get('degree')==degree:
            assert k['source_sha256']==sha(NLO/'candidate'/script) and k['parent_operator_sha256']==sha(parent/'operators.npz')
            assert read(p.parent/'RESULT.json')['finite'];matches.append(p.parent)
    if matches:
        assert len(matches)==1;save('compatible_projection_reused',script=script,degree=degree,bank=rel(matches[0]));return matches[0]
    return Path(run(parent.name[:12]+'_'+script+'_'+str(degree),NLO/'candidate'/script,[parent,'--degree',degree],120).strip())
try:
    save('pipeline_started',reused_output_rows=120,base_and_inner_rerun=False)
    raw=NLO/state['raw'];base=NLO/state['base'];inner=NLO/state['inner72']
    out_raw=assembly_path(run('output48',PREP/'source/refine_output_GG_coordinates.py',[raw,'--special','48','--cap','1200','--reuse-output',PARTIAL],1500,True))
    state['output48_raw']=rel(out_raw);save('output_raw_complete',inherited_rows_bitwise_preserved=120)
    output=Path(run('correct_output48',NLO/'candidate/correct_qg_rows.py',[out_raw],120).strip())
    state['output48']=rel(output);save('output_corrected')
    dirs={'weak10':rel(base)}
    for degree in (6,8):dirs['weak'+str(degree)]=rel(derive('restrict_nested.py',base,degree))
    for degree in (6,8,10):dirs['strong'+str(degree)]=rel(derive('collocation_reference.py',base,degree))
    for name,parent in [('inner72',inner),('output48',output)]:
        dirs[name+'_weak10']=rel(parent);dirs[name+'_strong10']=rel(derive('collocation_reference.py',parent,10))
    state['directories']=dirs;save('projections_complete')
    for script in ('check_raw_moments.py','check_physics.py','check_saved_action.py'):
        run('source_'+script,NLO/'candidate'/script,[base],180)
    run('source_NS_charge',PREP/'source/check_ns_charge_missing.py',[base],180)
    run('source_structure',PREP/'source/check_structure_missing.py',[base,'--q2','15.3'],180)
    from geometry_validation import validate
    point={'xi':.505,'directories':dirs}
    for field in ('smooth','Dterm_added'):
        for q in (3.4,7.1,15.3):
            path=OUT/f'GG_COORDINATE_{field}_Q2_{q}.json'
            result=validate(point,q,field,path)
            row={'field':field,'Q2':q,'pass':result['pass'],'gates':result['gates'],'file':str(path.relative_to(PREP))}
            state['validation'].append(row);save('point_validated',**row);print(json.dumps(row),flush=True)
            if not result['pass']:raise RuntimeError('Genuine numerical gate failure; no retuning')
    preserve();state['status']='FAMILY_VALIDATED';save('pipeline_completed',original_evidence_preserved=True)
except BaseException as error:
    state['status']='BLOCKED';state['error']=repr(error);save('pipeline_stopped',error=repr(error));raise

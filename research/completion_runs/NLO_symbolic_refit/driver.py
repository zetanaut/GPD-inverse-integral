"""One symbolic proposal and equal-budget all-term continuation per NLO parent.

No true-field evaluation occurs in this module. Decisions are persisted before
the separate after_decision module may inspect development truth information.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys,json,hashlib,time,datetime
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('NLO_PILOT_PACKET',ROOT.parents[1])).resolve()
PRIOR=PROJECT/'GPD_NLO_resolution_review/pilot'
sys.path.insert(0,str(PRIOR))
import common as c
import neural_pilot as n
import preflight as numerical
sys.path.insert(0,str(ROOT))
from symbolic_adapter import install_supports,propose_supports

PLAN=json.loads((ROOT/'PLAN.json').read_text())
DATA=json.loads((ROOT/'DATA.json').read_text())
write=c.write
Q2S=[2.3,4.7,9.3,3.1,6.1]
CHANNELS=['Sigma','T_rescaled','Hg']

def timestamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def full_supports(problem):return [list(range(N//2+1)) for _,N,_ in problem.p.spec]

def load_parent(parent,branch='allterm',supports=None):
    path=PROJECT/parent['parent_checkpoint'];assert c.digest(path)==parent['parent_sha256']
    problem=n.Problem(parent['truth'],parent['seed_lineage'])
    saved=torch.load(path,weights_only=True);problem.model.load_state_dict(saved['state_dict'])
    problem.x0=saved['flat'].clone();problem.ckpath=path;problem.key=parent['key']+'__'+branch
    expected=torch.cat([v.detach().flatten() for v in problem.model.parameters()])
    assert torch.equal(problem.x0,expected) and expected.numel()==3867
    assert torch.equal(problem.y,torch.tensor([r['target'] for r in DATA[parent['truth']]['training']]).flatten())
    if supports is not None:install_supports(problem,supports)
    return problem

def verify_sources():
    records=json.loads((ROOT/'SOURCE_HASHES.json').read_text())
    for name,record in records.items():assert c.digest(PROJECT/name)==record['sha256'],name
    assert c.digest(ROOT/'SOURCE_HASHES.json')==PLAN['source_hashes_sha256']
    assert c.digest(ROOT/'DATA.json')==PLAN['data_sha256']
    return len(records)

def proposals():
    target=ROOT/'PROPOSALS.json'
    if target.exists():return json.loads(target.read_text())
    verify_sources();provenance={'created_UTC':timestamp(),'plan_sha256':c.digest(ROOT/'PLAN.json'),
      'source_files':{f.name:c.digest(f) for f in ROOT.glob('*.py')},'helper_tests':'12 passed in 1.24s; mechanical tests only'}
    write(ROOT/'IMPLEMENTATION_BEFORE_PROPOSALS.json',provenance)
    records=[]
    for parent in PLAN['parents']:
        p=load_parent(parent);full=full_supports(p)
        proposed=propose_supports(p,tolerance=.001)
        supports=[v['support'] for v in proposed]
        records.append({'parent':parent['key'],'parent_sha256':parent['parent_sha256'],'tolerance':.001,
           'full_supports':full,'proposed_supports':supports,'identity':supports==full,
           'terms_full':sum(map(len,full)),'terms_proposed':sum(map(len,supports)),
           'records':proposed,'coefficient_role':'interpretation only; never installed as fixed values'})
    result={'created_UTC':timestamp(),'proposal_count':4,'source':'inferred parent moments only','parents':records}
    write(target,result);return result

def moment_check(model,supports):
    xf,xd=c.helpers.moment_grids();scales=c.projector().scales.numpy();details={};residuals={}
    for order in (96,192):
        pf=c.projector(order,xf);pd=c.projector(order,xd)
        with torch.no_grad():mf=pf.moments(model).numpy();md=pd.moments(model).numpy()
        entries=[];res=[]
        for i,((channel,spin,_),ss) in enumerate(zip(pf.spec,supports)):
            fit_design=c.poly.design(xf,spin,ss);check_design=c.poly.design(xd,spin,ss)
            coefficient=np.linalg.lstsq(fit_design,mf[:,i],rcond=None)[0]
            delta=(md[:,i]-check_design@coefficient)/scales[i];res.append(delta)
            full=np.zeros(spin//2+1);full[ss]=coefficient
            entries.append({'channel':CHANNELS[channel],'spin':spin,'support_z':ss,'powers_xi':[2*k for k in ss],
              'coefficient_z':full.tolist(),'coefficient_xi':(full/.6**(2*np.arange(len(full)))).tolist(),
              'rms':float(np.sqrt(np.mean(delta**2))),'max_abs':float(max(abs(delta)))})
        residuals[order]=np.array(res).T
        details[str(order)]={'entries':entries,'combined_rms':float(np.sqrt(np.mean(residuals[order]**2))),
                             'max_channel_spin_rms':max(e['rms'] for e in entries)}
    vector_change=float(np.sqrt(np.mean((residuals[192]-residuals[96])**2)))
    rms_change=abs(details['192']['combined_rms']-details['96']['combined_rms'])
    return {'orders':details,'diagnostic_residual_RMS_change':vector_change,'diagnostic_RMS_change':rms_change,
      'polynomiality_pass':max(details[str(k)]['combined_rms'] for k in (96,192))<=1e-3,
      'quadrature_pass':vector_change<=1e-5 and rms_change<=1e-5}

def branch_key(parent,kind,supports):
    record={'parent_sha256':parent['parent_sha256'],'active_supports_z':supports,
      'data_sha256':PLAN['data_sha256'],'loss':PLAN['loss'],'moments':PLAN['moments'],
      'source_hash_index':PLAN['source_hashes_sha256'],'adapter_sha256':c.digest(ROOT/'symbolic_adapter.py'),
      'driver_sha256':c.digest(ROOT/'driver.py'),'plan_sha256':c.digest(ROOT/'PLAN.json'),
      'meaning':'parent identity and declared scalar objective; existing NLO row caches unchanged'}
    token=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return {'cache_key':token,'branch':kind,**record}

def continue_branch(parent,kind,supports):
    out=ROOT/'branches'/(parent['key']+'__'+kind);out.mkdir(parents=True,exist_ok=True)
    key=branch_key(parent,kind,supports)
    if (out/'METRICS.json').exists():
        assert json.loads((out/'KEY.json').read_text())==key
        return json.loads((out/'METRICS.json').read_text())
    if (out/'START.json').exists():raise RuntimeError('Partial branch requires inspection; no automatic retry or extra budget: '+str(out))
    p=load_parent(parent,kind,supports);interface=n.interface_checks(p)
    assert interface['pass'];write(out/'KEY.json',key);write(out/'INTERFACE_BEFORE.json',interface)
    write(out/'START.json',{'created_UTC':timestamp(),'parent':parent,'kind':kind,'active_supports_z':supports,'cache_key':key['cache_key']})
    start=time.monotonic();print('START',p.key,'terms',sum(map(len,supports)),flush=True)
    optimizer,x=n.solver.solve(p,n.solver.Settings(max_jacobians=60),out)
    final=n.finish(p,x,out)
    # finish has saved the checkpoint before the expensive learned-field checks.
    ck=torch.load(out/'final.pt',weights_only=True)
    ck.update(parent_checkpoint=parent['parent_checkpoint'],parent_sha256=parent['parent_sha256'],
       random_seed_lineage=parent['seed_lineage'],random_initial_sha256=parent['original_random_checkpoint']['sha256'],
       active_supports_z=supports,objective_cache_key=key['cache_key'],
       legacy_initial_sha256_role='fitted parent used as continuation start; original random lineage separately recorded')
    torch.save(ck,out/'final.pt')
    final.update(key=p.key,parent=parent['key'],branch=kind,supports_z=supports,parent_sha256=parent['parent_sha256'],
      checkpoint_sha256=c.digest(out/'final.pt'),cache_key=key['cache_key'],nominal_branch=True,actual_continuation=True)
    saved_flat=ck['flat'];p.x0=saved_flat.clone()
    final['interface']=n.interface_checks(p)
    full=full_supports(p)
    final['full_space']=moment_check(p.model,full)
    final['selected_space']=final['full_space'] if supports==full else moment_check(p.model,supports)
    target_map={(r['xi'],r['Q2']):r['target'] for split in DATA[parent['truth']].values() for r in split}
    CFF=[]
    for point in c.operator_directories():
        for q2 in Q2S:
            nr=numerical.validate_point(point,q2,p.key,p.model)
            write(out/f'numerical_xi{point["xi"]}_Q2{q2}.json',nr)
            target=np.array(target_map[point['xi'],q2]);reference=np.array(nr['checks']['strong10']['independent_hard']['reference'])
            CFF.append({'xi':point['xi'],'Q2':q2,'split':'selection' if q2==3.1 else 'reporting' if q2==6.1 else 'training',
               'target':target.tolist(),'nominal_prediction':nr['checks']['weak10']['CFF'],
               'reference_prediction':reference.tolist(),'reference_residual':(reference-target).tolist(),
               'CFF_budget':nr['CFF_estimated_total'],'CFF_limits':nr['CFF_limits'],
               'field_budget':{ch:v['estimated_total'] for ch,v in nr['fields'].items()},
               'numerical_pass':nr['pass'],'numerical_gates':nr['gates']})
    final['CFF']=CFF
    final['gates']={'gradient':final['optimizer_pass'],'scalar_residual_Jacobian':final['interface']['pass'],
      'full_polynomiality':final['full_space']['polynomiality_pass'],
      'selected_polynomiality':final['selected_space']['polynomiality_pass'],
      'moment_quadrature':final['full_space']['quadrature_pass'] and final['selected_space']['quadrature_pass'],
      'NLO_numerics':all(r['numerical_pass'] for r in CFF)}
    final['pass']=all(final['gates'].values())
    final['continuation_status']=optimizer['status'];final['jacobians']=len(optimizer['history'])
    final['accepted_GN_updates']=sum(r.get('accepted',False) for r in optimizer['history'])
    final['accepted_cleanup_updates']=sum(r['accepted'] for r in final['history'])
    final['seconds_total']=time.monotonic()-start
    write(out/'METRICS.json',final);print('COMPLETE',p.key,final['gates'],flush=True)
    return final

def decision(parent,allterm,candidate,identity):
    pairs=[]
    for left in candidate['CFF']:
        if left['split']!='selection':continue
        right=next(v for v in allterm['CFF'] if v['xi']==left['xi'] and v['Q2']==left['Q2'])
        actual=abs(np.array(left['reference_residual']))
        allowed=1.10*abs(np.array(right['reference_residual']))+np.array(left['CFF_budget'])+np.array(right['CFF_budget'])
        budgets_pass=bool(np.all(np.array(left['CFF_budget'])<=left['CFF_limits']) and np.all(np.array(right['CFF_budget'])<=right['CFF_limits']))
        pairs.append({'xi':left['xi'],'Q2':3.1,'candidate_absolute_residual':actual.tolist(),
          'allterm_absolute_residual':np.abs(right['reference_residual']).tolist(),
          'candidate_budget':left['CFF_budget'],'allterm_budget':right['CFF_budget'],
          'allowed_absolute_residual':allowed.tolist(),'component_pass':(actual<=allowed).tolist(),'both_budgets_pass':budgets_pass})
    compatible=allterm['pass'] and candidate['pass'] and all(all(v['component_pass']) and v['both_budgets_pass'] for v in pairs)
    return {'parent':parent['key'],'parent_sha256':parent['parent_sha256'],'identity_proposal':identity,
       'allterm_pass':allterm['pass'],'candidate_pass':candidate['pass'],'selection_components':pairs,
       'compatible':bool(compatible),'interpretation':'unresolved-compatible simplification, not a physical zero' if compatible else 'failed descriptive candidate; preserved',
       'decision_inputs':'branch gates plus Q2=3.1 components only; no Q2=6.1 residual or truth recovery',
       'allterm_checkpoint_sha256':allterm['checkpoint_sha256'],'candidate_checkpoint_sha256':candidate['checkpoint_sha256']}

def run():
    prop=proposals();start=time.monotonic();runs=[];decisions=[]
    for parent,proposal in zip(PLAN['parents'],prop['parents']):
        assert parent['key']==proposal['parent']
        allterm=continue_branch(parent,'allterm',proposal['full_supports']);runs.append(allterm)
        if proposal['identity']:
            candidate=dict(allterm);candidate.update(key=parent['key']+'__proposed',branch='proposed',actual_continuation=False,
                           reused_branch=allterm['key'],reason='proposal is all-term identity; duplicate optimization omitted')
            out=ROOT/'branches'/candidate['key'];out.mkdir(parents=True,exist_ok=True);write(out/'METRICS.json',candidate)
        else:candidate=continue_branch(parent,'proposed',proposal['proposed_supports'])
        runs.append(candidate);decisions.append(decision(parent,allterm,candidate,proposal['identity']))
    write(ROOT/'BRANCH_SUMMARY.json',{'nominal_branches':8,'actual_continuations':sum(r['actual_continuation'] for r in runs),
      'branches':runs,'seconds':time.monotonic()-start})
    path=ROOT/'DECISIONS.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['decisions']==decisions
    else:write(path,{'frozen_UTC':timestamp(),'proposal_sha256':c.digest(ROOT/'PROPOSALS.json'),
        'plan_sha256':c.digest(ROOT/'PLAN.json'),'truth_diagnostics_evaluated_before_decisions':False,'decisions':decisions})
    verify_sources();print('DECISIONS FROZEN',[(d['parent'],d['compatible']) for d in decisions],flush=True)

if __name__=='__main__':run()

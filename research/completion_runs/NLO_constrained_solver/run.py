"""Locked two-continuation controller. No historical optimization entry points."""
from problem import *
import json,traceback,argparse,hashlib,platform
import scipy
from solver import solve,SETTINGS

ESTIMATOR='NLO_constrained_GN_development_v1'
original_checks=common.module('immutable_feasible_checks',OLD/'verify.py')

def verify_inputs():
    frozen=json.loads((HERE/'FREEZE.json').read_text())
    for rel,h in frozen['files'].items():assert sha(HERE/rel)==h,rel
    deps=json.loads((HERE/'DEPENDENCIES.json').read_text())
    for rel,h in deps['files'].items():assert sha(ROOT/rel)==h,rel
    for item in deps['prior_payload']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    return {'frozen':len(frozen['files']),'dependency_files':len(deps['files']),'prior_payload':len(deps['prior_payload'])}

def freeze():
    if (HERE/'FREEZE.json').exists():raise FileExistsError('Already frozen')
    assert json.loads((HERE/'MECHANICAL.json').read_text())['pass']
    for name,n in [('supplied_first_order_tests.log',12),('solver_tests.log',10),('inherited_kkt_tests.log',16)]:
        text=(HERE/'logs'/name).read_text();assert f'Ran {n} tests' in text and text.strip().endswith('OK')
    contract={'estimator':ESTIMATOR,'purpose':'DEVELOPMENT warm-start constrained solver continuation; not an end-to-end estimator or independent random starts',
      'algorithm':'Feasible trust-region SQP, Gauss-Newton objective quadratic, normalized-row primal active-set QP, nonlinear normal SOC',
      'settings':SETTINGS,'states':['inputs/initial31.pt','inputs/initial43.pt'],
      'unchanged':'Original FiLM 3867 parameters, all42 permitted moment coefficients, J denominators18/492, lambda1, d000 y/C, all166 inequalities, numerical margin0.95, NLO operators/contractions, all authoritative gates',
      'temporary_coordinates':'At each current state form an orthogonal span of full residual Jacobian, original gradient and working constraint normals. This is a changing numerical step subspace only, not a fixed GPD representation or parameter freeze.',
      'scaling':'Objective unscaled. Constraint rows divided by original-coordinate row norms only for the QP; multipliers divided by those norms when mapped back. Q is orthonormal; final KKT uses all3867 original coordinates.',
      'working_set':'All rows c>=-0.01 plus any newly violated trial rows. Every trial evaluates all166 and must satisfy strict c<=0; working selection never changes acceptance active_tolerance1e-6.',
      'trust_model':'J quadratic rTr+gTs+sT(RTR)s. Damping only selects a bounded proposal. SOC corrects nonlinear feasibility; acceptance compares actual original-J decrease to the uncorrected SQP model decrease. No penalty escalation or fixed extra term in J.',
      'SOC':'At most3 min-norm linear normal corrections, at most0.5 proposal norm, kept inside full trust radius; targets c=-1e-9 only as interior solver mechanics, not a new acceptance threshold.',
      'stationarity':'Unchanged original-coordinate KKT L2<=1e-6, strict primal, nonnegative multipliers, complementarity<=1e-8, active threshold1e-6 and active-row rank diagnostic. Library/QP success never promotes a candidate.',
      'budget':{'starts':2,'total_seconds_each':7200,'optimization_seconds_each':6600,'validation_reserve_seconds':600,
                'max_iterations':SETTINGS['max_iterations'],'max_evaluations':SETTINGS['max_evaluations'],
                'stop':'Three consecutive unchanged-state terminations, or tighter trust/finite/resource stop; no retry after a returned result'},
      'validation':'Each returned candidate gets all15 unchanged full-forward checks, original80 refined moments and specified new160 disjoint xi; no new160 values enter optimization. Independent final directional derivatives and KKT recomputed.',
      'forbidden':'No kernels/contraction reconstruction, AL, historical training, new noise/datasets, profiles, calibration, symbolic pruning, true-GPD selection or experimental fits'}
    write(HERE/'CONTRACT.json',contract)
    deps={}
    prior=json.loads((OLD/'MANIFEST.json').read_text())['files']
    for r in prior:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    for rel in ['review_NLO_feasible_domain.zip','GPD_NLO_feasible_domain_review.zip']:
        deps[rel]=sha(ROOT/rel)
    old_frozen=json.loads((OLD/'FREEZE.json').read_text())
    for rel,h in old_frozen['files'].items():
        assert sha(OLD/rel)==h;deps[str((OLD/rel).relative_to(ROOT))]=h
    for rel,h in json.loads((OLD/'EXTERNAL_HASHES.json').read_text()).items():
        assert sha(ROOT/rel)==h;deps[rel]=h
    for label in (31,43):
        p=OLD/f'start{label}/final.pt';deps[str(p.relative_to(ROOT))]=sha(p)
        q=ROOT/f'GPD_NLO_feasible_domain_review/results/FIRST_ORDER_EXTENDED_{label}.npz';deps[str(q.relative_to(ROOT))]=sha(q)
    write(HERE/'DEPENDENCIES.json',{'files':deps,'prior_payload':prior,
       'immutable_packet':'review_NLO_feasible_domain.zip; extract original paths in the same project for replay; no rebuilding required'})
    write(HERE/'ENVIRONMENT.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
              'scipy':scipy.__version__,'torch':torch.__version__,'threads':1})
    files={str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*.py')}
    for name in ('CONTRACT.json','DEPENDENCIES.json','ENVIRONMENT.json','INITIAL_INPUTS.json','MECHANICAL.json'):
        files[name]=sha(HERE/name)
    for p in (HERE/'inputs').glob('*'):files[str(p.relative_to(HERE))]=sha(p)
    write(HERE/'FREEZE.json',{'unix_time':time.time(),'before_continuations':True,'files':files})
    print('FROZEN',sha(HERE/'FREEZE.json'),flush=True)

def first_order(oracle,x,out):
    values=oracle.values(x);ids=np.flatnonzero(values['c']>=-SETTINGS['working_window'])
    g,R,A=oracle.derivatives(x,ids);check=kkt(g,values['c'],A,ids,oracle.names)
    full_lam=np.zeros(len(values['c']))
    full_lam[check['evaluated_jacobian_indices']]=check['inequality_multipliers']
    np.savez_compressed(out/'FIRST_ORDER.npz',flat=x,original_gradient=g,constraints=values['c'],
                        working_indices=ids,working_jacobian=A,multipliers=full_lam,
                        residual=values['residual'],names=np.array(oracle.names))
    check.update(J=values['J'],J_components=values['components'].tolist(),
                 mapped_multipliers_full=full_lam.tolist(),original_parameter_dimension=len(x))
    write(out/'KKT.json',check);return check

def final_checks(oracle,x,out):
    t=time.monotonic();p=oracle.p;d=oracle.domain;xx=tensor(x)
    check=first_order(oracle,x,out)
    moments=original.moment_validation(p,xx);write(out/'MOMENTS.json',moments)
    full=original.full_forward(p,xx,out/'full_forward');write(out/'FULL_FORWARD.json',full)
    derivatives=original_checks.derivative_record(p,d,xx,independent=True);write(out/'DERIVATIVES.json',derivatives)
    interior=True
    for r in full['records']:
        interior &= max(np.array(r['CFF_estimated_total'])/r['CFF_limits'])<1-1e-8
        interior &= max(r['field_totals'].values())<1e-3*(1-1e-8)
        detail=json.loads((out/'full_forward'/r['file']).read_text())
        for method in ('weak10','strong10'):
            interior &= max(detail['checks'][method]['moments']['relative_errors'])<1e-4*(1-1e-8)
    gates={'KKT':check['pass'],'moments_refinement_and_new160':moments['pass'],
           'all15_authoritative_forward':full['pass'],'derivatives':derivatives['pass'],
           'full_authoritative_inequalities_inactive':bool(interior)}
    result={'label':p.label,'estimator':ESTIMATOR,'accepted':all(gates.values()),'gates':gates,
            'J':check['J'],'J_components':check['J_components'],'original_gradient_L2':check['raw_gradient_L2'],
            'KKT_L2':check['stationarity_L2'],'primal_max_excess':check['full_maximum_excess'],
            'complementarity':check['complementarity_max'],'validation_seconds':time.monotonic()-t,
            'scope':'Warm-start development continuation only; no statistical coverage or end-to-end estimator certification'}
    write(out/'RESULT.json',result);return result

def run():
    verify_inputs()
    with (HERE/'STARTED.json').open('x') as f:
        json.dump({'unix_time':time.time(),'labels':[31,43],'freeze_sha256':sha(HERE/'FREEZE.json')},f)
    results=[]
    for label in (31,43):
        start=time.monotonic();out=HERE/f'start{label}';out.mkdir();oracle=Oracle(label);x=oracle.x0.copy()
        save(out/'initial.pt',{'flat':tensor(x),'source_sha256':sha(HERE/f'inputs/initial{label}.pt')})
        def record(a,rec,last_trial):
            number=rec['iteration'];folder=out/f'iteration{number:04d}';folder.mkdir()
            rec['process_peak_RSS_bytes']=rss();rec['total_seconds_from_start']=time.monotonic()-start
            rec['record_semantics']='J/gradient/KKT/constraints at linearization base; state.pt is accepted next state or unchanged base. Accepted trial contains next J/constraints.'
            save(folder/'state.pt',{'flat':tensor(a),'accepted':rec['accepted'],'freeze_sha256':sha(HERE/'FREEZE.json')})
            if not rec['accepted'] and last_trial is not None:
                save(folder/'last_failed_trial.pt',{'flat':tensor(last_trial),'reason':'all recorded trials failed'})
            write(folder/'ITERATION.json',rec)
            print(json.dumps({'label':label,'iteration':number,'J_base':rec['J'],'KKT_base':rec['KKT']['stationarity_L2'],
                              'accepted_step':rec['accepted'],'radius':rec.get('radius_end'),
                              'seconds':time.monotonic()-start,'RSS':rss()}),flush=True)
        try:
            x,status=solve(oracle,kkt,record);write(out/'OPTIMIZER.json',status)
        except Exception as e:
            saved=sorted(out.glob('iteration*/state.pt'))
            if saved:x=torch.load(saved[-1],weights_only=True)['flat'].numpy()
            status={'status':'exception_no_retry','reason':repr(e),'traceback':traceback.format_exc(),
                    'counts':oracle.counts,'timers':oracle.timers}
            write(out/'FAILURE.json',status);write(out/'OPTIMIZER.json',status)
        oracle.p.set_model(tensor(x));save(out/'final.pt',{'flat':tensor(x),'state_dict':oracle.p.model.state_dict(),
                  'initial_sha256':sha(HERE/f'inputs/initial{label}.pt'),'freeze_sha256':sha(HERE/'FREEZE.json')})
        try:result=final_checks(oracle,x,out)
        except Exception as e:
            result={'label':label,'accepted':False,'validation_failure':repr(e),'traceback':traceback.format_exc()}
            write(out/'VALIDATION_FAILURE.json',result)
        result.update(total_seconds=time.monotonic()-start,process_peak_RSS_bytes=rss(),
                      all_evaluation_counts=oracle.counts,all_evaluation_timers=oracle.timers)
        result['wall_budget_pass']=result['total_seconds']<=7200
        result['accepted']=bool(result['accepted'] and result['wall_budget_pass'] and not (out/'FAILURE.json').exists())
        write(out/'RESULT.json',result);results.append(result);print('FINAL',json.dumps(result),flush=True)
        del oracle
    write(HERE/'RESULT.json',{'estimator':ESTIMATOR,'accepted_count':sum(r['accepted'] for r in results),
          'starts':results,'hash_verification':verify_inputs(),'historical_training':'NOT_RUN','AL':'NOT_RUN',
          'kernels':'NOT_RUN','other_datasets':'NOT_RUN','profiles':'NOT_RUN','calibration':'NOT_RUN','experimental':'NOT_RUN'})

def replay():
    hashes=verify_inputs();records=[]
    for label in (31,43):
        oracle=Oracle(label);out=HERE/f'start{label}';x=torch.load(out/'final.pt',weights_only=True)['flat'].numpy()
        saved=np.load(out/'FIRST_ORDER.npz',allow_pickle=False);v=oracle.values(x);ids=saved['working_indices']
        g,R,A=oracle.derivatives(x,ids);check=kkt(g,v['c'],A,ids,oracle.names)
        errors={'J':abs(v['J']-json.loads((out/'RESULT.json').read_text())['J']),
                'gradient_max_abs':float(np.max(abs(g-saved['original_gradient']))),
                'constraints_max_abs':float(np.max(abs(v['c']-saved['constraints']))),
                'Jacobian_max_abs':float(np.max(abs(A-saved['working_jacobian']),initial=0.))}
        records.append({'label':label,'J':v['J'],'KKT':check,'errors':errors,
                        'pass':bool(errors['J']<=1e-10 and errors['gradient_max_abs']<=1e-8 and errors['constraints_max_abs']<=1e-6 and errors['Jacobian_max_abs']<=1e-5)})
    result={'pass':all(r['pass'] for r in records),'records':records,'hashes':hashes,
            'scope':'Saved-state replay only; no optimizer, profiles or full-forward campaign'}
    print(json.dumps(result,indent=2));
    if not result['pass']:raise SystemExit(1)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run','replay']);args=ap.parse_args()
    {'freeze':freeze,'run':run,'replay':replay}[args.mode]()

"""Single locked two-start feasible-domain experiment; verify mode never fits."""
from domain import *
from optimizer import SETTINGS,run_al,necessary_kkt
from verify import derivative_record,independent_moment
import argparse,traceback

ESTIMATOR='NLO_feasible_domain_FiLM_AL_v1'

class Oracle:
    def __init__(self,p,d,out):
        self.p=p;self.d=d;self.n_constraints=len(d.names);self.out=out;self.calls=0
    def set_augmented(self,lam,rho):self.lam=tensor(lam);self.rho=rho
    def __call__(self,a,weights):
        x=tensor(a).detach().requires_grad_(True)
        j=self.p.objective(x);c=self.d.constraints(self.p,x)
        g=torch.autograd.grad(j,x,retain_graph=True)[0]
        lag=g if not np.any(weights) else torch.autograd.grad(j+(tensor(weights)*c).sum(),x)[0]
        return float(j.detach()),c.detach().numpy(),g.detach().numpy(),lag.detach().numpy()
    def augmented(self,a):
        x=tensor(a).detach().requires_grad_(True)
        comp=self.p.components(x);j=comp.sum();c=self.d.constraints(self.p,x)
        v=torch.relu(self.lam+self.rho*c)
        al=j+(v.square()-self.lam.square()).sum()/(2*self.rho)
        grad=torch.autograd.grad(al,x)[0];self.calls+=1
        finite=bool(torch.isfinite(al) and torch.isfinite(grad).all() and torch.isfinite(c).all())
        entry={'evaluation':self.calls,'finite':finite,'peak_RSS_bytes':rss()}
        if finite:entry.update(J=float(j.detach()),J_components=comp.detach().tolist(),
              constraint_values=c.detach().tolist(),AL_value=float(al.detach()),AL_gradient_L2=float(grad.norm()),penalty=self.rho)
        with (self.out/'EVALUATIONS.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        if not finite:
            save(self.out/f'failed_evaluation{self.calls}.pt',{'flat':x.detach(),'reason':'nonfinite'})
        return float(al.detach()),grad.detach().numpy()

def first_order(p,d,x,out):
    z=x.detach().requires_grad_(True);comp=p.components(z);j=comp.sum();c=d.constraints(p,z)
    g=torch.autograd.grad(j,z,retain_graph=True)[0]
    cn=c.detach().numpy();indices=np.flatnonzero(cn>=-SETTINGS['active_tolerance'])
    rows=[torch.autograd.grad(c[int(i)],z,retain_graph=True)[0].detach().numpy() for i in indices]
    A=np.stack(rows) if rows else np.empty((0,len(z)))
    kkt=necessary_kkt(g.detach().numpy(),cn,A,indices,d.names)
    np.savez_compressed(out/'FIRST_ORDER.npz',gradient=g.detach().numpy(),constraints=cn,
                        active_indices=indices,active_jacobian=A,
                        constraint_names=np.array(d.names))
    kkt.update(J=float(j.detach()),J_components=comp.detach().tolist(),
               original_coordinates=3867,estimator=ESTIMATOR,
               scope='Necessary KKT of the declared 0.95-margin differentiable filter; full independent gates additionally required and must be inactive. No global optimum claim.')
    write(out/'KKT.json',kkt);return kkt

def validate(p,d,x,out):
    t=time.monotonic();kkt=first_order(p,d,x,out)
    moments=moment_validation(p,x);write(out/'MOMENTS.json',moments)
    forward=full_forward(p,x,out/'full_forward');write(out/'FULL_FORWARD.json',forward)
    deriv=derivative_record(p,d,x,independent=True);write(out/'FINAL_DERIVATIVES.json',deriv)
    full_interior=all(max(np.array(r['CFF_estimated_total'])/np.array(r['CFF_limits']))<1-1e-8 and
                      max(r['field_totals'].values())<1e-3*(1-1e-8) for r in forward['records'])
    # Transport and warning checks have their own unchanged authoritative gates.
    transport_interior=True
    for r in forward['records']:
        detail=json.loads((out/'full_forward'/r['file']).read_text())
        for m in ('weak10','strong10'):
            transport_interior &= max(detail['checks'][m]['moments']['relative_errors'])<1e-4*(1-1e-8)
    gates={'declared_domain_KKT':kkt['pass'],'refined_and_new_moments':moments['pass'],
           'all_original_forward_checks':forward['pass'],'directional_derivatives':deriv['pass'],
           'authoritative_numeric_inequalities_inactive':bool(full_interior and transport_interior)}
    result={'label':p.label,'estimator':ESTIMATOR,'accepted':all(gates.values()),'gates':gates,
            'J':kkt['J'],'J_components':kkt['J_components'],'raw_gradient_L2':kkt['raw_gradient_L2'],
            'KKT_stationarity_L2':kkt['stationarity_L2'],'validation_seconds':time.monotonic()-t,
            'peak_RSS_bytes':rss(),'qualification':'Development formulation only; no calibration, coverage, global optimality or all-order admissibility claim.'}
    write(out/'RESULT.json',result);return result

def freeze():
    if (HERE/'FREEZE.json').exists():raise FileExistsError('Frozen contract already exists')
    assert json.loads((HERE/'MECHANICAL.json').read_text())['pass']
    assert 'Ran 16 tests' in (HERE/'logs/supplied_kkt_tests.log').read_text()
    assert (HERE/'logs/supplied_kkt_tests.log').read_text().strip().endswith('OK')
    assert 'Ran 5 tests' in (HERE/'logs/mechanics_tests.log').read_text()
    assert (HERE/'logs/mechanics_tests.log').read_text().strip().endswith('OK')
    contract={
      'estimator':ESTIMATOR,'prospective_new_estimator':True,'data':'Only saved d000 y and C; no controller/noise/truth values read',
      'initials':['inputs/initial31.pt','inputs/initial43.pt'],
      'model':'Original 3867-parameter width24 FiLM; fixed reference scale 2 GeV^2, original parity and envelope',
      'objective':'J=(pred-y)^T C^-1(pred-y)/18 + sum(((I-QQ^T)M48)/fixed_scales)^2/492; lambda=1',
      'moments':'All 42 permitted coefficients variable-projected; no fixed, removed or penalized coefficients',
      'feasible_domain':{'moment':'192-point-per-side split Gaussian quadrature; all allowed coefficients fitted on original41 xi, predictive RMS on original80 xi <=1e-3',
          'numerical_filter':'All15 xi/Q2; evolved-field degree8/10, weak/strong, inner72/output48 additive budgets, hard192/384, arithmetic reserve and spin2 transport, with 0.95 times original limits',
          'omissions_from_filter':'Independent adaptive hard error/warnings and ordered-time crosschecks are absent from differentiable filter; unchanged complete routine is authoritative',
          'full_domain':'Intersection of filter inequalities and ALL original full learned-field gates. Conservative numerical margin is explicit; not a physics prior or proof of error bounds',
          'stationarity':'Original-coordinate KKT of explicit filter at 1e-6 L2, signed multipliers, strict c<=0, complementarity<=1e-8, active threshold1e-6, active-row LICQ diagnostic; original complete constraints must be strictly inactive for omitted derivatives to have zero multipliers',
          'nonsmoothness':'Absolute values and Euclidean norms retain original form; AD uses standard subgradients at exact zero, which is inactive for these budgets. Active-row derivative tests and rank checks required'},
      'algorithm':{'name':'Powell-Hestenes-Rockafellar inequality augmented Lagrangian, deterministic L-BFGS-B inner solves','settings':SETTINGS,
          'multipliers':'lambda <- max(0,lambda+rho*c); rho multiplied by5 only when violation fails factor0.5 reduction, capped1e8',
          'continuation':'Same start, same theta and multiplier sequence; no retries or other algorithm',
          'return':'Last outer iterate; every outer/failed candidate retained. No truth-based or best-J selection',
          'full_checks_during_search':'At every sixth outer iterate that passes differentiable filter; always on final candidate',
          'refinement':'Moment192 vs independent384 at every outer; stop on difference>1e-7; no automatic quadrature escalation'},
      'budget':{'starts':2,'maximum_total_seconds_per_start':7200,'optimization_wall_seconds':6600,'reserved_validation_seconds':600,'outer_iterations':24,'inner_iterations':80,'inner_evaluation_hard_cap':160},
      'independent_validation':'160 xi=.02+.96*(j+.25)/160, disjoint from41 and80; moments96/192/384 and combined<=1e-3; spins10/12 diagnostics; no final-truth evaluation',
      'forbidden':'No new draws/datasets, kernels, LO campaigns, experimental fits, target profiles, calibration, symbolic pruning, architecture change, retuning',
      'claim':'Bounded development formulation; a failed run stays failed; the old rejected gate remains immutable'}
    write(HERE/'CONTRACT.json',contract)
    # Preserve every existing source/evidence hash from the previous gate.
    prior=json.loads((ROOT/'completion_runs/NLO_noisy_feasibility/SOURCE_HASHES.json').read_text())['sources']
    external={}
    for rel,rec in prior.items():
        h=rec['sha256'] if isinstance(rec,dict) else rec
        assert sha(ROOT/rel)==h,rel
        external[rel]=h
    for folder in ('GPD_NLO_resolution_review/pilot','GPD_NLO_noisy_feasibility_review'):
        for p in (ROOT/folder).glob('*.py'):external[str(p.relative_to(ROOT))]=sha(p)
    for p in (common.WORK/'rows').glob('*.npz'):external[str(p.relative_to(ROOT))]=sha(p)
    for label in (31,43):
        p=ROOT/f'completion_runs/NLO_noisy_feasibility/datasets/d000/start{label}/initial.pt'
        external[str(p.relative_to(ROOT))]=sha(p)
    write(HERE/'EXTERNAL_HASHES.json',external)
    files={str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*.py')}
    for name in ('CONTRACT.json','INPUTS.json','EXTERNAL_HASHES.json','MECHANICAL.json','CONTRACTION_BUILD.json'):
        files[name]=sha(HERE/name)
    for p in (HERE/'inputs').iterdir():files[str(p.relative_to(HERE))]=sha(p)
    write(HERE/'FREEZE.json',{'estimator':ESTIMATOR,'before_optimization':True,'unix_time':time.time(),'files':files})
    print('FROZEN',sha(HERE/'FREEZE.json'),flush=True)

def verify_hashes():
    frozen=json.loads((HERE/'FREEZE.json').read_text())
    for rel,h in frozen['files'].items():assert sha(HERE/rel)==h,rel
    external=json.loads((HERE/'EXTERNAL_HASHES.json').read_text())
    for rel,h in external.items():assert sha(ROOT/rel)==h,rel
    return len(frozen['files']),len(external)

def run():
    verify_hashes()
    with (HERE/'STARTED.json').open('x') as f:
        json.dump({'unix_time':time.time(),'freeze_sha256':sha(HERE/'FREEZE.json'),'only_starts':[31,43]},f)
    results=[]
    for label in (31,43):
        t=time.monotonic();out=HERE/f'start{label}';out.mkdir()
        p=Problem(label);d=Domain();x=p.x0.clone()
        save(out/'initial.pt',{'flat':x,'state_dict':p.model.state_dict(),'input_sha256':sha(HERE/f'inputs/initial{label}.pt')})
        oracle=Oracle(p,d,out)
        def record(a,rec,failed):
            xx=tensor(a);idx=rec['outer'];loc=out/f'outer{idx:02d}';loc.mkdir()
            save(loc/'state.pt',{'flat':xx,'multipliers':tensor(rec['AL_multipliers']),
                                'penalty':rec['penalty'],'freeze_sha256':sha(HERE/'FREEZE.json')})
            if failed is not None:save(loc/'failed_trial.pt',{'flat':tensor(failed),'reason':rec['inner']['message']})
            with torch.no_grad():
                comp=p.components(xx).tolist();m192=float(d.mom.value(p,xx))
            m384=independent_moment(p,xx)
            rec.update(J_components=comp,peak_RSS_bytes=rss(),moment192=m192,moment384=m384,
                       moment_refinement_absolute=abs(m192-m384),seconds_from_start=time.monotonic()-t)
            if rec['strictly_feasible'] and (idx+1)%6==0:
                full=full_forward(p,xx,loc/'full_forward');write(loc/'FULL_FORWARD.json',full)
                rec['full_forward_pass']=full['pass']
            write(loc/'ITERATION.json',rec)
            print(json.dumps({'label':label,'outer':idx,'J':rec['J'],'max_c':max(rec['constraint_values']),
                              'gradient':rec['raw_gradient_L2'],'AL_KKT':rec['AL_stationarity_L2'],
                              'seconds':time.monotonic()-t,'RSS':rss()}),flush=True)
            if abs(m192-m384)>1e-7:raise RuntimeError('moment_constraint_quadrature_not_converged')
        try:
            a,status=run_al(x.numpy(),oracle,record);x=tensor(a)
            write(out/'OPTIMIZER.json',status)
        except Exception as e:
            # Do not retry or change the algorithm. Recover the last preserved outer candidate.
            states=sorted(out.glob('outer*/state.pt'))
            if states:x=torch.load(states[-1],weights_only=True)['flat']
            write(out/'FAILURE.json',{'reason':repr(e),'traceback':traceback.format_exc(),'seconds':time.monotonic()-t})
        p.set_model(x);save(out/'final.pt',{'flat':x,'state_dict':p.model.state_dict(),'freeze_sha256':sha(HERE/'FREEZE.json')})
        try:
            result=validate(p,d,x,out)
        except Exception as e:
            result={'label':label,'accepted':False,'validation_failure':repr(e),'traceback':traceback.format_exc()}
        result.update(total_seconds=time.monotonic()-t,peak_RSS_bytes=rss())
        result['wall_budget_pass']=result['total_seconds']<=7200
        result['accepted']=bool(result['accepted'] and result['wall_budget_pass'] and not (out/'FAILURE.json').exists())
        write(out/'RESULT.json',result);results.append(result)
        print('FINAL',json.dumps(result),flush=True)
        del oracle,p,d
    counts=verify_hashes()
    write(HERE/'RESULT.json',{'estimator':ESTIMATOR,'starts':results,
          'accepted_count':sum(r['accepted'] for r in results),'original_sources_unchanged':True,
          'hashes_verified':counts,'other_datasets':'NOT_RUN','profiles':'NOT_RUN','calibration':'NOT_RUN'})

def replay(label):
    verify_hashes();p=Problem(label);d=Domain();out=HERE/f'start{label}'
    x=torch.load(out/'final.pt',weights_only=True)['flat']
    # Read-only reproduction of stored first-order and objective/primal values.
    dat=np.load(out/'FIRST_ORDER.npz',allow_pickle=False)
    xx=x.clone().requires_grad_(True);j=p.objective(xx);g=torch.autograd.grad(j,xx)[0]
    with torch.no_grad():c=d.constraints(p,x).numpy()
    result={'label':label,'J':float(j.detach()),'original_gradient_L2':float(g.norm()),
            'maximum_constraint_excess':float(np.maximum(c,0).max()),
            'saved_gradient_max_abs_difference':float(np.max(abs(g.numpy()-dat['gradient']))),
            'saved_constraint_max_abs_difference':float(np.max(abs(c-dat['constraints']))),
            'saved_result':json.loads((out/'RESULT.json').read_text()),'no_optimization':True}
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--run',action='store_true')
    ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    if sum((a.freeze,a.run,a.verify))!=1:ap.error('Choose exactly one mode')
    if a.freeze:freeze()
    elif a.run:run()
    else:
        replay(31);replay(43)

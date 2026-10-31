"""Independent local audit of the returned P2B saved states; no fitting or NLO build.

Run with /Users/dustin/miniforge3/bin/python -B. Writes this review directory only.
The frozen model/truth/metric definitions are reused after hash verification.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import csv
import io
import json
from pathlib import Path
import resource
import sys
import time
from zipfile import ZipFile
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
PACKET = PROJECT/'review_GPD_P2B_exact_center_evaluation_v1'
CAP = PROJECT/'review_GPD_P1_advice/reference_capsule'
CAP_RETURN = PROJECT/'review_GPD_P2A_preservation_results_v1/GPD_P1_Rivanna'
PREP = PROJECT/'GPD_P2B_preparation_v3'
STUDY = CAP/'project/GPD_three_stage_closure'
RESULT = PACKET/'results/GPD_P2B_evaluation_20261028_a01/GPD_P2B_exact_center_s31'
CONTROL = PACKET/'control/GPD_P2B_evaluation_20261028_a01'
sys.path[:0] = [str(STUDY), str(STUDY/'vendor')]
from gpd_reh.model import ExperimentalFiLM
from closure.truths import Truth
from closure.metrics import field_metrics, split_grid

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def verify(root, entries):
    for entry in entries:
        p = Path(entry['path'])
        assert not p.is_absolute() and '..' not in p.parts
        assert (root/p).stat().st_size == entry['bytes'], str(p)
        assert sha(root/p) == entry['sha256'], str(p)
    return len(entries)

def checkpoint(path):
    # Only the NumPy constructors needed by these states are allowlisted.
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray,
                                          np.dtype, type(np.dtype('float64'))]):
        return torch.load(path, map_location='cpu', weights_only=True)

def same(a, b):
    if isinstance(a, (np.ndarray, torch.Tensor)):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

def main():
    start = time.monotonic()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    entries = read(PACKET/'RETAINED_MANIFEST.json')['files']
    payloads = verify(PACKET, entries)
    actual = {str(p.relative_to(PACKET)) for p in PACKET.rglob('*') if p.is_file()}
    assert actual == {r['path'] for r in entries} | {'RETAINED_MANIFEST.json'}
    archive = PROJECT/(PACKET.name+'.zip')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == payloads+1
        for member in z.namelist():
            assert z.read(member) == (PACKET/member).read_bytes()
    assert sha(PACKET/'GPD_P2B_preparation_v3.zip') == sha(PROJECT/'GPD_P2B_preparation_v3.zip')
    prep_count = verify(PREP, read(PREP/'MANIFEST.json')['files'])
    driver_count = verify(PACKET/'P2B_execution', read(PACKET/'P2B_execution/MANIFEST.json')['files'])
    eval_count = verify(PACKET/'P2B_evaluation', read(PACKET/'P2B_evaluation/MANIFEST.json')['files'])
    lock = read(RESULT/'RUN_LOCK.json')
    assert sha(CAP/'MANIFEST.json') == lock['capsule_manifest_sha256']
    assert sha(PREP/'MANIFEST.json') == lock['packet_manifest_sha256']
    assert sha(PACKET/'P2B_execution/MANIFEST.json') == lock['driver_manifest_sha256']
    original = {r['path']: r for r in read(CAP/'MANIFEST.json')['files']}
    source_rel = ['project/GPD_three_stage_closure/'+s for s in [
        'vendor/gpd_reh/model.py', 'vendor/gpd_reh/reference_model.py',
        'closure/truths.py', 'closure/geometry.py', 'closure/metrics.py',
        'prepared/FIT_CONFIG.json', 'prepared/GEOMETRY.json',
        'prepared/observed/exp_Re_exact_Z1.npz', 'runs/exp_Re_exact_Z1_s31/final.pt',
        'runs/exp_Re_exact_Z1_s31/RESULT.json']]
    source_rel += ['reference/FRESH_SEED31.npy', 'runtime/core.py', 'runtime/finalize.py', 'runtime/resume_support.py']
    def original_path(s):
        return next(root/s for root in (CAP, CAP_RETURN, PACKET/'GPD_P1_Rivanna') if (root/s).exists())
    for s in source_rel:
        f = original_path(s)
        assert f.stat().st_size == original[s]['bytes'] and sha(f) == original[s]['sha256']
    source_bindings = {str(original_path(s).relative_to(PROJECT)): sha(original_path(s)) for s in source_rel}
    obs_path = original_path('project/GPD_three_stage_closure/prepared/observed/exp_Re_exact_Z1.npz')
    initial_path = original_path('reference/FRESH_SEED31.npy')
    assert sha(obs_path) == lock['observed_sha256']
    assert sha(initial_path) == lock['initial_file_sha256']
    assert sha(PREP/'inputs/CENTRAL_FORWARD.npz') == lock['PDF_input_sha256']
    base_result = read(original_path('project/GPD_three_stage_closure/runs/exp_Re_exact_Z1_s31/RESULT.json'))
    assert sha(PROJECT/'GPD_P2A_preservation_UVA_v1/evidence/EXACT_BASELINE_RESULT.json') == sha(PACKET/'GPD_P1_Rivanna/project/GPD_three_stage_closure/runs/exp_Re_exact_Z1_s31/RESULT.json')
    assert sha(STUDY/'runs/exp_Re_exact_Z1_s31/final.pt') == base_result['scored_checkpoint_sha256']

    result, scores = read(RESULT/'RESULT.json'), read(RESULT/'SCORES.json')
    selected = result['selected']
    assert result['status'] == 'FINITE_BUDGET_FEASIBLE'
    assert selected['pass'] and selected['constraints_finite'] and selected['constraint_exceedances'] == 0
    assert selected['moments']['pass'] and selected['full_operator']['pass']
    assert sha(RESULT/'final.pt') == sha(RESULT/'STAGE60.pt') == result['scored_checkpoint_sha256']
    assert sha(RESULT/'RUN_LOCK.json') == result['run_lock_sha256']
    assert sha(RESULT/'SCORES.json') == result['scores_sha256']
    assert sha(RESULT/'FORWARD_SCORES.json') == result['forward_scores_sha256']
    assert sha(RESULT/'SELECTION.json') == scores['selection_sha256']
    forward = read(RESULT/'FORWARD_SCORES.json')
    assert forward['selection_sha256'] == scores['selection_sha256']
    for row in selected['full_operator']['coordinate_selected_rows']:
        assert row['pass'] and sha(RESULT/row['file']) == row['sha256']
    retained_count = verify(RESULT, read(RESULT/'TERMINAL_RETENTION.json')['files'])
    stage_manifest = read(CONTROL/'PARENT_STAGE_MANIFEST.json')['files']
    for row in stage_manifest:
        f = RESULT/row['destination']
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256']
    protected = read(RESULT/'NO_OPTIMIZATION_VERIFICATION.json')
    assert protected['pass']
    for name, digest in protected['protected_files'].items():
        assert sha(RESULT/name) == digest
    for name in ['SOLVER_DIAGNOSTICS.json', 'OPTIMIZER_COMPLETE.json']:
        assert sha(RESULT/name) == sha(CONTROL/('RECONSTRUCTED_'+name))
    supervisor = read(RESULT/'SUPERVISOR_22117473.json')
    assert supervisor['child_exit'] == 0 and supervisor['terminal_retention_verified']
    assert not any(supervisor[k] for k in ['resource_stop','monitor_error','cleanup_error','terminal_retention_error','provisional_copy_errors'])
    qualification = read(RESULT/'parent_attempt/preflight/QUALIFICATION.json')
    assert qualification['status'] == 'PASS'
    test_counts = dict(policy=8, continuation=3, operations=7, launch=8)
    for group in qualification['groups']:
        script = group['script']
        assert group['exit_code'] == 0 and sha(PACKET/'P2B_execution'/script) == group['source_sha256']
        for stream in ['stdout','stderr']:
            assert sha(RESULT/'parent_attempt/preflight'/(script+'.'+stream)) == group[stream+'_sha256']
        label = script.removeprefix('test_').removesuffix('.py')
        assert f"Ran {test_counts[label]} tests" in (RESULT/'parent_attempt/preflight'/(script+'.stderr')).read_text()
    eq = read(RESULT/'evaluation_preflight/QUALIFICATION.json')
    assert eq['status'] == 'PASS' and eq['script_sha256'] == sha(PACKET/'P2B_evaluation/test_evaluation.py')
    for stream in ['stdout','stderr']:
        assert sha(RESULT/'evaluation_preflight'/(stream+'.log')) == eq[stream+'_sha256']
    assert 'Ran 10 tests' in (RESULT/'evaluation_preflight/stderr.log').read_text()

    state = checkpoint(RESULT/'SOLVER_RESUME.pt')
    adam = checkpoint(RESULT/'ADAM_RESUME.pt')
    latest = checkpoint(RESULT/'LATEST_STATE.pt')
    assert adam['completed_updates'] == 400 and adam['remaining_updates'] == 0
    assert state['next_iteration'] == 60 and state['remaining_outer_iterations'] == 0
    assert latest['completed_updates'] == 60 and same(latest['flat'], state['x'])
    assert same(checkpoint(RESULT/'final.pt')['flat'], state['x'])
    for s in [state, adam, latest]:
        assert s['run_lock_sha256'] == sha(RESULT/'RUN_LOCK.json')
    journal = [json.loads(line) for line in (RESULT/'ITERATIONS.jsonl').read_text().splitlines()]
    diag = read(RESULT/'SOLVER_DIAGNOSTICS.json')
    assert len(journal) == len(state['history']) == len(diag['history']) == 60
    assert same(state['history'], diag['history'])
    assert [r['iteration'] for r in journal] == list(range(1,61))
    assert all(r['accepted'] == h['accepted'] for r,h in zip(journal, state['history']))
    assert same(state['oracle_counts'], diag['oracle_counts'])
    assert state['oracle_counts']['evaluations'] == 121 < 12000
    opt = read(RESULT/'OPTIMIZER_COMPLETE.json')
    assert opt == result['optimization']
    candidates = opt['candidates_by_supplied_J']
    assert candidates == sorted(candidates, key=lambda r: (r['J'],31,r['stage']))
    assert candidates[0]['stage'] == selected['stage'] == 60
    for row in candidates:
        assert sha(RESULT/row['path']) == row['sha256']
        assert checkpoint(RESULT/row['path'])['J'] == row['J']
    assert np.isclose(state['values']['J'], selected['J'], rtol=1e-12, atol=1e-14)

    cfg = read(STUDY/'prepared/FIT_CONFIG.json')
    model = ExperimentalFiLM(cfg['network']['t_mean'], cfg['network']['t_std']).double().eval()
    truth = Truth('Z1')
    geometry = read(STUDY/'prepared/GEOMETRY.json')
    with np.load(obs_path, allow_pickle=False) as z:
        obs = {k:z[k].copy() for k in z.files}
    with np.load(PREP/'inputs/CENTRAL_FORWARD.npz', allow_pickle=False) as z:
        info = {k:z[k].copy() for k in z.files}
    xpdf, central, cov = info['x'], info['observed'], info['covariance']
    assert np.allclose(xpdf, np.geomspace(.02,.8,12), rtol=0, atol=1e-15)
    assert np.allclose(central, truth.values(xpdf,0.,0.), rtol=0, atol=1e-15)
    scale = .1*np.maximum(abs(central), .05*np.max(abs(central), axis=0))
    kx = np.exp(-abs(np.log(xpdf)[:,None]-np.log(xpdf)[None,:])/.7)
    kc = np.array([[1.,.5,0.],[.5,1.,0.],[0.,0.,1.]])
    rebuilt = np.kron(kx,kc)*scale.ravel()[:,None]*scale.ravel()[None,:]
    assert np.allclose(rebuilt,cov,rtol=1e-14,atol=1e-16)
    chol = np.linalg.cholesky(cov)
    grids = [split_grid(float(xi),160) for xi in obs['xi']]
    xs, ws = np.array([x for x,w in grids]), np.array([w for x,w in grids])
    tf = truth.values(xs,obs['xi'][:,None],obs['t'][:,None])
    th = (2/9)*(tf[:,:,0]+tf[:,:,1])

    def load(f):
        ck = checkpoint(f)
        flat = ck['flat'].detach().double()
        assert flat.shape == (3867,) and torch.isfinite(flat).all()
        params, offset = {}, 0
        for name,p in model.named_parameters():
            params[name] = flat[offset:offset+p.numel()].reshape(p.shape)
            assert torch.equal(params[name],ck['state_dict'][name])
            offset += p.numel()
        assert offset == 3867
        model.load_state_dict(params)
        return ck

    def evaluate(x,xi,t):
        with torch.no_grad():
            return model(torch.tensor(np.asarray(x)[None]),torch.tensor([float(xi)]),
                         torch.tensor([float(t)]))[0].numpy().T

    def observed():
        chunks = []
        with torch.no_grad():
            for lo in range(0,160,8):
                p = model(torch.tensor(xs[lo:lo+8]),torch.tensor(obs['xi'][lo:lo+8]),
                          torch.tensor(obs['t'][lo:lo+8])).numpy()
                chunks.append((2/9)*(p[:,0]+p[:,1]))
        ph = np.concatenate(chunks)
        return {key:float(np.sqrt(np.sum(ws[:,s]*(ph[:,s]-th[:,s])**2)/np.sum(ws[:,s]*th[:,s]**2)))
                for key,s in [('all',slice(None)),('ERBL',slice(0,160)),('DGLAP',slice(160,None))]}

    def targets(evaluator):
        z,w = np.polynomial.legendre.leggauss(384)
        ans = []
        for lo,hi,is_a in [(.2,.4,True),(0.,1.,False)]:
            x = (lo+hi)/2+(hi-lo)*z/2
            f = evaluator(x,.275,-.363); h=(2/9)*(f[:,0]+f[:,1])
            ans.append(float(np.sum(w*(hi-lo)/2*h*(5 if is_a else x))))
        return np.array(ans)

    def boundary():
        pred = evaluate(xpdf,0.,0.)
        residual = np.linalg.solve(chol,(pred-central).reshape(-1))
        grid = np.linspace(.02,.8,201)
        pg,tg = evaluate(grid,0.,0.),truth.values(grid,0.,0.)
        rel = np.sqrt(np.sum((pg-tg)**2,axis=0)/np.sum(tg**2,axis=0))
        regions = {}
        for name,lo,hi in [('low_x_tail',0.,.02),('observed_support',.02,.8),('high_x_tail',.8,1.)]:
            orders={}
            for order in (80,160):
                z,w=np.polynomial.legendre.leggauss(order);x=(lo+hi)/2+(hi-lo)*z/2;w=w*(hi-lo)/2
                f,tr=evaluate(x,0.,0.),truth.values(x,0.,0.)
                h,ht=(2/9)*(f[:,0]+f[:,1]),(2/9)*(tr[:,0]+tr[:,1])
                orders[str(order)]=dict(forward_Hplus_first_moment=float(w@(x*h)),
                    truth_forward_Hplus_first_moment=float(w@(x*ht)),
                    integrated_squared_Hplus_error=float(w@((h-ht)**2)),
                    integrated_squared_truth_Hplus=float(w@(ht**2)))
            regions[name]=orders
        return dict(whitened_RMS=float(np.sqrt(np.mean(residual**2))),chi2_PDF=float(residual@residual),
                    J_PDF=float(residual@residual)/160,channel_relative_L2=rel.tolist(),
                    support_and_tails=regions),pred,residual,pg

    true_targets=targets(truth.values)
    assert np.max(abs(true_targets-np.array([.3619108615320205,.1105532776465300])))<1e-14
    rows, discrepancies = [], []
    baseline_scores=read(PROJECT/'GPD_P2A_preservation_UVA_v1/evidence/EXACT_BASELINE_SCORES.json')
    baseline_observed=read(PROJECT/'GPD_P2A_preservation_review_20261026/REVIEW_EVIDENCE.json')['selected'][0]['observed']
    cases=[('exact_baseline',60,STUDY/'runs/exp_Re_exact_Z1_s31/final.pt')]
    cases += [('P2B',n,RESULT/f'STAGE{n}.pt') for n in (0,20,40,60)]
    for name,n,f in cases:
        ck=load(f)
        if name=='P2B':
            assert ck['lambda_stab']==0. and ck['run_lock_sha256']==sha(RESULT/'RUN_LOCK.json')
            if n==0:
                assert np.array_equal(ck['flat'].numpy(),np.load(initial_path,allow_pickle=False))
        om=observed(); chart=field_metrics(evaluate,truth,geometry); tv=targets(evaluate)
        error=tv/true_targets-1
        pdf,pred,residual,pg=boundary()
        row=dict(case=name,stage=n,checkpoint=str(f.relative_to(PROJECT)),J_recorded=ck['J'],observed=om,
                 wider_chart=chart,targets=tv.tolist(),target_signed_relative_error=error.tolist(),boundary=pdf,
                 gates=dict(observed_Hplus=om['all']<=.02,target_A=bool(abs(error[0])<=.02),
                            target_B=bool(abs(error[1])<=.02),PDF_whitened_RMS=pdf['whitened_RMS']<=1.))
        if name=='P2B':
            assert abs(pdf['J_PDF']-ck['named_components']['J_PDF'])<1e-10
        expected=scores if name=='P2B' and n==60 else baseline_scores if name=='exact_baseline' else None
        if expected:
            expobs=scores['observed_slices']['relative_L2'] if name=='P2B' else baseline_observed
            discrepancies += [abs(om[k]-expobs[k]) for k in om]
            discrepancies += [abs(chart['charge_relative_L2'][k]-expected['field']['charge_relative_L2'][k]) for k in om]
            discrepancies += list(abs(np.array(chart['internal_channel_relative_L2'])-expected['field']['internal_channel_relative_L2']))
            expt=scores['target_384'] if name=='P2B' else base_result['selected']['target_384']
            discrepancies += list(abs(tv-expt))
        if name=='P2B' and n==60:
            discrepancies += [abs(pdf['whitened_RMS']-forward['observed_samples']['whitened_RMS'])]
            discrepancies += [float(np.max(abs(pred-forward['observed_samples']['prediction']))),
                              float(np.max(abs(residual-forward['observed_samples']['whitened_residual']))),
                              float(np.max(abs(pg-forward['supported_grid']['prediction'])))]
            for region,orders in pdf['support_and_tails'].items():
                for order,vals in orders.items():
                    discrepancies += [abs(v-forward['support_and_tails'][region][order][k]) for k,v in vals.items()]
            assert all(row['gates'].values())
        rows.append(row)
        print(name,n,'PASS','Hplus=',om['all'],'PDF_RMS=',pdf['whitened_RMS'],flush=True)
    assert max(discrepancies)<1e-10
    # The very small PDF residual is expected for this noiseless central control.
    names=result['named_components']
    assert abs(sum(selected['components'])-names['J_joint'])<1e-15
    assert abs(names['J_CFF']-scores['CFF']['input_matching']['RMS_original_sigma']**2)<1e-15
    assert abs(names['chi2_PDF']/160-names['J_PDF'])<1e-15
    assert abs(names['J_P1']+names['J_PDF']-names['J_joint'])<1e-15

    parent=read(PACKET/'control/GPD_P2B_exact_center_20261027_a01/hpc_terminal_verification_20261028T170240Z.json')
    final=read(CONTROL/'hpc_terminal_verification.json')
    spec=read(PACKET/'control/GPD_P2B_exact_center_20261027_a01/RUN_SPEC.json')
    missing_science=[p for p in spec['files'] if '/control/' in p or '/submit/' in p]
    def elapsed(text):
        answer=0.
        for part in text.split(':'):
            answer=60*answer+float(part)
        return answer
    accounting=list(csv.DictReader(io.StringIO(final['accounting']['stdout']),delimiter='|'))
    evaluation_job=next(r for r in accounting if r['JobID']=='22117473')
    assert evaluation_job['State']=='COMPLETED' and evaluation_job['ExitCode']=='0:0'
    assert parent['job']['state']=='FAILED' and parent['job']['exit_code']=='75:0'
    pwall,pcpu=elapsed(parent['job']['elapsed']),elapsed(parent['job']['total_cpu'])
    ewall,ecpu=elapsed(evaluation_job['Elapsed']),elapsed(evaluation_job['TotalCPU'])
    costs=dict(parent_job='22081808',parent_elapsed_seconds=pwall,parent_TotalCPU_seconds=pcpu,
               evaluation_job='22117473',evaluation_elapsed_seconds=ewall,evaluation_TotalCPU_seconds=ecpu,
               total_elapsed_seconds=pwall+ewall,total_CPU_seconds=pcpu+ecpu,
               parent_batch_MaxRSS_GiB=int(parent['job']['batch_max_rss'][:-1])/1024**2,
               peak_process_RSS_bytes=opt['peak_RSS_bytes'],solver_seconds=diag['seconds'],
               norm_subproblem_seconds=opt['binding_stats']['norm_subproblem_seconds'],
               norm_subproblem_fraction=opt['binding_stats']['norm_subproblem_seconds']/diag['seconds'],
               residual_jacobian_seconds=opt['oracle_timers']['residual_jacobian_seconds'],
               CPU_efficiency_fraction=pcpu/(int(parent['job']['cpus'])*pwall),gpus=0,
               includes_prior_profiling_jobs=False)
    verification=dict(status='PASS_RETURNED_PAYLOAD_AND_SAVED_FIELD_REPLAY',archive_sha256=sha(archive),
        payload_files_verified=payloads,result_retention_files_verified=retained_count,
        unchanged_staged_artifacts_verified=len(stage_manifest),prep_files_verified=prep_count,
        execution_source_files_verified=driver_count,evaluation_source_files_verified=eval_count,
        qualification_logs_verified=dict(execution_fixture_tests=26,evaluation_fixture_tests=10,rerun_locally=False,
                                         scope='Tiny software fixtures, not full model/derivative qualification'),
        original_local_source_bindings=source_bindings,missing_acceleration_manifest=lock['acceleration_manifest_sha256'],
        omitted_parent_control_and_submit_files=missing_science,
        limitation='Acceleration overlay, its CPU/GPU parity/performance review, parent readiness and submission script are hash-bound but absent; not independently audited. Full NLO/constraints and optimizer trajectory not rerun.')
    write('PACKET_VERIFICATION.json',verification)
    evidence=dict(status='PASS',scientific_decision='ACCEPT_EXACT_CENTER_PRESERVATION',development_released=False,
        archive_sha256=sha(archive),selected=rows[-1],baseline=rows[0],stages=rows[1:],truth_targets=true_targets.tolist(),
        max_replayed_metric_absolute_difference=max(discrepancies),
        covariance=dict(rank=int(np.linalg.matrix_rank(cov)),condition_number=float(np.linalg.cond(cov)),formula_replay=True),
        continuation=dict(completed_Adam_updates=400,completed_outer_updates=60,solver_evaluations=121,
                          history_consistency=True,additional_optimization=0,stage60_equals_final_and_terminal_state=True),
        last_prefinal_KKT=dict(iteration=diag['history'][-1]['iteration'],KKT=diag['history'][-1]['KKT']),
        numerical_eligibility=dict(reported_status=result['status'],reported_constraint_max=selected['constraint_max'],
                                  reported_moments=selected['moments'],reported_full_operator=selected['full_operator'],replayed_locally=False),
        costs=costs,seconds=time.monotonic()-start,review_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        environment=dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__)),
        scope='Saved-field/target/forward-covariance replay, hashes, checkpoint history, selection, objective arithmetic and returned cost receipts. No optimizer, NLO bank, new noise or cluster operation.')
    write('REVIEW_EVIDENCE.json',evidence)
    print('PASS max discrepancy',max(discrepancies),'seconds',evidence['seconds'],flush=True)

if __name__=='__main__':
    main()

"""Future source-locked campaign controller. Default mode is bookkeeping only.

No registered truth, noise stream, or fitter is evaluated at module import.
The current preparation invokes only --bookkeeping-only.
"""
from pathlib import Path
import sys,json,hashlib,argparse,math,dataclasses,subprocess
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]
sys.path.insert(0,str(ROOT/'GPD_direct_error_calibration'))
import direct_error as score
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(d):return json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False)
def roster():
 return [(phase,truth,i,('%s%03d'%('c' if phase=='calibration' else 't',2*i+j))) for phase,n in [('calibration',80),('test',240)] for i in range(n) for j,truth in enumerate(('A','B'))]
def bookkeeping():
 c=score.Contract('ANALYTIC_STUB_ESTIMATOR','ANALYTIC_STUB_GEOMETRY');pairs=[]
 for phase,truth,i,ident in roster():
  if phase=='calibration':
   fit=score.Fit(ident,'stub_input_'+ident,c.hash,(.05*i*(1 if truth=='A' else 2),.025*i))
   pairs.append(score.Pair(fit,truth,(0.,0.),phase))
 q=score.freeze_cutoff(c,pairs);assert q.q==144. and all(v['rank']==73 for v in q.per_stratum.values())
 changed=list(pairs);first=changed[0];changed[0]=score.Pair(score.Fit(first.fit.dataset_id,first.fit.input_hash,c.hash,None,'analytic_stub_failure'),first.stratum,first.truth,first.phase)
 failed=score.freeze_cutoff(c,changed);assert failed.q is None
 test=score.Fit('stub_test','stub_test_input',c.hash,(0.,0.))
 assert all(v is None for v in score.intervals(c,failed,test)['bounds'].values())
 fitfail=score.Fit('stub_test_fail','stub_test_fail_input',c.hash,None,'analytic_stub_failure')
 assert score.intervals(c,q,fitfail)['status']=='fit_failure'
 rr=roster();assert len(rr)==640 and len(set(x[3] for x in rr))==640
 result={'pass':True,'scope':'Deterministic analytic bookkeeping stubs only; no GPD truths, noise, or fits','counts':{'calibration_per_stratum':80,'test_per_stratum':240,'total':640},'rank':73,'common_max_verified':True,'failure_full_line_verified':True,'calibration_abstention_infinite_verified':True,'final_truths_evaluated':False,'final_noise_drawn':False}
 result['driver_sha256']=sha(Path(__file__))
 (PREP/'results/CAMPAIGN_BOOKKEEPING.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
def campaign(qualification,obligations,out):
 # Future invocation only, after scientific review and authorization.
 import os
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
 import numpy as np
 import torch
 torch.set_num_interop_threads(1)
 from environment_gate import verify
 actual_environment=verify()
 from development_controller import verify_lock
 verify_lock();ready=json.loads((PREP/'FINAL_READY_CONTRACT.json').read_text())
 if ready['status']!='READY_FOR_CALIBRATION_REVIEW':raise RuntimeError('Preparation is incomplete')
 qf=Path(qualification).resolve();qualified=json.loads(qf.read_text())
 if qualified['status']!='PASS':raise RuntimeError('Final-truth qualification has not passed')
 if qualified['estimator_lock_sha256']!=sha(PREP/'ESTIMATOR_LOCK.json'):raise ValueError('Qualification estimator mismatch')
 if qualified['truth_definition_sha256']!=sha(ROOT/'completion_runs/01/inference_contract.json'):raise ValueError('Final-truth definition mismatch')
 obligations=json.loads(Path(obligations).read_text())
 if obligations['estimator_lock_sha256']!=sha(PREP/'ESTIMATOR_LOCK.json'):raise ValueError('Control binding mismatch')
 for name in ('three_spin','truth_A_Dterm_counterfactual','training_bank_weak_direction'):
  control=obligations[name]
  if control['status'] not in ('PASS','PROSPECTIVELY_WAIVED_BY_REVIEW'):raise RuntimeError('Outstanding original control: '+name)
  path=Path(control['evidence_path']);assert sha(path)==control['evidence_sha256']
 registry=json.loads((PREP/'FGDEV_REGISTRY.json').read_text());geometry_hash=registry['geometry_sha256']
 c=score.Contract(sha(PREP/'ESTIMATOR_LOCK.json'),geometry_hash)
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 (out/'CONTRACT.json').write_text(json.dumps(dataclasses.asdict(c),indent=2)+'\n')
 (out/'ACTUAL_ENVIRONMENT.json').write_text(json.dumps(actual_environment,indent=2)+'\n')
 (out/'QUALIFICATION_BINDING.json').write_text(json.dumps({'qualification_path':str(qf),'qualification_sha256':sha(qf),'original_controls':obligations,'driver_sha256':sha(Path(__file__))},indent=2)+'\n')
 truths={}
 for truth in ('A','B'):
  entry=qualified['truths'][truth];path=Path(entry['path']);assert sha(path)==entry['sha256']
  with np.load(path,allow_pickle=False) as z:
   truths[truth]=(z['mean'].copy(),z['targets'].copy())
  mean,targets=truths[truth]
  assert mean.shape==(30,) and targets.shape==(2,) and np.isfinite(mean).all() and np.isfinite(targets).all()
 with np.load(PREP/'inputs/fit_bank.npz',allow_pickle=False) as z:ids=z['row_ids'].copy()
 pairs={'calibration':[],'test':[]};cutoff=None
 for phase,truth,index,ident in roster():
  if phase=='test' and cutoff is None:
   cutoff=score.freeze_cutoff(c,pairs['calibration'])
   (out/'CUTOFF.json').write_text(json.dumps(dataclasses.asdict(cutoff),indent=2)+'\n')
  key={'master':20260923,'stream':'noise','truth_id':truth,'phase':phase,'outer_index':index,'geometry_sha256':geometry_hash}
  seed=int.from_bytes(hashlib.sha256(canonical(key).encode()).digest()[:8],'big')>>1
  center,true_target=truths[truth];scales=.04*np.maximum(1,abs(center));L=np.zeros((30,30))
  for k in range(0,30,2):L[k:k+2,k:k+2]=[[scales[k],0],[.25*scales[k+1],scales[k+1]*math.sqrt(1-.25**2)]]
  y=center+L@np.random.Generator(np.random.PCG64(seed)).normal(size=30);C=L@L.T
  dst=out/ident;dst.mkdir();np.savez_compressed(dst/'input.npz',y=y,covariance=C,row_ids=ids)
  (dst/'CONTROLLER.json').write_text(json.dumps({'phase':phase,'truth_controller_only':truth,'noise_key':key,'seed':seed,'true_targets_controller_only':true_target.tolist()},indent=2)+'\n')
  cmd=['/usr/bin/python3',str(PREP/'source/fit_final_geometry.py'),'--input',str(dst/'input.npz'),'--out',str(dst/'fit')]
  with (dst/'fit.log').open('w') as log:ret=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
  result=json.loads((dst/'fit/RESULT.json').read_text()) if (dst/'fit/RESULT.json').exists() else None
  success=ret==0 and result is not None and result['status']=='finite_budget_feasible'
  fit=score.Fit(ident,sha(dst/'input.npz'),c.hash,tuple(result['selected']['targets']) if success else None,None if success else 'preserved_fit_failure_or_abstention')
  pair=score.Pair(fit,truth,tuple(true_target),phase);pairs[phase].append(pair)
  (dst/'SCORING_RECORD.json').write_text(json.dumps(dataclasses.asdict(pair),indent=2)+'\n')
  if phase=='test':(dst/'INTERVALS.json').write_text(json.dumps(score.intervals(c,cutoff,fit),indent=2)+'\n')
 report=score.evaluate(c,cutoff,pairs['test'],expected_per_stratum=240)
 (out/'TEST_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bookkeeping-only',action='store_true');p.add_argument('--run-authorized-campaign',action='store_true');p.add_argument('--qualified-final-truths');p.add_argument('--original-control-evidence');p.add_argument('--out');args=p.parse_args()
 if args.bookkeeping_only and not args.run_authorized_campaign:bookkeeping()
 elif args.run_authorized_campaign and all((args.qualified_final_truths,args.original_control_evidence,args.out)):
  campaign(args.qualified_final_truths,args.original_control_evidence,args.out)
 else:p.error('Only --bookkeeping-only is authorized during current preparation. Future campaign requires qualified final truths and original-control evidence.')

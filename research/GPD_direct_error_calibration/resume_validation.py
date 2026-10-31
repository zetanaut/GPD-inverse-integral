"""Recover ONLY post-optimization validation of the already-ranked candidate0.
No optimization, new candidate, noise, or revised acceptance criterion is allowed.
Both complete per-start solver records and the preexisting checkpoint are required.
"""
from pathlib import Path
import argparse,json,time
import fit_nlo as f

def recover(input_path,result_dir):
 root=Path(result_dir)
 if (root/'RESULT.json').exists():raise FileExistsError('Result already complete')
 records={seed:json.loads((root/f'seed{seed}_solver.json').read_text()) for seed in (31,43)}
 for seed,r in records.items():
  if r['status'] in ('wall_budget','wall_or_evaluation_budget'):raise RuntimeError('Time-aborted optimization cannot be a completed finite-budget estimator')
  if not r['history']:raise RuntimeError('Missing optimizer completion')
 cp=f.torch.load(root/'candidate0/checkpoint.pt',weights_only=True)
 with f.np.load(input_path,allow_pickle=False) as data:y=data['y'];cov=data['covariance']
 p=f.Problem(y,cov,cp['seed']);x=cp['flat'];d=f.Domain()
 if any(not f.torch.equal(p.parameters(x)[k],v) for k,v in cp['state_dict'].items()):raise RuntimeError('Checkpoint flat/state mismatch')
 t0=time.monotonic()
 with f.torch.no_grad():
  c=d.constraints(p,x).numpy();m=f.moment_validation(p,x);a=f.targets(p,x,192);b=f.targets(p,x,384);J=float(p.objective(x))
 ok=bool(f.np.isfinite(c).all() and f.np.all(c<=0) and m['pass'] and max(abs(f.np.array(a)-b))<=1e-7)
 ff={'pass':False,'not_run':'precheck failed'}
 if ok:ff=f.full_forward(p,x,root/'candidate0/full_forward')
 z=x.detach().requires_grad_(True);g=f.torch.autograd.grad(p.objective(z),z)[0]
 item={'seed':int(cp['seed']),'stage':int(cp['stage']),'J':J,'max_constraint':float(c.max()),'moment_pass':m['pass'],
       'forward_pass':ff['pass'],'targets':b,'target_refinement_max':float(max(abs(f.np.array(a)-b))),
       'raw_gradient_L2':float(f.torch.linalg.vector_norm(g)),'checkpoint':'candidate0/checkpoint.pt','pass':bool(ok and ff['pass'])}
 f.write(root/'candidate0/MOMENTS.json',m);f.write(root/'candidate0/FORWARD.json',ff);f.write(root/'candidate0/CHECK.json',item)
 if not item['pass']:
  f.write(root/'VALIDATION_RECOVERY_FAILURE.json',{'candidate0':item,'status':'operationally_incomplete; unsaved candidates not reconstructed'})
  raise RuntimeError('Candidate0 failed; preserve missing candidate roster rather than inventing another selection')
 starts=[]
 for seed,r in records.items():
  p0=f.Problem(y,cov,seed)
  starts.append({'seed':seed,'status':r['status'],'initial_hash':f.array_hash(p0.x0.numpy()),'record':f'seed{seed}_solver.json'})
 result={'status':'finite_budget_feasible','selected':item,'validations':[item],'starts':starts,'data_sha256':f.array_hash(y),'covariance_sha256':f.array_hash(cov),
 'seconds':None,'stationarity_required':False,'scope':'development finite-budget prediction, no convergence or coverage claim',
 'operational_recovery':{'what':'rerun validation only for saved preselected candidate0, both optimizations already complete','seconds':time.monotonic()-t0,
                         'new_optimization_steps':0,'new_candidate_selection':False}}
 f.write(root/'RESULT.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--result',required=True);a=p.parse_args();print(json.dumps(recover(a.input,a.result),indent=2))

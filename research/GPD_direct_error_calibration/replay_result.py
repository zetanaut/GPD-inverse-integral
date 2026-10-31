"""Fresh objective/feasibility/target replay; no optimizer or kernel rebuild."""
from pathlib import Path
import argparse,json
import fit_nlo as f

def replay(input_path,result_dir,full=False):
 root=Path(result_dir);result=json.loads((root/'RESULT.json').read_text())
 with f.np.load(input_path,allow_pickle=False) as data:y=data['y'];cov=data['covariance']
 if f.array_hash(y)!=result['data_sha256'] or f.array_hash(cov)!=result['covariance_sha256']:raise ValueError('Wrong observations')
 out={'scope':'fresh replay of selected finite-budget output only','status':result['status'],'full_forward_repeated':full}
 if result['status']=='abstain':out['pass']=True;out['meaning']='recorded algorithm failure retained';return out
 sel=result['selected'];p=f.Problem(y,cov,sel['seed']);d=f.Domain();cp=f.torch.load(root/sel['checkpoint'],weights_only=True);x=cp['flat']
 if any(not f.torch.equal(p.parameters(x)[n],v) for n,v in cp['state_dict'].items()):raise ValueError('Inconsistent checkpoint')
 a=x.clone().requires_grad_(True);J=p.objective(a);g=f.torch.autograd.grad(J,a)[0]
 with f.torch.no_grad():
  indep=float(p.independent_objective(x));c=d.constraints(p,x).numpy();mom=f.moment_validation(p,x);t=f.targets(p,x,384)
 full_rec={'pass':True,'not_rerun':True}
 if full:full_rec=f.full_forward(p,x,root/'audit_full_forward')
 out.update(J=float(J.detach()),J_replay_error=abs(float(J.detach())-sel['J']),independent_objective_error=abs(float(J.detach())-indep),
  gradient_L2=float(f.torch.linalg.vector_norm(g)),gradient_replay_error=abs(float(f.torch.linalg.vector_norm(g))-sel['raw_gradient_L2']),
  max_constraint=float(c.max()),moment_pass=mom['pass'],targets=t,target_replay_max=float(max(abs(f.np.array(t)-sel['targets']))),
  passed_original_full_forward=sel['forward_pass'],full_forward=full_rec)
 out['pass']=out['J_replay_error']<1e-10 and out['independent_objective_error']<1e-10 and out['gradient_replay_error']<1e-8 and out['max_constraint']<=0 and mom['pass'] and out['target_replay_max']<1e-9 and full_rec['pass']
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--result',required=True);p.add_argument('--out',required=True);p.add_argument('--full',action='store_true');a=p.parse_args()
 out=replay(a.input,a.result,a.full);Path(a.out).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
 if not out['pass']:raise SystemExit(1)

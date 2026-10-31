"""Replay only new fgdev outputs; no optimization, old-fit replay, or truths."""
import json,hashlib,argparse
from pathlib import Path
import fit_final_geometry as f
a=f.a;np,torch=a.np,a.torch
def replay(ident):
 assert ident in ('fgdev000','fgdev001')
 path=f.PREP/'results'/ident;result=json.loads((path/'RESULT.json').read_text())
 with np.load(f.PREP/'inputs'/ident/'input.npz',allow_pickle=False) as z:y=z['y'];C=z['covariance'];ids=z['row_ids'].tolist()
 assert a.smoke.array_hash(y)==result['data_sha256'] and a.smoke.array_hash(C)==result['covariance_sha256']
 out={'id':ident,'status':result['status'],'optimization_steps':0,'old_fit_replays':0,'full_forward_repeated':False}
 if result['status']=='abstain':out.update(pass_=True,meaning='Abstention retained; stop further development')
 else:
  sel=result['selected'];p=a.Problem(y,C,sel['seed'],f.load_bank(),ids)
  data=torch.load(f.PREP/'inputs/contractions.pt',weights_only=True);d=a.Domain(data['points'],data['spin2'])
  cp=torch.load(path/sel['checkpoint'],weights_only=True);x=cp['flat']
  assert all(torch.equal(p.parameters(x)[n],v) for n,v in cp['state_dict'].items())
  xx=x.clone().requires_grad_(True);j=p.objective(xx);g=torch.autograd.grad(j,xx)[0]
  with torch.no_grad():
   ind=float(p.independent_objective(x));cons=d.constraints(p,x).numpy();mom=a.smoke.moment_validation(p,x);targets=a.smoke.targets(p,x,384)
  fwpath=(path/sel['checkpoint']).parent/'FORWARD.json';fw=json.loads(fwpath.read_text())
  assert fw['pass'] and fw['all_24_points'] and len(fw['records'])==24
  for row in fw['records']:
   rp=fwpath.parent/'full_forward'/row['file'];assert f.sha(rp)==row['sha256']
   rec=json.loads(rp.read_text());assert rec['pass'] and all(rec['gates'].values())
  out.update(J=float(j.detach()),J_replay_error=abs(float(j.detach())-sel['J']),independent_objective_error=abs(float(j.detach())-ind),gradient_L2=float(torch.linalg.vector_norm(g)),gradient_replay_error=abs(float(torch.linalg.vector_norm(g))-sel['raw_gradient_L2']),max_constraint=float(cons.max()),constraints=len(cons),moment_pass=mom['pass'],targets=targets,target_replay_max=float(max(abs(np.asarray(targets)-sel['targets']))),forward_records_verified=24,checkpoint_sha256=f.sha(path/sel['checkpoint']))
  out['pass_']=out['J_replay_error']<1e-10 and out['independent_objective_error']<1e-10 and out['gradient_replay_error']<1e-8 and out['max_constraint']<=0 and mom['pass'] and out['target_replay_max']<1e-9
 dest=f.PREP/'results'/('REPLAY_'+ident+'.json');dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['pass_']
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('id',choices=('fgdev000','fgdev001'));replay(parser.parse_args().id)

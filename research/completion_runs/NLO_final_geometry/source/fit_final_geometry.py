"""Same accepted finite-budget run function with geometry-driven interfaces."""
import os
from pathlib import Path
import sys,json,hashlib,types,argparse
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]
os.environ['GPD_REVIEW_ROOT']=str(ROOT)
import geometry_adapter as a
import geometry_validation as gv
np,torch=a.np,a.torch
torch.set_num_interop_threads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_bank():
 with np.load(PREP/'inputs/fit_bank.npz',allow_pickle=False) as z:
  return a.Bank([{'xi':float(xi),'Q2':float(q),'x':z['x'][i].copy(),'rows':z['rows'][i].copy()} for i,(xi,q) in enumerate(zip(z['xi'],z['Q2']))])
def full_forward(p,x,out):
 p.set_model(x);out=Path(out);records=[]
 for point in json.loads((PREP/'inputs/GEOMETRY_POINTS.json').read_text()):
  for q in point['Q2']:
   result=gv.pf.validate_point(point,q,'candidate_numerical_check',p.model)
   filename=f'xi{point["xi"]}_Q2{q}.json';a.smoke.write(out/filename,result)
   records.append({k:result[k] for k in ('xi','Q2','pass','gates','CFF_estimated_total','CFF_limits')})
   records[-1].update(field_totals={k:v['estimated_total'] for k,v in result['fields'].items()},file=filename,sha256=sha(out/filename),geometry_role=point['role'])
 return {'pass':all(r['pass'] for r in records),'records':records,'all_24_points':len(records)==24,'diagnostic_truth_used':False}
def run(y,cov,input_row_ids,out):
 from environment_gate import verify
 environment=verify()
 lock=json.loads((PREP/'ESTIMATOR_LOCK.json').read_text())
 for name,expected in lock['source_hashes'].items():
  assert sha(ROOT/name)==expected,'Frozen source changed: '+name
 bank=load_bank();data=torch.load(PREP/'inputs/contractions.pt',weights_only=True)
 def problem(y,cov,seed):return a.Problem(y,cov,seed,bank,input_row_ids)
 def domain():return a.Domain(data['points'],data['spin2'])
 assert len(domain().names)==265
 a.smoke.write(Path(out)/'ACTUAL_ENVIRONMENT.json',environment)
 # The algorithm bytecode, optimizer settings, ranking, and four-candidate cap
 # are exactly the accepted frozen implementation; only its interfaces differ.
 scope=dict(a.smoke.run.__globals__,Problem=problem,Domain=domain,full_forward=full_forward,__file__=str(Path(__file__).resolve()))
 accepted_run=types.FunctionType(a.smoke.run.__code__,scope,'accepted_run',a.smoke.run.__defaults__,a.smoke.run.__closure__)
 a.smoke.write(Path(out)/'GEOMETRY_EXECUTION.json',{'estimator_lock_sha256':sha(PREP/'ESTIMATOR_LOCK.json'),'geometry_sha256':sha(PREP/'FINAL_GEOMETRY.json'),'runtime_environment_sha256':sha(PREP/'runtime_repair/ENVIRONMENT.json'),'accepted_run_source_sha256':sha(ROOT/'GPD_direct_error_calibration/fit_nlo.py'),'adapter_sha256':sha(Path(__file__)),'n_scalar':bank.n_scalar,'check_points':24,'constraints':265,'input_contains_truth_or_role':False})
 return accepted_run(y,cov,out)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
 with np.load(args.input,allow_pickle=False) as z:y=z['y'].copy();cov=z['covariance'].copy();ids=z['row_ids'].tolist()
 print(json.dumps(run(y,cov,ids,args.out),indent=2),flush=True)

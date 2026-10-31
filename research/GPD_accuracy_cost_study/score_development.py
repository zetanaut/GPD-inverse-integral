"""Post-selection evaluator for TWO OLD technical truths. Never imported by the fitter.
Does not open registered final/test truth functions or select an estimator.
"""
from pathlib import Path
import os,sys,json,argparse
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from runtime_adapter import Runtime,write
from study_core import file_sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--out',type=Path);args=ap.parse_args()
 root=args.project.resolve();out=(args.out or root/'completion_runs/NLO_accuracy_cost_v1').resolve();rt=Runtime(root)
 np,torch=rt.np,rt.torch
 from src.replay import analytic_input
 from scipy.integrate import quad
 rows=[]
 for ident,added in [('fgdev000',False),('fgdev001',True)]:
  def truth(xx,xi):
   values=analytic_input(np.asarray(xx),xi).copy()
   if added:
    z=np.asarray(xx)/xi;support=abs(z)<1
    values+=np.stack((.09*np.where(support,z*(1-z*z)**2,0.),-.02*np.where(support,z*(1-z*z)**2,0.),.03*np.where(support,xi*(1-z*z)**2,0.)))
   return values
  def ch(x):return float((2/9)*np.sum(truth(np.array([x]),.275)[:2,0]))
  ttrue=[quad(lambda x:5*ch(x),.2,.275,epsabs=1e-12)[0]+quad(lambda x:5*ch(x),.275,.4,epsabs=1e-12)[0],
         quad(lambda x:x*ch(x),0,.275,epsabs=1e-12)[0]+quad(lambda x:x*ch(x),.275,1,epsabs=1e-12)[0]]
  for label,lam in [('replay',1.),('lambda03',.3),('lambda3',3.)]:
   folder=out/(ident+'_'+label);path=folder/'RESULT.json'
   if not path.exists():continue
   result=json.loads(path.read_text());sel=result.get('selected')
   if sel is None:continue
   if file_sha(sel['checkpoint_path'])!=sel['checkpoint_sha256']:raise ValueError('Changed selected checkpoint')
   p=rt.problem(rt.prep/'inputs'/ident/'input.npz',sel['seed'],lam)
   x=rt.torch.load(sel['checkpoint_path'],weights_only=True)['flat']
   model_C=[];truth_C=[]
   with torch.no_grad():
    for point in rt.domain.points:
     if float(point['xi']) not in (.155,.305,.505):continue
     obj=point['data']['output48_strong10'];q=str(point['Q2']);n=obj['nodes'];qi=obj['qids'];gi=obj['gids']
     hm=p.field(x,n[None],rt.a.tensor([point['xi']]))[0];ht=rt.a.tensor(truth(n.numpy(),point['xi']))
     def push(h):return obj['hard'][q]@(obj['U'][q]@torch.cat((h[0,qi],h[1,qi],h[2,gi])))
     model_C.extend(push(hm).tolist());truth_C.extend(push(ht).tolist())
    # Independent fixed reference-field diagnostic grid; never an optimization target.
    z,w=np.polynomial.legendre.leggauss(128);xis=np.linspace(.02,.98,41);num=den=0.
    for xi in xis:
     xx=np.r_[xi*(z+1)/2,xi+(1-xi)*(z+1)/2];ww=np.r_[xi*w/2,(1-xi)*w/2]
     hm=p.field(x,rt.a.tensor(xx)[None],rt.a.tensor([xi]))[0].numpy();ht=truth(xx,xi)
     cm=(2/9)*(hm[0]+hm[1]);ct=(2/9)*(ht[0]+ht[1]);num+=np.sum(ww*(cm-ct)**2);den+=np.sum(ww*ct**2)
   ec=np.asarray(model_C)-truth_C
   row={'case':ident+'_'+label,'lambda_poly':lam,'targets':sel['targets'],'true_targets_POSTSELECTION':ttrue,
        'target_absolute_errors':abs(np.array(sel['targets'])-ttrue).tolist(),
        'holdout_CFF_absolute_RMS':float(np.sqrt(np.mean(ec**2))),'holdout_CFF_relative_L2':float(np.linalg.norm(ec)/np.linalg.norm(truth_C)),
        'charge_GPD_relative_L2':float(np.sqrt(num/den)),'selected_output_sha256':file_sha(path),
        'used_for_selection':False,'phase':'development_only','new_truth_or_noise_generated':False}
   write(folder/'POSTSELECTION_DEVELOPMENT_ERRORS.json',row);rows.append(row)
 write(out/'DEVELOPMENT_ERRORS.json',{'records':rows,'not_calibration_scores':True,'scope':'two old technical truths; fixed existing matrices; no new kernels'})
if __name__=='__main__':main()

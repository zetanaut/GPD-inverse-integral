"""Independent saved-state replay; no optimizer, NLO build, new noise or cluster action."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import json,time,sys,resource
from pathlib import Path
import numpy as np
import torch
from audit_packet import ROOT,OUT,PACKET,TRANSFER,RESULT,CONTROL,CAP,PREP,read,sha,write

def checkpoint(path):
 with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('float64'))]):return torch.load(path,map_location='cpu',weights_only=True)
def same(a,b):
 if isinstance(a,(np.ndarray,torch.Tensor)):return np.array_equal(np.asarray(a),np.asarray(b))
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 start=time.monotonic();torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 verification=read(OUT/'PACKET_VERIFICATION.json');assert verification['status']=='PASS'
 for rel,digest in verification['original_scoring_source_pins'].items():assert sha(CAP/rel)==digest
 # Recheck all imports used for polynomial-only value replay against frozen capsule manifest.
 original={r['path']:r for r in read(CAP/'MANIFEST.json')['files']}
 for name in ['moments.py','polynomiality_base.py']:
  rel='project/GPD_three_stage_closure/vendor/gpd_reh/'+name;assert sha(CAP/rel)==original[rel]['sha256']
 STUDY=CAP/'project/GPD_three_stage_closure';sys.path[:0]=[str(STUDY),str(STUDY/'vendor')]
 from gpd_reh.model import ExperimentalFiLM
 from gpd_reh.moments import FixedTMoments
 from closure.truths import Truth
 from closure.metrics import field_metrics,split_grid
 lock=read(RESULT/'RUN_LOCK.json');identity=sha(RESULT/'RUN_LOCK.json')
 state,adam,latest=[checkpoint(RESULT/n) for n in ['SOLVER_RESUME.pt','ADAM_RESUME.pt','LATEST_STATE.pt']]
 assert adam['completed_updates']==400 and adam['remaining_updates']==0
 assert state['next_iteration']==60 and state['remaining_outer_iterations']==0 and latest['completed_updates']==60
 assert same(latest['flat'],state['x']) and same(checkpoint(RESULT/'final.pt')['flat'],state['x'])
 assert all(s['run_lock_sha256']==identity for s in [state,adam,latest])
 journal=[json.loads(line) for line in (RESULT/'ITERATIONS.jsonl').read_text().splitlines()];diag=read(RESULT/'SOLVER_DIAGNOSTICS.json');opt=read(RESULT/'OPTIMIZER_COMPLETE.json')
 assert len(journal)==len(state['history'])==len(diag['history'])==60
 assert same(state['history'],diag['history']) and [r['iteration'] for r in journal]==list(range(1,61))
 assert all(r['accepted']==h['accepted'] for r,h in zip(journal,diag['history'])) and sum(r['accepted'] for r in journal)==60
 assert same(state['oracle_counts'],diag['oracle_counts']) and state['oracle_counts']['evaluations']==199<=12000
 assert diag['status']=='iteration_budget' and opt['solver_iterations']==60
 result,scores,forward=[read(RESULT/n) for n in ['RESULT.json','SCORES.json','FORWARD_SCORES.json']]
 assert opt==result['optimization'] and opt['field_and_rank_threads']==1 and opt['solver_blas_threads']==8
 candidates=opt['candidates_by_supplied_J'];assert candidates==sorted(candidates,key=lambda r:(r['J'],31,r['stage']))
 assert candidates[0]['stage']==result['selected']['stage']==60 and result['selected']['rank']==0
 for row in candidates:assert sha(RESULT/row['path'])==row['sha256'] and checkpoint(RESULT/row['path'])['J']==row['J']
 assert np.isclose(state['values']['J'],result['selected']['J'],rtol=1e-12,atol=1e-14)
 cfg=read(STUDY/'prepared/FIT_CONFIG.json');model=ExperimentalFiLM(cfg['network']['t_mean'],cfg['network']['t_std']).double().eval()
 pol=cfg['polynomiality'];mom=FixedTMoments(pol['t_nodes'],np.array(pol['xi_fit']),order=48,spins=tuple(pol['spins']))
 truth=Truth('Z1');geometry=read(STUDY/'prepared/GEOMETRY.json')
 with np.load(TRANSFER/'inputs/CFF_HALF.npz',allow_pickle=False) as z:obs={k:z[k].copy() for k in z.files}
 with np.load(TRANSFER/'inputs/CENTRAL_FORWARD.npz',allow_pickle=False) as z:central_info={k:z[k].copy() for k in z.files}
 with np.load(TRANSFER/'inputs/FORWARD_NOISY.npz',allow_pickle=False) as z:info={k:z[k].copy() for k in z.files}
 meta=read(TRANSFER/'inputs/NOISE_RECEIPT.json');noise=np.load(TRANSFER/'inputs/PDF_NOISE_Z.npy',allow_pickle=False)
 assert meta['draw_count']==1 and meta['seed']==2026102601 and meta['bit_generator']=='PCG64' and meta['ordering']=='x-major/channel-minor'
 for name,key in [('CENTRAL_FORWARD.npz','central_sha256'),('FORWARD_NOISY.npz','input_sha256'),('PDF_NOISE_Z.npy','noise_z_sha256')]:assert sha(TRANSFER/'inputs'/name)==meta[key]
 xpdf,central,cov=central_info['x'],central_info['observed'],central_info['covariance'];noisy=info['observed']
 assert np.array_equal(xpdf,info['x']) and np.array_equal(cov,info['covariance'])
 assert np.allclose(xpdf,np.geomspace(.02,.8,12),rtol=0,atol=1e-15) and np.allclose(central,truth.values(xpdf,0.,0.),rtol=0,atol=1e-15)
 scale=.1*np.maximum(abs(central),.05*np.max(abs(central),axis=0));kx=np.exp(-abs(np.log(xpdf)[:,None]-np.log(xpdf)[None,:])/.7);kc=np.array([[1.,.5,0.],[.5,1.,0.],[0.,0.,1.]])
 rebuilt=np.kron(kx,kc)*scale.ravel()[:,None]*scale.ravel()[None,:];assert np.allclose(rebuilt,cov,rtol=1e-14,atol=1e-16)
 chol=np.linalg.cholesky(cov);assert np.allclose(noisy.reshape(-1),central.reshape(-1)+chol@noise,rtol=1e-12,atol=2e-15)
 grids=[split_grid(float(xi),160) for xi in obs['xi']];xs,ws=np.array([x for x,w in grids]),np.array([w for x,w in grids])
 tf=truth.values(xs,obs['xi'][:,None],obs['t'][:,None]);th=(2/9)*(tf[:,:,0]+tf[:,:,1])
 assert np.allclose(obs['mass'],np.full(160,1/160),rtol=0,atol=1e-15)
 def load(path):
  ck=checkpoint(path);flat=ck['flat'].detach().double();assert flat.shape==(3867,) and torch.isfinite(flat).all()
  params={};offset=0
  for name,p in model.named_parameters():
   params[name]=flat[offset:offset+p.numel()].reshape(p.shape);assert torch.equal(params[name],ck['state_dict'][name]);offset+=p.numel()
  assert offset==3867;model.load_state_dict(params);return ck
 def evaluate(x,xi,t):
  with torch.no_grad():return model(torch.tensor(np.asarray(x)[None]),torch.tensor([float(xi)]),torch.tensor([float(t)]))[0].numpy().T
 def observed():
  chunks=[]
  with torch.no_grad():
   for lo in range(0,160,8):
    pred=model(torch.tensor(xs[lo:lo+8]),torch.tensor(obs['xi'][lo:lo+8]),torch.tensor(obs['t'][lo:lo+8])).numpy();chunks.append((2/9)*(pred[:,0]+pred[:,1]))
  pred=np.concatenate(chunks)
  return {k:float(np.sqrt(np.sum(ws[:,s]*(pred[:,s]-th[:,s])**2)/np.sum(ws[:,s]*th[:,s]**2))) for k,s in [('all',slice(None)),('ERBL',slice(0,160)),('DGLAP',slice(160,None))]}
 def targets(evaluator):
  z,w=np.polynomial.legendre.leggauss(384);ans=[]
  for lo,hi,is_a in [(.2,.4,True),(0.,1.,False)]:
   x=(lo+hi)/2+(hi-lo)*z/2;f=evaluator(x,.275,-.363);h=(2/9)*(f[:,0]+f[:,1]);ans.append(float(np.sum(w*(hi-lo)/2*h*(5 if is_a else x))))
  return np.array(ans)
 def boundary():
  pred=evaluate(xpdf,0.,0.);residual=np.linalg.solve(chol,(pred-noisy).reshape(-1));grid=np.linspace(.02,.8,201)
  pg,tg=evaluate(grid,0.,0.),truth.values(grid,0.,0.);rel=np.sqrt(np.sum((pg-tg)**2,axis=0)/np.sum(tg**2,axis=0));regions={}
  for name,lo,hi in [('low_x_tail',0.,.02),('observed_support',.02,.8),('high_x_tail',.8,1.)]:
   orders={}
   for order in (80,160):
    z,w=np.polynomial.legendre.leggauss(order);x=(lo+hi)/2+(hi-lo)*z/2;w=w*(hi-lo)/2
    f,tr=evaluate(x,0.,0.),truth.values(x,0.,0.);h,ht=(2/9)*(f[:,0]+f[:,1]),(2/9)*(tr[:,0]+tr[:,1])
    orders[str(order)]=dict(forward_Hplus_first_moment=float(w@(x*h)),truth_forward_Hplus_first_moment=float(w@(x*ht)),integrated_squared_Hplus_error=float(w@((h-ht)**2)),integrated_squared_truth_Hplus=float(w@(ht**2)))
   regions[name]=orders
  return dict(whitened_RMS=float(np.sqrt(np.mean(residual**2))),chi2_PDF=float(residual@residual),J_PDF=float(residual@residual)/160,channel_relative_L2=rel.tolist(),support_and_tails=regions),pred,residual,pg
 true_targets=targets(truth.values);assert np.max(abs(true_targets-np.array([.3619108615320205,.1105532776465300])))<1e-14
 base=TRANSFER/'evidence/baseline';base_scores=read(base/'SCORES.json');rows=[];differences=[]
 for name,n,path in [('half_baseline',60,base/'final.pt')]+[('P2B',n,RESULT/f'STAGE{n}.pt') for n in (0,20,40,60)]:
  ck=load(path)
  if name=='P2B':
   assert ck['lambda_stab']==0 and ck['run_lock_sha256']==identity
   if n==0:assert np.array_equal(ck['flat'].numpy(),np.load(CAP/'reference/FRESH_SEED31.npy',allow_pickle=False))
  om=observed();chart=field_metrics(evaluate,truth,geometry);tv=targets(evaluate);err=tv/true_targets-1;pdf,pred,residual,pg=boundary()
  with torch.no_grad():poly=float(mom(model))*cfg['polynomiality']['lambda']
  row=dict(case=name,stage=n,checkpoint=str(path.relative_to(ROOT)),J_recorded=ck['J'],observed=om,wider_chart=chart,targets=tv.tolist(),target_signed_relative_error=err.tolist(),boundary=pdf,J_poly_replayed=poly,
           J_joint_at_saved_state=ck['J']+(pdf['J_PDF'] if name=='half_baseline' else 0))
  if name=='P2B':
   differences += [abs(poly-ck['named_components']['J_poly']),abs(pdf['J_PDF']-ck['named_components']['J_PDF'])]
  expected=scores if name=='P2B' and n==60 else base_scores if name=='half_baseline' else None
  if expected:
   differences += [abs(om[k]-expected['observed_slices']['relative_L2'][k]) for k in om]
   differences += [abs(chart['charge_relative_L2'][k]-expected['field']['charge_relative_L2'][k]) for k in om]
   differences += list(abs(np.array(chart['internal_channel_relative_L2'])-expected['field']['internal_channel_relative_L2']))+list(abs(tv-expected['target_384']))
  if name=='P2B' and n==60:
   differences += [abs(pdf['whitened_RMS']-forward['observed_samples']['whitened_RMS']),float(np.max(abs(pred-forward['observed_samples']['prediction']))),float(np.max(abs(residual-forward['observed_samples']['whitened_residual']))),float(np.max(abs(pg-forward['supported_grid']['prediction'])))]
   for region,orders in pdf['support_and_tails'].items():
    for order,vals in orders.items():differences += [abs(v-forward['support_and_tails'][region][order][k]) for k,v in vals.items()]
  rows.append(row);print(name,n,'Hplus',om['all'],'A/B',err,'PDF_RMS',pdf['whitened_RMS'],flush=True)
 assert max(differences)<1e-10,max(differences)
 selected,baseline=rows[-1],rows[0]
 # Independently implement the predeclared comparison, using replayed values.
 tolerance=lambda a,b:1e-10+1e-6*max(abs(a),abs(b))
 s,bs=selected['observed']['all']**2,baseline['observed']['all']**2
 e,be=np.abs(selected['target_signed_relative_error']),np.abs(baseline['target_signed_relative_error'])
 gates=dict(observed_Hplus_squared_improves=bs-s>tolerance(s,bs),target_A_not_worse=bool(e[0]-be[0]<=tolerance(e[0],be[0])),target_B_not_worse=bool(e[1]-be[1]<=tolerance(e[1],be[1])))
 assessment=read(PACKET/'ASSESSMENT.json');assert gates==assessment['gates']
 decision='DEVELOPMENT_PASS_REVIEW_REQUIRED' if all(gates.values()) else 'STOP_DEVELOPMENT_FAILED';assert decision==assessment['decision'] and not assessment['assessment_released']
 names=result['named_components'];assert abs(sum(result['selected']['components'])-names['J_joint'])<1e-14
 assert abs(names['J_CFF']-scores['CFF']['input_matching']['RMS_original_sigma']**2)<1e-14
 assert abs(names['chi2_PDF']/160-names['J_PDF'])<1e-14 and abs(names['J_P1']+names['J_PDF']-names['J_joint'])<1e-14
 last=diag['history'][-1]
 history_summary=dict(completed_Adam_updates=400,completed_outer_updates=60,accepted_outer_updates=60,solver_evaluations=199,history_consistency=True,stage0_is_original_fresh_seed=True,stage60_equals_final_and_terminal_state=True,
  solver_status=diag['status'],solver_counts=diag['counts'],oracle_counts=diag['oracle_counts'],coordinate_cache_stats=opt['coordinate_cache_stats'],last_prefinal_KKT=dict(iteration=last['iteration'],KKT=last['KKT']))
 comparisons=dict(gates=gates,observed_squared_improvement_fraction=(bs-s)/bs,observed_L2_improvement_fraction=1-selected['observed']['all']/baseline['observed']['all'],
  target_B_error_increase=float(e[1]-be[1]),target_B_tolerance=tolerance(e[1],be[1]),target_B_error_increase_over_tolerance=float((e[1]-be[1])/tolerance(e[1],be[1])),
  primary_squared_error=dict(baseline=bs,selected=s,tolerance=tolerance(s,bs)))
 evidence=dict(status='PASS_INDEPENDENT_SAVED_STATE_REPLAY',scientific_decision=decision,independent_assessment_released=False,archive_sha256=read(OUT/'INTAKE.json')['sha256'],
  baseline=baseline,selected=selected,stages=rows[1:],truth_targets=true_targets.tolist(),comparisons=comparisons,max_replayed_metric_absolute_difference=max(differences),
  covariance=dict(rank=int(np.linalg.matrix_rank(cov)),condition_number=float(np.linalg.cond(cov)),formula_replayed=True,saved_noise_formula_verified=True,noise_redrawn=False),
  trajectory=history_summary,numerical_eligibility=dict(reported_status=result['status'],reported_constraint_max=result['selected']['constraint_max'],reported_moments=result['selected']['moments'],reported_full_operator=result['selected']['full_operator'],full_constraints_replayed_locally=False,finite_polynomial_objective_replayed=True),
  recorded_CFF_scores=dict(baseline=base_scores['CFF'],selected=scores['CFF'],NLO_replayed_locally=False),costs=verification['costs'],
  seconds=time.monotonic()-start,review_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,environment=dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__)),
  scope='Five saved fields: baseline plus stages0/20/40/60; field/targets/forward/noise/covariance/polynomial objective replay and saved optimizer history. No NLO/constraint Jacobian or optimizer rerun, no new noise, no cluster contact.')
 write('REVIEW_EVIDENCE.json',evidence);print('PASS',decision,'max discrepancy',max(differences),'seconds',evidence['seconds'],flush=True)
if __name__=='__main__':main()

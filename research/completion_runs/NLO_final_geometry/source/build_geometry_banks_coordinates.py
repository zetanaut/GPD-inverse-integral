"""Explicit final rows and numerical filters from validated saved generators."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np
import torch
from scipy.integrate import solve_ivp
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]
os.environ['GPD_REVIEW_ROOT']=str(ROOT)
import geometry_adapter as a
import geometry_validation as gv
from domain import LABELS,PAIRS,spin2_map
from src.ordered_evolution import coupling
NLO=gv.NLO;T=a.tensor
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def ordered_maps(k0,k1,qs):
 aa,end,_=coupling(max(qs));n=len(k0);times=np.log(np.array(sorted(qs))/2.)
 def rhs(t,v):
  u=v.reshape(n,n);z=2*float(aa(t));return (z*(k0@u)+z*z*(k1@u)).ravel()
 sol=solve_ivp(rhs,(0.,end),np.eye(n).ravel(),method='DOP853',t_eval=times,rtol=1e-13,atol=1e-15)
 if not sol.success or not np.isfinite(sol.y).all():raise RuntimeError('Given-matrix propagation failed')
 return {str(q):T(sol.y[:,i].reshape(n,n).copy()) for i,q in enumerate(sorted(qs))},sol.nfev
def main(available_only=False):
 geo=json.loads((PREP/'FINAL_GEOMETRY.json').read_text())
 old=torch.load(ROOT/'completion_runs/NLO_feasible_domain/inputs/contractions.pt',weights_only=True)
 olddata={p['xi']:p for p in old['points']};oldpoints={p['xi']:p for p in gv.common.operator_directories()}
 points=[];data=[];records=[];bank=[];original=a.original_bank();start=time.monotonic()
 for xi in geo['fit_xi']+geo['prediction_diagnostic_xi']:
  role='likelihood' if xi in geo['fit_xi'] else 'numerical_diagnostic'
  qs=geo['fit_Q2'] if role=='likelihood' else geo['prediction_diagnostic_Q2']
  if xi in olddata:
   point=oldpoints[xi];state=olddata[xi];source={'reused':'completion_runs/NLO_feasible_domain/inputs/contractions.pt','sha256':sha(ROOT/'completion_runs/NLO_feasible_domain/inputs/contractions.pt')}
  else:
   pipeline_path=PREP/f'geometry/xi{xi}'/('GG_COORDINATE_PIPELINE.json' if xi==.505 else 'PIPELINE.json')
   pipeline=json.loads(pipeline_path.read_text())
   if pipeline['status'] not in ('FIRST_FAMILY_VALIDATED','FAMILY_VALIDATED'):
    if available_only:print('PENDING_VALIDATION',xi,flush=True);continue
    raise RuntimeError('Unvalidated family: '+str(xi))
   point={'xi':xi,'directories':pipeline['directories']}
   cache=PREP/f'geometry/xi{xi}/contractions.pt'
   binding={'geometry_sha256':sha(PREP/'FINAL_GEOMETRY.json'),'builder_sha256':sha(Path(__file__)),'Q2':qs,'operators':{d:{'KEY':sha(NLO/d/'KEY.json'),'matrix':sha(NLO/d/'operators.npz')} for d in point['directories'].values()}}
   if cache.exists():
    item=torch.load(cache,weights_only=True);assert item['binding']==binding;state=item['point']
   else:
    ds={};bs={}
    for label in LABELS:
     d=point['directories'][label];b,k0,k1,key=gv.pf.op(d);bs[label]=b
     map_path=PREP/f'geometry/xi{xi}/maps_{label}.pt'
     if map_path.exists():
      prior=torch.load(map_path,weights_only=True);assert prior['binding']==binding and prior['label']==label
      ds[label]=prior['data'];continue
     U,nfev=ordered_maps(k0,k1,qs);nq=len(b.qids);n=len(k0);xx,ww=b.outputs(96)
     M=np.zeros((3,n));M[0,:nq]=(ww*xx)@b.basis(xx,'odd');M[1,2*nq:]=ww@b.basis(xx,'even');M[2,nq:2*nq]=(ww*xx)@b.basis(xx,'odd')
     ds[label]={'nodes':T(b.nodes),'qids':torch.tensor(b.qids),'gids':torch.tensor(b.gids),'U':U,'nq':nq,'M':T(M),
      'hard':{str(q):T(np.stack((gv.pf.hard(b,q,192).real,gv.pf.hard(b,q,192).imag))) for q in qs},
      'hard_delta':{str(q):T(np.stack(((gv.pf.hard(b,q,384)-gv.pf.hard(b,q,192)).real,(gv.pf.hard(b,q,384)-gv.pf.hard(b,q,192)).imag))) for q in qs}}
     save(map_path,{'binding':binding,'label':label,'data':ds[label],'nfev':nfev})
    pairs=[]
    for l,r,wgt in PAIRS:
     b,bb=bs[l],bs[r];edges=np.unique(np.r_[b.edges,bb.edges]);z,w=np.polynomial.legendre.leggauss(128)
     xx=np.concatenate([(lft+rgt)/2+(rgt-lft)*z/2 for lft,rgt in zip(edges[:-1],edges[1:])]);ww=np.concatenate([(rgt-lft)*w/2 for lft,rgt in zip(edges[:-1],edges[1:])])
     pairs.append({'left':l,'right':r,'weight':wgt,'w':T(ww),'lq':T(b.basis(xx,'odd')),'lg':T(b.basis(xx,'even')),'rq':T(bb.basis(xx,'odd')),'rg':T(bb.basis(xx,'even'))})
    state={'xi':xi,'data':ds,'pairs':pairs};save(cache,{'binding':binding,'point':state})
   source={'generated':str(cache.relative_to(ROOT)),'sha256':sha(cache)}
  points.append(dict(point,Q2=qs,role=role));data.extend(dict(state,Q2=q) for q in qs);records.append(dict(xi=xi,**source))
  if role=='likelihood':
   for q in qs:
    if xi in olddata:
     row=next(p for p in original.points if p['xi']==xi and p['Q2']==q);bank.append(row)
    else:
     d=point['directories']['weak10'];b,_,_,_=gv.pf.op(d)
     bank.append({'xi':xi,'Q2':q,'x':b.nodes,'rows':gv.pf.cached_row(d,q),'source':d,'role':'likelihood'})
 if len(data)!=24 and available_only:
  (PREP/'geometry/PARTIAL_CONTRACTION_BUILD.json').write_text(json.dumps({'status':'INCOMPLETE','source_records':records,'saved_check_points':len(data),'seconds':time.monotonic()-start,'production_inputs_exported':False},indent=2)+'\n')
  return
 final=a.Bank(bank);assert final.n_scalar==30 and len(data)==24
 spin={str(q):old['spin2'][str(q)] if str(q) in old['spin2'] else T(spin2_map(q)) for q in sorted(set(geo['fit_Q2']+geo['prediction_diagnostic_Q2']))}
 save(PREP/'inputs/contractions.pt',{'points':data,'spin2':spin,'sources':records})
 (PREP/'inputs').mkdir(exist_ok=True)
 np.savez_compressed(PREP/'inputs/fit_bank.npz',xi=[p['xi'] for p in bank],Q2=[p['Q2'] for p in bank],x=np.stack([p['x'] for p in bank]),rows=np.stack([p['rows'] for p in bank]),row_ids=final.ids)
 (PREP/'inputs/GEOMETRY_POINTS.json').write_text(json.dumps(points,indent=2)+'\n')
 (PREP/'inputs/BUILD.json').write_text(json.dumps({'source_records':records,'scalar_fit_rows':30,'diagnostic_scalar_rows':18,'numerical_constraints':265,'seconds':time.monotonic()-start,'existing_nine_complex_fit_rows_reused':True,'fit_bank_sha256':sha(PREP/'inputs/fit_bank.npz'),'contractions_sha256':sha(PREP/'inputs/contractions.pt')},indent=2)+'\n')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--available-only',action='store_true');args=parser.parse_args();main(args.available_only)

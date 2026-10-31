"""Two registered old-truth technical controls; no final-truth access."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,math,argparse
import numpy as np
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def verify_lock():
 lock=json.loads((PREP/'ESTIMATOR_LOCK.json').read_text())
 for name,h in lock['source_hashes'].items():assert sha(ROOT/name)==h,name
 return lock
def freeze():
 # Freeze after successful numerical construction and before any fresh stream.
 for xi in (.215,.155,.305,.365,.505):
  pipeline_path=PREP/f'geometry/xi{xi}'/('GG_LOCAL_PIPELINE.json' if xi==.505 else 'PIPELINE.json')
  status=json.loads(pipeline_path.read_text())['status']
  assert status in ('FIRST_FAMILY_VALIDATED','FAMILY_VALIDATED')
 assert (PREP/'inputs/BUILD.json').exists()
 assert not (PREP/'ESTIMATOR_LOCK.json').exists()
 geom=json.loads((PREP/'FINAL_GEOMETRY.json').read_text())
 design={k:geom[k] for k in ('fit_xi','fit_Q2','fit_n_scalar','prediction_diagnostic_xi','prediction_diagnostic_Q2','prediction_diagnostic_n_scalar','ordering','reference_scale_Q02','t')}
 gh=hashlib.sha256(canonical(design).encode()).hexdigest()
 entries=[]
 for ident,truth in [('fgdev000','smooth'),('fgdev001','Dterm_added')]:
  key={'master':20260923,'stream':'noise','truth_id':truth,'phase':'direct_error_final_geometry_development_v1','outer_index':0,'geometry_sha256':gh}
  entries.append({'opaque_id':ident,'truth_controller_only':truth,'noise_key':key,'noise_seed':int.from_bytes(hashlib.sha256(canonical(key).encode()).digest()[:8],'big')>>1,'starts':[31,43]})
 registry={'geometry':design,'geometry_sha256':gh,'geometry_document_sha256':sha(PREP/'FINAL_GEOMETRY.json'),'datasets':entries,'first63_rule':'int.from_bytes(SHA256(canonical_key)[:8],big) >> 1','noise_rng':'PCG64','streams_drawn_at_freeze':0,'scope':'Two technical controls only; not calibration or testing'}
 (PREP/'FGDEV_REGISTRY.json').write_text(json.dumps(registry,indent=2)+'\n')
 paths=list((PREP/'source').glob('*.py'))+[PREP/'FINAL_GEOMETRY.json',PREP/'FGDEV_REGISTRY.json',PREP/'runtime_repair/ENVIRONMENT.json',PREP/'runtime_repair/VERIFIED.json']+list((PREP/'inputs').glob('*'))
 paths += [ROOT/'GPD_direct_error_calibration/fit_nlo.py',ROOT/'GPD_direct_error_calibration/CONTRACT.json',ROOT/'GPD_direct_error_calibration/direct_error.py',ROOT/'completion_runs/01/inference_contract.json',ROOT/'completion_runs/00/baseline/gpd_poly_symbolic_v5/vendor/gpd_lo_baseline_v4/src/truths.py']
 paths += list((ROOT/'completion_runs/NLO_constrained_solver').glob('*.py'))+list((ROOT/'completion_runs/NLO_feasible_domain').glob('*.py'))+list((ROOT/'GPD_NLO_constrained_solver_review').glob('*.py'))+list((ROOT/'GPD_NLO_resolution_review/pilot').glob('*.py'))
 nlo=ROOT/'GPD_NLO_completion_package'
 for folder in ('candidate','src','frozen/python'):paths+=list((nlo/folder).glob('*.py'))
 paths+=list((ROOT/'GPD_task04_solver_repair/vendor/gpd_task04_local/vendor').glob('*.py'))
 # Retain every previously locked dependency and its accepted bytes.
 inherited_path=ROOT/'GPD_direct_error_calibration/RUNTIME_LOCK.json'
 inherited=json.loads(inherited_path.read_text())['files']
 for name,expected in inherited.items():
  assert sha(ROOT/name)==expected,'Accepted dependency changed: '+name
  paths.append(ROOT/name)
 paths.append(inherited_path)
 paths+=list((PREP/'runtime_repair').glob('*.json'))
 bundle=json.loads((PREP/'runtime_repair/BUNDLE.json').read_text())
 repaired=PREP/'runtime_repair/bundles'/bundle['key']
 assert repaired.resolve()==Path(bundle['bundle']).resolve()
 paths+=[repaired/'REPAIR.json']+list(repaired.glob('*.dylib'))
 points=json.loads((PREP/'inputs/GEOMETRY_POINTS.json').read_text())
 for point in points:
  for d in point['directories'].values():paths.extend([nlo/d/'KEY.json',nlo/d/'operators.npz'])
 addon=ROOT/'GPD_final_geometry_GG_local_repair'
 build=json.loads((addon/'ACTIVE_BUILD.json').read_text())
 paths+=list(addon.glob('*.py'))+list(addon.glob('*.cpp'))
 paths+=[addon/n for n in ('ACTIVE_BUILD.json','LOCAL_VERIFICATION.json','ASSEMBLY_COPIES.json')]
 paths+=[Path(build['library']),Path(build['library']).parent/'BUILD.json',Path(build['library']).parent/'mp_kernel.cpp',Path(build['library']).parent/'local_reference.cpp',nlo/'candidate/mp_kernel.cpp']
 paths+=list((addon/'fallback_cache').glob('*.json'))
 paths+=[PREP/'geometry/FAMILY_POINTERS.json',PREP/'geometry/xi0.505/GG_LOCAL_PIPELINE.json',PREP/'results/GG_LOCAL_USAGE.json']
 binding={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if p.is_file()}
 lock={'status':'FROZEN_BEFORE_FGDEV','source_hashes':binding,'accepted_estimator':json.loads((ROOT/'GPD_direct_error_calibration/CONTRACT.json').read_text()),'n_scalar':30,'constraints':265,'moment_denominator':492,'new_development_streams':2,'new_development_datasets_per_truth':1,'calibration_or_test_authorized':False}
 (PREP/'ESTIMATOR_LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
def prepare(ident):
 verify_lock();registry=json.loads((PREP/'FGDEV_REGISTRY.json').read_text())
 entry=next(e for e in registry['datasets'] if e['opaque_id']==ident)
 if ident=='fgdev001':assert json.loads((PREP/'results/fgdev000/RESULT.json').read_text())['status']=='finite_budget_feasible'
 out=PREP/'inputs'/ident
 if out.exists():raise FileExistsError('Preserve registered input; no new draw')
 import geometry_validation as gv
 means=[];sources=[]
 for point in json.loads((PREP/'inputs/GEOMETRY_POINTS.json').read_text()):
  if point['role']!='likelihood':continue
  d=point['directories']['output48_strong10'];b,k0,k1,key=gv.pf.op(d)
  for q in point['Q2']:
   v=gv.pf.initial(b,entry['truth_controller_only']=='Dterm_added');h,diag=gv.evolve(v,k0,k1,q,rtol=1e-13,atol=1e-15)
   row=gv.hard(b,q,384);c=row@h;ref,info=gv.reference(b,h,q,1e-9)
   error=np.abs([c.real-ref.real,c.imag-ref.imag])+np.asarray(info['reference_quad_error_estimate'])
   limit=1e-4*np.maximum(1,np.abs([ref.real,ref.imag]))
   assert not info['warnings'] and np.all(error<=limit),'Refined mean numerical failure'
   means.extend([c.real,c.imag]);sources.append({'xi':point['xi'],'Q2':q,'directory':d,'KEY_sha256':sha(gv.NLO/d/'KEY.json'),'operators_sha256':sha(gv.NLO/d/'operators.npz'),'hard_order':384,'ordered_rtol':1e-13,'ordered_atol':1e-15,'independent_hard':info,'absolute_component_errors':error.tolist()})
 center=np.array(means);assert len(center)==30
 s=.04*np.maximum(1,abs(center));L=np.zeros((30,30))
 for k in range(0,30,2):L[k:k+2,k:k+2]=[[s[k],0],[.25*s[k+1],s[k+1]*math.sqrt(1-.25**2)]]
 C=L@L.T;normal=np.random.Generator(np.random.PCG64(entry['noise_seed'])).normal(size=30);y=center+L@normal
 with np.load(PREP/'inputs/fit_bank.npz',allow_pickle=False) as z:ids=z['row_ids'].copy()
 out.mkdir();np.savez_compressed(out/'input.npz',y=y,covariance=C,row_ids=ids)
 np.savez_compressed(out/'controller_only.npz',mean=center,standard_normal=normal)
 (out/'CONTROLLER.json').write_text(json.dumps({'entry':entry,'input_sha256':sha(out/'input.npz'),'mean_and_noise_sha256':sha(out/'controller_only.npz'),'sources':sources,'fitter_input_keys':['y','covariance','row_ids'],'geometry_sha256':registry['geometry_sha256'],'estimator_lock_sha256':sha(PREP/'ESTIMATOR_LOCK.json')},indent=2)+'\n')
 print(out/'input.npz')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--id',choices=('fgdev000','fgdev001'));a=p.parse_args()
 if a.freeze:freeze()
 elif a.id:prepare(a.id)
 else:p.error('choose --freeze or --id')

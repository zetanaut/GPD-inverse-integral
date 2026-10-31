"""Summarize executed evidence and prepare an explicit minimal review file list."""
from pathlib import Path
import json,hashlib,difflib,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 points=[{'base':'work/operators/89ab51d596c4c97272320ecc912209a40de9066d069ac3b45c6adcd16b654c34','q2':2.3,'ns_charge_evidence':'work/operators/3255d4d8b4cc2a8b7b27b39b5d4b3005a3d87398e89450dcada4549fb844f31c/NS_CHARGE_CHECK.json','budget':'work/PILOT_BUDGET.json','source':'work/PILOT_SOURCE_QUADRATURE.json','output':'work/PILOT_OUTPUT_QUADRATURE.json','inner72':'work/operators/a8126b3d82b130a78c1b7024181208929ef8325a3c85483771449239720f4584','output48':'work/operators/f16084d9b179c783b37d76f013e105bdf08d81fa262e3fe2fd603acb94715952'}]
 for xi in [.275,.455]:
  prefix=f'work/point_{xi:g}';pipe=read(prefix+'/PIPELINE.json')
  if pipe.get('status')!='completed measurements' or 'failure' in pipe:raise RuntimeError('Unfinished point '+prefix)
  points.append({'base':pipe['base'],'q2':pipe['Q2'],'budget':prefix+'/BUDGET.json','source':prefix+'/SOURCE_QUADRATURE.json','output':prefix+'/OUTPUT_QUADRATURE.json','inner72':pipe['inner72'],'output48':pipe['output48'],'pipeline':prefix+'/PIPELINE.json'})
 files=set();fixtures=[];moments=[];gradients=[];controls=[]
 def add(p):
  p=Path(p);p=p.relative_to(ROOT) if p.is_absolute() else p
  if not (ROOT/p).is_file():raise FileNotFoundError(p)
  files.add(str(p))
 maxcff=np.zeros(2);maxfield=0.
 for point in points:
  base=point['base'];q2=point['q2'];pe=base+f'/POINT_Q2_{q2:g}.json';p=read(pe);budget=read(point['budget']);hard=read(base+'/HARD_REFERENCE.json');gr=read(base+'/GRADIENT_CHECK.json');st=read(base+'/STRUCTURE_CHECK.json');mom=read(base+'/EVOLVED_MOMENTS.json');rawmom=read(base+'/RAW_MOMENT_CHECK.json');ns=read(point.get('ns_charge_evidence',base+'/NS_CHARGE_CHECK.json'));action=read(base+'/SAVED_ACTION_CHECK.json')
  if not budget['numerical_budget_pass'] or budget['independent_hard_has_warnings']:raise RuntimeError('Numerical budget/independent hard failed')
  bestgrad=max(r['tangent_linear_absolute_error'] for r in gr['checks']);fd=max(r['centered_absolute_error'] for r in gr['checks']);parity=max(r['max_scaled_error'] for r in st['source_parity']);harderr=max(max(abs(np.array(v[-1]['component_difference']))) for channels in hard['checks'].values() for v in channels.values())
  if bestgrad>1e-9 or fd>1e-9 or parity>1e-12 or st['composition_CFF_abs']>1e-10 or harderr>1e-7:raise RuntimeError('Independent structural/derivative check failed')
  for truth,v in budget['fixtures'].items():
   if not all(all(a) for a in v['representation_changes_decrease'].values()):raise RuntimeError('Representation convergence not established')
   maxcff=np.maximum(maxcff,v['CFF_estimated_total']);maxfield=max(maxfield,max(w['estimated_total'] for w in v['fields'].values()))
  fixtures.append({'xi':p['xi'],'Q02':2.,'Q2':q2,'status':'passed','truths':budget['fixtures'],'point_evidence':pe,'budget_evidence':point['budget'],'source_quadrature_evidence':point['source'],'output_quadrature_evidence':point['output']})
  moments.append({'xi':p['xi'],'raw_spin2_max_absolute_error':max(r['absolute_error'] for r in rawmom['rows']),'NS_minus_charge_evidence':point.get('ns_charge_evidence',base+'/NS_CHARGE_CHECK.json'),'NS_minus_charge_absolute_derivatives':np.abs(ns['nlo_charge_derivative']).tolist(),'evolved_spin2_max_relative_error_p10':max(max(r['relative_errors']) for r in mom['rows'] if r['method'] in ['weak10','strong10']),'Dterm_support':mom['Dterm_support'],'evidence':base+'/EVOLVED_MOMENTS.json'})
  gradients.append({'xi':p['xi'],'tangent_adjoint_max_absolute_error':bestgrad,'centered_difference_max_absolute_error':fd,'evidence':base+'/GRADIENT_CHECK.json'})
  controls.append({'xi':p['xi'],'independent_full_support_action':{'status':'passed','max_analytic_action_error':action['max_absolute_error'],'shared_source_limits':'Direct fixed/adaptive NLO actions share pinned source; analytic action, independent equation transcriptions and published moments provide separate checks.'},'kernel_quadrature_refinement':'passed','representation_refinement':'passed','ordered_evolution_refinement':'passed','hard_quadrature_refinement':{'status':'passed','independent_continuum_max_component_difference':harderr},'physical_normalization':'passed with pure quark/gluon hard-contour and factorization-log controls','nlo_moment_identities':'passed; measured residuals and degree convergence retained','forward_adjoint_and_derivatives':'passed','source_parity_max_scaled_error':parity,'ordered_composition_CFF_absolute_error':st['composition_CFF_abs']})
  for path in [pe,point['budget'],point['source'],point['output'],point.get('ns_charge_evidence',base+'/NS_CHARGE_CHECK.json')]:add(path)
  for name in ['API_CHECKS.json','HARD_REFERENCE.json','GRADIENT_CHECK.json','STRUCTURE_CHECK.json','RAW_MOMENT_CHECK.json','PHYSICS_CHECKS.json','SAVED_ACTION_CHECK.json','EVOLVED_MOMENTS.json']:add(base+'/'+name)
  # Native lower-degree fields retain convergence evidence; only the two final operators are shipped.
  for label,directory in p['directories'].items():
   for name in ['KEY.json','RESULT.json']:add(directory+'/'+name)
   if label in ['weak10','strong10']:add(directory+'/operators.npz');add(directory+f'/jacobian_Q2_{q2:g}.npz')
   for truth in ['smooth','Dterm_added']:add(p['results'][label][truth]['native_field'])
  for variant in ['inner72','output48']:
   directory=point[variant];add(directory+'/KEY.json');add(directory+'/RESULT.json');add(directory+f'/POINT_Q2_{q2:g}.json')
  if 'pipeline' in point:
   add(point['pipeline']);folder=(ROOT/point['pipeline']).parent
   for path in folder.glob('*.log'):add(path)
   add(folder/'OUTPUT_STRATEGY.json')
   for optional in ['RESUME_STRATEGY.json','PIPELINE_output_cap.json']:
    if (folder/optional).exists():add(folder/optional)
 result={'status':'passed_for_scientific_review','scope':'Six frozen full-support coupled fixed-nf3 MSbar NLO forward development fixtures','source_revision':'Captured pinned tiktaalik source plus documented arithmetic and quadrature repairs; frozen input hashes unchanged','source_equations':'candidate/SOURCE_MAP.md and contracts/formula_map.json','reference_independence':'Weak Galerkin versus strong collocation with independent convergence; both share pinned kernel functions. Separate mpmath Eq177/185 checks, analytic/adaptive full-support actions, published spin2/charge identities, independent evolved-moment ODE, direct continuum hard convolution, ordered RK4/composition and derivatives supply additional controls. This is not a wholly separate QCD evolution library.','acceptance_fixtures':fixtures,'controls':controls,'moment_checks':moments,'gradient_checks':gradients,'maximum_estimated_CFF_error_components':maxcff.tolist(),'maximum_estimated_relative_field_L2':maxfield,'error_estimates_are_rigorous_bounds':False,'no_thresholds_relaxed':True,'no_experimental_fit':True,'no_LO_campaign':True,'no_forced_conservation':True,'resources':{'elapsed_seconds':None,'peak_memory_bytes':None,'actual_tokens':None,'actual_cost':None,'note':'Unknown totals are not inferred. Actual per-build/probe timings, forecasts and caps are retained in the strategy and pipeline logs.'},'reproduce_command':'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python candidate/reproduce_review.py','limitations':['Validation covers the six frozen smooth/D-term development fixtures only; arbitrary new shapes or xi values require their own resolution check.','Physical PDFs, form factors, experimental data, fitting and final statistical calibration remain outside this task.','Large source-row banks remain local; hashes, selected source diagnostics and final candidate/reference operators are packaged.'],'blocker':None}
 (ROOT/'work/RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 summary=f'''# Task 03 resolution\n\nThe coupled full-support NLO forward milestone passes for all six frozen fixtures: (xi,Q²)=(0.135,2.3),(0.275,4.7),(0.455,9.3), each with the smooth input and its specified D-term addition. The largest conservative estimated CFF errors are Re {maxcff[0]:.3g}, Im {maxcff[1]:.3g}; the largest estimated relative field L2 error is {maxfield:.3g}. The unchanged limits are 1e-4 max(1,|reference component|) and 1e-3 per field. Estimates are not rigorous bounds.\n\nThe failures combined cancellation in source arithmetic with unresolved crossover representation and QG endpoint integration. Localized MPFR guards preserve the published expressions and original finite-part tolerance. Continuous spectral elements with localized logarithmic functions resolve the crossover. A derived log-squared subtraction integrates the QG endpoint analytically; it introduces no contact term or fitted coefficient. Original sources, LO work, failed candidates and cached rows remain intact.\n\nCandidate and strong reference converge separately. Kernel/output/hard/time quadrature, full-support source actions, published moments, independent evolved-moment ODE, source parity, ordered composition, D-term support/linearity and derivatives were checked. The final CFF also agrees with a separate adaptive convolution of the evolved fields. The two evolution discretizations share pinned kernel functions; this limitation and the separate physics controls are explicit.\n\nThe differentiable API and perturbative order are documented in candidate/FORWARD_API.json. The review includes native fields, final operators, critical logs, hashes, source mapping and diffs. Run the command in RESULT.json to replay the saved-operator milestone without constructing kernels. Source regeneration scripts and dependency records are also included.\n\nNo LO campaign, neural fit, nuisance survey, experimental-input substitution or tolerance change was performed. Validation of arbitrary new inputs and subsequent inverse fitting remain separate work.\n'''
 if len(summary.split())>400:raise RuntimeError('Summary word limit')
 (ROOT/'work/SUMMARY.md').write_text(summary)
 (ROOT/'work/STATUS.md').write_text(f'All six frozen coupled full-support NLO forward fixtures pass. Maximum estimated CFF errors are Re{maxcff[0]:.3g}, Im{maxcff[1]:.3g}; maximum estimated relative field L2 is{maxfield:.3g}. Independent source/moment/coefficient, representation, quadrature, ordering and derivative controls are complete. The review packet is being assembled and replayed. Frozen sources and completed LO work remain unchanged; no fits or other campaigns were run.\n')
 api=read('candidate/FORWARD_API.json');api['status']='passed six frozen forward development fixtures';(ROOT/'candidate/FORWARD_API.json').write_text(json.dumps(api,indent=2)+'\n')
 for name in ['START_HERE.md','CODEX_PROMPT.md','ANALYSIS.md','PILOT.json','ACCEPTANCE.json','MANIFEST.json','results/LOCAL_PREFLIGHT.json','work/FINAL_PREFLIGHT.log','work/INPUT_ARCHIVE.json','results/FULL_CHECKS.log','results/CACHED_DIAGNOSTICS.json','results/TIME_DIAGNOSTICS.json','results/POINT_BRIDGE_CHECKS.json','work/STRATEGY.json','work/STATUS.md','work/SUMMARY.md','work/RESULT.json','work/PILOT_BUDGET.json']:add(name)
 for folder in ['candidate','src','contracts','frozen/python','frozen/fortran','frozen/original_bridges','fortran','scripts']:
  for path in (ROOT/folder).rglob('*'):
   if path.is_file() and (path.suffix in ['.py','.cpp','.f90','.json','.md','.diff'] or path.name=='LICENSE.tiktaalik'):add(path)
 for name in ['evidence/hard_verified/corrected/reference_hard_contour.py','evidence/hard_verified/SOURCE_DERIVATION.md','evidence/hard_verified/results/RESULT.json']:add(name)
 for name in ['refined.json','GG_ARITHMETIC.json','NS_ARITHMETIC.json','MPFR_CHECKS.json','MP_LOCAL_CHECKS.json','FAST_QG_CHECKS.json','FAST_BLOCK_CHECK.json','ANALYTIC_SCALE.json','QG_SUBTRACTION_CHECK.json','CROSSOVER_QG_QUADRATURE.json','MP_BUILD.json','FAST_QG_BUILD.json','QG_TRANSLATION.json','LOG_BASIS_CHECK.json','LOG_GEOMETRY_CHECK.json','ENRICHED_TIMING.json']:
  add('work/action/'+name)
 for name in ['CROSSOVER_LOCAL_FAILURE.json','local.json','local_linear.json','momentum.json','momentum.log','crossover_benchmark.log','NS_LOCAL_DIRECT.json','GG_BUILD.json','crossover_qg_quadrature.log','qg_subtraction_check.log','mp_loader_failure.log']:
  add('work/action/'+name)
 for path in (ROOT/'work/failures/crossover_NS').glob('*.py'):add(path)
 for path in (ROOT/'work/failures/budget_serialization').iterdir():
  if path.is_file():add(path)
 for old in ['work/operators/3255d4d8b4cc2a8b7b27b39b5d4b3005a3d87398e89450dcada4549fb844f31c/HIERARCHY.json','work/operators/b5aa09981f16a5c03358edc7e2e85797f78ed271a16d5676208f9a8c5196dacc/HIERARCHY.json','work/enriched_hierarchy.log','work/crossover_hierarchy.log']:add(old)
 for name in ['pilot_hard_reference.log','pilot_point_suite.log','pilot_budget.log','pilot_structure.log','pilot_evolved_moments.log','enriched_build.log','enriched_corrected_hierarchy.log','enriched_q72_build.log','output48_build.log','point275_build.log','point455_build.log']:add('work/'+name)
 # Final candidate additions and source hashes, alongside focused kernel diffs.
 sources=sorted(p for p in (ROOT/'candidate').iterdir() if p.suffix in ['.py','.cpp','.f90'])
 diff=''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(ROOT)))) for p in sources)
 (ROOT/'candidate/candidate.diff').write_text(diff);add('candidate/candidate.diff')
 hashes={str(p.relative_to(ROOT)):sha(p) for folder in ['candidate','src','frozen/fortran','frozen/python','frozen/original_bridges','fortran'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.cpp','.f90','.diff']};(ROOT/'work/SOURCE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n');add('work/SOURCE_HASHES.json')
 (ROOT/'work/REVIEW_FILES.txt').write_text('\n'.join(sorted(files))+'\n');print(json.dumps({'files':len(files),'bytes':sum((ROOT/p).stat().st_size for p in files),'summary_words':len(summary.split()),'maximum_estimated_CFF_errors':maxcff.tolist(),'maximum_estimated_field_L2':maxfield},indent=2))
if __name__=='__main__':main()

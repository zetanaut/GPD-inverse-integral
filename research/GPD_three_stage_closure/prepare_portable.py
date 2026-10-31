"""Manifest exact portable source/array inputs; no native binaries or old histories."""
from __future__ import annotations
import hashlib,json,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PKG=Path(__file__).resolve().parent
files=set()
def add(path):
 path=Path(path)
 if not path.is_file():raise FileNotFoundError(path)
 if path.name in ('PORTABLE_MANIFEST.json','EXECUTION_LOCK.json','DESTINATION_COMPARISON.json'):
  return
 files.add(path.relative_to(ROOT))
def suffixes(path,allowed):
 for p in Path(path).rglob('*'):
  if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts:add(p)
# Closure sources, original source evidence, prepared arrays, exact rows.
for p in PKG.iterdir():
 if p.is_file() and p.suffix in ('.py','.json','.md','.patch'):add(p)
for d in ('closure','vendor','prepared','inputs','scale_rows','numerical_contractions','tests','readiness'):
 suffixes(PKG/d,{'.py','.json','.npz','.pt','.csv','.patch','.md'})
# The accepted solver and the original numerical prescription.
for d in ('GPD_NLO_completion_package/candidate','GPD_NLO_completion_package/src',
          'GPD_NLO_completion_package/frozen/python','GPD_NLO_resolution_review/pilot',
          'GPD_task04_solver_repair/vendor/gpd_task04_local/vendor'):
 suffixes(ROOT/d,{'.py'})
for name in ('GPD_ReH_experimental_run/experimental_domain.py',
             'GPD_ReH_experimental_run/build_experimental_contractions.py',
             'completion_runs/NLO_final_geometry/source/geometry_validation.py',
             'completion_runs/NLO_final_geometry/runtime_repair/BUNDLE.json',
             'completion_runs/NLO_feasible_domain/domain.py',
             'completion_runs/NLO_feasible_domain/optimizer.py',
             'completion_runs/NLO_constrained_solver/problem.py',
             'completion_runs/NLO_constrained_solver/solver.py',
             'GPD_direct_error_calibration/fit_nlo.py',
             'GPD_NLO_constrained_solver_review/bind_solver.py',
             'GPD_NLO_constrained_solver_review/moment_linearization.py',
             'GPD_NLO_constrained_solver_review/residual_subproblem.py'):
 add(ROOT/name)
# Each required xi family exactly once, with its completed comparison arrays.
family_root=ROOT/'GPD_ReH_experimental_run/families'
required=set();families=[]
for pipeline in sorted(family_root.glob('*/PIPELINE.json')):
 folder=pipeline.parent;state=json.loads(pipeline.read_text())
 add(pipeline);add(folder/'CONTRACTIONS.pt')
 for check in state['validation']:add(folder/check['file'])
 required.update(state['directories'].values())
 families.append({'xi_hex':state['xi_hex'],'directory':str(folder.relative_to(ROOT)),
                  'operator_directories':sorted(set(state['directories'].values()))})
for directory in sorted(required):
 add(ROOT/'GPD_NLO_completion_package'/directory/'KEY.json')
 add(ROOT/'GPD_NLO_completion_package'/directory/'operators.npz')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
items=[]
for name in sorted(files):
 p=ROOT/name
 items.append({'path':str(name),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'status':'PORTABLE_INPUTS_PREPARED_NO_SCIENTIFIC_FIT',
 'basis_generator_families':families,'distinct_operator_directories':len(required),
 'exact_map_pairs':319,'observed_cases':12,
 'included_files':items,'total_bytes':sum(x['bytes'] for x in items),
 'local_versions':{'python':sys.version,'platform':platform.platform()},
 'destination_versions':'record and compare after transfer',
 'excluded':'macOS native binaries, old directory histories, timing checkpoint, all completed historical fit checkpoints',
 'missing_source_paths':[],
 'binding_checks':'GPD_three_stage_closure/readiness',
 'scientific_fit_count':0}
out=PKG/'readiness/PORTABLE_MANIFEST.json';out.write_text(json.dumps(manifest,indent=2)+'\n')
filelist=PKG/'readiness/PORTABLE_FILE_LIST.txt'
filelist.write_text('\n'.join(str(item['path']) for item in items)+'\n'+str(out.relative_to(ROOT))+'\n')
print(json.dumps({'families':len(families),'operators':len(required),'files':len(items),'total_bytes':manifest['total_bytes'],'manifest':str(out),'filelist':str(filelist)}))

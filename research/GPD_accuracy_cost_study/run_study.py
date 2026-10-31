#!/usr/bin/env python3
"""Bounded development only. Four new fits maximum; no noise or calibration draws."""
from __future__ import annotations
import argparse,json,sys,subprocess,time,traceback,zipfile
from pathlib import Path
from study_core import file_sha

CASES={
 'c024_replay':('c024',1.,False),
 'fgdev000_replay':('fgdev000',1.,False),
 'fgdev001_replay':('fgdev001',1.,False),
 'fgdev000_lambda03':('fgdev000',.3,True),
 'fgdev001_lambda03':('fgdev001',.3,True),
 'fgdev000_lambda3':('fgdev000',3.,True),
 'fgdev001_lambda3':('fgdev001',3.,True),
}

def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)

def run_case(root,out,name):
 from runtime_adapter import Runtime,safe_extract
 rt=Runtime(root);ident,lam,new=CASES[name];dest=out/name
 if ident=='c024':
  old=rt.prep/'final_campaign/campaign/c024';inp=old/'input.npz'
  if (old/'fit_evidence.zip').exists():
   archive=old/'fit_evidence.zip';fit=dest/'reused_fit';safe_extract(archive,fit)
   # Validate extracted source files against captured archive manifest, when supplied.
   manifest=old/'FIT_ARCHIVE.json'
   if manifest.exists() and json.loads(manifest.read_text())['sha256']!=file_sha(archive):raise ValueError('Original fit archive hash mismatch')
  else:fit=old/'fit'
 else:
  inp=rt.prep/'inputs'/ident/'input.npz';fit=rt.prep/'results'/ident
 if new:
  fit=dest/'fit';rt.optimize(inp,fit,lam)
 result=rt.diagnose(inp,fit,dest/'diagnostics',lam)
 write(dest/'RESULT.json',result)
 return result

def package(out,destination):
 with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file() and p.suffix in ('.json','.md','.csv','.log') and 'reused_fit' not in p.parts:
    if p.stat().st_size<=5*1024**2:z.write(p,'worker_results/'+str(p.relative_to(out)))
  # Include only selected new checkpoints; keep full histories local.
  for name in CASES:
   p=out/name/'RESULT.json'
   if p.exists():
    result=json.loads(p.read_text());sel=result.get('selected')
    if sel:
     cp=Path(sel['checkpoint_path'])
     if cp.exists():z.write(cp,'selected_checkpoints/'+name+'.pt')
  for p in Path(__file__).parent.glob('*.py'):z.write(p,'executed_tools/'+p.name)
  for name in ('PLAN.md','PROTOCOL_AMENDMENT.md','SOURCE_CONTRACT.json'):
   p=Path(__file__).with_name(name);z.write(p,'executed_tools/'+name)
 print(str(destination),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--out',type=Path)
 ap.add_argument('--case',choices=tuple(CASES));ap.add_argument('--all',action='store_true');ap.add_argument('--package',action='store_true');args=ap.parse_args()
 root=args.project.resolve();out=(args.out or root/'completion_runs/NLO_accuracy_cost_v1').resolve()
 if out==root or not out.is_relative_to(root):raise ValueError('Use an isolated output directory inside the project')
 if args.package:
  package(out,root/'review_NLO_accuracy_cost.zip');return
 if args.case:
  try:run_case(root,out,args.case)
  except Exception as exc:
   write(out/args.case/'EXECUTION_FAILURE.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'no_automatic_retry':True,'phase':'development'})
   raise
  return
 if not args.all:ap.error('Choose --all, --case, or --package')
 out.mkdir(parents=True,exist_ok=True)
 fence={'phase':'development_only','cases':CASES,'new_fits_maximum':4,'new_initializations_maximum':8,
        'exposed_old_campaign_ids':[f'c{i:03}' for i in range(25)],
        'no_final_calibration_reuse':True,'new_noise_draws':0,
        'code_hashes':{p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}}
 freeze=out/'DEVELOPMENT_LOCK.json'
 if freeze.exists():
  if json.loads(freeze.read_text())!=json.loads(json.dumps(fence)):raise ValueError('Study lock changed; use a new reviewed branch')
 else:write(freeze,fence)
 ledger=[]
 for name in CASES:
  command=[sys.executable,__file__,'--project',str(root),'--out',str(out),'--case',name]
  log=out/(name+'.log');start=time.monotonic()
  # No unconditional repeated optimization. Completed output is reused at the case layer.
  with log.open('a') as handle:
   ret=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,cwd=root).returncode
  ledger.append({'case':name,'returncode':ret,'seconds_this_invocation':time.monotonic()-start,'command':command})
  write(out/'EXECUTION_LEDGER.json',ledger)
  if ret:
   failure=out/name/'EXECUTION_FAILURE.json'
   data=json.loads(failure.read_text()) if failure.exists() else {'type':'ProcessTermination','error':'No terminal record'}
   # Known infrastructure/corrupt-source errors affect the other cases as well.
   if data['type'] in ('FileNotFoundError','ImportError','ModuleNotFoundError','MemoryError','ProcessTermination') or 'lock' in data.get('error','').lower() or 'source' in data.get('error','').lower():break
   # A dataset numerical outcome is retained; it does not poison all other trials.
 package(out,root/'review_NLO_accuracy_cost.zip')
if __name__=='__main__':main()

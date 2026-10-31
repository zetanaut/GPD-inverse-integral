"""Retain the existing reflection/composition and NS-minus charge diagnostics."""
from pathlib import Path
import os,json,time,subprocess,argparse
PREP=Path(__file__).resolve().parents[1];ROOT=PREP.parents[1]
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
parser=argparse.ArgumentParser();parser.add_argument('--validated-only',action='store_true');options=parser.parse_args()
for xi,qs in [(.215,(2.3,4.7,9.3)),(.155,(3.4,7.1,15.3)),(.305,(3.4,7.1,15.3)),(.365,(2.3,4.7,9.3)),(.505,(3.4,7.1,15.3))]:
 out=PREP/f'geometry/xi{xi}';state=json.loads((out/'PIPELINE.json').read_text())
 if state['status'] not in ('FIRST_FAMILY_VALIDATED','FAMILY_VALIDATED'):
  if options.validated_only:print('PENDING_VALIDATION',xi,flush=True);continue
  raise RuntimeError('Family not validated: '+str(xi))
 dest=out/'ADDITIONAL_SOURCE_DIAGNOSTICS.json'
 if dest.exists():
  prior=json.loads(dest.read_text());assert len(prior)==2 and all(r['returncode']==0 for r in prior),'Preserved incomplete/failed source diagnostic'
  continue
 base=ROOT/'GPD_NLO_completion_package'/state['base'];records=[]
 for name,script,args in [('source_NS_charge','check_ns_charge_missing.py',[]),('source_structure','check_structure_missing.py',['--q2',str(max(qs))])]:
  cmd=['/usr/bin/python3',str(PREP/'source'/script),str(base),*args];start=time.monotonic()
  with (out/(name+'.log')).open('w') as log:ret=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
  records.append({'name':name,'command':cmd,'returncode':ret,'seconds':time.monotonic()-start});dest.write_text(json.dumps(records,indent=2)+'\n')
  if ret:raise RuntimeError('Preserved source diagnostic failure at '+str(xi))
 print('SOURCE_DIAGNOSTICS_COMPLETE',xi,flush=True)

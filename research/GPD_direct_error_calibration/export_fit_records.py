"""Bind existing selected outputs to this DEVELOPMENT estimator, without fitting.
These records are not calibration pairs and contain no true targets/stratum.
"""
from pathlib import Path
from dataclasses import asdict
import json,hashlib
from direct_error import Contract,Fit
R=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
version_files=['fit_nlo.py','fit_nlo_initial.py','CONTRACT.json','OPERATIONAL_PATCH.json','RUNTIME_LOCK.json','ENVIRONMENT.json']
estimator_hash=h(canon({f:h((R/f).read_bytes()) for f in version_files}))
registry=json.loads((R/'DEVELOPMENT_REGISTRY.json').read_text())
design_hash=h(canon({'geometry':registry['geometry'],'order':'NLO fixed-nf3 MSbar','noise':'.04 max(1,abs(mean)); within-pair rho=.25; independent kinematic points','stage':'DEVELOPMENT_ONLY'}))
c=Contract(estimator_hash,design_hash)
(R/'MODEL_BINDING.json').write_text(json.dumps({'core_contract':asdict(c),'contract_hash':c.hash,'stage':'DEVELOPMENT_ONLY_NOT_CALIBRATION_APPROVED'},indent=2)+'\n')
for d in ('d000','d001','d002','d003'):
 p=R/f'results/development_{d}/RESULT.json'
 if not p.exists():continue
 r=json.loads(p.read_text());inp=h(canon({'y':r['data_sha256'],'covariance':r['covariance_sha256']}))
 if r['status']=='finite_budget_feasible':f=Fit(d,inp,c.hash,tuple(r['selected']['targets']))
 else:f=Fit(d,inp,c.hash,None,'no_feasible_output')
 f.checked(c);(p.parent/'FIT_RECORD.json').write_text(json.dumps(asdict(f),indent=2,allow_nan=False)+'\n')
print('Exported development fit records; no cutoff generated.')

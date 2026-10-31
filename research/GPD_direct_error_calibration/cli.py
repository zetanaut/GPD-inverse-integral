"""Data-only calibration/reporting CLI. Does not execute any neural fit.
Input JSON contains a core Contract, a prospective registry and Pair records.
The current development package does not authorize a final campaign.
"""
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path
from direct_error import Contract,Fit,Pair,Cutoff,freeze_cutoff,intervals,evaluate

def contract(x):
 x=dict(x);x['targets']=tuple(x['targets']);x['scales']=tuple(x['scales']);x['strata']=tuple(x['strata']);return Contract(**x)
def pair(x):
 f=dict(x['fit']);f['values']=None if f['values'] is None else tuple(f['values'])
 return Pair(Fit(**f),x['stratum'],tuple(x['truth']),x['phase'])
def cutoff(x):
 x=dict(x);x['calibration_ids']=tuple(x['calibration_ids']);x['calibration_inputs']=tuple(x['calibration_inputs']);return Cutoff(**x)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('calibrate','apply','test'));p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 dest=Path(a.out)
 if dest.exists():raise FileExistsError('Refusing to overwrite scientific result')
 x=json.loads(Path(a.input).read_text());c=contract(x['contract'])
 if a.mode=='calibrate':
  if x.get('authorization')!='APPROVED_FROZEN_CALIBRATION':raise ValueError('Campaign freeze/approval missing')
  pairs=[pair(r) for r in x['pairs']]
  actual={(r.fit.dataset_id,r.stratum) for r in pairs};expected={(r['dataset_id'],r['stratum']) for r in x['registered_calibration']}
  if len(expected)!=len(x['registered_calibration']) or actual!=expected:raise ValueError('Calibration registry mismatch')
  out={'contract':asdict(c),'cutoff':asdict(freeze_cutoff(c,pairs))}
 elif a.mode=='test':
  if x.get('authorization')!='APPROVED_FROZEN_TEST':raise ValueError('Test freeze/approval missing')
  pairs=[pair(r) for r in x['pairs']]
  actual={(r.fit.dataset_id,r.stratum) for r in pairs};expected={(r['dataset_id'],r['stratum']) for r in x['registered_test']}
  if len(expected)!=len(x['registered_test']) or actual!=expected:raise ValueError('Test registry mismatch')
  out=evaluate(c,cutoff(x['cutoff']),pairs,x.get('expected_per_stratum',240))
 else:
  if 'truth' in x or 'stratum' in x:raise ValueError('Deployment does not receive truth or stratum')
  f=dict(x['fit']);f['values']=None if f['values'] is None else tuple(f['values'])
  out=intervals(c,cutoff(x['cutoff']),Fit(**f))
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()

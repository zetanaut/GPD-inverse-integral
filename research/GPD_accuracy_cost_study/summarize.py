from pathlib import Path
import argparse,json,csv
from run_study import CASES

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();root=args.out
 rows=[]
 for name in CASES:
  p=root/name/'RESULT.json'
  if not p.exists():
   rows.append({'case':name,'status':'NOT_RUN_or_execution_failure'});continue
  r=json.loads(p.read_text());s=r.get('selected')
  if not s:rows.append({'case':name,'status':r['status']});continue
  d=s['diagnostics'];row={'case':name,'status':r['status'],'seed':s['seed'],'stage':s['stage'],
    'J_original_lambda1':s['J_original_lambda1'],'J_weighted':s['J_weighted'],'data_chi2_per_scalar':s['data_chi2_per_scalar'],
    'moment_RMS':d['moment_RMS'],'moment_percent':100*d['moment_RMS'],'CFF_error_over_noise_sigma':s['numerics_in_noise_units']['max_component_sigma_units'],
    'raw_gradient_L2':s['raw_gradient_L2'],'target0':s['targets'][0],'target1':s['targets'][1],
    'quality_flags':';'.join(d['quality_flags']),'numerical_refinement_over_old_target':d['max_CFF_budget_over_legacy_limit']}
  marker=root/name/'fit/OPTIMIZATION_COMPLETE_STUDY.json'
  row['new_fit_seconds']=json.loads(marker.read_text())['optimization_seconds'] if marker.exists() else 0.
  err=root/name/'POSTSELECTION_DEVELOPMENT_ERRORS.json'
  if err.exists():
   e=json.loads(err.read_text());row.update(target0_abs_error=e['target_absolute_errors'][0],target1_abs_error=e['target_absolute_errors'][1],holdout_CFF_relative=e['holdout_CFF_relative_L2'],charge_GPD_relative=e['charge_GPD_relative_L2'])
  rows.append(row)
 keys=sorted(set().union(*(r.keys() for r in rows)))
 with (root/'COMPARISON.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 finite=sum(r.get('status')=='finite_approximation_with_diagnostics' for r in rows)
 (root/'SUMMARY.md').write_text(f'''# Accuracy–cost development study\n\n{finite}/{len(CASES)} planned outputs are finite diagnostic predictions. See COMPARISON.csv for continuous errors, warning flags and measured costs. Missing runs and execution failures remain in the denominator.\n\nHistorical pass/fail records are unchanged. No result is automatically promoted to a physically admissible extraction or a calibrated interval. No winner is chosen by this script. The two known technical truths are development controls only. The c024 truth is not read.\n\nReview the accuracy–cost tradeoffs before freezing a deployment rule. No fresh calibration or test dataset was generated. Do not pool earlier c000–c024 scores with a changed estimator.\n''')
 print(root/'SUMMARY.md')
if __name__=='__main__':main()

"""Independent statistical-mechanics control; NO GPD fits or calibration data.
Independent repetitions each draw fresh calibration scores and one test score.
The second vector component is non-Gaussian and biased; validity does not require
an unbiased estimator or normal errors when score exchangeability holds.
"""
import json
from pathlib import Path
import numpy as np
from direct_error import rank,wilson
R=Path(__file__).resolve().parent/'results';R.mkdir(exist_ok=True)
rng=np.random.default_rng(202609270901);B=10000;m=80;k=rank(m,'0.10')
result={'scope':'synthetic error-vector control only; not NLO calibration or inference validation','replications':B,'m':m,'rank':k,'cases':{}}
def scores(shape,scale):
 e1=rng.standard_normal(shape)*scale+.4
 e2=(rng.exponential(size=shape)-1.)*.7*scale-.2
 return np.maximum(abs(e1),abs(e2))
a=scores((B,m+1),1.);cut=np.partition(a[:,:m],k-1,axis=1)[:,k-1];hit=a[:,m]<=cut
result['cases']['single_stratum']={'covered':int(hit.sum()),'rate':float(hit.mean()),'wilson95':wilson(int(hit.sum()),B),'continuous_rank_expectation':k/(m+1)}
a=scores((B,m+1),1.);b=scores((B,m+1),2.);q=np.maximum(np.partition(a[:,:m],k-1,axis=1)[:,k-1],np.partition(b[:,:m],k-1,axis=1)[:,k-1])
result['cases']['common_max']={name:{'covered':int((x[:,m]<=q).sum()),'rate':float((x[:,m]<=q).mean()),'wilson95':wilson(int((x[:,m]<=q).sum()),B)} for name,x in [('A',a),('B',b)]}
# Theoretical counterexample to truth-specific deployment: use the narrow
# stratum's cutoff for the broad stratum; data-visible truth label is not allowed.
qa=np.partition(a[:,:m],k-1,axis=1)[:,k-1]
result['cases']['wrong_narrow_cutoff_for_B']={'coverage':float((b[:,m]<=qa).mean()),'valid_procedure':False}
# A single learned target rule and calibration/test source mismatch can also fail.
c=scores((B,1),4.)[:,0]
result['cases']['unregistered_broader_distribution']={'coverage':float((c<=q).mean()),'covered_by_claim':False}
# Every run has independent calibration AND test draws; this control is not
# the final project's fixed calibration/240-test experiment.
(R/'STATISTICAL_CONTROL.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

"""Artificial error records illustrating the API, not a GPD calibration campaign."""
from dataclasses import asdict
import json
from gpd_tools.direct_error import Contract, Fit, Pair, freeze_cutoff, intervals

c = Contract('demo-estimator', 'demo-observation-law', calibration_count=9)
pairs = []
for stratum in c.strata:
    for i in range(9):
        key = f'calibration-{stratum}-{i}'
        fit = Fit(key, 'input-'+key, c.hash, (.005*i, 0.))
        pairs.append(Pair(fit, stratum, (0., 0.), 'calibration'))
cutoff = freeze_cutoff(c, pairs)
prediction = Fit('new-observation', 'new-input-hash', c.hash, (.2, .1))
print(json.dumps({'role':'artificial error-record demo', 'cutoff':asdict(cutoff),
                  'intervals':intervals(c, cutoff, prediction)}, indent=2, allow_nan=False))

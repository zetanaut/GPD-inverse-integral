#!/usr/bin/env python3
"""Read-only science preflight: inputs, geometry, bank index. No fits or kernels."""
import argparse,json
from pathlib import Path
from gpd_reh.inputs import prepare,atomic_json
from gpd_reh.bank import preflight
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,default=ROOT/'inputs/final_summary_by_DNN.csv')
p.add_argument('--prepared',type=Path,default=ROOT/'prepared')
p.add_argument('--bank-dir',type=Path,default=ROOT/'banks')
a=p.parse_args()
s=prepare(a.source,a.prepared);r=preflight(a.prepared,a.bank_dir)
atomic_json(a.prepared/'CACHE_PREFLIGHT.json',r)
print(json.dumps({'rows':s['n_selected'],'xi_families':s['n_exact_evolution_families'],**r},indent=2))

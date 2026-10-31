"""Replay stored arithmetic/certification, not a new native calculation."""
from pathlib import Path
import json,hashlib
import numpy as np
from parent_policy import explicit_per_flavor,NODES,TOLERANCE
ROOT=Path(__file__).resolve().parent

def main():
    d=json.loads((ROOT/'results/RESULT.json').read_text())
    for row in d['native_cases']:
        v=np.array(row['integrals']);e=np.max(abs(np.diff(v,axis=0)),axis=0)
        assert row['nodes']==list(NODES)
        assert np.all(e<=TOLERANCE*np.maximum(1.,abs(v[-1])))
        assert np.allclose(e,row['max_successive_change'],rtol=0,atol=0)
    p=d['native_cases'][0];v=np.array(p['integrals'])[-1];local=v[0]+3*(v[1]+explicit_per_flavor())
    assert abs(local-d['failed_point_local'])<1e-14
    v2=np.array(d['adaptive']['integrals']);local2=v2[0]+3*(v2[1]+explicit_per_flavor())
    assert abs(local-local2)<1e-9 and d['adaptive']['estimated_error']<1e-9
    assert d['adaptive']['all_success']
    assert d['point_ABI_max_absolute_difference']==0.
    assert d['precision_192_256_max_difference']==0.
    print('PASS stored certificate arithmetic; see RESULT scope for actually executed native checks')
if __name__=='__main__':main()

"""Bounded worker certification of the exact failed point; no matrix construction."""
from pathlib import Path
import argparse,json,time,warnings
import numpy as np
from scipy.integrate import quad_vec
from mapped_local import Reference
from parent_policy import explicit_per_flavor
from build_addon import read,sha
HERE=Path(__file__).resolve().parent

def adaptive_reference(ref,x,xi):
    values=[];errors=[];evaluations=0
    for panel in range(ref.count_native(x,xi)):
        def f(u):return ref.mapped_panel(x,xi,panel,[u])[0]
        with warnings.catch_warnings(record=True) as caught:
            v,e,info=quad_vec(f,0.,1.,epsabs=1e-12,epsrel=1e-11,quadrature='gk21',limit=100,full_output=True)
        if caught or not info.success or not np.isfinite(v).all():
            raise FloatingPointError(f'Adaptive check failed on panel {panel}: {info.message}; {[str(w.message) for w in caught]}')
        values.append(v);errors.append(e);evaluations+=int(info.neval)
    v=np.sum(values,axis=0);return {'integrals':v.tolist(),'estimated_error':float(sum(errors)),
       'neval':evaluations,'local':float(v[0]+3*(v[1]+explicit_per_flavor()))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=HERE.parent);args=ap.parse_args()
    build=read(HERE/'ACTIVE_BUILD.json');ref=Reference(build['library'])
    if sha(build['library'])!=build['binary_sha256']:raise ValueError('Binary changed')
    x=float.fromhex('0x1.028f5c4564889p-1');xi=float.fromhex('0x1.028f5c28f5c29p-1');t0=time.monotonic()
    fixed=ref.certify(x,xi);ind=adaptive_reference(ref,x,xi)
    discrepancy=abs(ind['local']-fixed['local'])
    if discrepancy>1e-9 or ind['estimated_error']>1e-9:raise FloatingPointError('Independent local-integral check fails')
    # One formerly verified point protects the preceding local repair's regime.
    old=ref.certify(.5050607301889878,.505)
    if abs(old['local']-(-3.975474426644551))>1e-9:raise FloatingPointError('Previous point disagreement')
    record={'pass':True,'build_key':build['key'],'binary_sha256':build['binary_sha256'],
       'fixed':fixed,'adaptive':ind,'independent_local_difference':discrepancy,
       'previous_point':old,'seconds':time.monotonic()-t0,
       'scope':'Local coefficient only; neither missing rows, evolved fields nor final geometry certified'}
    (HERE/'LOCAL_VERIFICATION.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()

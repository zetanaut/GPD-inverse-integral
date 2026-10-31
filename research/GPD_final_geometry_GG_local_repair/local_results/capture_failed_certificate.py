"""One evidence-only replay of the unchanged failed certificate; no row retry.

The production guard did not serialize its intermediate values. Read them from
the exception traceback of the identical fixed-resolution certificate call.
Never write a fallback cache or accept a failed local coefficient.
"""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys,json,hashlib,time,platform
ADDON=Path(__file__).resolve().parents[1];ROOT=ADDON.parent
sys.path.insert(0,str(ADDON))
from gg_local import Reference,NODES,TOLERANCE
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    out=ADDON/'local_results/FAILED_CERTIFICATE_VALUES.json'
    assert not out.exists(),'Do not repeat this diagnostic'
    failure_path=ROOT/'GPD_NLO_completion_package/work/operators/3469941c41e7591df44efe185683c5039e6ce759f832b125bdb02a56f29ff21d/FAILURE.json'
    failure=read(failure_path);build=read(ADDON/'ACTIVE_BUILD.json')
    assert failure['i']==120 and failure['x']==0.5050000033100029
    assert tuple(NODES)==(24,48,96) and TOLERANCE==1e-9
    assert sha(build['library'])==build['binary_sha256']
    result={'purpose':'One unchanged fixed-resolution failure-certificate replay to capture values omitted by the original guard; no row retry, cache write or cap escalation',
            'production_failure_sha256':sha(failure_path),'production_failure':failure,
            'x_hex':failure['x'].hex(),'xi_hex':failure['xi'].hex(),
            'build_key':build['key'],'binary_sha256':build['binary_sha256'],
            'algorithm_sha256':sha(ADDON/'gg_local.py'),'node_counts':list(NODES),'tolerance':TOLERANCE,
            'invocation':'/usr/bin/python3','sys_executable':sys.executable,'python':sys.version,'platform':platform.platform()}
    start=time.monotonic()
    try:
        result['unexpected_success']=Reference(build['library']).certify(failure['x'],failure['xi'])
        result['reproduced_failure']=False
    except FloatingPointError as error:
        result['reproduced_failure']=str(error)=='GG local reference refinement failed; no cap escalation'
        result['error']=repr(error)
        trace=error.__traceback__
        while trace:
            frame=trace.tb_frame
            if frame.f_code.co_name=='certify' and Path(frame.f_code.co_filename).resolve()==(ADDON/'gg_local.py').resolve():
                for source,dest in [('values','integrals'),('errors','max_successive_change'),('limits','limits')]:
                    result[dest]=frame.f_locals[source].tolist()
            trace=trace.tb_next
    result['seconds']=time.monotonic()-start
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    assert result['reproduced_failure'],'Unexpected diagnostic outcome; preserve without retry'
if __name__=='__main__':main()

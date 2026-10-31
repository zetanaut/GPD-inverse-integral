"""One failed local call, independent quadrature, and two bypass controls. No banks."""
from pathlib import Path
import argparse,ctypes as ct,importlib.util,json,sys,time
import numpy as np
from scipy.integrate import quad_vec
from gg_local import Reference,FallbackSource,panels,explicit_per_flavor,sha
HERE=Path(__file__).resolve().parent
X=.5050607301889878;XI=.505

def adaptive(reference,x,xi):
    reg,pairs=panels(x,xi);total=np.zeros(2);error=0.;neval=0
    for a,b in reg:
        v,e,info=quad_vec(lambda y:reference.values(x,xi,[y])[0],a,b,
                          epsabs=2e-10/len(reg),epsrel=1e-10,limit=150,full_output=True)
        if not info.success:raise RuntimeError('Independent regular quadrature: '+info.message)
        total+=v;error+=e;neval+=info.neval
    for pole,radius in pairs:
        v,e,info=quad_vec(lambda u:reference.values(x,xi,[radius*u*u],pole,True)[0]*2*radius*u,
                          0,1,epsabs=2e-10/len(reg),epsrel=1e-10,limit=150,full_output=True)
        if not info.success:raise RuntimeError('Independent paired quadrature: '+info.message)
        total+=v;error+=e;neval+=info.neval
    return {'integrals':total.tolist(),'estimated_error':error,'evaluations':neval,
            'local':float(total[0]+3*(total[1]+explicit_per_flavor()))}

def verify(source,reference,cache,out,build_key):
    t=time.monotonic();record={'pass':False,'build_key':build_key,'x':X,'xi':XI}
    try:
        print('START ORIGINAL',flush=True)
        try:
            _,_,local=source.evaluate(1,'GG',X,XI,[.813579],True)
            record['original']={'status':'finite','local':local}
        except FloatingPointError as exc:
            r,p,_=source.evaluate(1,'GG',X,XI,[.813579],False)
            record['original']={'status':'nonfinite_local','error':repr(exc),'regular':r.tolist(),'plus':p.tolist()}
        print('START WRAPPER',time.monotonic()-t,flush=True)
        wrapper=FallbackSource(source,reference,{'build_key':build_key},cache)
        r,p,new=wrapper.evaluate(1,'GG',X,XI,[.813579],True)
        cached,_=wrapper._path(X,XI)
        robust=(json.loads(cached.read_text())['reference'] if cached.exists() else reference.certify(X,XI))
        record['paired_Gauss']=robust
        print('START ADAPTIVE',time.monotonic()-t,flush=True)
        independent=adaptive(reference,X,XI);record['independent_adaptive']=independent
        delta=abs(robust['local']-independent['local'])
        if delta>1e-9 or independent['estimated_error']>1e-9:raise RuntimeError('Independent local comparison fails')
        if record['original']['status']=='nonfinite_local':
            if abs(new-robust['local'])>1e-12:raise RuntimeError('Fallback did not reproduce verified value')
        rr,pp,_=source.evaluate(1,'GG',X,XI,[.813579],False)
        if not np.array_equal(r,rr) or not np.array_equal(p,pp):raise RuntimeError('Point kernel changed')
        record['independent_local_difference']=delta
        record['fallback_return']=new;record['point_terms_unchanged']=True
        record['finite_bypass_controls']=[]
        print('START BYPASS',time.monotonic()-t,flush=True)
        for x,xi in [(.19,.135),(.4,.275)]:
            before=source.evaluate(1,'GG',x,xi,[.813579],False)
            after=wrapper.evaluate(1,'GG',x,xi,[.813579],False)
            if not all(np.array_equal(a,b) for a,b in zip(before,after)):raise RuntimeError('Successful original call changed')
            record['finite_bypass_controls'].append({'x':x,'xi':xi,'include_local':False,'bitwise_unchanged':True})
        for x,xi in [(XI,XI),(0.,XI)]:
            try:reference.certify(x,xi)
            except ValueError:pass
            else:raise RuntimeError('Singular reference point was accepted')
        record.update({'pass':True,'seconds':time.monotonic()-t,
                       'scope':'source-local quadrature only; no matrix, evolution, CFF, fit, or calibration'})
    except Exception as exc:
        record.update({'error':repr(exc),'seconds':time.monotonic()-t})
        Path(out).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');raise
    Path(out).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--project',type=Path,default=HERE.parent);args=a.parse_args()
    root=args.project.resolve();prep=root/'completion_runs/NLO_final_geometry';nlo=root/'GPD_NLO_completion_package'
    build=json.loads((HERE/'ACTIVE_BUILD.json').read_text());meta=json.loads((prep/'runtime_repair/MP_BUILD.json').read_text())
    spec=importlib.util.spec_from_file_location('captured_source_parts',nlo/'src/source_parts.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    parts=nlo/'work/mp_guard_bridge'/meta['key']/'parts.dylib'
    if sha(parts)!=meta['binary_sha256']:raise RuntimeError('Native library changed')
    result=verify(mod.SourceParts(parts),Reference(build['library']),HERE/'verification_cache',HERE/'LOCAL_VERIFICATION.json',build['key'])
    print(json.dumps(result,indent=2))

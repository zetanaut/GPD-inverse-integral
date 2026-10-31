#!/usr/bin/env python3
"""Compile a scalar/vector SOURCE diagnostic ABI. Never builds kernel matrices.

Uses the existing compiler; hash-keyed out-of-tree build; no fast math,
contraction, architecture flags, executable trampolines, or silent fallback.
"""
from pathlib import Path
import hashlib,json,os,platform,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
FILES=['qcd/constants.f90','qcd/alpha_qcd.f90','helpers/gridspace.f90','helpers/specfun.f90','helpers/integration.f90','helpers/pixelation.f90','helpers/software_wide.f90','kernels/kernels_common.f90','kernels/kernels_lo.f90','kernels/qg_wide.f90','kernels/kernels_nlo.f90','evolution/convolution.f90','evolution/matevo.f90']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
    fc=shutil.which(os.environ.get('FC','gfortran'))
    if not fc:raise RuntimeError('Existing gfortran not found; set FC. No auto-install.')
    flags=['-O2','-fno-fast-math','-ffp-contract=off','-ffree-line-length-none','-Werror=trampolines','-fPIC','-shared']
    if sys.platform.startswith('linux'):flags+=['-Wl,-z,noexecstack']
    paths=[(ROOT/'candidate/kernels_nlo.f90' if p=='kernels/kernels_nlo.f90' else ROOT/'frozen/fortran'/p) for p in FILES]+[ROOT/'frozen/original_bridges/bridge.f90',ROOT/'fortran/source_parts.f90']
    paths.insert(next(i for i,p in enumerate(paths) if p.name=='kernels_nlo.f90'),ROOT/'candidate/gg_wide.f90')
    sig={'compiler':fc,'version':subprocess.check_output([fc,'--version'],text=True),'flags':flags,'platform':platform.platform(),'machine':platform.machine(),'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths},'environment':{k:os.environ.get(k) for k in ['SDKROOT','MACOSX_DEPLOYMENT_TARGET','LIBRARY_PATH','CPATH']}}
    key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest()
    wd=ROOT/'work/gg_guard_bridge'/key;wd.mkdir(parents=True,exist_ok=True)
    lib=wd/('parts.dylib' if sys.platform=='darwin' else 'parts.so');meta=wd/'BUILD.json'
    if meta.exists():
        old=json.loads(meta.read_text())
        if sha(lib)!=old['binary_sha256']:raise RuntimeError('Cached binary changed; preserving evidence')
        return lib,{**old,'cache_hit':True}
    started=time.monotonic();cmd=[fc,*flags,*map(str,paths),'-o',str(lib)]
    with (wd/'compile.log').open('w') as log:
        subprocess.run(cmd,cwd=wd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    if sys.platform=='darwin':subprocess.run(['codesign','--force','--sign','-',str(lib)],capture_output=True,check=True,timeout=20)
    data={**sig,'key':key,'binary_sha256':sha(lib),'seconds':time.monotonic()-started,'command':cmd,'cache_hit':False,'scope':'source ABI build only; no kernel matrix built'}
    meta.write_text(json.dumps(data,indent=2)+'\n');return lib,data
if __name__=='__main__':
    lib,data=build();(ROOT/'work/action/GG_BUILD.json').write_text(json.dumps(data,indent=2)+'\n');print(lib)

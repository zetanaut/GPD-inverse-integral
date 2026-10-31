"""Use existing C++/Fortran/MPFR toolchain; no automatic installations."""
from pathlib import Path
import hashlib,json,os,platform,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
FILES=['qcd/constants.f90','qcd/alpha_qcd.f90','helpers/gridspace.f90','helpers/specfun.f90','helpers/integration.f90','helpers/pixelation.f90','helpers/software_wide.f90','kernels/kernels_common.f90','kernels/kernels_lo.f90','kernels/qg_wide.f90','kernels/kernels_nlo.f90','evolution/convolution.f90','evolution/matevo.f90']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
 fc=shutil.which(os.environ.get('FC','gfortran'));cxx=shutil.which(os.environ.get('CXX','clang++'));prefix=Path(os.environ.get('MPFR_PREFIX','/Users/dustin/miniforge3/envs/box'))
 if not fc or not cxx:raise RuntimeError('Existing compiler required; no install')
 suffix='.dylib' if sys.platform=='darwin' else '.so';deps=[prefix/'include/mpfr.h',prefix/f'lib/libmpfr{suffix}',prefix/f'lib/libgmp{suffix}']
 for dep in deps:
  if not dep.exists():raise FileNotFoundError(dep)
 flags=['-O2','-fno-fast-math','-ffp-contract=off','-ffree-line-length-none','-Werror=trampolines','-fPIC','-shared'];cppflags=['-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-fno-exceptions','-fno-rtti','-DGPD_MP_BITS=192','-fPIC']
 if sys.platform.startswith('linux'):flags+=['-Wl,-z,noexecstack']
 paths=[(ROOT/'candidate/kernels_nlo_mp.f90' if p=='kernels/kernels_nlo.f90' else ROOT/'frozen/fortran'/p) for p in FILES]+[ROOT/'frozen/original_bridges/bridge.f90',ROOT/'fortran/source_parts.f90']
 paths.insert(next(i for i,p in enumerate(paths) if p.name=='kernels_nlo_mp.f90'),ROOT/'candidate/mp_guard.f90')
 sources=paths+[ROOT/'candidate/mp_kernel.cpp',Path(__file__)]
 sig={'compiler':fc,'version':subprocess.check_output([fc,'--version'],text=True),'cxx':cxx,'cxx_version':subprocess.check_output([cxx,'--version'],text=True),'flags':flags,'cppflags':cppflags,'platform':platform.platform(),'machine':platform.machine(),'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources},'external_dependencies':{str(p):sha(p) for p in deps},'environment':{k:os.environ.get(k) for k in ['SDKROOT','MACOSX_DEPLOYMENT_TARGET','LIBRARY_PATH','CPATH']}}
 key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();wd=ROOT/'work/mp_guard_bridge'/key;wd.mkdir(parents=True,exist_ok=True);lib=wd/('parts'+suffix);meta=wd/'BUILD.json'
 if meta.exists():
  old=json.loads(meta.read_text())
  if sha(lib)!=old['binary_sha256']:raise RuntimeError('Cache hash mismatch')
  return lib,{**old,'cache_hit':True}
 cppcmd=[cxx,*cppflags,'-I'+str(prefix/'include'),'-c',str(ROOT/'candidate/mp_kernel.cpp'),'-o',str(wd/'guard.o')]
 cmd=[fc,*flags,*map(str,paths),str(wd/'guard.o'),'-L'+str(prefix/'lib'),'-lmpfr','-lgmp',*(['-Wl,-rpath,'+str(prefix/'lib')] if sys.platform!='darwin' else []),'-o',str(lib)]
 t=time.monotonic()
 with (wd/'compile.log').open('w') as log:
  subprocess.run(cppcmd,cwd=wd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180);subprocess.run(cmd,cwd=wd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
 if sys.platform=='darwin':subprocess.run(['codesign','--force','--sign','-',str(lib)],capture_output=True,check=True,timeout=20)
 data={**sig,'key':key,'binary_sha256':sha(lib),'seconds':time.monotonic()-t,'commands':[cppcmd,cmd],'cache_hit':False};meta.write_text(json.dumps(data,indent=2)+'\n');return lib,data
if __name__=='__main__':
 lib,data=build();(ROOT/'work/action/MP_BUILD.json').write_text(json.dumps(data,indent=2)+'\n');print(lib)

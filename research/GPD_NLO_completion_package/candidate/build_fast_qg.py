"""Build/adopt the measured MPFR QG guard with full provenance; no installs."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,platform,shutil,argparse,time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(adopt=False):
 cxx=shutil.which(os.environ.get('CXX','clang++'));prefix=Path(os.environ.get('MPFR_PREFIX','/Users/dustin/miniforge3/envs/box'));suffix='.dylib' if sys.platform=='darwin' else '.so';flags=['-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-DGPD_MP_BITS=192','-fno-exceptions','-fno-rtti','-fPIC','-shared']
 if not cxx:raise RuntimeError('Existing C++ compiler required')
 deps=[prefix/'include/mpfr.h',prefix/f'lib/libmpfr{suffix}',prefix/f'lib/libgmp{suffix}'];sources=[ROOT/'candidate/mp_qg.cpp',ROOT/'candidate/mp_kernel.cpp']
 sig={'compiler':cxx,'version':subprocess.check_output([cxx,'--version'],text=True),'flags':flags,'external_dependencies':{str(p):sha(p) for p in deps},'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'platform':platform.platform(),'environment':{k:os.environ.get(k) for k in ['SDKROOT','MACOSX_DEPLOYMENT_TARGET','LIBRARY_PATH','CPATH']}}
 key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();wd=ROOT/'work/fast_qg_bridge'/key;wd.mkdir(parents=True,exist_ok=True);lib=wd/('qg'+suffix);metadata=wd/'BUILD.json';t=time.monotonic()
 cmd=[cxx,*flags,'-I'+str(prefix/'include'),str(ROOT/'candidate/mp_qg.cpp'),'-L'+str(prefix/'lib'),'-lmpfr','-lgmp','-Wl,-rpath,'+str(prefix/'lib'),'-o',str(lib)]
 if metadata.exists():
  m=json.loads(metadata.read_text());assert sha(lib)==m['binary_sha256'];return lib,m
 if adopt:
  check=json.loads((ROOT/'work/action/FAST_QG_CHECKS.json').read_text());assert check['max_scaled_error']==0
  for p in sources:assert check['source_sha256'][p.name]==sha(p)
  shutil.copy2(ROOT/'work/mpfr_probe'/('qg'+suffix),lib)
 else:
  with (wd/'compile.log').open('w') as log:subprocess.run(cmd,cwd=wd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
 m={**sig,'key':key,'binary_sha256':sha(lib),'seconds':time.monotonic()-t,'command':cmd,'adopted_measured_probe':adopt,'point_control':'work/action/FAST_QG_CHECKS.json'};metadata.write_text(json.dumps(m,indent=2)+'\n');return lib,m
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--adopt-probe',action='store_true');a=ap.parse_args();lib,m=build(a.adopt_probe);(ROOT/'work/action/FAST_QG_BUILD.json').write_text(json.dumps(m,indent=2)+'\n');print(lib)

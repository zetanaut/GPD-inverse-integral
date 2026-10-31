"""Build only the small paired-GG evaluator using the EXISTING repaired runtime."""
from pathlib import Path
import hashlib,json,os,platform,shutil,subprocess,sys
from gg_local import sha,CPP_SHA256
HERE=Path(__file__).resolve().parent

def build(project):
    project=Path(project).resolve();prep=project/'completion_runs/NLO_final_geometry'
    nlo=project/'GPD_NLO_completion_package';meta=json.loads((prep/'runtime_repair/MP_BUILD.json').read_text())
    cpp=nlo/'candidate/mp_kernel.cpp'
    if sha(cpp)!=CPP_SHA256:raise RuntimeError('Unexpected captured GG source; do not substitute')
    header=next(Path(p) for p in meta['external_dependencies'] if p.endswith('/mpfr.h'))
    if not header.is_file() or sha(header)!=meta['external_dependencies'][str(header)]:
        raise RuntimeError('Existing production MPFR header does not match')
    repair=json.loads((prep/'runtime_repair/BUNDLE.json').read_text());bundle=Path(repair['bundle'])
    provenance=json.loads((bundle/'REPAIR.json').read_text())
    if sys.platform!='darwin':raise RuntimeError('Worker builder expects captured macOS runtime; Linux reviewer used a separate diagnostic build')
    deps=[bundle/'libmpfr.6.dylib',bundle/'libgmp.10.dylib']
    for dep in deps:
        if sha(dep)!=provenance['libraries'][dep.name]['sha256']:raise RuntimeError('Repaired dependency hash mismatch')
    cxx=meta['cxx']
    if not Path(cxx).is_file():raise RuntimeError('Original compiler missing; no installation authorized')
    sig={'original_point_source':CPP_SHA256,'addon_source':sha(HERE/'local_reference.cpp'),
         'quadrature_source':sha(HERE/'gg_local.py'),'builder':sha(Path(__file__)),
         'header':sha(header),'dependencies':{str(p):sha(p) for p in deps},
         'compiler':cxx,'version':subprocess.check_output([cxx,'--version'],text=True),
         'platform':platform.platform(),'bits':192,'source_build_key':meta['key']}
    key=hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest();wd=HERE/'work'/key;wd.mkdir(parents=True,exist_ok=True)
    dest=wd/'BUILD.json';lib=wd/'gg_local.dylib'
    if dest.exists():
        result=json.loads(dest.read_text())
        if sha(lib)!=result['binary_sha256']:raise RuntimeError('Addon binary changed')
    else:
        shutil.copy2(cpp,wd/'mp_kernel.cpp');shutil.copy2(HERE/'local_reference.cpp',wd/'local_reference.cpp')
        cmd=[cxx,'-std=c++17','-O2','-fPIC','-shared','-Wl,-headerpad_max_install_names','-fno-fast-math','-ffp-contract=off',
             '-fno-exceptions','-fno-rtti','-DGPD_MP_BITS=192','-I'+str(header.parent),
             str(wd/'local_reference.cpp'),*map(str,deps),'-o',str(lib)]
        commands=[cmd]
        with (wd/'BUILD.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
            # Fix load commands on the NEW addon only. Never mutate a dependency.
            for dep in deps:
                cmd=['/usr/bin/install_name_tool','-change','@loader_path/'+dep.name,str(dep),str(lib)]
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=15);commands.append(cmd)
            cmd=['/usr/bin/codesign','--force','--sign','-',str(lib)]
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=15);commands.append(cmd)
        result={'key':key,'library':str(lib),'binary_sha256':sha(lib),'signature':sig,'commands':commands}
        dest.write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'ACTIVE_BUILD.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parent
    print(json.dumps(build(root),indent=2))

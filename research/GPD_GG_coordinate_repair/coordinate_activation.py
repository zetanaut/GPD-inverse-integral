"""Nested fallback: retain every successful original AND previous-addon result."""
from pathlib import Path
import importlib.util,json,sys
from mapped_local import Reference
from parent_policy import FallbackSource
from build_addon import sha,read,PARENT_SHA
HERE=Path(__file__).resolve().parent

def parent_activation(project):
    old=Path(project).resolve()/'GPD_final_geometry_GG_local_repair'
    if sha(old/'gg_local.py')!=PARENT_SHA:raise RuntimeError('Parent quadrature changed')
    # The parent module imports gg_local. Reject an unrelated existing binding.
    mod=sys.modules.get('gg_local')
    if mod is not None and Path(mod.__file__).resolve()!=old/'gg_local.py':raise RuntimeError('Unrelated gg_local module loaded')
    sys.path.insert(0,str(old))
    spec=importlib.util.spec_from_file_location('_gg_coordinate_parent_activation',old/'activation.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def identity(project=None):
    project=HERE.parent if project is None else Path(project).resolve()
    record=read(HERE/'ACTIVE_BUILD.json');v=read(HERE/'LOCAL_VERIFICATION.json')
    if v.get('pass') is not True or v['build_key']!=record['key']:raise RuntimeError('Verify exact new build before activation')
    if sha(record['library'])!=record['binary_sha256']:raise RuntimeError('Coordinate library changed')
    for filename,key in [('local_reference.cpp','cpp'),('mapped_local.py','python'),('parent_policy.py','parent_policy_copy')]:
        if sha(HERE/filename)!=record['signature'][key]:raise RuntimeError('Coordinate algorithm changed')
    return {'build_key':record['key'],'binary_sha256':record['binary_sha256'],
      'algorithm_sha256':sha(HERE/'mapped_local.py'),'activation_sha256':sha(__file__),
      'parent':parent_activation(project).identity()}

def wrap_source(source,project=None):
    project=HERE.parent if project is None else Path(project).resolve()
    sig=identity(project);old=parent_activation(project).wrap_source(source)
    return FallbackSource(old,Reference(read(HERE/'ACTIVE_BUILD.json')['library']),sig,HERE/'fallback_cache')

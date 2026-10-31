"""Explicit activation after verification; integrate into an isolated assembly copy."""
from pathlib import Path
import json
from gg_local import Reference,FallbackSource,sha
HERE=Path(__file__).resolve().parent

def identity():
    record=json.loads((HERE/'ACTIVE_BUILD.json').read_text())
    if not (HERE/'LOCAL_VERIFICATION.json').is_file():
        raise RuntimeError('Run verify_native.py before activation')
    verified=json.loads((HERE/'LOCAL_VERIFICATION.json').read_text())
    if verified.get('pass') is not True or verified['build_key']!=record['key']:
        raise RuntimeError('The exact addon build must pass local verification')
    if sha(record['library'])!=record['binary_sha256']:raise RuntimeError('Addon binary changed')
    if sha(HERE/'gg_local.py')!=record['signature']['quadrature_source']:
        raise RuntimeError('Fallback algorithm changed')
    return {'build_key':record['key'],'binary_sha256':record['binary_sha256'],
            'algorithm_sha256':sha(HERE/'gg_local.py'),'activation_sha256':sha(Path(__file__))}

def wrap_source(source):
    sig=identity();record=json.loads((HERE/'ACTIVE_BUILD.json').read_text())
    return FallbackSource(source,Reference(record['library']),sig,HERE/'fallback_cache')

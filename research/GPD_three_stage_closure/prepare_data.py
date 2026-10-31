#!/usr/bin/env python3
"""Generate synthetic observations. No experimental ReH means used; no fits.

Reference-map targets and nominal-map discrepancies are kept distinct. The
fitter consumes observed/<case>.npz only. Ground truth lives in evaluation_only/.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from closure.bank import RowBank
from closure.truths import Truth
from closure.io import atomic_json,atomic_npz,sha256
ROOT=Path(__file__).resolve().parent

def preserve_supplied_npz(path, **arrays):
    """Keep packaged arrays if regeneration differs only by float roundoff."""
    path=Path(path)
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays):raise FileExistsError(f'Conflicting immutable arrays: {path}')
            for key,value in arrays.items():
                value=np.asarray(value)
                prior=old[key]
                if prior.shape!=value.shape or prior.dtype.kind!=value.dtype.kind:
                    raise FileExistsError(f'Conflicting immutable arrays: {path} ({key})')
                if value.dtype.kind=='f':
                    same=np.allclose(prior,value,rtol=0,atol=1e-13,equal_nan=False)
                else:same=np.array_equal(prior,value)
                if not same:raise FileExistsError(f'Conflicting immutable arrays: {path} ({key})')
        return
    atomic_npz(path,**arrays)

def encode_observations(rows,mean,components,perturbation=None):
    ci=np.asarray(components,int)
    if ci.tolist() not in ([0],[0,1]):raise ValueError('Real or complex only')
    y=np.asarray(mean,float)[:,ci].copy()
    if perturbation is not None:
        if ci.tolist()!=[0]:raise ValueError('This plan only adds Re noise')
        if np.asarray(perturbation).shape!=(len(rows),):raise ValueError('Bad noise shape')
        y[:,0]+=perturbation
    return dict(row_id=np.array([r['row_id'] for r in rows]),
        xi=np.array([r['xi'] for r in rows]),Q2=np.array([r['Q2_GeV2'] for r in rows]),
        t=np.array([r['t_GeV2'] for r in rows]),mass=np.array([r['mass'] for r in rows]),
        components=ci,y=y,sigma=np.array([[r['sigma_Re'],r['sigma_Im']] for r in rows])[:,ci])

def prepare(prepared,extra_banks=None,scope='all'):
    prepared=Path(prepared);spec=json.loads((prepared/'STUDY.json').read_text());geom=json.loads((prepared/'GEOMETRY.json').read_text())
    bank=RowBank(ROOT/'inputs/accepted_experimental_banks',*(extra_banks or []))
    tasks=[task for task in spec['core_tasks'] if scope=='all' or task['geometry']=='experimental']
    required={task['geometry'] for task in tasks}
    if scope=='all':required.add('surface_check')
    missing=sorted({(r['xi'],r['Q2_GeV2']) for g in required for r in geom['grids'][g] if not bank.has(r['xi'],r['Q2_GeV2'])})
    if missing:raise RuntimeError(f'{len(missing)} exact scale rows missing; run export_missing_rows.py on the source-generator machine. No interpolation fallback.')
    summaries=[]
    for truth_name in ('Z1','Z2'):
        truth=Truth(truth_name);means={}
        for geometry_name in sorted(required):
            rows=geom['grids'][geometry_name];ref=bank.field_cff(rows,truth,'reference');nom=bank.field_cff(rows,truth,'nominal')
            means[geometry_name]=ref
            preserve_supplied_npz(prepared/'evaluation_only'/f'{truth_name}_{geometry_name}.npz',
                row_id=[r['row_id'] for r in rows],mean_reference=ref,mean_nominal=nom,
                xi=[r['xi'] for r in rows],Q2=[r['Q2_GeV2'] for r in rows],t=[r['t_GeV2'] for r in rows])
            sig=np.array([[r['sigma_Re'],r['sigma_Im']] for r in rows]);delta=nom-ref
            summaries.append({'truth':truth_name,'geometry':geometry_name,'n_rows':len(rows),
                'max_abs_component_difference':np.max(abs(delta),axis=0).tolist(),
                'relative_L2_component_difference':(np.linalg.norm(delta,axis=0)/np.maximum(np.linalg.norm(ref,axis=0),1e-15)).tolist(),
                'RMS_difference_in_numerical_weight_units':np.sqrt(np.mean((delta/sig)**2,axis=0)).tolist(),
                'interpretation':'independent representation/resolution, SHARED physical kernels; not a continuum bound'})
        rows=geom['grids']['experimental'];sigma=np.array([r['sigma_Re'] for r in rows])
        rng=np.random.Generator(np.random.PCG64(spec['observation_rule']['noise_seeds'][truth_name]))
        delta=sigma*rng.standard_normal(len(rows))
        preserve_supplied_npz(prepared/'evaluation_only'/f'{truth_name}_noise.npz',noise=delta,sigma=sigma)
        for task in tasks:
            if task['truth']!=truth_name:continue
            rows=geom['grids'][task['geometry']];noise=delta if task['branch']=='exp_Re_noisy' else None
            arrays=encode_observations(rows,means[task['geometry']],task['components'],noise)
            # No true function, withheld imaginary values or true coefficients in fit files.
            preserve_supplied_npz(prepared/'observed'/f"{task['branch']}_{truth_name}.npz",**arrays)
    record={'status':'OBSERVATIONS_PREPARED_NO_FITS','scope':scope,'study_sha256':sha256(prepared/'STUDY.json'),
        'bank_indices':bank.index_hashes,'comparisons':summaries}
    atomic_json(prepared/f'DATA_{scope}_AUDIT.json',record)
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,default=ROOT/'prepared');p.add_argument('--bank',action='append',type=Path,default=[])
    p.add_argument('--scope',choices=('experimental','all'),default='all');a=p.parse_args()
    print(json.dumps(prepare(a.prepared,a.bank,a.scope),indent=2))

#!/usr/bin/env python3
"""Read-only experimental-result export. No optimizer, noise sampler, or kernel builder.

Default output is replayed/; the shipped results/ are never overwritten. Saved
ReH tables are arithmetically checked, NOT newly recomputed through native NLO.
H+ curves and targets ARE evaluated afresh from unchanged NumPy weights.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, sys, time
from pathlib import Path
from typing import Iterable
import numpy as np

ROOT = Path(__file__).resolve().parent
STATES = ('parent7','control7','central13','zero','plus','minus')
CENTRAL = 'central13'  # Existing scientific nomination, not minimum-J reselection.
QUANTITIES = ('bin_mean_x_02_04','positive_x_first_moment')

def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read_csv(p: Path) -> list[dict[str,str]]:
    with p.open(newline='') as f: return list(csv.DictReader(f))

def write_csv(p: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError('Empty output table')
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dump(p: Path, value) -> None:
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def finite(values, label='values') -> np.ndarray:
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all(): raise ValueError(f'Nonfinite {label}')
    return a

def residual_summary(y,p,s) -> dict:
    y,p,s=finite(y),finite(p),finite(s)
    if y.shape!=p.shape or s.shape!=y.shape or np.any(s<=0):
        raise ValueError('Mismatched arrays or nonpositive reported errors')
    pulls=(y-p)/s
    return dict(n=len(y),weighted_residual_sum=float(pulls@pulls),
                mean_squared_error_units=float(np.mean(pulls**2)),
                RMS_error_units=float(np.sqrt(np.mean(pulls**2))),
                mean_signed_pull=float(np.mean(pulls)),
                n_abs_pull_gt3=int(np.sum(abs(pulls)>3)),
                max_abs_pull=float(np.max(abs(pulls))))

def decompose(parent,zero,plus,minus):
    parent,zero,plus,minus=map(finite,(parent,zero,plus,minus))
    if len({a.shape for a in (parent,zero,plus,minus)})!=1:
        raise ValueError('Incompatible response shapes')
    return zero-parent,(plus-minus)/2,(plus+minus)/2-zero

def field(weights: dict[str,np.ndarray], x, xi, t, t_mean, t_std):
    """Independent NumPy inference of the frozen t-conditioned FiLM; output B,3,N."""
    x,xi,t=finite(x),finite(xi),finite(t)
    if x.ndim!=2 or xi.shape!=(len(x),) or t.shape!=xi.shape:
        raise ValueError('Expected x[B,N], xi[B], t[B]')
    if np.any(abs(x)>1) or np.any((xi<=0)|(xi>=1)) or not math.isfinite(t_std) or t_std<=0:
        raise ValueError('Invalid coordinate domain or normalization')
    c=np.stack(((xi**2-.18)/.18,(t-t_mean)/t_std),axis=-1)
    def linear(a,k):return a@weights[k+'.weight'].T+weights[k+'.bias']
    def raw(q):
        h=np.tanh(linear(q[...,None],'input'))
        for j in range(2):
            v=linear(np.tanh(linear(c,f'conditioners.{j}.0')),f'conditioners.{j}.2')
            g,b=np.split(v,2,axis=-1)
            h=h+np.tanh((1+g[:,None,:])*linear(h,f'blocks.{j}')+b[:,None,:])/2
        return linear(h,'output')
    p,m=raw(x),raw(-x)
    h=np.concatenate(((p[...,:2]-m[...,:2])/2,(p[...,2:]+m[...,2:])/2),axis=-1)
    return (h*np.maximum(1-x**2,0)[...,None]**2).transpose(0,2,1)

def charge(weights,x,xi,t,meta):
    a=field(weights,x,xi,t,meta['t_mean'],meta['t_std'])
    return (2/9)*(a[:,0,:]+a[:,1,:])

def load_weights(root:Path):
    meta=json.loads((root/'models/index.json').read_text());out={}
    if set(meta)!=set(STATES):raise ValueError('Unexpected checkpoint set')
    for s in STATES:
        m=meta[s];p=root/m['npz']
        if digest(p)!=m['npz_sha256']:raise ValueError(f'Changed converted weights: {s}')
        if digest(root/m['original_checkpoint'])!=m['checkpoint_sha256']:
            raise ValueError(f'Changed original checkpoint: {s}')
        with np.load(p,allow_pickle=False) as f: out[s]={k:finite(f[k],s) for k in f.files}
        if sum(a.size for a in out[s].values())!=3867:raise ValueError('Changed architecture')
    return out,meta

def source_check(root:Path):
    m=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    for r in m['files']:
        p=root/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
            raise ValueError(f'Changed source member: {r["path"]}')
    return len(m['files'])

def gquad(lo,hi,xi,order):
    """Split integration at crossover; do not infer derivatives there."""
    z,w=np.polynomial.legendre.leggauss(order)
    edges=sorted(set([lo,hi]+([xi] if lo<xi<hi else [])))
    return tuple(np.concatenate(parts) for parts in zip(*[
        ((b-a)*z/2+(a+b)/2,(b-a)*w/2) for a,b in zip(edges[:-1],edges[1:])]))

def target_values(weights,meta,xi,t,order):
    result=[]
    for lo,hi,power,normal in [(.2,.4,0,.2),(0,1,1,1.)]:
        x,w=gquad(lo,hi,xi,order)
        h=charge(weights,x[None,:],np.array([xi]),np.array([t]),meta)[0]
        result.append(float(w@(x**power*h)/normal))
    return np.array(result)

def normalize_predictions(root,observations):
    source={r['row_id']:r for r in observations};out=[]
    rows=read_csv(root/'evidence/branch_predictions.csv')
    label={'parent':'parent7','control':'control7','enriched':'central13'}
    for r in rows:
        if r['state'] in label:
            out.append((label[r['state']],r,'ReH_source','ReH_source'))
    for s in ['zero','plus','minus']:
        out.extend((s,r,'ReH_original_source','ReH_case_source') for r in read_csv(root/f'evidence/{s}_row_predictions.csv'))
    result=[];seen=set()
    for s,r,original,own in out:
        rid=r['row_id'];src=source[rid]
        if (s,rid) in seen:raise ValueError('Duplicate prediction row')
        seen.add((s,rid))
        for c in ['xi','Q2_GeV2','t_GeV2','sigma_reported']:
            if float(r[c])!=float(src[c]):raise ValueError(f'Coordinate/error changed: {s}/{rid}/{c}')
        if float(r[original])!=float(src['ReH_source']):raise ValueError('Changed central CFF input')
        pred=float(r['ReH_pred_source']);ref=float(r['ReH_reference_source']);y=float(src['ReH_source']);sd=float(src['sigma_reported'])
        result.append(dict(state=s,row_id=rid,source_set=src['source_set'],method=src['method'],
            xi=float(src['xi']),Q2_GeV2=float(src['Q2_GeV2']),t_GeV2=float(src['t_GeV2']),
            leading_power_boundary_flag=int(src['leading_power_boundary_flag']),
            ReH_source=y,sigma_reported_combined=sd,ReH_case_source=float(r[own]),
            ReH_fitted_source=pred,ImH_unfitted_source=float(r['ImH_unfitted_source']),
            ReH_reference_source=ref,nominal_reference_Re_abs=abs(pred-ref),
            pull_original=(y-pred)/sd,pull_own=(float(r[own])-pred)/sd))
    if len(result)!=6*len(observations):raise ValueError('Incomplete six-state row export')
    return result

def main(out:Path):
    start=time.perf_counter();count=source_check(ROOT);w,meta=load_weights(ROOT)
    obs=read_csv(ROOT/'evidence/observations.csv');anchors=read_csv(ROOT/'evidence/plot_anchors.csv')
    if len(obs)!=160 or len({r['row_id'] for r in obs})!=160:raise ValueError('Changed observation roster')
    if len(anchors)!=6:raise ValueError('Changed plot-anchor roster')
    out=out.resolve()
    if out in (ROOT,ROOT/'evidence',ROOT/'models',ROOT/'results') or ROOT/'evidence' in out.parents or ROOT/'models' in out.parents:
        raise ValueError('Refuse to overwrite frozen source or supplied results')
    out.mkdir(parents=True,exist_ok=True)
    pred=normalize_predictions(ROOT,obs);write_csv(out/'ReH_predictions.csv',pred)
    metrics={}
    for s in STATES:
        r=[a for a in pred if a['state']==s]
        z=residual_summary([a['ReH_source'] for a in r],[a['ReH_fitted_source'] for a in r],[a['sigma_reported_combined'] for a in r])
        z['own_input_RMS']=residual_summary([a['ReH_case_source'] for a in r],[a['ReH_fitted_source'] for a in r],[a['sigma_reported_combined'] for a in r])['RMS_error_units']
        z['max_two_map_abs_Re']=max(a['nominal_reference_Re_abs'] for a in r)
        metrics[s]=z
    write_csv(out/'fit_diagnostics.csv',[dict(state=s,**metrics[s]) for s in STATES])
    # Source branch curves and input-response parent are checked against each other.
    bc=read_csv(ROOT/'evidence/branch_curves.csv');rc=read_csv(ROOT/'evidence/response_curves.csv')
    curves={}
    for r in bc:
        s={'parent':'parent7','control':'control7','enriched':'central13'}.get(r['state'])
        if s: curves[(s,r['anchor_row_id'],float(r['x']))]=float(r['Hcharge_plus'])
    source_cross=0.
    for r in rc:
        key=('central13',r['anchor_row_id'],float(r['x']))
        source_cross=max(source_cross,abs(curves[key]-float(r['parent'])))
        for s in ['zero','plus','minus']:curves[(s,r['anchor_row_id'],float(r['x']))]=float(r[s])
    if source_cross>5e-12:raise ValueError('Response parent differs from nominated central curve')
    cr=[];ct=[];diffs=[];replaymax=0.;quadmax=0.;target_csv_max=0.
    target_raw={s:{r['anchor_row_id']:r for r in read_csv(ROOT/f'evidence/{s}_anchor_targets.csv')} for s in ['zero','plus','minus']}
    for a in anchors:
        rid=a['row_id'];xi=float(a['xi']);t=float(a['t_GeV2'])
        x=np.array(sorted(k[2] for k in curves if k[:2]==('central13',rid)))
        hs={};tv={}
        for s in STATES:
            h=charge(w[s],x[None,:],np.array([xi]),np.array([t]),meta[s])[0];hs[s]=h
            old=np.array([curves[s,rid,xx] for xx in x]);replaymax=max(replaymax,float(np.max(abs(h-old))))
            for xx,hh in zip(x,h):cr.append(dict(state=s,anchor_row_id=rid,xi=xi,t_GeV2=t,Q02_GeV2=2.,x=float(xx),Hcharge_plus=float(hh)))
            ta=target_values(w[s],meta[s],xi,t,192);tb=target_values(w[s],meta[s],xi,t,384);tv[s]=tb
            qdiff=float(np.max(abs(ta-tb)));quadmax=max(quadmax,qdiff)
            ct.append(dict(state=s,anchor_row_id=rid,xi=xi,t_GeV2=t,Q02_GeV2=2.,**dict(zip(QUANTITIES,map(float,tb))),quadrature_192_384_abs=qdiff))
            if s in target_raw:
                target_csv_max=max(target_csv_max,max(abs(tb[j]-float(target_raw[s][rid][q])) for j,q in enumerate(QUANTITIES)))
        drift,odd,even=decompose(hs['central13'],hs['zero'],hs['plus'],hs['minus'])
        for i,xx in enumerate(x):
            diffs.append(dict(anchor_row_id=rid,xi=xi,t_GeV2=t,Q02_GeV2=2.,x=float(xx),
              central13=float(hs['central13'][i]),method_central13_minus_control7=float(hs['central13'][i]-hs['control7'][i]),
              zero_continuation_drift=float(drift[i]),one_direction_odd_response=float(odd[i]),one_direction_even_response=float(even[i])))
    if replaymax>5e-12 or target_csv_max>5e-12:raise ValueError(f'Saved field/target mismatch: {replaymax}, {target_csv_max}')
    write_csv(out/'Hplus_curves.csv',cr);write_csv(out/'anchor_targets.csv',ct);write_csv(out/'signed_method_and_response.csv',diffs)
    # Uncertainty interpretation is a schema, not a formula adding unlike diagnostics.
    status={'central_state':CENTRAL,'central_selection':'Existing thirteen-t nomination; not changed by this export',
      'input_sigma':'Author-designated reported combined statistical/systematic errors, unchanged and used once',
      'input_covariance_for_pilot':'Declared marginal-Gaussian diagonal approximation; no empirical covariance invented',
      'Hplus_confidence_band':None,'total_GPD_uncertainty':None,'posterior':False,
      'method_difference':'Signed central13-control7; two deterministic formulation choices, not an ensemble variance',
      'continuation_drift':'zero-central13, not input noise and not silently subtracted',
      'one_pair_response':'One frozen antithetic pair; odd/even response diagnostics, not a standard deviation',
      'calibration_transfer':'No A/B synthetic cutoff or Z1 closure percentage assigned to experiment',
      'interpretation':'Conditional finite-budget central reconstruction and sensitivity diagnostics only'}
    dump(out/'UNCERTAINTY_STATUS.json',status)
    report={'sources_checked':count,'immutable_checkpoints':6,'fitter_or_kernel_executions':0,
      'saved_prediction_rows_arithmetically_checked':len(pred),'new_native_evolution_checks':0,
      'fresh_numpy_curve_values':len(cr),'fresh_target_values_order384':len(ct)*2,
      'max_curve_difference_from_saved':replaymax,'max_target_difference_from_saved':target_csv_max,
      'max_target_quadrature_192_384':quadmax,'response_parent_crosscheck_max':source_cross,
      'central_state':CENTRAL,'fit_diagnostics':metrics,'python':sys.version.split()[0],'numpy':np.__version__,
      'platform':platform.platform(),'elapsed_seconds':time.perf_counter()-start,
      'scope':'New saved-output export and NumPy inference/target quadrature; no fitting, kernel changes, full NLO replay, new moments, uncertainty calibration, or manuscript edits.'}
    dump(out/'AUDIT.json',report)
    print(json.dumps({k:report[k] for k in ['sources_checked','fresh_numpy_curve_values','fresh_target_values_order384','max_curve_difference_from_saved','max_target_difference_from_saved','elapsed_seconds']},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'replayed')
    args=p.parse_args();main(args.out)

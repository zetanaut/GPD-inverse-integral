#!/usr/bin/env python3
"""Read-only pilot replay. No fitting, kernel construction, or noise generation.
Shared numerical maps are replayed; native/time/adjoint checks are not rerun.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys
from functools import lru_cache
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import numpy as np
import torch

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def dump(p,d):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def weights(cp):
    w={k:v.detach().cpu().numpy() for k,v in cp['state_dict'].items()}
    flat=np.concatenate([v.ravel() for v in w.values()])
    if flat.size!=3867 or not np.array_equal(flat,cp['flat'].detach().cpu().numpy()):
        raise ValueError('Invalid FiLM state/flat pair')
    return w

def field(w,x,xi,t,meta):
    """Independent direct NumPy evaluation, B x 3 x N."""
    x,xi,t=map(lambda a:np.asarray(a,dtype=float),(x,xi,t))
    c=np.column_stack(((xi**2-.18)/.18,(t-meta['t_mean'])/meta['t_std_population']))
    def lin(v,k): return v@w[k+'.weight'].T+w[k+'.bias']
    def raw(z):
        h=np.tanh(lin(z[...,None],'input'))
        for j in range(2):
            g,b=np.split(lin(np.tanh(lin(c,f'conditioners.{j}.0')),f'conditioners.{j}.2'),2,axis=-1)
            h=h+.5*np.tanh((1+g[:,None,:])*lin(h,f'blocks.{j}')+b[:,None,:])
        return lin(h,'output')
    plus,minus=raw(x),raw(-x)
    h=np.concatenate((.5*(plus[...,:2]-minus[...,:2]),.5*(plus[...,2:]+minus[...,2:])),axis=-1)
    return (h*np.maximum(1-x*x,0)[...,None]**2).transpose(0,2,1)

@lru_cache(maxsize=12)
def nodes(n): return np.polynomial.legendre.leggauss(n)
def split_nodes(xi,n):
    a,b=nodes(n);u=(a+1)/2;v=b/2;xi=np.asarray(xi)
    return (np.concatenate((xi[:,None]*u,xi[:,None]+(1-xi[:,None])*u),axis=1),
            np.concatenate((xi[:,None]*v,(1-xi[:,None])*v),axis=1))
def moment_scale(ch,spin):
    return 16/((spin+1)*(spin+3)*(spin+5)) if ch<2 else 16/((spin-1)*(spin+1)*(spin+3))
def raw_moments(w,meta,xi,t,n):
    x,g=split_nodes(xi,n);h=field(w,x,xi,np.full(len(xi),t),meta)
    return np.column_stack([2*np.sum(g*x**(s-(1 if c<2 else 2))*h[:,c],axis=1)
                           for c in range(3) for s in (2,4,6,8)])
def polynomial_matrix(xi,s):
    return np.column_stack([np.asarray(xi)**(2*k) for k in range(s//2+1)])
def moment_loss(w,meta,ts,n=48):
    xi=np.linspace(.02,.98,41);total=0.
    for t in ts:
        m=raw_moments(w,meta,xi,t,n)
        for i,(c,s) in enumerate((c,s) for c in range(3) for s in (2,4,6,8)):
            p=polynomial_matrix(xi,s)
            r=m[:,i]-p@np.linalg.lstsq(p,m[:,i],rcond=None)[0]
            total+=np.sum((r/moment_scale(c,s))**2)
    return float(total/(len(ts)*len(xi)*12))
def diagnostic_moments(w,meta,ts,n=192):
    fitxi=np.linspace(.02,.98,41);chkxi=.02+.96*(np.arange(80)+.5)/80;out=[]
    for t in ts:
        mf=raw_moments(w,meta,fitxi,t,n);mc=raw_moments(w,meta,chkxi,t,n);errs=[]
        for i,(c,s) in enumerate((c,s) for c in range(3) for s in (2,4,6,8)):
            coef=np.linalg.lstsq(polynomial_matrix(fitxi,s),mf[:,i],rcond=None)[0]
            r=(mc[:,i]-polynomial_matrix(chkxi,s)@coef)/moment_scale(c,s);errs.append(r)
            out.append(dict(t=float(t),channel=['Sigma','T_rescaled','Hg'][c],spin=s,RMS=float(np.sqrt(np.mean(r*r)))))
        for i,s in enumerate((2,4,6,8)):
            r=(2/9)*(errs[i]+errs[i+4])
            out.append(dict(t=float(t),channel='Hcharge_plus_same_quark_scale',spin=s,RMS=float(np.sqrt(np.mean(r*r)))))
    return out

def main(args):
    p=args.pilot.resolve();project=args.project.resolve();exp=project/'GPD_ReH_experimental_run'
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if out==p or p in out.parents: raise ValueError('Output must not be inside immutable pilot')
    manifest=json.loads((p/'PACKAGE_MANIFEST.json').read_text())
    for n,h in manifest['artifacts'].items():
        if sha(p/n)!=h: raise ValueError('Changed pilot member '+n)
    meta=json.loads((exp/'prepared/INPUT_SUMMARY.json').read_text());obs=readcsv(exp/'prepared/observations.csv')
    with np.load(p/'prepared/surface_observations.npz',allow_pickle=False) as f: surf={k:f[k] for k in f.files}
    source=exp/'inputs/final_summary_by_DNN.csv'
    if sha(source)!='8d7274db9dd7e4fcc3ab84c0230701242274e42868f9c3e66e3d61a1a6838696':raise ValueError('Wrong source')
    sigma=np.array([float(r['sigma_reported']) for r in obs]);y=np.array([float(r['ReH_source']) for r in obs]);target=-surf['y']
    np.testing.assert_array_equal(sigma,surf['sigma'])
    if len(obs)!=160: raise ValueError('Changed input count')
    index=json.loads((exp/'banks/index.json').read_text());banks={};nb=0
    for e in index['entries']:
        for a,b in [('file','sha256'),('validation_file','validation_sha256')]:
            if sha(exp/'banks'/e[a])!=e[b]: raise ValueError('Changed map '+e[a])
        with np.load(exp/'banks'/e['file'],allow_pickle=False) as f: banks[(float(e['xi']),float(e['Q2']))]={k:f[k] for k in f.files}
        nb+=1
    def predict(w,method):
        values=[]
        for r in obs:
            xi,t,q=float(r['xi']),float(r['t_GeV2']),float(r['Q2_GeV2']);b=banks[(xi,q)]
            h=field(w,b[method+'_x'][None],np.array([xi]),np.array([t]),meta)[0]
            values.append(-np.einsum('ocn,cn->o',b[method+'_weights'],h))
        return np.array(values)
    saved=readcsv(p/'stage_predictions_160.csv');print('saved_stage_columns',list(saved[0]),flush=True)
    expected_metrics={int(r['stage']):r for r in readcsv(p/'stage_metrics.csv')}
    max_pred=0.;max_J=0.;metrics=[];curves=[];moment_states={};predictions={}
    states={f'stage{i:02}':p/f'STAGE_{i:02d}.pt' for i in range(21)}
    states['zero5']=project/'GPD_experimental_closeout/models/zero.pt'
    for name,path in states.items():
        cp=torch.load(path,weights_only=True,map_location='cpu');w=weights(cp);pred=predict(w,'nominal');ref=predict(w,'reference')
        predictions[name]=pred
        dl=np.mean(((pred[:,0]-target)/sigma)**2);local=np.mean(((pred[:,0]-y)/sigma)**2)
        if name!='zero5':
            i=int(name[5:]);poly=moment_loss(w,meta,cp['t_nodes']);row=expected_metrics[i]
            max_J=max(max_J,abs(dl+poly-float(row['J'])))
        else:i=None;poly=None
        entry=dict(state=name,surface_RMS=float(np.sqrt(dl)),local_RMS=float(np.sqrt(local)),surface_data_loss=float(dl),
                   local_data_loss=float(local),poly_loss=poly,max_nominal_reference_Re=float(np.max(abs(pred[:,0]-ref[:,0]))),
                   surface_abs_pull_gt3=int(np.sum(abs((pred[:,0]-target)/sigma)>3)),
                   local_abs_pull_gt3=int(np.sum(abs((pred[:,0]-y)/sigma)>3)))
        metrics.append(entry)
        if name in ['stage00','stage05','stage10','stage20','zero5']:
            moment_states[name]=(w,cp)
        print(name,entry['surface_RMS'],entry['local_RMS'],flush=True)
    # Replay saved source-sign stage predictions rather than trusting the aggregate table.
    keys=list(saved[0]); pred_col=next(k for k in ['ReH_GPD_source','ReH_pred_source'] if k in keys)
    ids={r['row_id']:i for i,r in enumerate(obs)}
    for r in saved:
        max_pred=max(max_pred,abs(float(r[pred_col])-predictions[f"stage{int(r['stage']):02d}"][ids[r['row_id']],0]))
    # Selected/reference parent/control plotted curves.
    plot=readcsv(p/'Hplus_anchor_curves.csv');mapnames={'central13':'stage00','zero5':'zero5','surface_selected':'stage20'}
    curve_max=0.;curve_counts=0
    for name,label in mapnames.items():
        rr=[r for r in plot if r['state']==name];w=moment_states[label][0]
        for rid in dict.fromkeys(r['anchor_row_id'] for r in rr):
            z=[r for r in rr if r['anchor_row_id']==rid];x=np.array([float(r['x']) for r in z]);xi=float(z[0]['xi']);t=float(z[0]['t_GeV2'])
            hh=field(w,x[None],np.array([xi]),np.array([t]),meta)[0];h=(2/9)*(hh[0]+hh[1]);old=np.array([float(r['Hcharge_plus']) for r in z])
            curve_max=max(curve_max,float(np.max(abs(h-old))));curve_counts+=len(h)
    # Saved complete-forward summaries: arithmetic/source identity only, no native replay.
    fwd=json.loads((p/'FULL_FORWARD_SUMMARY.json').read_text());saved_full_count=len(fwd['records'])
    # Compute all 79 t diagnostics afresh for selected and matched fifth-step branch.
    ts=sorted({float(r['t_GeV2']) for r in obs});mom={};mmax=0.
    for name in ['stage05','stage20']:
        records=diagnostic_moments(moment_states[name][0],meta,ts,192)
        m=np.array([r['RMS'] for r in records if r['channel']!='Hcharge_plus_same_quark_scale'])
        mom[name]=dict(combined_RMS=float(np.sqrt(np.mean(m*m))),worst=max(records,key=lambda r:r['RMS']),
                       worst_charge=max([r for r in records if r['channel']=='Hcharge_plus_same_quark_scale'],key=lambda r:r['RMS']))
        if name=='stage20':
            src={(float(r['t_GeV2']),r['channel'],int(r['spin'])):float(r['RMS']) for r in readcsv(p/'moment_diagnostics_79t.csv') if int(r['order_per_side'])==192}
            mmax=max(abs(r['RMS']-src[(r['t'],r['channel'],r['spin'])]) for r in records)
        dump(out/(name+'_moments.json'),records)
        print('moments',name,mom[name],flush=True)
    cp20=torch.load(p/'STAGE_20.pt',weights_only=True,map_location='cpu');term=torch.load(p/'TERMINAL.pt',weights_only=True,map_location='cpu')
    np.testing.assert_array_equal(cp20['flat'],term['flat'])
    report=dict(pilot_hash_count=len(manifest['artifacts']),map_pairs_hash_checked=nb,map_file_hashes=2*nb,
       checkpoint_records_replayed=len(states),terminal_parameter_identity=True,cached_point_method_predictions=len(states)*160*2,
       maximum_saved_prediction_difference=max_pred,maximum_objective_difference=max_J,curve_values_replayed=curve_counts,
       max_saved_curve_difference=curve_max,selected_moment_records_replayed=79*16,maximum_saved_moment_RMS_difference=mmax,
       moments=mom,metrics=metrics,full_forward_reports_inspected=saved_full_count,
       new_native_full_forward_checks=0,new_fits=0,source_table_sha256=sha(source),
       interpretation='Cached row and numerical field replay, not independent QCD or full constraint/KKT certification')
    dump(out/'AUDIT.json',report)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=metrics[0].keys());writer.writeheader();writer.writerows(metrics)
    np.savez_compressed(out/'replayed_predictions.npz',**predictions,y_source=y,y_surface_source=target,sigma=sigma)
    print('AUDIT_COMPLETE',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pilot',type=Path,required=True);a.add_argument('--project',type=Path,required=True);a.add_argument('--out',type=Path,required=True);main(a.parse_args())

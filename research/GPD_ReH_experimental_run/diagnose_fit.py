#!/usr/bin/env python3
"""Saved experimental checkpoint diagnostics; no optimization or uncertainty claims."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
from gpd_reh.problem import ExperimentalProblem
from gpd_reh.bank import ExperimentalForward
from gpd_reh.inputs import digest,atomic_json,write_csv
from gpd_reh.polynomiality_base import make_grid,design
from gpd_reh.moments import scale

ROOT=Path(__file__).resolve().parent

def diagnose(prepared,bank_dir,checkpoint,out):
    prepared=Path(prepared);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if 'state_dict' not in ck:raise ValueError('Checkpoint lacks state_dict')
    meta=json.loads((prepared/'INPUT_SUMMARY.json').read_text())
    for key,value in (('source_sha256',meta['source_sha256']),('t_mean',meta['t_mean']),('t_std_population',meta['t_std_population'])):
        if ck.get('experiment_contract',{}).get(key)!=value:raise ValueError('Checkpoint experimental contract mismatch: '+key)
    p=ExperimentalProblem(prepared,bank_dir,seed=int(ck['seed']));p.model.load_state_dict(ck['state_dict'])
    flat=torch.cat([q.detach().flatten() for q in p.model.parameters()])
    if 'flat' in ck and not torch.equal(flat,ck['flat'].cpu()):raise ValueError('Flat/state mismatch')
    ref=ExperimentalForward(prepared,bank_dir,method='reference')
    with torch.no_grad():nom=p.forward(p.model).numpy();fine=ref(p.model).numpy()
    with (prepared/'observations.csv').open(newline='') as f:observations=list(csv.DictReader(f))
    rows=[]
    for i,obs in enumerate(observations):
        rows.append({'row_id':obs['row_id'],'set':obs['source_set'],'Q2_GeV2':obs['Q2_GeV2'],
            'xi':obs['xi'],'t_GeV2':obs['t_GeV2'],'ReH_source':obs['ReH_source'],
            'sigma_reported':obs['sigma_reported'],'ReH_fit_source_convention':-nom[i,0],
            'unfitted_ImH_prediction_source_convention':-nom[i,1],
            'Re_pull':(-nom[i,0]-float(obs['ReH_source']))/float(obs['sigma_reported']),
            'Re_nominal_reference_abs':abs(nom[i,0]-fine[i,0]),'Im_nominal_reference_abs':abs(nom[i,1]-fine[i,1])})
    write_csv(out/'ReH_reproduction.csv',rows,list(rows[0]))
    # Quantities in new rows below are predictions; no experimental error band is inferred.
    with (prepared/'plot_anchors.csv').open(newline='') as f:anchors=list(csv.DictReader(f))
    field_rows=[]
    with torch.no_grad():
        for a in anchors:
            xi=float(a['xi']);t=float(a['t_GeV2']);x=np.unique(np.r_[np.linspace(-1,1,501),-xi,xi])
            h=p.model(torch.tensor(x)[None],torch.tensor([xi],dtype=torch.float64),torch.tensor([t],dtype=torch.float64))[0].numpy()
            for j,xx in enumerate(x):field_rows.append({'anchor_row_id':a['row_id'],'xi':xi,'t_GeV2':t,'Q02_GeV2':2.,'x':xx,'Hcharge_plus':(2/9)*(h[0,j]+h[1,j])})
    write_csv(out/'Hplus_slices.csv',field_rows,list(field_rows[0]))
    diagnostic_xi=.02+.96*(np.arange(80)+.5)/80
    # Coefficients refitted only to moments on fixed 41 xi nodes at EACH diagnostic t.
    # This is a polynomiality diagnostic, not a new CFF fit or fixed moment prior.
    specs=p.p.spec;moment_rows=[]
    with torch.no_grad():
        for t in sorted(set(float(o['t_GeV2']) for o in observations)):
            coefficients=[]
            fitM=p.p.raw_moments(p.model,torch.tensor(t,dtype=torch.float64)).numpy()
            for i,(_,N,_) in enumerate(specs):
                coefficients.append(np.linalg.lstsq(design(p.p.xi.numpy(),N),fitM[:,i],rcond=None)[0])
            for order in (96,192):
                x,w=make_grid(diagnostic_xi,order)
                h=p.model(torch.tensor(x),torch.tensor(diagnostic_xi),torch.full((80,),t,dtype=torch.float64)).numpy()
                residuals=[]
                for i,(ch,N,m) in enumerate(specs):
                    M=(h[:,ch]*(2*w*x**m)).sum(-1);pred=design(diagnostic_xi,N)@coefficients[i]
                    e=(M-pred)/scale(ch,N);residuals.extend(e)
                    moment_rows.append({'t_GeV2':t,'spin':N,'channel':ch,'quadrature_per_side':order,'normalized_RMS':float(np.sqrt(np.mean(e**2))),
                        'fit_t_collocation':int(t in p.p.t_nodes.tolist())})
    write_csv(out/'moment_diagnostics.csv',moment_rows,list(moment_rows[0]))
    x=flat.clone().requires_grad_();loss=p.objective(x);grad=torch.autograd.grad(loss,x)[0]
    r=p.residual(flat).detach().numpy();diff=nom-fine
    result={'role':'conditional real-ReH central extraction diagnostics; no calibrated intervals',
        'checkpoint_sha256':digest(checkpoint),'source_sha256':meta['source_sha256'], 'seed':int(ck['seed']),
        'J':float(loss.detach()),'data_loss':float(np.sum(r[:len(p.y)]**2)),'polynomiality_loss':float(np.sum(r[len(p.y):]**2)),
        'raw_gradient_L2_diagnostic':float(grad.norm()),'n_observations':len(p.y),
        'max_Re_nominal_reference_abs':float(np.max(abs(diff[:,0]))),'max_Im_nominal_reference_abs':float(np.max(abs(diff[:,1]))),
        'max_Re_refinement_in_reported_sigma':float(np.max(abs(diff[:,0])/p.sigma.numpy())),
        'moment_RMS_max_at192':max(v['normalized_RMS'] for v in moment_rows if v['quadrature_per_side']==192),
        'finite':bool(np.isfinite(nom).all() and np.isfinite(fine).all() and np.isfinite(r).all()),
        'scope':'Two-map refinement only; full source/quad/time validation remains separate. No Im input, E fit, or Htilde fit.'}
    atomic_json(out/'RESULT.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,default=ROOT/'prepared');p.add_argument('--bank-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);print(json.dumps(diagnose(a.prepared,a.bank_dir,a.checkpoint,a.out),indent=2))

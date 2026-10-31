"""Same-source diagnostics at cumulative budgets 5, 10 and 20; no fitting."""
from __future__ import annotations
import csv,json,os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXP=ROOT/'GPD_ReH_experimental_run'
PRIOR=ROOT/'completion_runs/ReH_coherent_GPD_pilot'
OLD=ROOT/'completion_runs/ReH_central_t_comparison'
PILOT=ROOT/'completion_runs/ReH_input_response_pilot'
sys.path[:0]=[str(HERE),str(PRIOR),str(PILOT),str(OLD),str(EXP),str(ROOT/'GPD_experimental_t_enriched_review'),str(ROOT/'GPD_direct_error_calibration')]
from run_budget import CASES,problem,BranchDomain,atomic,sha
from diagnose import field_for,moments,basis,moment_scale
from pilot_diagnose import target_values
from gpd_reh.bank import ExperimentalForward
from diagnose_candidates import full_forward


def write_csv(path,records):
    if not records:raise RuntimeError('Empty table: '+str(path))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def original_inputs():
    with np.load(EXP/'prepared/observations.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def targets():
    out={}
    for name in CASES:
        with np.load(PRIOR/'prepared'/f'{name}.npz',allow_pickle=False) as z:out[name]={k:z[k].copy() for k in z.files}
    return out


def load_budget(name,budget):
    if budget==5:path=PRIOR/'cases'/name/'STAGE_05.pt'
    else:path=HERE/name/f'STAGE_{budget:02}.pt'
    if budget>5:
        r=json.loads((HERE/name/'RESULT.json').read_text())
        item=next(x for x in r['all_finite_stages'] if x['stage']==budget)
        if sha(path)!=item['sha256']:raise RuntimeError('Checkpoint hash changed')
    cp=torch.load(path,weights_only=True,map_location='cpu')
    if cp['stage']!=budget or cp['case']!=name:raise RuntimeError('Checkpoint identity changed')
    return path,cp


def forward_predictions(cp,meta,nominal,reference=None):
    field=field_for(cp,meta)
    h=field(nominal.x.numpy(),nominal.xi.numpy(),nominal.t.numpy())
    pred=np.einsum('bocn,bcn->bo',nominal.weights.numpy(),h)
    if not np.isfinite(pred).all():raise RuntimeError('Nonfinite nominal CFF')
    if reference is None:return -pred[:,0],None
    hr=field(reference.x.numpy(),reference.xi.numpy(),reference.t.numpy())
    fine=np.einsum('bocn,bcn->bo',reference.weights.numpy(),hr)
    if not np.isfinite(fine).all():raise RuntimeError('Nonfinite reference CFF')
    return -pred[:,0],-fine[:,0]


def moment_diagnostics(name,budget,cp,meta,rows):
    field=field_for(cp,meta);fit_xi=np.linspace(.02,.98,41);diag_xi=.02+.96*(np.arange(80)+.5)/80
    ts=sorted({float(r['t_GeV2']) for r in rows})
    if len(ts)!=79:raise RuntimeError('Original t roster changed')
    records=[]
    for j,t in enumerate(ts):
        for order in (192,384):
            fit=moments(field,fit_xi,t,order);check=moments(field,diag_xi,t,order);errors=[]
            for i,(ch,spin) in enumerate((ch,spin) for ch in range(3) for spin in (2,4,6,8)):
                coeff=np.linalg.lstsq(basis(fit_xi,spin),fit[:,i],rcond=None)[0]
                e=(check[:,i]-basis(diag_xi,spin)@coeff)/moment_scale(ch,spin);errors.append(e)
                records.append({'case':name,'budget':budget,'t_GeV2':t,'order_per_side':order,
                    'channel':('Sigma','T_rescaled','Hg')[ch],'spin':spin,
                    'RMS':float(np.sqrt(np.mean(e*e))),'max_abs':float(np.max(abs(e))),
                    'peak_xi':float(diag_xi[np.argmax(abs(e))])})
            for k,spin in enumerate((2,4,6,8)):
                e=(2/9)*(errors[k]+errors[k+4])
                records.append({'case':name,'budget':budget,'t_GeV2':t,'order_per_side':order,
                    'channel':'Hcharge_plus_same_quark_scale','spin':spin,
                    'RMS':float(np.sqrt(np.mean(e*e))),'max_abs':float(np.max(abs(e))),
                    'peak_xi':float(diag_xi[np.argmax(abs(e))])})
        if (j+1)%20==0:print('MOMENTS',name,budget,j+1,79,flush=True)
    out=HERE/name/f'budget_{budget:02}';out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'moment_diagnostics_79t.csv',records)
    subset=[r for r in records if r['order_per_side']==384 and r['channel']!='Hcharge_plus_same_quark_scale']
    return {'actual_t_values':79,'orders_per_side':[192,384],'off_node_xi':80,
            'combined_internal_RMS':float(np.sqrt(np.mean([r['RMS']**2 for r in subset]))),
            'max_channel_spin_t':max(subset,key=lambda r:r['RMS'])}


def hplus(states,meta,anchors):
    curves=[];targets=[];response=[]
    for (name,budget),(_,cp) in states.items():
        field=field_for(cp,meta)
        for anchor in anchors:
            xi,t=float(anchor['xi']),float(anchor['t_GeV2'])
            x=np.unique(np.r_[np.linspace(-1,1,501),-xi,xi]);h=field(x[None],np.array([xi]),np.array([t]))[0]
            charge=(2/9)*(h[0]+h[1])
            for xx,yy in zip(x,charge):curves.append({'case':name,'budget':budget,'anchor_row_id':anchor['row_id'],
                      'xi':xi,'t_GeV2':t,'x':float(xx),'Hcharge_plus':float(yy)})
            a=target_values(field,xi,t,192);b=target_values(field,xi,t,384)
            targets.append({'case':name,'budget':budget,'anchor_row_id':anchor['row_id'],'xi':xi,'t_GeV2':t,
                 'bin_mean_x_02_04':b[0],'positive_x_first_moment':b[1],
                 'maximum_target_quadrature_change':float(np.max(abs(np.asarray(a)-b)))})
    write_csv(HERE/'Hplus_anchor_curves.csv',curves);write_csv(HERE/'anchor_targets.csv',targets)
    fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    colors={'ensemble_mean':'#087d79','replica_000':'#a63a5d','replica_001':'#5362a5'}
    styles={5:':',10:'--',20:'-'}
    for ax,anchor in zip(axs.flat,anchors):
        row=anchor['row_id'];xi=float(anchor['xi'])
        groups={(name,b):[r for r in curves if r['case']==name and r['budget']==b and r['anchor_row_id']==row]
                for name in CASES for b in (5,10,20)}
        for name in CASES:
            for b in (5,10,20):
                g=groups[(name,b)];ax.plot([r['x'] for r in g],[r['Hcharge_plus'] for r in g],
                       color=colors[name],ls=styles[b],lw=1,label=f'{name} {b}')
        x=np.array([r['x'] for r in groups[('ensemble_mean',5)]])
        for b in (5,10,20):
            mu=np.array([r['Hcharge_plus'] for r in groups[('ensemble_mean',b)]])
            for name in ('replica_000','replica_001'):
                member=np.array([r['Hcharge_plus'] for r in groups[(name,b)]])
                for region,mask in [('full',np.ones(len(x),bool)),('ERBL',abs(x)<=xi),('DGLAP',abs(x)>xi)]:
                    diff=member-mu
                    response.append({'case':name,'budget':b,'anchor_row_id':row,'xi':xi,
                      't_GeV2':float(anchor['t_GeV2']),'region':region,
                      'RMS_Hplus_member_minus_mean':float(np.sqrt(np.mean(diff[mask]**2))),
                      'max_abs_Hplus_member_minus_mean':float(np.max(abs(diff[mask])))})
        ax.axvline(-xi,color='0.6',lw=.6,ls='--');ax.axvline(xi,color='0.6',lw=.6,ls='--')
        ax.set(title=f'{row}: ξ={xi:.3f}, t={float(anchor["t_GeV2"]):.3f}',xlabel='x',ylabel='Hcharge_plus')
    axs[0,0].legend(fontsize=6,ncol=2);fig.savefig(HERE/'Hplus_budget_overlays.png',dpi=150);plt.close(fig)
    write_csv(HERE/'Hplus_matched_responses.csv',response)
    tmap={(r['case'],r['budget'],r['anchor_row_id']):r for r in targets}
    differences=[]
    for anchor in anchors:
        row=anchor['row_id']
        for b in (5,10,20):
            ref=tmap[('ensemble_mean',b,row)]
            for name in ('replica_000','replica_001'):
                r=tmap[(name,b,row)]
                differences.append({'case':name,'budget':b,'anchor_row_id':row,
                  'delta_bin_mean_x_02_04':r['bin_mean_x_02_04']-ref['bin_mean_x_02_04'],
                  'delta_positive_x_first_moment':r['positive_x_first_moment']-ref['positive_x_first_moment']})
    write_csv(HERE/'anchor_target_matched_differences.csv',differences)
    return response


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    plan=json.loads((HERE/'PLAN.json').read_text());lock=json.loads((HERE/'RUN_LOCK.json').read_text())
    if lock['plan_sha256']!=sha(HERE/'PLAN.json'):raise RuntimeError('Plan changed')
    for rel,digest in lock['sources'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Bound input changed: '+rel)
    rows=list(csv.DictReader((EXP/'prepared/observations.csv').open(newline='')))
    anchors=list(csv.DictReader((EXP/'prepared/plot_anchors.csv').open(newline='')))
    if len(rows)!=160 or len(anchors)!=6:raise RuntimeError('Row/anchor roster changed')
    obs=original_inputs();inp=targets();sigma=obs['sigma'];local=-obs['y'];mean=-inp['ensemble_mean']['y']
    with np.load(ROOT/'completion_runs/ReH_coherent_surface_pilot/coherent_surfaces.npz',allow_pickle=False) as z:
        central=z['fitted_central_source'].copy()
    meta=json.loads((EXP/'prepared/INPUT_SUMMARY.json').read_text())
    nominal=ExperimentalForward(EXP/'prepared',EXP/'banks')
    reference=ExperimentalForward(EXP/'prepared',EXP/'banks',method='reference')
    states={(name,b):load_budget(name,b) for name in CASES for b in (5,10,20)}
    pred={};fine={};row_records=[];metric_records=[];stage_records=[]
    for (name,b),(path,cp) in states.items():
        p,f=forward_predictions(cp,meta,nominal,reference);pred[(name,b)]=p;fine[(name,b)]=f
        own=-inp[name]['y']
        for label,target in [('own_input',own),('ensemble_mean_input',mean),('fitted_central_input',central),('original_local_data',local)]:
            diff=p-target
            metric_records.append({'case':name,'budget':b,'comparison':label,
              'RMS_absolute':float(np.sqrt(np.mean(diff**2))),
              'RMS_original_sigma':float(np.sqrt(np.mean((diff/sigma)**2))),
              'mean_signed_original_sigma':float(np.mean(diff/sigma)),
              'n_abs_pull_gt3':int((abs(diff/sigma)>3).sum())})
        for i,row in enumerate(rows):
            row_records.append({'case':name,'budget':b,'row_id':row['row_id'],'source_set':row['source_set'],
              'method':row['method'],'xi':row['xi'],'t_GeV2':row['t_GeV2'],'Q2_GeV2':row['Q2_GeV2'],
              'sigma_reported':sigma[i],'ReH_original_local_source':local[i],
              'ReH_fitted_central_source':central[i],'ReH_ensemble_mean_target_source':mean[i],
              'ReH_own_input_source':own[i],'ReH_GPD_source':p[i],'ReH_reference_source':f[i],
              'own_input_pull':(p[i]-own[i])/sigma[i],
              'ensemble_mean_pull':(p[i]-mean[i])/sigma[i],
              'nominal_reference_Re_abs':abs(p[i]-f[i])})
        print('PREDICTIONS',name,b,flush=True)
    write_csv(HERE/'comparison_budget_160.csv',row_records)
    write_csv(HERE/'comparison_metrics.csv',metric_records)
    matched=[];matched_metrics=[]
    for b in (5,10,20):
        mu=pred[('ensemble_mean',b)]
        for name in ('replica_000','replica_001'):
            idiff=-inp[name]['y']-mean;odiff=pred[(name,b)]-mu
            for i,row in enumerate(rows):
                matched.append({'case':name,'budget':b,'row_id':row['row_id'],'sigma_reported':sigma[i],
                  'CFF_input_member_minus_mean_source':idiff[i],
                  'GPD_output_member_minus_mean_source':odiff[i],
                  'CFF_input_difference_over_original_sigma':idiff[i]/sigma[i],
                  'GPD_output_difference_over_original_sigma':odiff[i]/sigma[i]})
            matched_metrics.append({'case':name,'budget':b,
                'input_RMS_absolute':float(np.sqrt(np.mean(idiff**2))),
                'output_RMS_absolute':float(np.sqrt(np.mean(odiff**2))),
                'input_RMS_original_sigma':float(np.sqrt(np.mean((idiff/sigma)**2))),
                'output_RMS_original_sigma':float(np.sqrt(np.mean((odiff/sigma)**2))),
                'output_over_input_RMS_original_sigma':float(np.sqrt(np.mean((odiff/sigma)**2))/np.sqrt(np.mean((idiff/sigma)**2)))})
    write_csv(HERE/'matched_input_output_160.csv',matched)
    write_csv(HERE/'matched_input_output_metrics.csv',matched_metrics)
    for name in CASES:
        r=json.loads((HERE/name/'RESULT.json').read_text())
        for item in r['all_finite_stages']:
            b=item['stage'];path=(PRIOR/'cases'/name/'STAGE_05.pt') if b==5 else (HERE/name/item['path'])
            cp=torch.load(path,weights_only=True,map_location='cpu');p,_=forward_predictions(cp,meta,nominal)
            own=-inp[name]['y']
            stage_records.append({'case':name,'cumulative_stage':b,'J':item['J'],
                'data_loss':item['data_loss'],'poly_loss':item['poly_loss'],
                'own_input_RMS_original_sigma':float(np.sqrt(np.mean(((p-own)/sigma)**2))),
                'ensemble_mean_RMS_original_sigma':float(np.sqrt(np.mean(((p-mean)/sigma)**2))),
                'original_local_RMS_original_sigma':float(np.sqrt(np.mean(((p-local)/sigma)**2)))})
    write_csv(HERE/'stage_metrics.csv',stage_records)
    response=hplus(states,meta,anchors)
    print('CURVES_AND_TARGETS_COMPLETE',flush=True)
    cases={}
    for name in CASES:
        cases[name]={}
        for b in (10,20):
            path,cp=states[(name,b)];out=HERE/name/f'budget_{b:02}';out.mkdir(parents=True,exist_ok=True)
            mm=moment_diagnostics(name,b,cp,meta,rows)
            p=problem(inp[name]);p.set_model(cp['flat']);d=BranchDomain('enriched')
            with torch.no_grad():components=p.components(cp['flat']).numpy();constraints=d.constraints(p,cp['flat']).numpy()
            full=full_forward(p,path,out/'full_forward')
            cases[name][str(b)]={'checkpoint_sha256':sha(path),'J':float(sum(components)),
              'data_loss':float(components[0]),'poly_loss':float(components[1]),
              'maximum_constraint':float(np.max(constraints)),'moment_constraint':float(constraints[0]),
              'nominal_reference_Re_max_abs':float(np.max(abs(pred[(name,b)]-fine[(name,b)]))),
              'moment_diagnostics':mm,
              'full_forward':{'rows':full['rows'],'finite':full['finite'],
                'max_CFF_budget_over_legacy_limit':full['max_CFF_budget_over_legacy_limit'],
                'max_field_estimated_total':full['max_field_estimated_total'],
                'legacy_guides_pass_count':full['legacy_guides_pass_count'],
                'warning_rows':full['warning_rows'],
                'summary_sha256':sha(out/'FULL_FORWARD_SUMMARY.json')}}
            atomic(HERE/'DIAGNOSTICS_PROGRESS.json',cases)
            print('CASE_BUDGET_DIAGNOSTICS_COMPLETE',name,b,flush=True)
    prior_diag=json.loads((PRIOR/'DIAGNOSTICS.json').read_text())
    atomic(HERE/'DIAGNOSTICS.json',{'cases_new_budgets':cases,
      'budget5_source':'completion_runs/ReH_coherent_GPD_pilot/DIAGNOSTICS.json',
      'budget5_source_sha256':sha(PRIOR/'DIAGNOSTICS.json'),
      'budget5_selected_diagnostics':{k:prior_diag['cases'][k] for k in CASES},
      'comparison_metrics':metric_records,'matched_input_output_metrics':matched_metrics,
      'matched_Hplus_response_rows':len(response),'source_sigma_unchanged':True,
      'interpretation':'Prospective cumulative 5/10/20 comparison; two members are not a GPD uncertainty band.'})
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for b in (5,10,20):
        group=[r for r in row_records if r['case']=='ensemble_mean' and r['budget']==b]
        ax.plot(np.arange(160),[r['ensemble_mean_pull'] for r in group],'.',ms=3,label=f'mean budget {b}')
    ax.axhline(0,color='0.3',lw=.7);ax.axhline(3,color='0.6',ls='--',lw=.7);ax.axhline(-3,color='0.6',ls='--',lw=.7)
    ax.set(xlabel='Original selected row order',ylabel='GPD mean minus same CFF mean / original quoted σ');ax.legend()
    fig.savefig(HERE/'mean_target_pulls_by_budget.png',dpi=150);plt.close(fig)
    print('DIAGNOSTICS_COMPLETE',flush=True)
if __name__=='__main__':main()

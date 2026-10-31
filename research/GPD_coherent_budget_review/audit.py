#!/usr/bin/env python3
"""Audit saved budget results only: no fitting, native build or new random draws.
The input packet, old experimental row maps and coherent target vectors are required.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import torch
from replay_math import numpy_field,poly_loss,target_values,moments_check,rms,response_metrics,drift_metrics
CASES=('ensemble_mean','replica_000','replica_001')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with Path(p).open('w',newline='') as f:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def load(p):
    a=torch.load(p,weights_only=True,map_location='cpu')
    vec=torch.cat([x.ravel() for x in a['state_dict'].values()])
    if vec.numel()!=3867 or not torch.equal(vec,a['flat']):raise ValueError('Parameter binding '+str(p))
    return a

def main(args):
    started=time.monotonic();torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    b=args.packet;project=args.project;exp=project/'GPD_ReH_experimental_run';o=args.out;o.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((b/'PACKAGE_MANIFEST.json').read_text())['files']
    for n,h in manifest.items():
        if Path(n).is_absolute() or '..' in Path(n).parts or sha(b/n)!=h:raise ValueError('Manifest '+n)
    lock=json.loads((b/'RUN_LOCK.json').read_text());plan=json.loads((b/'PLAN.json').read_text())
    if sha(b/'PLAN.json')!=lock['plan_sha256']:raise ValueError('Plan changed')
    meta=json.loads((exp/'prepared/INPUT_SUMMARY.json').read_text())
    old=dict(np.load(exp/'prepared/observations.npz',allow_pickle=False))
    inputs={c:dict(np.load(args.prior/'prepared'/f'{c}.npz',allow_pickle=False)) for c in CASES}
    with np.load(args.cohort,allow_pickle=False) as z:cohort={k:z[k].copy() for k in z.files}
    srcs={c:-inputs[c]['y'] for c in CASES};sigma=old['sigma']
    for c in CASES:
        for k in old:
            if k!='y' and not np.array_equal(inputs[c][k],old[k]):raise ValueError('Changed input '+c+' '+k)
    np.testing.assert_allclose(srcs['ensemble_mean'],cohort['mean'],atol=2e-15,rtol=0)
    for c,i in [('replica_000',0),('replica_001',1)]:np.testing.assert_array_equal(srcs[c],cohort['predictions_source'][i])
    for rel,h in lock['sources'].items():
        if rel.startswith('GPD_ReH_experimental_run/') and (project/rel).is_file():
            if sha(project/rel)!=h:raise ValueError('Frozen input '+rel)
    sys.path.insert(0,str(exp))
    from gpd_reh.bank import ExperimentalForward,preflight
    from gpd_reh.model import ExperimentalFiLM
    maps={m:ExperimentalForward(exp/'prepared',exp/'banks',m) for m in ('nominal','reference')}
    model=ExperimentalFiLM(meta['t_mean'],meta['t_std_population']).double()
    stage_tab={(r['case'],int(r['cumulative_stage'])):r for r in readcsv(b/'stage_metrics.csv')}
    saved={(r['case'],int(r['budget']),r['row_id']):r for r in readcsv(b/'comparison_budget_160.csv')}
    curve_tab=readcsv(b/'Hplus_anchor_curves.csv');target_tab=readcsv(b/'anchor_targets.csv')
    anchors=readcsv(exp/'prepared/plot_anchors.csv')
    metrics=[];predictions={};fields={};cps={};targets=[];maxes=dict(objective=0.,poly=0.,prediction=0.,numpy_torch_field=0.,curve=0.,target=0.,target_refinement=0.)
    for c in CASES:
        job=next(x for x in plan['jobs'] if x['case']==c)
        if sha(b/c/'STAGE_05.pt')!=job['start_sha256']:raise ValueError('Stage5 changed')
        result=json.loads((b/c/'RESULT.json').read_text())
        if result['terminal_cumulative_stage']!=20 or result['new_iterations_recorded']!=15:raise ValueError('Budget')
        for step in range(5,21):
            cp=load(b/c/f'STAGE_{step:02}.pt')
            if cp['case']!=c or cp['stage']!=step:raise ValueError('Checkpoint label')
            if step>5 and (cp['immediate_resume_sha256']!=job['start_sha256'] or cp['original_common_parent_sha256']!=job['original_same_parent_sha256']):raise ValueError('Lineage')
            if step>5 and torch.equal(cp['flat'],last['flat']):raise ValueError('Unchanged accepted state')
            last=cp;field=numpy_field(cp,meta);methods={}
            for m,layer in maps.items():
                if m=='reference' and step not in (5,10,20):continue
                val=field(layer.x.numpy(),layer.xi.numpy(),layer.t.numpy())
                methods[m]=np.einsum('bocn,bcn->bo',layer.weights.numpy(),val)
                if not np.isfinite(methods[m]).all():raise ValueError('Nonfinite replay')
                if step in (5,10,20):
                    col='ReH_GPD_source' if m=='nominal' else 'ReH_reference_source'
                    sv=np.array([float(saved[c,step,str(rid)][col]) for rid in old['row_id']])
                    maxes['prediction']=max(maxes['prediction'],float(np.max(abs(-methods[m][:,0]-sv))))
                    if m=='nominal':
                        model.load_state_dict(cp['state_dict'])
                        with torch.no_grad():val2=model(layer.x,layer.xi,layer.t).numpy()
                        maxes['numpy_torch_field']=max(maxes['numpy_torch_field'],float(np.max(abs(val-val2))))
            p=-methods['nominal'][:,0];dl=rms((p-srcs[c])/sigma)**2;pl=poly_loss(field,cp['t_nodes']);J=dl+pl
            maxes['objective']=max(maxes['objective'],abs(J-cp['J']));maxes['poly']=max(maxes['poly'],abs(pl-cp['poly_loss']))
            if abs(J-float(stage_tab[c,step]['J']))>1e-10:raise ValueError('J mismatch')
            row=dict(case=c,budget=step,J=J,poly_loss=pl,own_RMS=rms((p-srcs[c])/sigma),mean_RMS=rms((p-srcs['ensemble_mean'])/sigma),local_RMS=rms((p+old['y'])/sigma))
            metrics.append(row)
            if step in (5,10,20):
                predictions[c,step]=p;cps[c,step]=cp
                allh=[]
                for anchor in anchors:
                    ar=anchor['row_id'];rows=[r for r in curve_tab if r['case']==c and int(r['budget'])==step and r['anchor_row_id']==ar]
                    x=np.array([float(r['x']) for r in rows]);xi=float(anchor['xi']);t=float(anchor['t_GeV2'])
                    h=field(x[None],np.array([xi]),np.array([t]))[0];q=(2/9)*(h[0]+h[1]);allh.extend(q)
                    maxes['curve']=max(maxes['curve'],float(np.max(abs(q-np.array([float(r['Hcharge_plus']) for r in rows])))))
                    a=target_values(field,xi,t,192);v=target_values(field,xi,t,384)
                    sr=next(r for r in target_tab if r['case']==c and int(r['budget'])==step and r['anchor_row_id']==ar)
                    sv=np.array([float(sr['bin_mean_x_02_04']),float(sr['positive_x_first_moment'])])
                    maxes['target']=max(maxes['target'],float(np.max(abs(v-sv))));maxes['target_refinement']=max(maxes['target_refinement'],float(np.max(abs(a-v))))
                    targets.append(dict(case=c,budget=step,anchor=ar,xi=xi,t=t,bin_mean=float(v[0]),first_moment=float(v[1])))
                fields[c,step]=np.array(allh)
        np.testing.assert_array_equal(load(b/c/'TERMINAL.pt')['flat'],last['flat'])
        print('REPLAYED',c,'16 states',flush=True)
    response=[];shape=[];drift=[]
    for step in (5,10,20):
        for c in CASES[1:]:
            m=response_metrics((srcs[c]-srcs['ensemble_mean'])/sigma,(predictions[c,step]-predictions['ensemble_mean',step])/sigma)
            m.update(case=c,budget=step);response.append(m)
            d=fields[c,step]-fields['ensemble_mean',step];d5=fields[c,5]-fields['ensemble_mean',5]
            h=drift_metrics(d5,d);h.update(case=c,budget=step);shape.append(h)
    for c in CASES:
        d=drift_metrics(fields[c,5],fields[c,20]);d.update(case=c);drift.append(d)
    for c in CASES[1:]:
        a=fields[c,10]-fields['ensemble_mean',10];bb=fields[c,20]-fields['ensemble_mean',20]
        row=drift_metrics(a,bb);row.update(case=c,comparison='member-minus-mean:10_to_20');drift.append(row)
    # Journal diagnosis is record analysis, not re-execution of full numerical constraints.
    obsrows=readcsv(exp/'prepared/observations.csv');names=['moment_predictive']
    for r in obsrows:
        names += [r['row_id']+'_CFF_'+x for x in ('Re','Im')]
        names += [r['row_id']+'_field_'+x for x in ('Sigma','T','Hg')]
        names += [r['row_id']+'_transport_'+m+'_'+x for m in ('weak10','strong10') for x in ('Sigma','Hg','T')]
    journal={};rejrows=[]
    for c in CASES:
        s=json.loads((b/c/'SOLVER.json').read_text());last=s['history'][-1];r=json.loads((b/c/'RESULT.json').read_text())
        for h in s['history']:
            for tr in h['trials']:
                if not tr['accepted']:
                    a=np.asarray(tr['constraints']);idx=int(np.argmax(a))
                    rejrows.append(dict(case=c,cumulative_stage=6+h['iteration'],reason=tr.get('reason'),max_violation=float(a[idx]),index=idx,name=names[idx],actual_J_decrease=tr.get('actual_decrease')))
        journal[c]=dict(status=s['status'],additional_solver_seconds=s['seconds'],local_subproblem_seconds=r['binding_stats']['norm_subproblem_seconds'],
            peak_RSS_bytes_reported=r['peak_memory_bytes_macos'],final_radius=s['final_radius'],accepted_steps=s['counts']['accepted_steps'],rejections=s['counts']['trial_rejections'],
            last_pre_step_KKT=last['KKT']['stationarity_L2'],last_pre_step_raw_gradient=last['KKT']['raw_gradient_L2'],
            last_pre_step_active_names=last['KKT']['evaluated_constraint_names'],
            stage19_to20_actual_decrease=float(last['trials'][-1]['actual_decrease']),
            subspace_rank_range=[min(tr['subproblem']['subspace_rank'] for h in s['history'] for tr in h['trials']),max(tr['subproblem']['subspace_rank'] for h in s['history'] for tr in h['trials'])])
    diagnostics=json.loads((b/'DIAGNOSTICS.json').read_text());bindings=0;forward_rows=0;forward_pred_diff=0.;forward_sum_diff=0.; forward_warning_rows=[]
    for c in CASES:
        for step in (10,20):
            fpath=b/c/f'budget_{step:02}'/'FULL_FORWARD_SUMMARY.json'
            d=diagnostics['cases_new_budgets'][c][str(step)]
            if sha(fpath)!=d['full_forward']['summary_sha256'] or sha(b/c/f'STAGE_{step:02}.pt')!=d['checkpoint_sha256']:raise ValueError('Forward binding')
            # Preserve full worker reports. The maps are independently replayed; native source routines are NOT rerun.
            f=json.loads(fpath.read_text());bindings+=1
            if len(f['records'])!=160:raise ValueError('Incomplete full-forward record list')
            oblookup={r['row_id']:r for r in obsrows}
            for record in f['records']:
                path=fpath.parent/'full_forward'/record['file']
                if sha(path)!=record['file_sha256']:raise ValueError('Record hash')
                fr=json.loads(path.read_text());bind=fr['binding'];rr=fr['result'];ob=oblookup[record['row_id']]
                if bind['checkpoint_sha256']!=d['checkpoint_sha256'] or bind['row_id']!=record['row_id']:raise ValueError('Record source/state link')
                for k,col in [('xi','xi'),('Q2','Q2_GeV2'),('t','t_GeV2')]:
                    if float.fromhex(bind[k+'_hex'])!=float(ob[col]):raise ValueError('Record coordinate')
                total=np.sum([np.asarray(v) for v in rr['CFF_terms'].values()],axis=0)
                forward_sum_diff=max(forward_sum_diff,float(np.max(abs(total-np.asarray(rr['CFF_estimated_total'])))))
                i=next(i for i,x in enumerate(old['row_id']) if str(x)==record['row_id'])
                for method,col in [('weak10','ReH_GPD_source'),('strong10','ReH_reference_source')]:
                    forward_pred_diff=max(forward_pred_diff,abs(rr['checks'][method]['CFF'][0]+float(saved[c,step,record['row_id']][col])))
                if any(record['warnings'].values()):forward_warning_rows.append(dict(case=c,budget=step,row_id=record['row_id'],warnings=record['warnings']))
                forward_rows+=1
    between=rms((cohort['predictions_source']-cohort['mean'])/sigma)**2
    center=[r for r in metrics if r['case']=='ensemble_mean' and r['budget'] in (5,10,20)]
    decomposition=[dict(budget=r['budget'],mean_squared_distance=r['mean_RMS']**2,cohort_dispersion=between,mean_displacement_fraction=(r['mean_RMS']**2)/(r['mean_RMS']**2+between)) for r in center]
    a=dict(packet_manifest_count=len(manifest),exact_map_preflight=preflight(exp/'prepared',exp/'banks'),checkpoint_records_replayed=48,
        nominal_map_point_evaluations=48*len(sigma),reference_map_point_evaluations=9*len(sigma),curve_values_replayed=sum(len(h) for h in fields.values()),
        targets_replayed=len(targets)*2,target_quadrature_orders=[192,384],maximum_differences=maxes,
        full_forward_records_binding_checked=forward_rows,full_forward_summaries_bound=bindings,full_native_checks_reexecuted=0,
        full_forward_record_sum_max_difference=forward_sum_diff,full_forward_record_Re_map_max_difference=forward_pred_diff,
        finite_warning_records=forward_warning_rows,
        source_scope='Actual experimental small rows and original source; NumPy FiLM / least-squares moments independent of production Torch calculations.',
        environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__),elapsed_seconds=time.monotonic()-started,
        no_fitting=True,no_new_noise=True,no_kernels=True,no_manuscript_edits=True)
    if any(maxes[k]>1e-10 for k in maxes):raise ValueError('Unexpected replay difference '+str(maxes))
    for name,rows in [('STAGES.csv',metrics),('CFF_RESPONSE.csv',response),('HPLUS_RESPONSE.csv',shape),('HPLUS_BUDGET_CHANGES.csv',drift),('TARGETS.csv',targets),('REJECTED_TRIAL_LOG.csv',rejrows)]:csvout(o/name,rows)
    np.savez_compressed(o/'SAVED_FIELD_REPLAY.npz',**{f'{c}_b{s}':v for (c,s),v in fields.items()})
    dump(o/'JOURNAL_DIAGNOSIS.json',journal);dump(o/'MEAN_DISTANCE_DECOMPOSITION.json',decomposition);dump(o/'AUDIT.json',a)
    if args.moments:
        for c in CASES:
            fld=numpy_field(cps[c,20],meta);ts=sorted(set(float(r['t_GeV2']) for r in obsrows));t0=time.monotonic()
            rows=moments_check(fld,ts,192);sv=readcsv(b/c/'budget_20'/'moment_diagnostics_79t.csv');lookup={(float(r['t_GeV2']),r['channel'],int(r['spin'])):r for r in sv if int(r['order_per_side'])==192}
            err=max(abs(r['RMS']-float(lookup[r['t_GeV2'],r['channel'],r['spin']]['RMS'])) for r in rows)
            combined=float(np.sqrt(np.mean([r['RMS']**2 for r in rows if r['channel']!='Hcharge_plus_same_quark_scale'])))
            csvout(o/f'MOMENTS_{c}_20.csv',rows)
            md=dict(rows=len(rows),t_values=len(ts),xi_check_nodes=80,order_per_side=192,combined_internal_RMS=combined,maximum_saved_RMS_difference=err,elapsed=time.monotonic()-t0)
            dump(o/f'MOMENT_AUDIT_{c}.json',md);print('MOMENTS_REPLAYED',c,md,flush=True)
    a['total_elapsed_seconds_including_requested_moments']=time.monotonic()-started
    a['fresh_moment_checks']=len(CASES) if args.moments else 0
    dump(o/'AUDIT.json',a)
    print(json.dumps(a,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--moments',action='store_true');main(p.parse_args())

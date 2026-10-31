#!/usr/bin/env python3
"""32 source-table -> smooth-CFF replicas. NO GPD fits, NLO, or new datasets.
Uses the exact previously supplied CFF smoother functions without calling main().
The source-matched unperturbed control runs first. Resume only completed cases.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import pandas as pd
import torch
from replica_tools import draw_table,summarize,comparison

SOURCE_SHA='8d7274db9dd7e4fcc3ab84c0230701242274e42868f9c3e66e3d61a1a6838696'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,d):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save_torch(p,d):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');torch.save(d,tmp);tmp.replace(p)
def safe_import(path):
    spec=importlib.util.spec_from_file_location('frozen_CFF_smoother',path)
    if not spec or not spec.loader:raise RuntimeError('Cannot load source fitter')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--project',type=Path,required=True);a=arg.parse_args()
    project=a.project.resolve();surface=project/'completion_runs/ReH_selected_surface'
    pilot=project/'completion_runs/ReH_surface_mean_pilot'
    out=project/'completion_runs/ReH_coherent_surface_pilot';out.mkdir(parents=True,exist_ok=True)
    files={'source':project/'GPD_ReH_experimental_run/inputs/final_summary_by_DNN.csv',
           'observations':project/'GPD_ReH_experimental_run/prepared/observations.csv',
           'fitter':surface/'fit_and_compare.py','protocol':surface/'FIT_PROTOCOL.json',
           'central_surface':surface/'ReH_fitted_central_surface.pt',
           'central_comparison':surface/'comparison_160.csv','GPD_comparison':pilot/'comparison_states_160.csv'}
    expected=json.loads((Path(__file__).parent/'SOURCE_LOCK.json').read_text())
    actual={k:sha(p) for k,p in files.items()}
    if actual!=expected['required']:raise RuntimeError('Source identity mismatch; preserve inputs and report exact changed item')
    obs=pd.read_csv(files['observations']);c=pd.read_csv(files['central_comparison'])
    if len(obs)!=160 or len(set(obs.row_id))!=160 or obs.row_id.tolist()!=c.row_id.tolist():raise ValueError('Wrong/reordered observations')
    if not np.array_equal(obs.method.to_numpy(),c.method.to_numpy()):raise ValueError('Changed method labels')
    y=obs.ReH_source.to_numpy();sigma=obs.sigma_reported.to_numpy()
    cp=torch.load(files['central_surface'],weights_only=True,map_location='cpu')
    for k in ['source_sha256','prepared_sha256','protocol_sha256']:
        ref={'source_sha256':'source','prepared_sha256':'observations','protocol_sha256':'protocol'}[k]
        if cp[k]!=actual[ref]:raise ValueError('Incompatible surface checkpoint')
    m=safe_import(files['fitter']);torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    x=m.features(obs);perm=np.random.default_rng(m.SEED).permutation(160);val,train=perm[:32],perm[32:]
    rule={'name':'selected_table_coherent_CFF_replica_pilot_v1','members':32,'source_hashes':actual,
      'random_rule':'PCG64 SeedSequence([20261011,391870,replica_id])','replica_ids':list(range(32)),
      'sampling':'y_rep=y_selected+sigma_reported*z, independent rows; stated marginal-Gaussian approximation',
      'fit_rule':'Unchanged source setup/fit; same fixed split/model seed; rerun registered early stop per replica then all-row refit',
      'no_additional_errors':True,'all_outputs':'ReH_source only; save whole function per ID',
      'scripts':{Path(__file__).name:sha(__file__), 'replica_tools.py':sha(Path(__file__).with_name('replica_tools.py'))},
      'environment':{'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'torch':str(torch.__version__)}}
    rp=out/'RELEASE.json'
    if rp.exists():
        if json.loads(rp.read_text())!=rule:raise RuntimeError('Release drift; no automatic overwrite/restart')
    else:atomic_json(rp,rule)

    def one(rid):
        label='unperturbed' if rid is None else f'r{rid:03d}';d=out/label;d.mkdir(exist_ok=True)
        if (d/'RESULT.json').exists():
            rec=json.loads((d/'RESULT.json').read_text())
            if rec['status']=='SUCCESS':
                if sha(d/'surface.pt')!=rec['checkpoint_sha256'] or sha(d/'arrays.npz')!=rec['array_sha256']:raise RuntimeError('Changed saved member')
                with np.load(d/'arrays.npz',allow_pickle=False) as f:pred=f['pred_source'].copy()
                return rec,pred
            return rec,None
        if (d/'STARTED.json').exists():raise RuntimeError('Interrupted case '+label+': preserve it, report partial, no automatic redraw/restart')
        yr,z=(y.copy(),np.zeros_like(y)) if rid is None else draw_table(y,sigma,rid)
        atomic_json(d/'STARTED.json',{'id':label,'release_sha256':sha(rp),'time_unix':time.time()})
        t0=time.perf_counter()
        try:
            model,xt,yt,st,scl=m.setup(x,yr,sigma,train)
            model,epoch,history=m.fit(model,xt,yt,st,train,2000,val)
            if not 1<=epoch<=2000:raise RuntimeError('Invalid selected epoch')
            with torch.no_grad(): vpred=model(xt).numpy()*scl[3]+scl[2]
            vmetric=comparison(vpred[val],yr[val],sigma[val])
            model,xt,yt,st,sc=m.setup(x,yr,sigma,np.arange(160))
            model,_,_=m.fit(model,xt,yt,st,np.arange(160),epoch);model.eval()
            with torch.no_grad():pred=model(xt).numpy()*sc[3]+sc[2]
            if not np.isfinite(pred).all():raise RuntimeError('Nonfinite surface')
            saved={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'input_mean':sc[0].tolist(),'input_std':sc[1].tolist(),
                'target_mean':float(sc[2]),'target_std':float(sc[3]),'feature_order':cp['feature_order'],
                'best_validation_epoch':epoch,'source_sha256':actual['source'],'prepared_sha256':actual['observations'],
                'protocol_sha256':actual['protocol'],'seed':m.SEED,'replica_id':label,'release_sha256':sha(rp)}
            save_torch(d/'surface.pt',saved)
            tmp=d/'arrays.tmp.npz';np.savez_compressed(tmp,pred_source=pred,observed_source=yr,perturbation_standard_normal=z,
                row_id=obs.row_id.to_numpy(dtype=str),sigma_reported=sigma);tmp.replace(d/'arrays.npz')
            if history:pd.DataFrame(history,columns=['epoch','training_mse','validation_mse']).to_csv(d/'split_history.csv',index=False)
            rec={'id':label,'status':'SUCCESS','selected_epoch':int(epoch),'elapsed_seconds':time.perf_counter()-t0,
                 'validation':vmetric,'all_row_fit':comparison(pred,yr,sigma),'checkpoint_sha256':sha(d/'surface.pt'),'array_sha256':sha(d/'arrays.npz')}
            if rid is None:
                rec['central_replay_max_abs']=float(np.max(abs(pred-c.ReH_smooth_source.to_numpy())))
                rec['central_replay_epoch_match']=int(epoch)==int(cp['best_validation_epoch'])
                # Mechanical replay tolerance, not a scientific quality threshold.
                rec['central_replay_pass']=bool(rec['central_replay_epoch_match'] and np.allclose(pred,c.ReH_smooth_source,rtol=1e-7,atol=1e-9))
            atomic_json(d/'RESULT.json',rec);print(label,json.dumps(rec),flush=True);return rec,pred
        except Exception as e:
            atomic_json(d/'RESULT.json',{'id':label,'status':'FAILED','error_type':type(e).__name__,'error':str(e),'elapsed_seconds':time.perf_counter()-t0})
            raise
    control,_=one(None)
    if not control.get('central_replay_pass',False):raise RuntimeError('Unperturbed source fitter did not reproduce saved central surface; do not launch members')
    records=[];preds=[]
    for rid in range(32):
        rec,pred=one(rid);records.append(rec)
        if pred is not None:preds.append((rid,pred))
    if len(preds)!=32:raise RuntimeError('Incomplete cohort: no claimed final spread; keep every failed/missing ID')
    matrix=np.stack([v for _,v in preds]);summ=summarize(matrix)
    rawc=c.ReH_smooth_source.to_numpy()
    table=obs[['row_id','source_line','source_set','method','xi','t_GeV2','Q2_GeV2','ReH_source','sigma_reported']].copy()
    table['fitted_central_ReH']=rawc
    for k in ['mean','std','q16','q50','q84']:table['replica_'+k]=summ[k]
    p=pd.read_csv(files['GPD_comparison']);metrics={}
    for label in ['central13','zero5','surface_selected']:
        z=p[p.state==label].set_index('row_id').loc[obs.row_id,'ReH_GPD_source'].to_numpy()
        table['GPD_'+label]=z;metrics[label]={'vs_fitted_central':comparison(z,rawc,sigma),'vs_replica_mean':comparison(z,summ['mean'],sigma)}
    table.to_csv(out/'surface_summary_160.csv',index=False,float_format='%.17g')
    np.savez_compressed(out/'coherent_surfaces.npz',replica_ids=np.arange(32),row_id=obs.row_id.to_numpy(dtype=str),
                        predictions_source=matrix,predictions_internal=-matrix,fitted_central_source=rawc,**summ)
    final={'status':'COMPLETE_CFF_ONLY_PILOT','members':32,'failed':0,'unperturbed_replay':control,
       'recorded_fit_seconds':sum(r['elapsed_seconds'] for r in records),'central_vs_replica_mean':comparison(summ['mean'],rawc,sigma),
       'median_replica_std':float(np.median(summ['std'])),'median_replica_std_over_original_sigma':float(np.median(summ['std']/sigma)),
       'comparison':metrics,'replica_records':records,'GPD_fits':0,'NLO_builds':0,
       'scope':'Spread of coherently fitted CFF functions under the stated table-error approximation; no GPD band or calibrated confidence claim.'}
    atomic_json(out/'SUMMARY.json',final)
    filesout=[f for f in out.rglob('*') if f.is_file() and f.name!='MANIFEST.json']
    atomic_json(out/'MANIFEST.json',{'files':{str(f.relative_to(out)):sha(f) for f in sorted(filesout)}})
    print('COMPLETE_CFF_ONLY_PILOT',flush=True)

if __name__=='__main__':main()

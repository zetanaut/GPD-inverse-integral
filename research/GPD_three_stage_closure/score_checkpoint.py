#!/usr/bin/env python3
"""Read-only scoring. No selection, fitting, cutoffs, or curve-based retuning."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
from closure.problem import ClosureProblem,GroupedForward
from closure.bank import RowBank
from closure.truths import Truth
from closure.metrics import field_metrics,cff_metrics
from closure.io import atomic_json,sha256
ROOT=Path(__file__).resolve().parent

def score(task_id,checkpoint,extra_banks,out):
    study=json.loads((ROOT/'prepared/STUDY.json').read_text());geom=json.loads((ROOT/'prepared/GEOMETRY.json').read_text())
    task=next(t for t in study['core_tasks'] if t['id']==task_id)
    bank=RowBank(ROOT/'inputs/accepted_experimental_banks',*extra_banks)
    problem=ClosureProblem(ROOT/'prepared/observed'/f"{task['branch']}_{task['truth']}.npz",bank,study,task['seed'])
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'flat' not in ck:raise ValueError('Export final checkpoint with flat vector and state_dict')
    flat=ck['flat'].detach().double()
    model=problem.set_model(flat)
    if 'state_dict' in ck:
        for k,v in model.state_dict().items():
            if not torch.equal(v,ck['state_dict'][k].double()):raise ValueError('Flat/state mismatch')
    def evaluator(x,xi,t):
        with torch.no_grad():return model(torch.tensor(x[None],dtype=torch.float64),torch.tensor([xi],dtype=torch.float64),torch.tensor([t],dtype=torch.float64))[0].numpy().T
    truth=Truth(task['truth']);metrics=field_metrics(evaluator,truth,geom)
    cffs={}
    with torch.no_grad():
        for role in [task['geometry'],'surface_check']:
            rows=geom['grids'][role]
            if any(not bank.has(r['xi'],r['Q2_GeV2']) for r in rows):
                cffs[role]={'status':'MISSING_CHECK_MAPS'};continue
            ref=GroupedForward(rows,bank,'reference')(model).numpy()
            nom=GroupedForward(rows,bank,'nominal')(model).numpy()
            target=bank.field_cff(rows,truth,'reference')
            sigma=np.array([[r['sigma_Re'],r['sigma_Im']] for r in rows])
            cffs[role]={'truth_error':cff_metrics(ref,target,sigma),'numerical_refinement':cff_metrics(nom,ref,sigma),
                'noisy_input_used_in_fit':task['branch']=='exp_Re_noisy' and role==task['geometry']}
    record={'task':task,'checkpoint_sha256':sha256(checkpoint),'study_sha256':sha256(ROOT/'prepared/STUDY.json'),
        'field':metrics,'CFF':cffs,'objective':float(problem.objective(flat)),
        'interpretation':'truth-error measurement after fixed selection; no confidence band or unique inversion claim'}
    atomic_json(out,record);return record
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',action='append',type=Path,default=[]);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(score(a.task,a.checkpoint,a.bank,a.out),indent=2))

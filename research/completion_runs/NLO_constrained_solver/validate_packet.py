"""Integrity and consistency checks on completed saved evidence; no fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,hashlib
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    records=[]
    def check(name,value):
        records.append({'name':name,'pass':bool(value)})
        if not value:raise AssertionError(name)
    frozen=read(HERE/'FREEZE.json')
    for rel,h in frozen['files'].items():check('frozen '+rel,sha(HERE/rel)==h)
    deps=read(HERE/'DEPENDENCIES.json')
    for rel,h in deps['files'].items():check('dependency '+rel,sha(ROOT/rel)==h)
    for r in deps['prior_payload']:check('prior evidence '+r['path'],sha(ROOT/r['path'])==r['sha256'])
    started=read(HERE/'STARTED.json');check('freeze precedes both continuations',frozen['unix_time']<started['unix_time'])
    check('exactly two authorized continuations',started['labels']==[31,43])
    check('two state directories',sorted(p.name for p in HERE.glob('start*') if p.is_dir())==['start31','start43'])
    check('mechanical preflight passed',read(HERE/'MECHANICAL.json')['pass'])
    for name,count in [('supplied_first_order_tests.log',12),('inherited_kkt_tests.log',16),('solver_tests.log',10)]:
        s=(HERE/'logs'/name).read_text();check(name,f'Ran {count} tests' in s and s.strip().endswith('OK'))
    for i,d in enumerate(read(HERE/'ANALYTIC_CONTROLS.json')):check('analytic actual KKT '+str(i),d['KKT']['pass'])
    total_accepted=0;counts=[]
    for label in (31,43):
        out=HERE/f'start{label}';r=read(out/'RESULT.json');opt=read(out/'OPTIMIZER.json');k=read(out/'KKT.json')
        source=read(HERE/'INITIAL_INPUTS.json')[f'inputs/initial{label}.pt']
        check(f'exact warm-start file {label}',sha(HERE/f'inputs/initial{label}.pt')==source['sha256']==sha(ROOT/source['source']))
        before=torch.load(HERE/f'inputs/initial{label}.pt',weights_only=True)['flat']
        check(f'exact initial vector {label}',torch.equal(before,torch.load(out/'initial.pt',weights_only=True)['flat']))
        iteration_files=sorted(out.glob('iteration*/ITERATION.json'))
        check(f'iteration ceiling {label}',len(iteration_files)<=600)
        accepted=0;terminations=0;max_terminations=0
        for p in iteration_files:
            d=read(p);after=torch.load(p.parent/'state.pt',weights_only=True)['flat']
            if d['accepted']:
                accepted+=1;terminations=0
                check(f'accepted state changed {label} {d["iteration"]}',not torch.equal(before,after))
                passed=[t for t in d['trials'] if t['accepted']]
                check(f'one accepted trial {label} {d["iteration"]}',len(passed)==1)
                trial=passed[0]
                check(f'strict all166 feasibility {label} {d["iteration"]}',len(trial['constraints'])==166 and max(trial['constraints'])<=0)
                check(f'original J decreases {label} {d["iteration"]}',trial['J']<d['J'] and trial['actual_decrease']>0)
            else:
                check(f'failed/terminal state unchanged {label} {d["iteration"]}',torch.equal(before,after))
                if d.get('termination')!='original_coordinate_KKT_pass':terminations+=1
                max_terminations=max(max_terminations,terminations)
            before=after
        final=torch.load(out/'final.pt',weights_only=True)['flat']
        check(f'last iterate returned {label}',torch.equal(before,final))
        check(f'all3867 parameters finite {label}',final.numel()==3867 and torch.isfinite(final).all().item())
        check(f'unchanged-state limit {label}',max_terminations<=3)
        if 'counts' in opt and 'accepted_steps' in opt['counts']:
            check(f'accepted-step count {label}',opt['counts']['accepted_steps']==accepted)
        if 'oracle_counts' in opt:check(f'evaluation cap {label}',opt['oracle_counts']['evaluations']<=12000)
        check(f'wall ceiling {label}',r['total_seconds']<=7200)
        check(f'J normalization components {label}',abs(sum(r['J_components'])-r['J'])<=1e-12)
        z=np.load(out/'FIRST_ORDER.npz',allow_pickle=False)
        check(f'first-order shapes {label}',z['original_gradient'].shape==(3867,) and z['constraints'].shape==(166,) and z['multipliers'].shape==(166,))
        check(f'working row coverage {label}',np.array_equal(z['working_indices'],np.flatnonzero(z['constraints']>=-.01)))
        check(f'strict primal rule {label}',k['full_primal_feasible']==bool(np.all(z['constraints']<=0)))
        check(f'raw gradient norm {label}',abs(np.linalg.norm(z['original_gradient'])-r['original_gradient_L2'])<=1e-12)
        check(f'nonnegative final multipliers {label}',np.all(z['multipliers']>=0))
        station=z['original_gradient']+z['working_jacobian'].T@z['multipliers'][z['working_indices']]
        check(f'original-coordinate stationarity recomposed {label}',abs(np.linalg.norm(station)-r['KKT_L2'])<=1e-10)
        check(f'complementarity recomposed {label}',abs(np.max(abs(z['multipliers']*z['constraints']))-r['complementarity'])<=1e-12)
        full=read(out/'FULL_FORWARD.json');expected={(xi,q) for xi in (.135,.275,.455) for q in (2.3,4.7,9.3,3.1,6.1)}
        check(f'all15 original forward points {label}',len(full['records'])==15 and {(v['xi'],v['Q2']) for v in full['records']}==expected)
        for v in full['records']:check(f'forward detail {label} {v["file"]}',sha(out/'full_forward'/v['file'])==v['sha256'])
        m=read(out/'MOMENTS.json');grid=np.array(m['new_grid']);xf=.02+.96*np.arange(41)/40;xd=.02+.96*(np.arange(80)+.5)/80
        check(f'new160 exact grid {label}',np.array_equal(grid,.02+.96*(np.arange(160)+.25)/160))
        check(f'new grid disjoint {label}',np.min(abs(grid[:,None]-xf))>1e-12 and np.min(abs(grid[:,None]-xd))>1e-12)
        check(f'final acceptance conjunction {label}',r['accepted']==bool(all(r['gates'].values()) and r['wall_budget_pass'] and not (out/'FAILURE.json').exists()))
        total_accepted+=r['accepted'];counts.append({'label':label,'accepted_steps':accepted,'max_unchanged_terminations':max_terminations})
    overall=read(HERE/'RESULT.json');check('aggregate acceptance',overall['accepted_count']==total_accepted)
    check('forbidden campaigns not run',all(overall[key]=='NOT_RUN' for key in ('historical_training','AL','kernels','other_datasets','profiles','calibration','experimental')))
    check('summary <=250 words',len((HERE/'SUMMARY.md').read_text().split())<=250)
    check('saved-state replay passed',read(HERE/'logs/replay.json')['pass'])
    result={'pass':all(r['pass'] for r in records),'checks_count':len(records),'checks':records,
            'continuations':counts,'scope':'Saved evidence integrity and consistency; no additional model fits'}
    (HERE/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pass':result['pass'],'checks':len(records)}))

if __name__=='__main__':main()

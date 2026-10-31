"""Check saved evidence and immutable sources; never launches an optimization."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    checks=[]
    def check(name,value):
        checks.append({'name':name,'pass':bool(value)})
        if not value:raise AssertionError(name)
    freeze=read(HERE/'FREEZE.json')
    for rel,h in freeze['files'].items():check('frozen '+rel,sha(HERE/rel)==h)
    for rel,h in read(HERE/'EXTERNAL_HASHES.json').items():check('external unchanged '+rel,sha(ROOT/rel)==h)
    for rel,rec in read(HERE/'SNAPSHOT_INDEX.json').items():check('snapshot '+rel,sha(HERE/rec['snapshot'])==rec['sha256'])
    for rec in read(ROOT/'completion_runs/NLO_noisy_feasibility/MANIFEST.json')['files']:
        check('original evidence '+rec['path'],sha(ROOT/rec['path'])==rec['sha256'])
    check('freeze predates starts',freeze['unix_time']<read(HERE/'STARTED.json')['unix_time'])
    check('exactly two starts',read(HERE/'STARTED.json')['only_starts']==[31,43])
    check('no extra start directories',sorted(p.name for p in HERE.glob('start*') if p.is_dir())==['start31','start43'])
    check('mechanical derivative checks',read(HERE/'MECHANICAL.json')['pass'])
    check('16 supplied KKT tests','Ran 16 tests' in (HERE/'logs/supplied_kkt_tests.log').read_text() and (HERE/'logs/supplied_kkt_tests.log').read_text().strip().endswith('OK'))
    check('5 algorithm/scoring safeguards','Ran 5 tests' in (HERE/'logs/mechanics_tests.log').read_text() and (HERE/'logs/mechanics_tests.log').read_text().strip().endswith('OK'))
    analytic=read(HERE/'ANALYTIC_CONTROL.json')
    check('actual analytic boundary solver KKT',analytic['actual_algorithm_result']['KKT']['pass'])
    reports=[]
    for label in (31,43):
        out=HERE/f'start{label}';r=read(out/'RESULT.json');opt=read(out/'OPTIMIZER.json')
        original=torch.load(HERE/f'inputs/initial{label}.pt',weights_only=True)['flat']
        copied=torch.load(out/'initial.pt',weights_only=True)['flat']
        check(f'original start {label}',torch.equal(original,copied) and original.numel()==3867)
        final=torch.load(out/'final.pt',weights_only=True)['flat']
        states=sorted(out.glob('outer*/state.pt'))
        check(f'last registered iterate returned {label}',torch.equal(final,torch.load(states[-1],weights_only=True)['flat']))
        check(f'outer count {label}',0<len(states)<=24 and len(states)==len(opt['history']))
        check(f'inner evaluation caps {label}',all(v['evaluations']<=160 for v in opt['history']))
        check(f'no objective change {label}',abs(sum(r['J_components'])-r['J'])<=1e-12)
        check(f'wall ceiling {label}',r['total_seconds']<=7200)
        full=read(out/'FULL_FORWARD.json');expected={(xi,q) for xi in (.135,.275,.455) for q in (2.3,4.7,9.3,3.1,6.1)}
        check(f'full15 points {label}',len(full['records'])==15 and {(d['xi'],d['Q2']) for d in full['records']}==expected)
        for rec in full['records']:
            check(f'forward detail hash {label} {rec["file"]}',sha(out/'full_forward'/rec['file'])==rec['sha256'])
        moments=read(out/'MOMENTS.json');grid=np.array(moments['new_grid']);xf=.02+.96*np.arange(41)/40;xd=.02+.96*(np.arange(80)+.5)/80
        check(f'new160 exact grid {label}',len(grid)==160 and np.array_equal(grid,.02+.96*(np.arange(160)+.25)/160))
        check(f'new grid disjoint {label}',np.min(abs(grid[:,None]-xf))>1e-12 and np.min(abs(grid[:,None]-xd))>1e-12)
        kkt=read(out/'KKT.json');raw=np.load(out/'FIRST_ORDER.npz',allow_pickle=False)
        check(f'KKT vector shapes {label}',raw['gradient'].shape==(3867,) and raw['constraints'].shape==(166,) and raw['active_jacobian'].shape==(len(raw['active_indices']),3867))
        check(f'no hidden primal relaxation {label}',kkt['full_primal_feasible']==bool(np.all(raw['constraints']<=0)))
        check(f'acceptance exactly required gates {label}',r['accepted']==(all(r['gates'].values()) and r['wall_budget_pass'] and not (out/'FAILURE.json').exists()))
        check(f'finite final state {label}',torch.isfinite(final).all().item())
        check(f'gradient norm matches record {label}',abs(np.linalg.norm(raw['gradient'])-r['raw_gradient_L2'])<=1e-12)
        old96=moments['grids']['enforced80']['quadrature_RMS']['96']
        reports.append({'label':label,'accepted':r['accepted'],'original80_moment_96_gate':old96<=1e-3,
                        'original80_moment_96_RMS':old96,'full_forward_pass':full['pass'],
                        'final_derivative_pass':read(out/'FINAL_DERIVATIVES.json')['pass']})
    result=read(HERE/'RESULT.json')
    check('aggregate accepted count',result['accepted_count']==sum(r['accepted'] for r in reports))
    check('no remaining datasets or campaigns',all(result[k]=='NOT_RUN' for k in ('other_datasets','profiles','calibration')))
    check('SUMMARY <=250 words',len((HERE/'SUMMARY.md').read_text().split())<=250)
    record={'pass':all(r['pass'] for r in checks),'checks_count':len(checks),'checks':checks,
            'old_moment_gate_reports':reports,'scope':'Integrity and consistency of completed evidence; no additional fit'}
    (HERE/'VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'pass':record['pass'],'checks':len(checks)}))

if __name__=='__main__':main()

"""Read-only objective/gradient/decision replay. Optional full NLO numerical replay."""
import sys
sys.dont_write_bytecode=True
from driver import *
import argparse,platform

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--numerical',action='store_true');args=parser.parse_args()
    start=time.monotonic();inputs=verify_sources()
    before=c.digest(ROOT/'DECISIONS.json');decisions=json.loads((ROOT/'DECISIONS.json').read_text())
    summary=json.loads((ROOT/'BRANCH_SUMMARY.json').read_text());props=json.loads((ROOT/'PROPOSALS.json').read_text())
    provenance=json.loads((ROOT/'IMPLEMENTATION_BEFORE_PROPOSALS.json').read_text())
    for name,h in provenance['source_files'].items():assert c.digest(ROOT/name)==h,name
    assert provenance['plan_sha256']==c.digest(ROOT/'PLAN.json')
    parents={r['key']:r for r in PLAN['parents']};results=[]
    for r in summary['branches']:
        if not r['actual_continuation']:continue
        parent=parents[r['parent']];p=load_parent(parent,r['branch'],r['supports_z'])
        out=ROOT/'branches'/r['key'];saved=torch.load(out/'final.pt',weights_only=True)
        assert c.digest(out/'final.pt')==r['checkpoint_sha256']
        assert saved['parent_sha256']==parent['parent_sha256'] and saved['active_supports_z']==r['supports_z']
        p.model.load_state_dict(saved['state_dict'])
        x=torch.cat([v.detach().flatten() for v in p.model.parameters()]);assert torch.equal(x,saved['flat'])
        p.x0=x;interface=n.interface_checks(p);assert interface['pass']
        x=x.requires_grad_(True);f=p.original_objective(x);g=torch.autograd.grad(f,x)[0]
        j=float(f.detach());gn=float(g.norm())
        assert abs(j-r['J'])<=1e-12*max(1,abs(j))
        assert abs(gn-r['gradient_l2'])<=1e-9*max(1,abs(gn))
        full=moment_check(p.model,full_supports(p));selected=moment_check(p.model,r['supports_z'])
        for actual,stored in [(full,r['full_space']),(selected,r['selected_space'])]:
            for order in ('96','192'):
                assert abs(actual['orders'][order]['combined_rms']-stored['orders'][order]['combined_rms'])<=1e-11
        with torch.no_grad():pred=n.NLOForward(Q2S)(p.model).numpy().reshape(-1,2)
        difference=float(np.max(abs(pred-np.array([v['nominal_prediction'] for v in r['CFF']]))))
        assert difference<=1e-10
        max_num_change=0.;checks=0
        for rec in r['CFF']:
            nr=json.loads((out/f'numerical_xi{rec["xi"]}_Q2{rec["Q2"]}.json').read_text())
            assert nr['CFF_estimated_total']==rec['CFF_budget'] and nr['pass']==rec['numerical_pass']
            if args.numerical:
                point=next(v for v in c.operator_directories() if v['xi']==rec['xi'])
                fresh=numerical.validate_point(point,rec['Q2'],r['key'],p.model)
                assert fresh['gates']==nr['gates']
                change=float(np.max(abs(np.array(fresh['CFF_estimated_total'])-nr['CFF_estimated_total'])))
                assert change<=1e-8;max_num_change=max(max_num_change,change);checks+=1
        results.append({'key':r['key'],'J':j,'fresh_gradient_l2':gn,'interface':interface,
            'full_RMS_192':full['orders']['192']['combined_rms'],'selected_RMS_192':selected['orders']['192']['combined_rms'],
            'moment_quadrature_change':max(full['diagnostic_residual_RMS_change'],selected['diagnostic_residual_RMS_change']),
            'nominal_CFF_replay_max_difference':difference,'numerical_checks_replayed':checks,
            'numerical_budget_max_replay_change':max_num_change,'saved_gates':r['gates']})
    bykey={r['key']:r for r in summary['branches']}
    for parent,proposal,stored in zip(PLAN['parents'],props['parents'],decisions['decisions']):
        fresh=decision(parent,bykey[parent['key']+'__allterm'],bykey[parent['key']+'__proposed'],proposal['identity'])
        assert fresh==stored
    after=json.loads((ROOT/'AFTER_DECISION_TRUTH.json').read_text())
    assert after['decisions_sha256_before_truth_evaluation']==before
    expressions=json.loads((ROOT/'EXPRESSIONS.json').read_text())['expressions']
    for e in expressions:
        z=np.array(e['coefficient_z']);xi=np.array(e['coefficient_xi'])
        np.testing.assert_allclose(z,xi*.6**(2*np.arange(len(z))),rtol=1e-12,atol=1e-14)
    assert c.digest(ROOT/'DECISIONS.json')==before
    verify_sources()
    print(json.dumps({'pass':True,'read_only':True,'input_files_verified':inputs,'nominal_branches':8,
      'actual_continuations':summary['actual_continuations'],'replayed_branches':results,
      'decisions_reproduced':True,'truth_diagnostics_after_decisions_hash_verified':True,
      'original_sources_unchanged':True,'optimization_steps':0,'kernels_built':0,
      'seconds':time.monotonic()-start,'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform()},
      'actual_tokens_cost':'unknown'},indent=2))

if __name__=='__main__':main()

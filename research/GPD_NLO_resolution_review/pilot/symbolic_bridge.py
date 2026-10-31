"""Existing v5 symbolic hypotheses -> same-CFF-loss refit interface.

Support selection is never called by the four primary fits. This bridge exposes
the existing v5 compression rule; it does not validate a symbolic hypothesis.
"""
from common import *
from neural_pilot import Problem

def hypotheses(xi, moments, spec, tolerance):
    return poly.symbolic_candidates(xi, moments, spec, tolerance)

def refit_problem(truth, seed, state_dict, supports):
    p=Problem(truth,seed);p.model.load_state_dict(state_dict)
    p.x0=torch.cat([v.detach().flatten() for v in p.model.parameters()])
    p.p=poly.MomentProjector(p.p.xi.numpy(),order=48,scales=p.p.scales,supports=supports)
    p.complements=[]
    for (_,n,_),support in zip(p.p.spec,supports):
        design=poly.design(p.p.xi.numpy(),n,support)
        q=np.linalg.qr(design,mode='complete')[0]
        p.complements.append(torch.tensor(q[:,design.shape[1]:]))
    return p

def unit_test():
    summary=json.loads((WORK/'FIT_SUMMARY.json').read_text())
    assert summary['pass'],'All primary gates must pass before symbolic interface test'
    p=Problem('smooth',31)
    saved=torch.load(WORK/'fits/smooth_seed31/final.pt',weights_only=True)
    p.model.load_state_dict(saved['state_dict']);p.x0=saved['flat']
    # Exact toy polynomial moments test plumbing only, not learned-field support selection.
    xi=p.p.xi.numpy();spec=p.p.spec
    toy=np.column_stack([(i+1)*np.ones(len(xi)) for i in range(len(spec))])
    candidates=hypotheses(xi,toy,spec,1e-10)
    assert all(c['support']==[0] for c in candidates)
    full=[list(range(n//2+1)) for _,n,_ in spec]
    same=refit_problem('smooth',31,saved['state_dict'],full)
    a=p.original_objective(p.x0);b=same.original_objective(same.x0)
    restricted=refit_problem('smooth',31,saved['state_dict'],[c['support'] for c in candidates])
    x=restricted.x0.detach().requires_grad_(True)
    value=restricted.original_objective(x);g=torch.autograd.grad(value,x)[0]
    norm=restricted.residual(x).square().sum()
    assert torch.isfinite(value) and torch.isfinite(g).all()
    assert abs(float(a-b))<1e-12 and abs(float(value-norm))<1e-10
    result={'pass':True,'existing_symbolic_candidates':True,'toy_expected_support':[0],
       'all_allowed_refit_objective_difference':abs(float(a-b)),
       'restricted_refit_objective_residual_difference':abs(float(value-norm)),
       'restricted_gradient_finite':True,'optimization_steps':0,
       'primary_support_selection':'OFF','learned_moment_support_selection':'not executed'}
    write(WORK/'SYMBOLIC_INTERFACE_TEST.json',result)
    return result

if __name__=='__main__':print(json.dumps(unit_test(),indent=2))

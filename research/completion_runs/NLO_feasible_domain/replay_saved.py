"""Portable audit of frozen states. Invoked by reproduce.py in a temporary tree."""
from domain import *
from optimizer import necessary_kkt,SETTINGS
import scipy

def main():
    results=[];d=Domain()
    for label in (31,43):
        p=Problem(label);out=HERE/f'start{label}'
        x=torch.load(out/'final.pt',weights_only=True)['flat'].detach().requires_grad_(True)
        j=p.objective(x);g=torch.autograd.grad(j,x)[0].detach().numpy()
        c=d.constraints(p,x);cn=c.detach().numpy()
        indices=np.flatnonzero(cn>=-SETTINGS['active_tolerance'])
        rows=[torch.autograd.grad(c[int(i)],x,retain_graph=True)[0].detach().numpy() for i in indices]
        A=np.stack(rows) if rows else np.empty((0,len(x)))
        current=necessary_kkt(g,cn,A,indices,d.names)
        with np.load(out/'FIRST_ORDER.npz',allow_pickle=False) as saved:
            assert np.array_equal(indices,saved['active_indices'])
            errors={'objective':abs(float(j.detach())-json.loads((out/'RESULT.json').read_text())['J']),
                    'gradient_max_abs':float(np.max(abs(g-saved['gradient']))),
                    'constraints_max_abs':float(np.max(abs(cn-saved['constraints']))),
                    'active_jacobian_max_abs':float(np.max(abs(A-saved['active_jacobian']),initial=0.))}
        ok=errors['objective']<=1e-10 and errors['gradient_max_abs']<=1e-8 and errors['constraints_max_abs']<=1e-6 and errors['active_jacobian_max_abs']<=1e-5
        results.append({'label':label,'reproduction_pass':bool(ok),'J':float(j.detach()),
                        'KKT':current,'differences_from_saved':errors,
                        'saved_full_result':json.loads((out/'RESULT.json').read_text())})
    result={'scope':'Independent reevaluation of original J, gradient, all filter constraints and active Jacobians. Full-forward gate outcomes read from hashed saved records; no new optimization.',
            'runtime':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
            'pass':all(r['reproduction_pass'] for r in results),'starts':results}
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result['pass']:raise SystemExit(1)

if __name__=='__main__':main()

"""Read-only saved-state replay with captured dependency-source verification.
The immutable previous packet supplies domain.py, data and contractions.
"""
from problem import *
import json,zipfile,hashlib
import scipy

def main():
    freeze=json.loads((HERE/'FREEZE.json').read_text());deps=json.loads((HERE/'DEPENDENCIES.json').read_text())
    for rel,h in freeze['files'].items():assert sha(HERE/rel)==h,rel
    for rec in deps['prior_payload']:assert sha(ROOT/rec['path'])==rec['sha256'],rec['path']
    index=json.loads((OLD/'SNAPSHOT_INDEX.json').read_text());captured=0;referenced=[]
    for rel,h in deps['files'].items():
        path=ROOT/rel
        if Path(rel).is_absolute():
            path=OLD/index[rel]['snapshot'];captured+=1
        if path.exists():assert sha(path)==h,rel
        elif rel in ('review_NLO_feasible_domain.zip','GPD_NLO_feasible_domain_review.zip'):
            # Prior scientific payload was verified above. Input archive hashes
            # remain explicit even if a reviewer retained only extracted files.
            referenced.append({'path':rel,'sha256':h,'status':'archive explicitly referenced; extracted runtime payload verified'})
        elif rel.startswith('GPD_NLO_feasible_domain_review/'):
            local=HERE/'review_input'/rel.split('/',1)[1]
            assert sha(local)==h,rel
        else:raise FileNotFoundError('Missing immutable dependency '+rel)
    records=[]
    for label in (31,43):
        o=Oracle(label);out=HERE/f'start{label}';x=torch.load(out/'final.pt',weights_only=True)['flat'].numpy()
        saved=np.load(out/'FIRST_ORDER.npz',allow_pickle=False);ids=saved['working_indices'];v=o.values(x)
        g,R,A=o.derivatives(x,ids);d=kkt(g,v['c'],A,ids,o.names)
        prior=json.loads((out/'KKT.json').read_text())
        errors={'J':abs(v['J']-prior['J']),'gradient_max_abs':float(np.max(abs(g-saved['original_gradient']))),
                'constraints_max_abs':float(np.max(abs(v['c']-saved['constraints']))),
                'Jacobian_max_abs':float(np.max(abs(A-saved['working_jacobian']),initial=0.))}
        passed=errors['J']<=1e-10 and errors['gradient_max_abs']<=1e-8 and errors['constraints_max_abs']<=1e-6 and errors['Jacobian_max_abs']<=1e-5
        records.append({'label':label,'J':v['J'],'KKT':d,'saved_KKT_pass':prior['pass'],
                        'same_gate_decision':d['pass']==prior['pass'],'errors':errors,
                        'pass':bool(passed and d['pass']==prior['pass'])})
    result={'pass':all(r['pass'] for r in records),'records':records,
            'dependency_payload_verified':len(deps['prior_payload']),
            'captured_environment_sources_verified':captured,'referenced_archives':referenced,
            'runtime':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
            'scope':'J, original-coordinate gradient, constraints and relevant Jacobian rows recomputed; no optimization or historical full-forward audit'}
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result['pass']:raise SystemExit(1)

if __name__=='__main__':main()

"""Verify saved checkpoints/results; no training, iterations or new kernels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
from common import *
import time, platform
from neural_pilot import Problem, NLOForward, moment_diagnostics

def main():
    t=time.monotonic();sources=json.loads((WORK/'SOURCE_HASHES.json').read_text())
    for f,m in sources.items():
        assert digest(PROJECT/f)==m['sha256'],f
    manifest=json.loads((HERE.parent/'MANIFEST.json').read_text())
    for f,m in manifest.items(): assert digest(HERE.parent/f)==m['sha256'],f
    records=[];plan=json.loads((HERE/'PILOT_PLAN.json').read_text())
    for truth in plan['truths']:
        for seed in plan['seeds']:
            out=WORK/'fits'/f'{truth}_seed{seed}'
            saved=torch.load(out/'final.pt',weights_only=True);metrics=json.loads((out/'METRICS.json').read_text())
            p=Problem(truth,seed);p.model.load_state_dict(saved['state_dict'])
            flat=torch.cat([v.detach().flatten() for v in p.model.parameters()])
            assert torch.equal(flat,saved['flat'])
            x=flat.requires_grad_(True);J=p.original_objective(x);g=torch.autograd.grad(J,x)[0]
            j=float(J.detach());gn=float(g.norm())
            assert abs(j-metrics['J'])<=1e-13*max(1,abs(j))
            assert abs(gn-metrics['gradient_l2'])<=1e-10
            poly_diag=moment_diagnostics(p.model)
            assert abs(poly_diag['combined_rms']-metrics['moments']['combined_rms'])<=1e-12
            forward=NLOForward(plan['training_Q2']+plan['heldout_Q2'])
            with torch.no_grad():pred=forward(p.model).numpy().reshape(-1,2)
            stored=np.array([r['nominal_prediction'] for r in metrics['CFF_metrics']])
            delta=float(np.max(abs(pred-stored)));assert delta<=1e-10
            records.append({'key':p.key,'J':j,'fresh_gradient_l2':gn,'polynomiality_RMS':poly_diag['combined_rms'],
              'saved_CFF_max_absolute_difference':delta,'checkpoint_sha256':digest(out/'final.pt'),'saved_gates':metrics['gates']})
    result={'saved_results_reproduced':True,'immutable_source_files_verified':len(sources),'supplied_manifest_files_verified':len(manifest),
      'runs':records,'optimizer_steps':0,'kernels_built':0,'seconds':time.monotonic()-t,
      'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'torch':torch.__version__},
      'cost_tokens':'unknown; runtime does not expose accounting'}
    write(WORK/'REPLAY_VERIFICATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

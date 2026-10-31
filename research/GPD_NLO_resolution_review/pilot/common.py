"""Reference-scale neural adapter; all evolution matrices are read-only inputs."""
from pathlib import Path
import sys, json, hashlib, importlib.util
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
NLO = PROJECT / 'GPD_NLO_completion_package'
TASK04 = PROJECT / 'GPD_task04_solver_repair'
WORK = HERE / 'work'
WORK.mkdir(exist_ok=True)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); tmp.replace(path)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

VENDOR = TASK04 / 'vendor/gpd_task04_local/vendor'
models = module('original_film', VENDOR/'models.py')
poly = module('original_polynomiality', VENDOR/'polynomiality.py')
helpers = module('original_moment_helpers', VENDOR/'review_helpers.py')
ReferenceNetwork = models.ReferenceNetwork

def projector(order=48, xi=None):
    if xi is None: xi = helpers.moment_grids()[0]
    scales = [helpers.moment_scale(c,n) for c in ('Sigma','T','Hg') for n in (2,4,6,8)]
    return poly.MomentProjector(xi, order=order, scales=scales)

def operator_directories():
    base = 'work/operators/'
    points = [{
        'xi': .135, 'frozen_q2': 2.3,
        'base': base+'89ab51d596c4c97272320ecc912209a40de9066d069ac3b45c6adcd16b654c34',
        'inner72': base+'a8126b3d82b130a78c1b7024181208929ef8325a3c85483771449239720f4584',
        'output48': base+'f16084d9b179c783b37d76f013e105bdf08d81fa262e3fe2fd603acb94715952'}]
    for xi,q2 in [(.275,4.7),(.455,9.3)]:
        p = json.loads((NLO/f'work/point_{xi}/PIPELINE.json').read_text())
        points.append(dict(xi=xi,frozen_q2=q2,**{k:p[k] for k in ('base','inner72','output48')}))
    for p in points:
        dirs = json.loads((NLO/p['base']/f"POINT_Q2_{p['frozen_q2']:g}.json").read_text())['directories']
        for kind in ('inner72','output48'):
            d = json.loads((NLO/p[kind]/f"POINT_Q2_{p['frozen_q2']:g}.json").read_text())['directories']
            dirs.update({kind+'_'+k:d[k] for k in ('weak10','strong10')})
        p['directories'] = dirs
    return points

def packing(b, values):
    """(...,node,channel) -> [...Sigma[qids], T[qids], Hg[gids]]."""
    return np.concatenate((values[...,b.qids,0], values[...,b.qids,1], values[...,b.gids,2]),axis=-1)

def dense_row(b, jac):
    out=np.zeros((2,3,len(b.nodes)));nq=len(b.qids)
    out[:,0,b.qids]=jac[:,:nq];out[:,1,b.qids]=jac[:,nq:2*nq];out[:,2,b.gids]=jac[:,2*nq:]
    return out

def sample(model,b):
    with torch.no_grad():
        return model(torch.tensor(b.nodes[None,:]),torch.tensor([b.xi]))[0].T.numpy()

def source_inventory():
    paths = [TASK04/n for n in ('solver.py','stationarity.py','run_fresh_smoke.py','CANDIDATE_OPTIMIZER_CONTRACT.json')]
    paths += [VENDOR/n for n in ('models.py','polynomiality.py','review_helpers.py')]
    paths += [NLO/f for d in ('candidate','src','frozen/python','evidence/hard_verified/corrected','contracts')
              for f in [str(p.relative_to(NLO)) for p in (NLO/d).rglob('*') if p.is_file() and p.suffix in ('.py','.json')]]
    paths += [PROJECT/'review_03_resolution.zip',HERE.parent/'data/nlo_rows.npz',HERE.parent/'WORKER_PROMPT.md',HERE.parent/'DECISION.md']
    for p in operator_directories():
        for d in p['directories'].values():
            paths += [NLO/d/'KEY.json',NLO/d/'operators.npz']
    return {str(p.relative_to(PROJECT)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(set(paths))}

def prepare():
    """Freeze plan and random states before any optimization."""
    if (HERE/'PILOT_PLAN.json').exists():
        return json.loads((HERE/'PILOT_PLAN.json').read_text())
    hashes=source_inventory();write(WORK/'SOURCE_HASHES.json',hashes)
    import zipfile
    verified=[]; retained=[]
    with zipfile.ZipFile(PROJECT/'review_03_resolution.zip') as z:
        accepted=json.loads(z.read('REVIEW_MANIFEST.json'))
        for path,entry in hashes.items():
            prefix='GPD_NLO_completion_package/'
            if path.startswith(prefix):
                rel=path[len(prefix):]
                if rel in accepted:
                    assert entry==accepted[rel], rel
                    verified.append(rel)
                else: retained.append(rel)
    checkpoints={}
    for seed in (31,43):
        torch.manual_seed(seed);model=ReferenceNetwork('film').double()
        path=WORK/f'initial_seed{seed}.pt'
        torch.save({'state_dict':model.state_dict(),'seed':seed,'origin':'fresh random initialization; no truth evaluation'},path)
        checkpoints[str(seed)]={'path':str(path.relative_to(HERE)),'sha256':digest(path)}
    plan={
      'purpose':'Bounded noiseless NLO FiLM integration; scale interpolation, not held-out skewness or uncertainty calibration',
      'source_hashes':'work/SOURCE_HASHES.json','source_hashes_sha256':digest(WORK/'SOURCE_HASHES.json'),
      'accepted_archive_verified_files':verified,'other_retained_source_files':retained,
      'checkpoints':checkpoints,'model':'Original width-24 FiLM; unchanged parity and (1-x^2)^2 envelope; only x,xi and fixed t=0 conditioning; Q02=2; no learned Q2',
      'channels':['Sigma','T_rescaled','Hg=xg'],
      'packing':'network [xi,channel,node] transposed to [xi,node,channel]; state Sigma[qids],T[qids],Hg[gids]; dense row zeros at unused nodes; charges 2/9 and gluon charges already in rows',
      'truths':['smooth','Dterm_added'],'seeds':[31,43],'xi':[.135,.275,.455],
      'training_Q2':[2.3,4.7,9.3],'heldout_Q2':[3.1,6.1],
      'loss':{'formula':'mean(((pred-y)/0.04)^2) + mean((I-QQ^T)M / fixed_moment_scales)^2',
              'data_scale':.04,'interpretation':'common fixed unit-reference numerical normalization for every truth, seed and Re/Im component; no noise and no statistical covariance',
              'lambda_poly':1.,'coefficients':'all 0,2,...,N free; no symbolic selection'},
      'moments':{'spins':[2,4,6,8],'fit_xi':list(helpers.moment_grids()[0]),'diagnostic_xi':list(helpers.moment_grids()[1]),
                 'fit_quadrature':48,'diagnostic_quadrature':96,'quadrature':'positive-half split GL at xi, exact parity factor2',
                 'scales':[helpers.moment_scale(c,n) for c in ('Sigma','T','Hg') for n in (2,4,6,8)],'diagnostic_normalized_RMS_limit':1e-3},
      'solver':{'source':'GPD_task04_solver_repair/CANDIDATE_OPTIMIZER_CONTRACT.json',
                'Adam':{'steps':400,'lr':.003,'betas':[.9,.999],'eps':1e-8,'weight_decay':0},
                'LBFGS':{'max_iter':400,'max_eval':500,'history_size':40,'tolerance_grad':1e-10,'tolerance_change':1e-13,'line_search_fn':'strong_wolfe'},
                'Gauss_Newton':{'max_jacobians':60,'max_trials':12,'initial_damping_relative':1e-4,'damping_min_relative':1e-14,'accept_ratio':1e-4,'acceleration':True,'acceleration_limit':.5,'acceleration_fd':.1},
                'stationarity_cleanup_max_steps':5,'fresh_original_gradient_L2_limit':1e-6},
      'numerics':{'CFF_component_limit':'1e-4*max(1,abs(reference_component))','relative_physical_field_L2_limit':1e-3,
                  'budget':'same additive empirical estimates as accepted budget_suite.py, with existing inner72/output48 operators and p6,p8,p10 products; hard192->384; independent hard eps1e-9',
                  'time':'DOP853 1e-11/1e-13 vs 1e-13/1e-15 and existing independent RK4 steps128',
                  'spin2_relative_moment_limit':1e-4,'Dterm_DGLAP_absolute_limit':1e-6,
                  'gradient_inputs':['seed31 original non-polynomial FiLM','seed43 original non-polynomial FiLM'],
                  'gradient_point':{'xi':.275,'Q2':4.7},'direction_seed':1703,'finite_difference_steps':[1e-4,1e-5,1e-6],
                  'gradient_component_tolerance':'1e-6*max(1,abs(directional_derivative))',
                  'initial_fields':'both random seeds, all 15 xi/Q2 combinations, before fitting'},
      'stop':'truth preflight failure stops affected truth pilot; initial numerical failure stops affected seed; no point removal, retuning, extra iterations or kernels',
      'symbolic':'OFF; interface unit test only if all four primary numerical/solver gates pass',
      'token_cost':'not available from runtime; unknown'}
    write(HERE/'PILOT_PLAN.json',plan)
    return plan

if __name__=='__main__':
    print(json.dumps(prepare(),indent=2))

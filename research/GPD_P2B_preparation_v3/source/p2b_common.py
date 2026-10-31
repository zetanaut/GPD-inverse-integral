"""Read-only binding to P1; distinct module name preserves its legacy common."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def setup(capsule):
    capsule = Path(capsule).resolve()
    if sha(capsule/'MANIFEST.json') != '53e6a2e8237285a5ac91ba9ffed2b916d8bc86c7deffba6910d9f36ddb6066bd':
        raise RuntimeError('Original P1 manifest identity mismatch')
    binding = ROOT/'SOURCE_BINDINGS.json'
    if binding.exists():
        for item in json.loads(binding.read_text())['capsule_files']:
            if sha(capsule/item['path']) != item['sha256']:
                raise RuntimeError('Bound dependency mismatch: '+item['path'])
    sys.path.insert(0, str(capsule/'runtime'))
    import core
    if core.CAPSULE.resolve() != capsule:
        raise RuntimeError('Wrong imported core capsule')
    core.setup()
    core.check_scientific_bindings()
    return core


def field_only(core):
    import numpy as np
    import torch
    from closure.problem import ClosureProblem
    from gpd_reh.model import ExperimentalFiLM
    cfg = json.loads((core.STUDY/'prepared/FIT_CONFIG.json').read_text())
    base = ClosureProblem.__new__(ClosureProblem)
    with torch.random.fork_rng():
        torch.manual_seed(31)
        base.model = ExperimentalFiLM(cfg['network']['t_mean'], cfg['network']['t_std']).double()
    base.spec = [(n, p.shape, p.numel()) for n, p in base.model.named_parameters()]
    initial = np.load(core.CAPSULE/'reference/FRESH_SEED31.npy', allow_pickle=False)
    if hashlib.sha256(initial.tobytes()).hexdigest() != core.INITIAL_SHA:
        raise RuntimeError('Fresh initialization changed')
    base.x0 = torch.tensor(initial.copy(), dtype=torch.float64)
    base.set_model(base.x0)
    return base


def information(base):
    import numpy as np
    from pdf_information import ForwardInformation
    meta = json.loads((ROOT/'inputs/OBSERVATION_METADATA.json').read_text())
    if sha(ROOT/'DESIGN.json') != meta['design_sha256'] or sha(ROOT/'inputs/CENTRAL_FORWARD.npz') != meta['input_sha256']:
        raise RuntimeError('Frozen forward input/design identity mismatch')
    if (meta['Q2_GeV2'], meta['xi'], meta['t_GeV2']) != (2., 0., 0.):
        raise RuntimeError('Wrong forward-boundary scale or coordinates')
    with np.load(ROOT/'inputs/CENTRAL_FORWARD.npz', allow_pickle=False) as z:
        return ForwardInformation(base, z['x'], z['observed'], z['covariance'])

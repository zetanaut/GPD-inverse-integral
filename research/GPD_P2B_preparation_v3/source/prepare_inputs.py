"""One-time deterministic preparation; no scientific noise draws or joint fits."""
import argparse
from p2b_common import ROOT, setup, write, sha
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    if (ROOT/'inputs/CENTRAL_FORWARD.npz').exists():
        raise RuntimeError('Prepared input already exists; do not silently replace it')
    core = setup(args.capsule)
    verified = core.verify_manifest(full=True)
    import numpy as np
    import torch
    from closure.truths import Truth
    cfg = json.loads((ROOT/'DESIGN.json').read_text())
    x = np.geomspace(cfg['x_grid']['minimum'], cfg['x_grid']['maximum'], cfg['x_grid']['count'])
    central = Truth('Z1').values(x, 0., 0.)
    cc = cfg['covariance']
    scale = cc['relative_sigma']*np.maximum(abs(central), cc['floor_fraction_of_channel_grid_maximum']*abs(central).max(axis=0))
    Kx = np.exp(-abs(np.log(x[:, None])-np.log(x[None, :]))/cc['log_length'])
    Kc = np.array(cc['channel_correlation'])
    correlation = np.kron(Kx, Kc)
    cov = scale.reshape(-1, 1)*correlation*scale.reshape(1, -1)
    L = np.linalg.cholesky(cov)
    (ROOT/'inputs').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'inputs/CENTRAL_FORWARD.npz', x=x, observed=central, covariance=cov)
    write(ROOT/'inputs/OBSERVATION_METADATA.json', dict(
        experiment=cfg['version'], role='central synthetic control; finite covariance retained',
        design_sha256=sha(ROOT/'DESIGN.json'), input_sha256=sha(ROOT/'inputs/CENTRAL_FORWARD.npz'),
        Q2_GeV2=2., Q_GeV=2.**.5, xi=0., t_GeV2=0.,
        external_PDF_set=None, vector_order='x-major, channel-minor', dimension=36,
        covariance_eigenvalues=np.linalg.eigvalsh(cov).tolist(),
        covariance_condition_number=float(np.linalg.cond(cov)), cholesky_max_error=float(abs(L@L.T-cov).max()),
        synthetic_scales=scale.tolist(), new_noise_draws=0,
        unmeasured='No constraints outside sampled boundary locations; no precision assigned to continuum gaps, tails, or finite-xi/t directions'))
    # Flat tensors only, exported from identified prior states for diagnostics.
    locations = {
        'exact_baseline': 'review_GPD_P1_advice/reference_capsule/project/GPD_three_stage_closure/runs/exp_Re_exact_Z1_s31/final.pt',
        'half_baseline': 'GPD_P2A_preflight_v1/evidence/HALF_STAGE60.pt',
        'P2A_preservation': 'review_GPD_P2A_preservation_results_v1/results/GPD_P2A_preservation_resume_20261023_a01/GPD_P2A_preserve_low_exact_s31/STAGE60.pt',
    }
    values = {}
    records = []
    safe = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype('float64'))]
    for name, relative in locations.items():
        path = args.project/relative
        with torch.serialization.safe_globals(safe):
            ck = torch.load(path, map_location='cpu', weights_only=True)
        values[name] = ck['flat'].detach().double().numpy().copy()
        records.append(dict(name=name, source=relative, sha256=sha(path)))
    (ROOT/'evidence').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'evidence/DIAGNOSTIC_STATES.npz', **values)
    write(ROOT/'evidence/DIAGNOSTIC_STATES.json', dict(states=records, scientific_initializers=False,
         scope='Boundary diagnostics only; do not initialize a scientific run with these fitted states'))
    manifest = json.loads((args.capsule/'MANIFEST.json').read_text())
    required = [r for r in manifest['files'] if r['path'].endswith('.py') or r['path'] in (
        'reference/FRESH_SEED31.npy', 'project/GPD_three_stage_closure/prepared/FIT_CONFIG.json',
        'project/GPD_three_stage_closure/prepared/observed/exp_Re_exact_Z1.npz',
        'project/GPD_operating_range_plan/prepared/precision160_Z1_a050_s31.npz')]
    write(ROOT/'SOURCE_BINDINGS.json', dict(
        capsule_archive='GPD_P1_Rivanna_run.zip',
        capsule_archive_sha256='98d6a1161218a075d0556d7e811eda0a6ffcbc2250345a28893488a82fff0e25',
        capsule_manifest_sha256=sha(args.capsule/'MANIFEST.json'),
        capsule_manifest_files_verified=verified, capsule_files=required,
        scientific_sources_modified=False, standalone_payload_does_not_include_capsule=True))
    print(json.dumps(dict(status='PREPARED', verified_capsule_files=verified, covariance_condition=float(np.linalg.cond(cov)), new_noise_draws=0)))


if __name__ == '__main__':
    main()

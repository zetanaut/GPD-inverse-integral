"""Evaluate saved model weights; no optimization or native-kernel rebuild."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import HERE, REPO, DATA, DEFAULT_OUT, verify_inputs, write_json

# Reuse the published field and the original post-fit scoring definition.
sys.path.insert(0, str(REPO / 'research' / 'GPD_three_stage_closure'))
from gpd_reh.model import ExperimentalFiLM
from gpd_reh.bank import ExperimentalForward
from closure.metrics import field_metrics
from closure.truths import Truth


def replay(out):
    count = verify_inputs()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = json.loads((HERE / 'config' / 'FIT_CONFIG.json').read_text())['network']
    geometry = json.loads((HERE / 'config' / 'GEOMETRY.json').read_text())
    models = json.loads((HERE / 'checkpoints' / 'index.json').read_text())['models']
    model = ExperimentalFiLM(config['t_mean'], config['t_std']).double().eval()
    layer = ExperimentalForward(REPO / 'data/experimental/prepared', REPO / 'data/experimental/banks')
    stages = pd.read_csv(DATA / 'budget_stages.csv').set_index(['case', 'budget'])
    cffs = pd.read_csv(DATA / 'experimental_cff.csv')
    curves = pd.read_csv(DATA / 'experimental_family.csv')
    nominal = pd.read_csv(DATA / 'experimental_nominal.csv')
    nominal = nominal[nominal.state == 'central13']
    local = pd.read_csv(DATA / 'experimental_local_fit_diagnostics.csv').set_index('state').loc['central13']
    with np.load(DATA / 'sampled_surfaces.npz', allow_pickle=False) as z:
        surfaces = {k: z[k].copy() for k in z.files}
    maxima = dict(field_score=0., curves=0., surfaces=0., residual_RMS=0., cff_predictions=0.)
    fields, residuals, predictions, regenerated = [], [], [], {}

    def compare(kind, actual, expected, tolerance=2e-11):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=2e-10)
        maxima[kind] = max(maxima[kind], float(np.max(np.abs(actual - expected))))

    def evaluate(x, xi, t):
        with torch.no_grad():
            return model(torch.as_tensor(x, dtype=torch.float64),
                         torch.as_tensor(xi, dtype=torch.float64),
                         torch.as_tensor(t, dtype=torch.float64)).numpy()

    def charge(x, xi, t):
        f = evaluate(x, xi, t)
        return (2 / 9) * (f[:, 0] + f[:, 1])

    def forward():
        pieces = []
        with torch.no_grad():
            for start in range(0, len(layer.row_ids), 8):
                sl = slice(start, start + 8)
                f = model(layer.x[sl], layer.xi[sl], layer.t[sl])
                pieces.append(torch.einsum('bocn,bcn->bo', layer.weights[sl], f).numpy())
        return np.concatenate(pieces)

    for record in models:
        with np.load(HERE / record['path'], allow_pickle=False) as z:
            state = {k: torch.from_numpy(z[k].copy()) for k in z.files}
        if sum(v.numel() for v in state.values()) != 3867:
            raise ValueError('Unexpected parameter count')
        model.load_state_dict(state, strict=True)
        case, iteration = record['case'], record['iterations']
        if record['kind'] == 'closure':
            def evaluator(x, xi, t):
                return evaluate(x[None], [xi], [t])[0].T
            measured = field_metrics(evaluator, Truth('Z1'), geometry)
            saved = json.loads((DATA / (case + '_scores.json')).read_text())['field']
            for region in ('all', 'ERBL', 'DGLAP'):
                compare('field_score', measured['charge_relative_L2'][region], saved['charge_relative_L2'][region])
            compare('field_score', measured['internal_channel_relative_L2'], saved['internal_channel_relative_L2'])
            compare('field_score', measured['charge_max_slice_relative_L2'], saved['charge_max_slice_relative_L2'])
            fields.append(dict(case=case, **measured))
            with np.load(DATA / (case + '.npz'), allow_pickle=False) as z:
                h = charge(np.tile(z['x'], (3, 1)), np.full(3, .275), [-.38, -.8, -1.2])
                compare('curves', h, z['x_fit'])
                hxi = charge(np.full((len(z['xi_slice']), 1), .5), z['xi_slice'], np.full(len(z['xi_slice']), -.8))[:, 0]
                ht = charge(np.full((len(z['t_slice']), 1), .5), np.full(len(z['t_slice']), .275), z['t_slice'])[:, 0]
                compare('curves', hxi, z['xi_fit'])
                compare('curves', ht, z['t_fit'])
            print('Replayed closure field:', case, flush=True)
        else:
            source_prediction = -forward()[:, 0]
            target_case = 'ensemble_mean' if case == 'local_cff' else case
            target = cffs[(cffs.case == target_case) & (cffs.budget == 5)].set_index('row_id').loc[list(layer.row_ids)]
            sigma = target.sigma_reported.to_numpy()
            vals = {}
            for label, col in [('own', 'ReH_own_input_source'), ('mean', 'ReH_ensemble_mean_target_source'),
                               ('local', 'ReH_original_local_source')]:
                vals[label + '_RMS'] = float(np.sqrt(np.mean(((source_prediction - target[col].to_numpy()) / sigma) ** 2)))
            if case == 'local_cff':
                vals['own_RMS'] = vals['local_RMS']
                compare('residual_RMS', vals['local_RMS'], local.RMS_error_units)
                rows = nominal
            else:
                saved = stages.loc[(case, iteration)]
                compare('residual_RMS', list(vals.values()), saved[list(vals)].to_numpy(float))
                rows = curves[(curves.case == case) & (curves.budget == iteration)]
                saved_cff = cffs[(cffs.case == case) & (cffs.budget == iteration)]
                if len(saved_cff):
                    saved_cff = saved_cff.set_index('row_id').loc[list(layer.row_ids)]
                    compare('cff_predictions', source_prediction, saved_cff.ReH_GPD_source.to_numpy())
            for anchor, group in rows.groupby('anchor_row_id'):
                group = group.sort_values('x')
                h = charge(group.x.to_numpy()[None], [group.iloc[0].xi], [group.iloc[0].t_GeV2])[0]
                compare('curves', h, group.Hcharge_plus.to_numpy())
            residuals.append(dict(case=case, additional_iterations=iteration, **vals))
            predictions.extend(dict(case=case, additional_iterations=iteration, row_id=rid, ReH_source=float(value))
                               for rid, value in zip(layer.row_ids, source_prediction))
            if iteration in (5, 10, 20) or case == 'local_cff':
                print('Replayed experimental fit:', case, iteration, flush=True)
        label = {'ideal_L0_Z1_s31': 'ideal'}.get(case)
        if iteration == 20:
            label = {'ensemble_mean': 'mean', 'replica_000': 'r000', 'replica_001': 'r001'}[case]
        if label:
            h = charge(np.tile(surfaces['x'], (len(surfaces['t']), 1)),
                       np.full(len(surfaces['t']), float(surfaces['xi'])), surfaces['t'])
            compare('surfaces', h, surfaces[label])
            regenerated[label] = h
    xx = np.tile(surfaces['x'], (len(surfaces['t']), 1))
    f = Truth('Z1').field(xx, np.full(len(surfaces['t']), float(surfaces['xi'])), surfaces['t'])
    regenerated['truth'] = (2 / 9) * (f[:, 0] + f[:, 1])
    compare('surfaces', regenerated['truth'], surfaces['truth'])
    np.savez_compressed(out / 'sampled_surfaces.npz', x=surfaces['x'], t=surfaces['t'], xi=surfaces['xi'], **regenerated)
    write_json(out / 'closure_field_scores.json', fields)
    pd.DataFrame(residuals).to_csv(out / 'experimental_residuals.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'experimental_predictions.csv', index=False)
    result = dict(status='PASS', input_files_verified=count, models_replayed=len(models),
                  closure_fields=len(fields), experimental_predictions=len(predictions),
                  maximum_absolute_differences=maxima,
                  scope='Saved weights evaluated on field grids and bundled experimental NLO rows. Closure CFF operators, P2B trajectories, and native kernels are not rebuilt.',
                  no_training=True, no_new_noise=True)
    write_json(out / 'CHECKPOINT_REPLAY.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT / 'replay')
    print(json.dumps(replay(parser.parse_args().out), indent=2))

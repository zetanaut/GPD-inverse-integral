"""Read-only checks and isolated targets for the selected ReH surface.

No optimizer, kernel construction, new noise, or source-file edits occur here.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_SHA = '8d7274db9dd7e4fcc3ab84c0230701242274e42868f9c3e66e3d61a1a6838696'
SURFACE_SHA = 'eeff3b842bfc7e27f849e564ea0e26f0af847313781c05e340cf3ac57e841161'
CENTRAL_SHA = '86babd3689dd4d81054a75f5893709495ad6b713041e925f166a5b66d0e7d11a'
ZERO_SHA = '28f4781a73ca13a75b2f884029bdf62aeccbb0b8c82a82bd7f3c9ac1b8472336'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_relative(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe relative file reference: {name}')
    path = Path(root).resolve() / relative
    if not path.is_file() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f'Missing or outside-root file: {name}')
    return path


def same(a, b, name, atol=1e-11):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'Invalid shape/nonfinite values: {name}')
    if not np.allclose(a, b, rtol=0, atol=atol):
        raise ValueError(f'Value mismatch: {name}')


def predict_numpy(checkpoint, rows):
    if checkpoint['feature_order'] != ['xi', 't_GeV2', 'log(Q2_GeV2)']:
        raise ValueError('Unsupported feature convention')
    q = rows.Q2_GeV2.to_numpy(dtype=float)
    if np.any(q <= 0):
        raise ValueError('Nonpositive Q2')
    x = np.column_stack([rows.xi, rows.t_GeV2, np.log(q)])
    scales = np.asarray(checkpoint['input_std'], dtype=float)
    if scales.shape != (3,) or np.any(scales <= 0):
        raise ValueError('Invalid input scaling')
    a = (x - np.asarray(checkpoint['input_mean'])) / scales
    state = checkpoint['state_dict']
    for k in (0, 2, 4):
        a = a @ state[f'net.{k}.weight'].detach().cpu().numpy().T + state[f'net.{k}.bias'].detach().cpu().numpy()
        if k != 4:
            a = np.tanh(a)
    result = a[:, 0] * checkpoint['target_std'] + checkpoint['target_mean']
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite fitted surface')
    return result


def residual_stats(residual, sigma):
    residual, sigma = np.asarray(residual, float), np.asarray(sigma, float)
    if residual.shape != sigma.shape or residual.ndim != 1 or len(residual) == 0:
        raise ValueError('Residual and sigma must be same nonempty vector shape')
    if not np.isfinite(residual).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('Invalid residual/error')
    pull = residual / sigma
    return {
        'mean_abs_ReH': float(np.mean(np.abs(residual))),
        'median_abs_ReH': float(np.median(np.abs(residual))),
        'rms_ReH': float(np.sqrt(np.mean(residual ** 2))),
        'mean_abs_pull': float(np.mean(np.abs(pull))),
        'rms_pull': float(np.sqrt(np.mean(pull ** 2))),
        'count_abs_pull_gt_3': int(np.count_nonzero(np.abs(pull) > 3)),
    }


def signed_mean_arrays(original, rows, mean_source):
    """Replace only y in an isolated copy, source sign -> internal exactly once."""
    expected = np.asarray(rows.row_id.tolist(), dtype=str)
    if not np.array_equal(np.asarray(original['row_id'], dtype=str), expected):
        raise ValueError('Original NPZ row order differs')
    same(original['y'], -rows.ReH_source.to_numpy(), 'baseline internal sign')
    same(original['sigma'], rows.sigma_reported.to_numpy(), 'quoted errors')
    mean_source = np.asarray(mean_source, dtype=np.float64)
    if mean_source.shape != np.asarray(original['y']).shape or not np.isfinite(mean_source).all():
        raise ValueError('Invalid central-surface target vector')
    arrays = {key: np.array(value, copy=True) for key, value in original.items()}
    arrays['y'] = -mean_source.copy()
    return arrays


def replace_problem_targets(problem, arrays):
    """For the existing BranchProblem: change data values, not loss or domain.

    Before calling, construct the original problem and source-check the input NPZ.
    The caller must separately preflight residual/constraint identities at the parent.
    """
    sigma = np.asarray(arrays['sigma'])
    same(problem.sigma.detach().cpu().numpy(), sigma, 'problem sigma', atol=0)
    ids = tuple(np.asarray(arrays['row_id'], dtype=str).tolist())
    if tuple(problem.forward.row_ids) != ids:
        raise ValueError('Forward row identity differs')
    y = np.asarray(arrays['y'], dtype=np.float64)
    if y.shape != tuple(problem.y.shape) or not np.isfinite(y).all():
        raise ValueError('Invalid replacement y')
    problem.y = torch.as_tensor(y.copy(), dtype=problem.y.dtype, device=problem.y.device)
    return problem


def audit(surface, project):
    surface, project = Path(surface), Path(project)
    manifest = json.loads((surface / 'MANIFEST.json').read_text())
    for name, sha in manifest['artifact_sha256'].items():
        if digest(checked_relative(surface, name)) != sha:
            raise ValueError('Surface artifact mismatch: ' + name)
    for name, sha in manifest['source_references_sha256'].items():
        if digest(checked_relative(project, name)) != sha:
            raise ValueError('Source reference mismatch: ' + name)
    source = pd.read_csv(project / 'GPD_ReH_experimental_run/inputs/final_summary_by_DNN.csv', float_precision='round_trip')
    rows = pd.read_csv(project / 'GPD_ReH_experimental_run/prepared/observations.csv', float_precision='round_trip')
    table = pd.read_csv(surface / 'comparison_160.csv', float_precision='round_trip')
    if len(source) != 189 or len(rows) != 160 or len(table) != 160:
        raise ValueError('Unexpected source/selection counts')
    cuts = (source.QQ > 1.5) & (source.t.abs() / source.QQ <= .25)
    if not np.array_equal(rows.source_line.to_numpy(), np.flatnonzero(cuts) + 2):
        raise ValueError('Selected rows changed')
    selected = source.iloc[rows.source_line.to_numpy() - 2]
    if rows.method.tolist() != selected.DNN.tolist():
        raise ValueError('Method selection changed')
    if table.row_id.tolist() != rows.row_id.tolist():
        raise ValueError('Comparison row order changed')
    for col in ['xi', 't_GeV2', 'Q2_GeV2', 'ReH_source', 'sigma_reported']:
        same(table[col], rows[col], col)
    same(rows.ReH_source, selected['Pred ReH'], 'selected central ReH')
    same(rows.sigma_reported, selected['Pred ReH Std'], 'selected sigma')
    same(rows.xi, rows.xB / (2 - rows.xB), 'xi')
    same(rows.ReH_internal, -rows.ReH_source, 'internal sign')
    cp = torch.load(surface / 'ReH_fitted_central_surface.pt', map_location='cpu', weights_only=True)
    if cp['source_sha256'] != SOURCE_SHA or digest(surface / 'ReH_fitted_central_surface.pt') != SURFACE_SHA:
        raise ValueError('Selected surface identity mismatch')
    if cp['protocol_sha256'] != digest(surface / 'FIT_PROTOCOL.json'):
        raise ValueError('Checkpoint/protocol mismatch')
    if cp['prepared_sha256'] != digest(project / 'GPD_ReH_experimental_run/prepared/observations.csv'):
        raise ValueError('Checkpoint/input mismatch')
    y = rows.ReH_source.to_numpy(); sigma = rows.sigma_reported.to_numpy()
    prediction = predict_numpy(cp, rows)
    same(prediction, table.ReH_smooth_source, 'numpy surface replay', atol=1e-12)
    raw = np.column_stack([rows.xi, rows.t_GeV2, np.log(rows.Q2_GeV2)])
    same(raw.mean(0), cp['input_mean'], 'all-row feature centering', atol=1e-14)
    same(raw.std(0), cp['input_std'], 'all-row feature scaling', atol=1e-14)
    same(y.mean(), cp['target_mean'], 'all-row target centering', atol=1e-14)
    same(y.std(), cp['target_std'], 'all-row target scaling', atol=1e-14)
    recorded = json.loads((surface / 'METRICS.json').read_text())
    groups = {
        'full_surface_vs_selected_local': prediction - y,
        'saved_central13_vs_selected_local': table.ReH_central13_source.to_numpy() - y,
        'saved_central13_vs_full_surface': table.ReH_central13_source.to_numpy() - prediction,
    }
    stats = {key: residual_stats(value, sigma) for key, value in groups.items()}
    metric_max = 0.
    for key in stats:
        for name, value in stats[key].items():
            same(value, recorded[key][name], key + '/' + name, atol=2e-12)
            metric_max = max(metric_max, abs(value - recorded[key][name]))
    permutation = np.random.default_rng(20261011).permutation(160)
    if recorded['split']['validation_rows'] != permutation[:32].tolist() or recorded['split']['training_rows'] != permutation[32:].tolist():
        raise ValueError('Declared split differs')
    history = pd.read_csv(surface / 'fit_history.csv', float_precision='round_trip')
    best, epoch, wait, ended = math.inf, 0, 0, None
    for item in history.itertuples(index=False):
        if item.validation_weighted_mse < best - 1e-5:
            best, epoch, wait = item.validation_weighted_mse, int(item.epoch), 0
        else:
            wait += 1
        if wait >= 200:
            ended = int(item.epoch)
            break
    if epoch != 299 or cp['best_validation_epoch'] != epoch or ended != 499 or len(history) != 499:
        raise ValueError('History selection/stopping differs')
    same(np.sqrt(best), recorded['split']['validation_at_selected_epoch']['rms_pull'], 'selected validation RMS', atol=2e-12)
    pred = pd.read_csv(project / 'GPD_experimental_closeout/results/ReH_predictions.csv', float_precision='round_trip')
    existing = {}
    for state in ['parent7', 'control7', 'central13', 'zero', 'plus', 'minus']:
        values = pred[pred.state == state].set_index('row_id').loc[rows.row_id, 'ReH_fitted_source'].to_numpy()
        if state == 'central13':
            same(values, table.ReH_central13_source, 'stored central13 linkage', atol=1e-14)
        existing[state] = {
            'vs_original_rows': residual_stats(values - y, sigma),
            'vs_fitted_surface': residual_stats(values - prediction, sigma),
        }
    result = {
        'status': 'SOURCE_AND_SAVED_SURFACE_REPLAYED',
        'artifact_hashes_verified': len(manifest['artifact_sha256']),
        'source_reference_hashes_verified': len(manifest['source_references_sha256']),
        'rows': len(rows), 'source_sha256': SOURCE_SHA, 'surface_checkpoint_sha256': SURFACE_SHA,
        'source_method_counts': source.DNN.value_counts().to_dict(),
        'selected_method_counts': rows.method.value_counts().to_dict(),
        'surface_parameters': sum(v.numel() for v in cp['state_dict'].values()),
        'maximum_numpy_prediction_difference': float(np.max(np.abs(prediction - table.ReH_smooth_source.to_numpy()))),
        'maximum_recomputed_metric_difference': metric_max,
        'selected_epoch_from_saved_history': epoch, 'stop_epoch_from_saved_history': ended,
        'selected_validation_rms': float(np.sqrt(best)),
        'validation_scope': 'Used for epoch selection; not an untouched final test. Split checkpoint absent; validation predictions not independently replayed.',
        'comparison': stats, 'existing_checkpoints_new_reference_comparison': existing,
        'rerun': ['saved surface inference', 'input identity/preparation checks', 'summary arithmetic', 'saved history selection rule'],
        'not_rerun': ['surface training', 'GPD training', 'NLO forward integrations', 'statistical calibration', 'replica construction'],
    }
    return result

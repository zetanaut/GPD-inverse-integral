#!/usr/bin/env python3
"""Prepare a separate central-surface input. Original data and caches stay untouched."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from surface_tools import audit, digest, signed_mean_arrays, CENTRAL_SHA, ZERO_SHA, SURFACE_SHA


def prepare(project, surface, out):
    project, surface, out = Path(project).resolve(), Path(surface).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Output exists; verify/reuse its lock, do not overwrite: {out}')
    # Do not import/run the submitted training script: it writes model/audit files.
    result = audit(surface, project)
    root = project / 'GPD_ReH_experimental_run'
    rows = pd.read_csv(root / 'prepared/observations.csv', float_precision='round_trip')
    comparisons = pd.read_csv(surface / 'comparison_160.csv', float_precision='round_trip')
    with np.load(root / 'prepared/observations.npz', allow_pickle=False) as z:
        original = {k: z[k].copy() for k in z.files}
    changed = signed_mean_arrays(original, rows, comparisons.ReH_smooth_source.to_numpy())
    if not all(np.array_equal(original[k], changed[k]) for k in original if k != 'y'):
        raise ValueError('Non-target NPZ data changed')
    parent = project / 'GPD_experimental_closeout/models/central13.pt'
    control = project / 'GPD_experimental_closeout/models/zero.pt'
    if digest(parent) != CENTRAL_SHA or digest(control) != ZERO_SHA:
        raise ValueError('Parent/control checkpoints changed')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'surface_observations.npz', **changed)
    exported = rows.copy()
    exported['ReH_local_source_unchanged'] = exported['ReH_source']
    exported['ReH_fit_target_source'] = comparisons.ReH_smooth_source.to_numpy()
    exported['ReH_fit_target_internal'] = changed['y']
    exported.to_csv(out / 'surface_input_160.csv', index=False, float_format='%.17g')
    lock = {
        'id': 'selected_ReH_surface_input_v1',
        'original_source': str(root / 'inputs/final_summary_by_DNN.csv'),
        'original_source_sha256': result['source_sha256'],
        'original_observations_npz': str(root / 'prepared/observations.npz'),
        'original_observations_npz_sha256': digest(root / 'prepared/observations.npz'),
        'surface_checkpoint': str(surface / 'ReH_fitted_central_surface.pt'),
        'surface_checkpoint_sha256': SURFACE_SHA,
        'surface_protocol_sha256': digest(surface / 'FIT_PROTOCOL.json'),
        'surface_comparison_sha256': digest(surface / 'comparison_160.csv'),
        'parent': str(parent), 'parent_sha256': CENTRAL_SHA,
        'matched_zero_control': str(control), 'matched_zero_control_sha256': ZERO_SHA,
        'changed_npz_fields': ['y'],
        'unchanged_npz_fields': [key for key in original if key != 'y'],
        'target_rule': 'y_internal = -ReH_smooth_source; originals preserved',
        'weights': 'Same quoted combined marginal sigma, 1/160 mean squared residual; numerical matching weights, not new surface-error likelihood',
        'status': 'PREPARED_ONLY_NO_GPD_FIT',
        'outputs': {name: digest(out / name) for name in ['surface_observations.npz', 'surface_input_160.csv']},
    }
    (out / 'INPUT_LOCK.json').write_text(json.dumps(lock, indent=2) + '\n')
    (out / 'SOURCE_REPLAY.json').write_text(json.dumps(result, indent=2) + '\n')
    return lock


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--surface', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    surface = a.surface or a.project / 'completion_runs/ReH_selected_surface'
    out = a.out or a.project / 'completion_runs/ReH_surface_mean_pilot/prepared'
    lock = prepare(a.project, surface, out)
    print(json.dumps({'output': str(out), 'status': lock['status'], 'only_changed_field': 'y'}, indent=2))

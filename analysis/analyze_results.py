"""Recompute tables and input-response diagnostics from bundled saved arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE, DATA, DEFAULT_OUT, verify_inputs, write_json
from metrics import aligned, rms, relative_l2, response, mean_distance_decomposition

CASES = ('ideal_L0_Z1_s31', 'exp_Re_exact_Z1_s31', 'exp_Re_exact_Z1_s43',
         'exp_Re_noisy_Z1_s31', 'exp_Re_noisy_Z1_s43')


def analyze(out):
    count = verify_inputs()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    closure = []
    for case in CASES:
        score = json.loads((DATA / (case + '_scores.json')).read_text())
        row = dict(case=case, field_error_percent=100 * score['field']['charge_relative_L2']['all'],
                   ERBL_error_percent=100 * score['field']['charge_relative_L2']['ERBL'],
                   DGLAP_error_percent=100 * score['field']['charge_relative_L2']['DGLAP'])
        with np.load(DATA / (case + '.npz'), allow_pickle=False) as z:
            for part, role in [('fit', score['task']['geometry']), ('holdout', 'surface_check')]:
                error = relative_l2(z['cff_' + part + '_pred'], z['cff_' + part + '_truth'])
                np.testing.assert_allclose(error, score['CFF'][role]['truth_error']['relative_L2'], rtol=1e-10, atol=1e-12)
                for component, value in zip(('Re', 'Im'), error):
                    row[part + '_' + component + '_error_percent'] = 100 * value
        closure.append(row)
    pd.DataFrame(closure).to_csv(out / 'closure_errors.csv', index=False)

    cff = pd.read_csv(DATA / 'experimental_cff.csv')
    stages = pd.read_csv(DATA / 'budget_stages.csv')
    residuals = []
    for (case, iteration), group in cff.groupby(['case', 'budget'], sort=True):
        if len(group) != 160 or group.row_id.duplicated().any():
            raise ValueError('Expected 160 distinct CFF observations')
        sigma = group.sigma_reported.to_numpy()
        if np.any(sigma <= 0) or not np.isfinite(sigma).all():
            raise ValueError('Invalid original quoted errors')
        prediction = group.ReH_GPD_source.to_numpy()
        row = dict(case=case, additional_iterations=int(iteration), n=len(group))
        for label, column in [('own', 'ReH_own_input_source'), ('mean', 'ReH_ensemble_mean_target_source'),
                              ('local', 'ReH_original_local_source')]:
            row[label + '_RMS'] = rms((prediction - group[column].to_numpy()) / sigma)
        stored = stages[(stages.case == case) & (stages.budget == iteration)].iloc[0]
        np.testing.assert_allclose([row[k] for k in ('own_RMS', 'mean_RMS', 'local_RMS')],
                                   stored[['own_RMS', 'mean_RMS', 'local_RMS']].to_numpy(float), atol=1e-12, rtol=1e-12)
        residuals.append(row)
    pd.DataFrame(residuals).to_csv(out / 'experimental_residuals.csv', index=False)
    local = pd.read_csv(DATA / 'experimental_local_fit_diagnostics.csv').set_index('state').loc['central13']
    comparison = [dict(case='local_cff', additional_iterations='', own_RMS=float(local.own_input_RMS),
                       local_RMS=float(local.RMS_error_units))]
    comparison += [{k: row[k] for k in ('case', 'additional_iterations', 'own_RMS', 'local_RMS')}
                   for row in residuals if row['additional_iterations'] == 20]
    pd.DataFrame(comparison).to_csv(out / 'experimental_comparison.csv', index=False)

    curves = pd.read_csv(DATA / 'experimental_family.csv')
    field_rows, cff_rows, decompositions = [], [], []
    saved_field = pd.read_csv(DATA / 'budget_response.csv').set_index(['case', 'budget'])
    saved_cff = pd.read_csv(DATA / 'cff_response.csv').set_index(['case', 'budget'])
    with np.load(DATA / 'coherent_cff_replicas.npz', allow_pickle=False) as z:
        replicas = z['predictions_source'].copy()
        replica_ids = z['row_id'].astype(str)
    for iteration in (5, 10, 20):
        mean = cff[(cff.case == 'ensemble_mean') & (cff.budget == iteration)]
        for case in ('replica_000', 'replica_001'):
            left = curves[(curves.case == case) & (curves.budget == iteration)]
            right = curves[(curves.case == 'ensemble_mean') & (curves.budget == iteration)]
            left, right = aligned(left, right, ['anchor_row_id', 'x'], ['xi', 't_GeV2'])
            value = rms(left.Hcharge_plus.to_numpy() - right.Hcharge_plus.to_numpy())
            np.testing.assert_allclose(value, saved_field.loc[(case, iteration), 'late_rms'], atol=1e-12)
            field_rows.append(dict(case=case, additional_iterations=iteration, field_response_RMS=value, n=len(left)))
            member = cff[(cff.case == case) & (cff.budget == iteration)]
            member, center = aligned(member, mean, ['row_id'], ['xi', 't_GeV2', 'Q2_GeV2', 'sigma_reported'])
            inp = (member.ReH_own_input_source.to_numpy() - center.ReH_own_input_source.to_numpy()) / center.sigma_reported.to_numpy()
            pred = (member.ReH_GPD_source.to_numpy() - center.ReH_GPD_source.to_numpy()) / center.sigma_reported.to_numpy()
            values = response(inp, pred)
            for key, value in values.items():
                np.testing.assert_allclose(value, saved_cff.loc[(case, iteration), key], atol=1e-12)
            cff_rows.append(dict(case=case, additional_iterations=iteration, **values))
        indexed = mean.set_index('row_id')
        if set(replica_ids) != set(indexed.index) or len(set(replica_ids)) != len(replica_ids):
            raise ValueError('CFF replica coordinate identifiers differ')
        indexed = indexed.loc[replica_ids]
        np.testing.assert_allclose(replicas.mean(axis=0), indexed.ReH_own_input_source.to_numpy(), atol=2e-14, rtol=0)
        parts = mean_distance_decomposition(indexed.ReH_GPD_source.to_numpy(), replicas, indexed.sigma_reported.to_numpy())
        if parts['identity_error'] > 1e-12:
            raise ValueError('Mean-distance identity failed')
        decompositions.append(dict(additional_iterations=iteration, replicas=len(replicas), **parts))
    pd.DataFrame(field_rows).to_csv(out / 'field_response.csv', index=False)
    pd.DataFrame(cff_rows).to_csv(out / 'cff_response.csv', index=False)
    pd.DataFrame(decompositions).to_csv(out / 'mean_distance_decomposition.csv', index=False)

    boundary = []
    for noise, name in [('exact', 'p2b_exact'), ('half', 'p2b_noisy')]:
        record = json.loads((DATA / (name + '.json')).read_text())
        for fit, key in [('CFF', 'baseline'), ('CFF+PDF', 'selected')]:
            row = record[key]
            boundary.append(dict(noise=noise, fit=fit,
                observed_field_error_percent=100 * row['observed']['all'],
                A_error_percent=100 * abs(row['target_signed_relative_error'][0]),
                B_error_percent=100 * abs(row['target_signed_relative_error'][1]),
                wider_field_error_percent=100 * row['wider_chart']['charge_relative_L2']['all']))
    pd.DataFrame(boundary).to_csv(out / 'forward_information_comparison.csv', index=False)
    result = dict(status='PASS', input_files_verified=count, closure_cases=len(closure),
                  experimental_residual_groups=len(residuals), field_response_groups=len(field_rows),
                  cff_response_groups=len(cff_rows), decomposition_checks=len(decompositions),
                  forward_information_rows=len(boundary),
                  scope='CFF and response arithmetic recomputed from saved arrays. Wider-chart field errors and P2B diagnostics read from bound score records; checkpoint replay is a separate command.')
    write_json(out / 'ANALYSIS_CHECK.json', result)
    lines = ['# Recomputed analysis tables', '', result['scope'], '']
    for title, rows in [('Closure errors (%)', closure), ('Experimental CFF residuals', comparison),
                        ('Forward information errors (%)', boundary)]:
        frame = pd.DataFrame(rows)
        lines += ['## ' + title, '', '| ' + ' | '.join(frame.columns) + ' |',
                  '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
        for row in frame.itertuples(index=False, name=None):
            lines.append('| ' + ' | '.join(f'{x:.5f}' if isinstance(x, float) else str(x) for x in row) + ' |')
        lines.append('')
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT / 'tables')
    print(json.dumps(analyze(parser.parse_args().out), indent=2))

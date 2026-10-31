"""Checks for coordinate alignment and statistical identities in public analysis."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('analysis_metrics', Path(__file__).resolve().parents[1] / 'analysis/metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_alignment_reorders_but_rejects_missing_or_changed_coordinates():
    frame = pd.DataFrame({'id': ['a', 'b'], 'x': [.2, .8], 'value': [1., 2.]})
    left, right = metrics.aligned(frame, frame.iloc[::-1], ['id'], ['x'])
    np.testing.assert_array_equal(left.value, right.value)
    with pytest.raises(ValueError, match='Unmatched'):
        metrics.aligned(frame, frame.iloc[:1], ['id'], ['x'])
    changed = frame.copy()
    changed.loc[1, 'x'] = .9
    with pytest.raises(ValueError, match='Coordinate'):
        metrics.aligned(frame, changed, ['id'], ['x'])
    with pytest.raises(ValueError, match='Duplicate'):
        metrics.aligned(frame, pd.concat([frame, frame]), ['id'])


def test_nonuniform_error_mean_distance_identity():
    replicas = np.array([[1., -3.], [2., 4.], [0., 5.]])
    result = metrics.mean_distance_decomposition(np.array([.8, 1.5]), replicas, np.array([.5, 3.]))
    assert result['identity_error'] < 1e-13
    assert result['mean_squared_distance_to_replicas'] == pytest.approx(result['mean_squared_distance'] + result['replica_dispersion'])
    with pytest.raises(ValueError, match='Positive'):
        metrics.mean_distance_decomposition(np.ones(2), replicas, np.array([1., 0.]))


def test_norm_response_does_not_imply_directional_recovery():
    result = metrics.response([1., 0.], [0., 1.])
    assert result['norm_ratio'] == 1
    assert result['gain'] == 0
    assert result['cosine'] == 0
    assert metrics.response([0., 0.], [1., 0.])['norm_ratio'] is None


def test_relative_error_is_componentwise_and_rejects_undefined_norm():
    np.testing.assert_allclose(metrics.relative_l2([[2., 3.], [4., 6.]], [[1., 1.], [2., 2.]]), [1., 2.])
    with pytest.raises(ValueError, match='nonzero'):
        metrics.relative_l2(np.ones((2, 2)), np.zeros((2, 2)))

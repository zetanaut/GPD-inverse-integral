"""Metric definitions; all differences require matched coordinates."""
import numpy as np


def rms(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Expected finite nonempty values')
    return float(np.sqrt(np.mean(values ** 2)))


def relative_l2(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError('Prediction and target shapes differ')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite field or CFF values')
    denominator = np.linalg.norm(target, axis=0)
    if np.any(denominator == 0):
        raise ValueError('Relative error requires a nonzero target norm')
    return np.linalg.norm(prediction - target, axis=0) / denominator


def aligned(left, right, keys, coordinates=()):
    """Sort, then require an exact one-to-one correspondence of observations."""
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError('Duplicate observation keys')
    left, right = (x.sort_values(keys).reset_index(drop=True) for x in (left, right))
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise ValueError('Unmatched observation keys')
    for col in coordinates:
        if not np.array_equal(left[col].to_numpy(), right[col].to_numpy()):
            raise ValueError('Coordinate or error mismatch: ' + col)
    return left, right


def response(input_delta, output_delta):
    a, b = np.asarray(input_delta, float), np.asarray(output_delta, float)
    if a.shape != b.shape:
        raise ValueError('Response vectors must have the same shape')
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    return dict(input_rms=rms(a), output_rms=rms(b),
                norm_ratio=None if an == 0 else float(bn / an),
                difference_rms=rms(b - a),
                gain=None if an == 0 else float(a @ b / an ** 2),
                cosine=None if an * bn == 0 else float(a @ b / (an * bn)))


def mean_distance_decomposition(prediction, replicas, sigma):
    prediction, replicas, sigma = map(np.asarray, (prediction, replicas, sigma))
    if replicas.ndim != 2 or prediction.shape != sigma.shape or replicas.shape[1:] != sigma.shape:
        raise ValueError('Expected prediction[n], replicas[m,n], sigma[n]')
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('Positive finite uncertainties required')
    center = replicas.mean(axis=0)
    mismatch = rms((prediction - center) / sigma) ** 2
    dispersion = rms((replicas - center) / sigma) ** 2
    total = rms((prediction - replicas) / sigma) ** 2
    return dict(mean_squared_distance=mismatch, replica_dispersion=dispersion,
                mean_squared_distance_to_replicas=total,
                identity_error=abs(total - mismatch - dispersion))

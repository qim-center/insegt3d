import numpy as np


def coarsest_small_level(shapes, max_voxels=128 ** 3):
    """Finest level with at most max_voxels voxels, else the coarsest level."""
    for i, shape in enumerate(shapes):
        if np.prod(shape) <= max_voxels:
            return i
    return len(shapes) - 1


def robust_percentile_range(array):
    """(0.5th, 99.5th) percentile range of `array`, estimated from every 4th voxel per axis."""
    array = np.asarray(array)

    sample_index = tuple(slice(None, None, 4) for _ in range(array.ndim))
    sample = np.nan_to_num(
        array[sample_index].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    lo, hi = (float(v) for v in np.percentile(sample, [0.5, 99.5]))
    if hi <= lo:
        hi = lo + 1e-6

    return lo, hi


def robust_normalize(array):
    array = np.asarray(array)
    lo, hi = robust_percentile_range(array)

    normalized = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.subtract(normalized, lo, out=normalized)
    np.multiply(normalized, 1.0 / (hi - lo), out=normalized)
    np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized

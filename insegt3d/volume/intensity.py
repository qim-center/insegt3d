import numpy as np


def coarsest_small_level(shapes, max_voxels=128 ** 3):
    """
    Index of the largest level where total voxel count is <= max_voxels.
    """
    for i, shape in enumerate(shapes):
        if np.prod(shape) <= max_voxels:
            return i
    return len(shapes) - 1


def native_intensity_range(volume):
    array = np.asarray(volume)
    return float(array.min()), float(array.max())


# Default percentile window for robust_percentile_range() / robust_normalize().
ROBUST_LOW_PERCENTILE = 0.5
ROBUST_HIGH_PERCENTILE = 99.5


def robust_percentile_range(array, low_percentile=ROBUST_LOW_PERCENTILE, high_percentile=ROBUST_HIGH_PERCENTILE, param_downscale=4):
    """
    Robust (low_percentile, high_percentile) value range for `array`, ignoring outliers.
    """
    array = np.asarray(array)

    sample_index = tuple(slice(None, None, param_downscale) for _ in range(array.ndim))
    sample = np.nan_to_num(
        array[sample_index].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    lo, hi = (float(v) for v in np.percentile(sample, [low_percentile, high_percentile]))
    if hi <= lo:
        hi = lo + 1e-6

    return lo, hi


def robust_normalize(array, low_percentile=ROBUST_LOW_PERCENTILE, high_percentile=ROBUST_HIGH_PERCENTILE, param_downscale=4):
    """
    Normalizes `array` to [0, 1] for model input using percentile clipping.
    """
    array = np.asarray(array)
    lo, hi = robust_percentile_range(array, low_percentile, high_percentile, param_downscale)

    normalized = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.subtract(normalized, lo, out=normalized)
    np.multiply(normalized, 1.0 / (hi - lo), out=normalized)
    np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized

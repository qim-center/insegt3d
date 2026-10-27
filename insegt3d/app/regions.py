import json
import os
import numpy as np
from pathlib import Path


def region_box(lo, hi, shape):
    """Box from corners in any order, in whole voxels inside a volume of `shape` and at least one voxel wide."""
    lo, hi = np.round(np.minimum(lo, hi)), np.round(np.maximum(lo, hi))
    lo = np.clip(lo, 0, np.subtract(shape, 1))
    hi = np.clip(np.maximum(hi, lo + 1), 1, shape)
    return [int(v) for v in (*lo, *hi)]

class Regions:
    """
    Boxes to predict, per volume, saved in regions.json. Each is a dict with the 'box' [z0, y0, x0, z1, y1, x1] in
    full-resolution voxels and whether to 'predict' it.
    """

    def __init__(self, project_path):
        self.path = Path(project_path) / 'regions.json'
        self._regions = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def of(self, volume_path):
        """The volume's regions, which callers change in place before saving."""
        return self._regions.setdefault(str(volume_path), [])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({volume: regions for volume, regions in self._regions.items() if regions}, indent=2))
        os.replace(temporary, self.path)

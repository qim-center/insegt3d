import itertools
import numpy as np

from insegt3d.app.regions import region_box
from insegt3d.tools.base_tool import BaseTool

# Viewport pixels within which an edge of a region is grabbed
GRAB_PX = 8

REGION_COLOR = '#fc7323'
SELECTED_COLOR = '#e63f47'

# Box corners in itertools.product order, so the 12 edges join corners whose indices differ in one bit
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count('1') == 1]

def section(lo, hi, camera):
    """Corners of the polygon where the slicing plane cuts the box from world `lo` to `hi`, in order around it, or None."""
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    depth = (corners - camera.origin) @ camera.u
    points = np.array([corners[a] + (corners[b] - corners[a]) * depth[a] / (depth[a] - depth[b]) for a, b in EDGES if (depth[a] < 0) != (depth[b] < 0)])
    if len(points) < 3:
        return None
    rel = points - points.mean(axis=0)
    return points[np.argsort(np.arctan2(rel @ camera.v, rel @ camera.w))]

class RegionTool(BaseTool):
    """
    Outlines the prediction regions where the slice cuts them. In 'regions' mode, dragging an edge or corner of one moves
    those faces and dragging inside it moves the whole region, in axis-aligned views.
    """

    def __init__(self, state, services, renderer, scheduler, callbacks):
        super().__init__(state, services, renderer, scheduler, callbacks)

        self._drag = None

        self.scheduler.register_async("region_overlay", self._push_overlay, max_hz=60)

    async def on_pointer(self, e):
        if self.state.annot.mode != 'regions' or self.state.train.predicting:
            return

        if e.up and self._drag is not None:
            self._drag = None
            self.services.regions.save()
            self.callbacks.refresh_regions()

        elif e.down and e.primary and not e.ctrl:
            self._drag = self._grab(e.x, e.y)
            self.state.regions.selected = None if self._drag is None else self._drag[0]
            self.callbacks.refresh_regions()

        elif e.move and self._drag is not None:
            self._move(e.x, e.y)
            self.scheduler.request("region_overlay")

    async def _push_overlay(self, camera=None):
        self.renderer.update_svg(self._outlines(camera or self.state.camera), part="regions")

    def _regions(self):
        slicer = self.services.slicer
        return [] if slicer.images is None else self.services.regions.of(slicer.zarr_path)

    def _world_box(self, region):
        return np.array(region['box'], float).reshape(2, 3) * self.services.slicer.voxel_sizes[0]

    def _outlines(self, camera):
        if not self.state.regions.predict:
            return ''

        view_h, view_w = self.state.ui.viewport_shape
        shapes = []

        for i, region in enumerate(self._regions()):
            corners = section(*self._world_box(region), camera)
            if corners is None:
                continue

            corners = self._to_view(corners, camera)
            points = ' '.join(f'{x:.1f},{y:.1f}' for x, y in corners)
            color = SELECTED_COLOR if i == self.state.regions.selected else REGION_COLOR
            # Regions left out of prediction are dashed
            dash = '' if region['predict'] else ' stroke-dasharray="6 4"'
            x, y = corners[np.lexsort(corners.T)[0]]
            shapes.append(
                f'<polygon points="{points}" fill="none" stroke="black" stroke-width="3" stroke-opacity="0.4"{dash} />'
                f'<polygon points="{points}" fill="none" stroke="{color}" stroke-width="1.5"{dash} />'
                f'<text x="{x + 4:.1f}" y="{y + 14:.1f}" fill="{color}" font-size="12">{i + 1}</text>'
            )

        # The image's own size sets the layer's coordinates, and previews are smaller, so the outlines use the viewport's
        return f'<svg width="100%" height="100%" viewBox="0 0 {view_w} {view_h}" preserveAspectRatio="none">{"".join(shapes)}</svg>'

    def _to_view(self, points, camera):
        (slice_h, slice_w), (view_h, view_w) = self.state.nav.slice_shape, self.state.ui.viewport_shape
        rel = (points - camera.origin) / camera.zoom
        return np.stack([(rel @ camera.w + slice_w / 2) * view_w / slice_w, (rel @ camera.v + slice_h / 2) * view_h / slice_h], axis=-1)

    def _to_world(self, x, y, camera):
        (slice_h, slice_w), (view_h, view_w) = self.state.nav.slice_shape, self.state.ui.viewport_shape
        return camera.origin + ((y * slice_h / view_h - slice_h / 2) * camera.v + (x * slice_w / view_w - slice_w / 2) * camera.w) * camera.zoom

    def _grab(self, x, y):
        """(index, faces as (axis, side), pointer in world, box) for the region under the pointer, or None."""
        camera = self.state.camera
        if not self.state.regions.predict or np.max(np.abs(camera.u)) < 0.999:
            return None

        normal = int(np.argmax(np.abs(camera.u)))
        in_plane = [axis for axis in range(3) if axis != normal]
        point = self._to_world(x, y, camera)
        reach = GRAB_PX * camera.zoom * self.state.nav.slice_shape[0] / self.state.ui.viewport_shape[0]

        regions = self._regions()
        # Where regions overlap, the selected one is grabbed first
        for i in sorted(range(len(regions)), key=lambda i: i != self.state.regions.selected):
            lo, hi = self._world_box(regions[i])
            if not lo[normal] <= point[normal] <= hi[normal]:
                continue

            faces = []
            for axis, other in (in_plane, in_plane[::-1]):
                distance = np.abs(point[axis] - np.array([lo[axis], hi[axis]]))
                if distance.min() < reach and lo[other] - reach <= point[other] <= hi[other] + reach:
                    faces.append((axis, int(distance.argmin())))

            if faces or all(lo[axis] <= point[axis] <= hi[axis] for axis in in_plane):
                return i, faces, point, np.array(regions[i]['box'], float)

        return None

    def _move(self, x, y):
        index, faces, start, box = self._drag
        slicer = self.services.slicer
        shape = slicer.shapes[0]
        lo, hi = box.reshape(2, 3).copy()

        point = self._to_world(x, y, self.state.camera) / slicer.voxel_sizes[0]
        if faces:
            for axis, side in faces:
                (lo, hi)[side][axis] = point[axis]
        else:
            # The whole region moves, staying inside the volume
            shift = np.clip(point - start / slicer.voxel_sizes[0], -lo, shape - hi)
            lo, hi = lo + shift, hi + shift

        self._regions()[index]['box'] = region_box(lo, hi, shape)

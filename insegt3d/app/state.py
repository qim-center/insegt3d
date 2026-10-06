import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Callable, Optional

from insegt3d.volume.slicer import VolumeSlicer
from insegt3d.volume.camera import Camera
from insegt3d.ml.annotation_tracker import AnnotationTracker
from insegt3d.ml import metrics
from insegt3d.app.history import EditHistory


@dataclass
class OverlayState:
    visible: bool = False
    alpha: float = 0.25

@dataclass
class UIState:
    viewport_shape: tuple[int, int] = (768, 768)
    # UI names: mask is the "Annotation overlay", saved_prediction the "Prediction overlay" and prediction
    # the "Live prediction overlay", while annotation styles the unsaved stroke and flood fill previews
    mask: OverlayState = field(default_factory=lambda: OverlayState(visible=True))
    annotation: OverlayState = field(default_factory=OverlayState)
    prediction: OverlayState = field(default_factory=OverlayState)
    saved_prediction: OverlayState = field(default_factory=OverlayState)
    jpeg_quality: int = 80
    intensity_low: float = 0.0
    intensity_high: float = 255.0

    def max_brush_size(self, max_scale=0.25):
        return min(self.viewport_shape) * max_scale

@dataclass
class NavigationState:
    slice_shape: tuple[int, int] = (768, 768)
    max_slice_megapixels: float = 2.0  # Caps the slice resolution sampled for large viewports

    @property
    def extent(self):
        h, w = self.slice_shape
        return (0, 0, -(h // 2), h // 2, -(w // 2), w // 2)

@dataclass
class AnnotationState:
    annotating: bool = False
    mode: str = 'draw'  # 'draw' | 'erase' | 'mask_fill' | 'flood' | 'keep'
    brush_size: int = 3
    colors: List[str] = field(default_factory=lambda: [
        'rgba(230, 25, 75, 1)', 'rgba(60, 180, 75, 1)',
        'rgba(255, 225, 25, 1)', 'rgba(0, 130, 200, 1)',
        'rgba(245, 130, 48, 1)', 'rgba(145, 30, 180, 1)',
        'rgba(70, 240, 240, 1)', 'rgba(240, 50, 230, 1)',
        'rgba(210, 245, 60, 1)', 'rgba(170, 255, 195, 1)',
    ])
    color_idx: int = 0

    def __post_init__(self):
        self.set_colors(self.colors)

    def set_colors(self, colors):
        rgb = [tuple(map(int, rgba[5:-1].split(",")[:3])) for rgba in colors]
        # Mask label 0 is unannotated, and class i is stored as label i + 1
        self.colors, self.palette_rgb = colors, np.array([(0, 0, 0)] + rgb, dtype=np.uint8)

    def remove_color(self, i):
        # Later classes keep their colours, and the removed colour moves to the end for reuse
        self.set_colors(self.colors[:i] + self.colors[i + 1:] + self.colors[i:i + 1])
        self.color_idx -= self.color_idx > i

    def next_color(self, num_classes):
        self.color_idx = (self.color_idx + 1) % num_classes

@dataclass
class DataState:
    project_path: Path = field(default_factory=lambda: Path.cwd() / "default_project")
    input_path: str = ''
    zarr_files: List[Path] = field(default_factory=list)
    zarr_idx: int = 0
    cache_size_mb: int = 16384  # Total tensorstore cache limit, shared by viewing and training

    @property
    def active_zarr(self) -> Optional[Path]:
        if not self.zarr_files:
            return None
        idx = max(0, min(self.zarr_idx, len(self.zarr_files) - 1))
        return self.zarr_files[idx]

@dataclass
class TrainState:
    input_size: int = 512
    lr: float = 1e-3
    batch_size: int = 4
    num_classes: int = 2
    slab_size: int = 5  # Neighbouring slices fed to the model as channels ("2.5D depth")
    level: int = 0
    level_available: bool = True
    architecture: str = 'Unet'
    encoder_name: str = 'efficientnet-b3'
    steps_per_epoch: int = 20
    scnp_enabled: bool = True
    scnp_size: int = 3
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = metrics.dice_ce_loss
    model_locked: bool = False
    predicting: bool = False
    changing_classes: bool = False
    export_tiff: bool = False
    live_training_enabled: bool = True

@dataclass
class AppState:
    ui: UIState = field(default_factory=UIState)
    nav: NavigationState = field(default_factory=NavigationState)
    camera: Camera = field(default_factory=Camera)
    annot: AnnotationState = field(default_factory=AnnotationState)
    data: DataState = field(default_factory=DataState)
    train: TrainState = field(default_factory=TrainState)

@dataclass
class AppServices:
    state: AppState
    slicer: VolumeSlicer = field(init=False)
    tracker: AnnotationTracker = field(init=False)
    history: EditHistory = field(init=False)

    def __post_init__(self):
        self.slicer = VolumeSlicer(cache_size_mb=self.state.data.cache_size_mb)
        self.tracker = AnnotationTracker(str(self.state.data.project_path))
        self.history = EditHistory(self.slicer, self.tracker)

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Callable, Optional

from insegt3d.volume.slicer import VolumeSlicer
from insegt3d.volume.camera import Camera
from insegt3d.ml.annotation_tracker import AnnotationTracker
from insegt3d.ml import metrics


# ------------------------- State (values only) -------------------------

@dataclass
class OverlayState:
    visible: bool = False
    alpha: float = 0.25

@dataclass
class UIState:
    viewport_shape: tuple[int, int] = (768, 768)
    # Persisted annotation mask ("Annotation overlay" in the UI), on by default.
    mask: OverlayState = field(default_factory=lambda: OverlayState(visible=True))
    annotation: OverlayState = field(default_factory=OverlayState)
    # Live, model-in-progress prediction ("Live prediction overlay" in the UI).
    prediction: OverlayState = field(default_factory=OverlayState)
    # Bulk prediction read from disk ("Prediction overlay" in the UI).
    saved_prediction: OverlayState = field(default_factory=OverlayState)
    show_orientation: bool = False
    jpeg_quality: int = 80
    intensity_low: float = 0.0
    intensity_high: float = 255.0

    def max_brush_size(self, max_scale=0.25):
        """
        Largest brush diameter (px) that still fits entirely inside the
        viewport: the smaller of its two dimensions, scaled by given factor.
        """
        return min(self.viewport_shape) * max_scale

@dataclass
class NavigationState:
    slice_shape: tuple[int, int] = (768, 768)
    max_slice_megapixels: float = 2.0  # Limits how much data is read from the zarr volume per slice
    revision: int = 0  # increments on any change that invalidates results

    def bump(self):
        self.revision += 1
        return self.revision

@dataclass
class AnnotationState:
    annotating: bool = False
    mode: str = 'draw'  # 'draw' | 'save' | 'flood' | 'mask_fill'
    brush_size: int = 3
    colors: List[str] = field(default_factory=lambda: [
        'rgba(230, 25, 75, 1)', 'rgba(60, 180, 75, 1)',
        'rgba(255, 225, 25, 1)', 'rgba(0, 130, 200, 1)',
        'rgba(245, 130, 48, 1)', 'rgba(145, 30, 180, 1)',
        'rgba(70, 240, 240, 1)', 'rgba(240, 50, 230, 1)',
        'rgba(210, 245, 60, 1)', 'rgba(170, 255, 195, 1)',
    ])
    color_idx: int = 0
    _palette_rgb: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def next_color(self, num_classes):
        self.color_idx = (self.color_idx + 1) % num_classes

    def previous_color(self, num_classes):
        self.color_idx = (self.color_idx - 1) % num_classes

    @property
    def palette_rgb(self):
        if self._palette_rgb is None:
            pal = np.zeros((len(self.colors) + 1, 3), dtype=np.uint8)
            for i, rgba in enumerate(self.colors, start=1):
                pal[i] = tuple(map(int, rgba[5:-1].split(",")[:3]))
            self._palette_rgb = pal
        return self._palette_rgb

    @property
    def palette_bgr(self):
        return self.palette_rgb[:, ::-1]

    @property
    def color_css(self):
        return str(self.colors[self.color_idx])

    @property
    def color_rgb(self):
        return np.array(self.palette_rgb[self.color_idx + 1])

@dataclass
class DataState:
    project_path: Path = field(default_factory=lambda: Path.cwd() / "default_project")
    zarr_files: List[Path] = field(default_factory=list)
    zarr_idx: int = 0
    cache_size_mb: int = 24000 # Total tensorstor cache limit for viewing and reading data

    @property
    def active_zarr(self) -> Optional[Path]:
        if not self.zarr_files:
            return None
        idx = max(0, min(self.zarr_idx, len(self.zarr_files) - 1))
        return self.zarr_files[idx]

@dataclass
class PointerState:
    x: int = 0
    y: int = 0

@dataclass
class TrainState:
    input_size: int = 512
    epochs: int = 1
    lr: float = 4e-3
    batch_size: int = 4
    num_classes: int = 2
    architecture: str = 'U-Net++'
    encoder_name: str = 'resnet50'
    pretrained: bool = True
    recency_temp: float = 250.0
    steps_per_epoch: int = 20
    ema_decay: float = 0.9
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = metrics.mcc_ce_loss
    model_locked: bool = False
    predicting: bool = False
    export_tiff: bool = False  # Also write predictions as a tiff stack
    live_training_enabled: bool = True
    cache_size_mb: int = 8000 # Total tensorstor cache limit for training

@dataclass
class AppState:
    ui: UIState = field(default_factory=UIState)
    nav: NavigationState = field(default_factory=NavigationState)
    camera: Camera = field(default_factory=Camera)
    annot: AnnotationState = field(default_factory=AnnotationState)
    data: DataState = field(default_factory=DataState)
    pointer: PointerState = field(default_factory=PointerState)
    train: TrainState = field(default_factory=TrainState)

    @classmethod
    def from_project(cls, project_folder: str | Path | None = None) -> "AppState":
        if project_folder is None:
            project_folder = Path.cwd() / "default_project"
        else:
            project_folder = Path(project_folder)

        return cls(data=DataState(project_path=project_folder))

@dataclass
class AppServices:
    state: AppState
    slicer: VolumeSlicer = field(init=False)
    tracker: AnnotationTracker = field(init=False)

    def __post_init__(self):
        self.slicer = VolumeSlicer(cache_size_mb=self.state.data.cache_size_mb)
        self.tracker = AnnotationTracker(str(self.state.data.project_path))

# InSegt3D

Interactive Segmentation of 3D volumes.

InSegt3D is a browser-based tool for annotating and segmenting large 3D volumetric images stored in OME-ZARR format. You annotate along 2D slices through the volume while a machine learning model trains live in the background on what you have annotated, and its prediction is shown back to you as an overlay while you keep annotating.

The OME-Zarr format allows efficient and low-RAM annotation of large 3D volumes. Tested with up to 200GB OME-Zarr volumes on a laptop with 32GB RAM and an RTX A3000 GPU with 6GB VRAM.

---

## Requirements

- Python **3.13** or newer
- A CUDA GPU is strongly recommended (training and prediction fall back to CPU, but will be VERY slow)
- Data stored as multiscale OME-Zarr 0.5 (Zarr v3) - see [Preparing data](docs/preparing-data.md)

## Installation with conda

Create and activate a conda environment named `insegt3d`, then install the package:

```bash
conda create --name insegt3d python=3.13
conda activate insegt3d
pip install git+https://github.com/qim-center/insegt3d
```

## Quick start

```bash
conda activate insegt3d
insegt3d --project_folder "path/to/project_folder" --num_classes 2
```

This creates the project folder if it does not exist, starts a server on a random free port, and prints a link to open in any web browser.

In the interface:

1. Put the path to your data in **Path to data** and press **Load**. This accepts a single `.zarr` store, a folder containing several `.zarr` stores, an `http(s)` URL to a remote store, or a comma-separated list of such URLs.
2. Pick a volume under **Scan**.
3. Navigate and begin annotating (see [Controls](#controls)). Once at least one annotation has been made for each class, a model will begin training in the background.
4. Watch the **Live prediction overlay** improve as the model trains. Press <kbd>D</kbd> to toggle it on and off, or <kbd>Shift</kbd> + <kbd>Left Click</kbd> to accept part of the prediction as ground-truth annotation.
5. When you are happy with the model, press **Predict** to run it over the whole volume. Tick **Also export tiff stack** first to write a tiff copy of each prediction alongside the zarr.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--project_folder` | `./default_project` | Where masks, annotations, checkpoints and predictions are stored |
| `--num_classes` | `2` | Number of classes to segment (2–10) |
| `--port` | random | Port to serve the interface on |

Use `insegt3d --help` for the full list of options.

## Preparing data

InSegt3D reads multiscale [OME-Zarr 0.5](https://ngff.openmicroscopy.org/0.5/) stores.
To convert a single tiff:

```python
from insegt3d.volume.io import convert_tiff_file

convert_tiff_file('path/to/volume.tif', 'path/to/output.zarr')
```

See [docs/preparing-data.md](docs/preparing-data.md) for tiff folders, numpy arrays, 3D/4D/5D
data, axis order, and converting volumes larger than RAM.

## Controls

### Annotating

| Input | Action |
| --- | --- |
| Left Click + Drag | Paint with the selected class |
| Shift + Left Click | Push the displayed prediction overlay into the annotation map |
| Mouse Wheel | Adjust brush size |
| C / X | Next / previous class colour |
| D | Toggle the live prediction overlay |
| Ctrl + Z | Undo last stroke |
| Ctrl + Y | Redo last stroke |

### Navigating

| Input | Action |
| --- | --- |
| Ctrl + Left Click + Drag | Pan |
| Ctrl + Right Click + Drag | Scroll through slices |
| Ctrl + Middle Click + Drag | Rotate the slicing plane |
| Ctrl + Mouse Wheel | Zoom in and out |
| Space | Randomize the orientation of the slicing plane |

Touch input is supported as well: one finger pans, two fingers rotate and pinch-zoom.

### Annotation modes

The **Mode** toggle in the **Annotation** panel switches between four ways of painting:

- **Draw** - freehand brush strokes in the selected class.
- **Overlay** - brush strokes accept the current prediction inside them as annotation
  (the same thing Shift does temporarily while held).
- **Flood** - click a seed point and drag to grow an intensity-based flood fill; the drag
  distance sets the tolerance.
- **Fill** - click inside a region fully enclosed by existing annotations to fill it.

## Training

Live training runs in the background while you annotate and is on by default. Under
**Advanced settings** you can turn it off, pick a different architecture or encoder (any
combination supported by
[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)),
change the learning rate, batch size and steps per training burst, or reset the model and
the annotations.

Note that the architecture and encoder are locked once training has started. Use **Reset
model** to change them.

## Batch prediction

Predictions can also be run headlessly from a trained checkpoint, which is useful for
applying a model to a whole set of volumes on a cluster:

```bash
insegt3d predict \
    --checkpoint "path/to/project_folder/model.ckpt" \
    --data "path/to/zarrs" \
    --output "path/to/output"
```

`--data` accepts a single Zarr store, a folder of Zarr stores, or an `http(s)` URL.
Results are written to `<output>/predictions/<volume_name>`.

| Option | Default | Description |
| --- | --- | --- |
| `--checkpoint` | *required* | Trained checkpoint (`model.ckpt`) |
| `--data` | *required* | Volume, folder of volumes, or `http(s)` URL |
| `--output` | *required* | Output directory |
| `--num-classes` | from checkpoint | Number of classes |
| `--input-size` | `512` | Cubic block size used for inference |
| `--batch-size` | auto | Inference batch size; defaults to the largest that fits in memory |
| `--overlap` | `0.25` | Fractional overlap between adjacent blocks |
| `--axes` | `0,1,2` | Axes to predict along and average over |
| `--export-tiff` | off | Also write each prediction as a tiff stack |
| `--temp-dir` | `<output>/temp` | Scratch directory for accumulation buffers |

Run `insegt3d predict --help` for the same list from the terminal.

### Exporting to tiff

Both **Also export tiff stack** in the interface and `--export-tiff` on the command line
write a second copy of the prediction next to the zarr, as
`<output>/predictions/<volume_name>_tiff/`. The folder holds one tiff per z slice, each
a channel-last `(y, x, c)` image whose channels are the per-class scores - the same data
the zarr holds, laid out so that no file is more than 3D.

The stacks are written slice by slice, so exporting costs no more memory than the
prediction itself. Note that tiff is uncompressed here: a stack takes roughly one byte per
voxel per class.

## Project folder layout

```
project_folder/
├── annotations.json     # record of every annotated region and its camera pose
├── model.ckpt           # latest trained model checkpoint
├── masks/               # per-volume annotation masks (zarr)
├── predictions/         # full-volume predictions (zarr, plus optional tiff stacks)
└── temp/                # scratch space used during prediction
```

A project folder can be reopened at any time. The annotations, masks and the trained model are all picked up again on start-up.

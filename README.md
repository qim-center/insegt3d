# InSegt3D

Interactive Segmentation of 3D volumes.

InSegt3D is a browser-based tool for annotating and segmenting large 3D volumetric images stored in OME-Zarr format. You annotate along 2D slices through the volume while a machine learning model trains live in the background on what you have annotated, and its prediction is shown back to you as an overlay while you keep annotating.

The OME-Zarr format allows efficient and low-RAM annotation of large 3D volumes. Tested with up to 200GB OME-Zarr volumes on a laptop with 32GB RAM and an RTX A3000 GPU with 6GB VRAM.

---

## Requirements

- Python **3.13**
- A CUDA GPU or an Apple Silicon Mac is strongly recommended (training and prediction fall back to CPU, but will be VERY slow). Intel Macs are not supported. On a Mac the GPU shares the system memory, so lower `--cache_gb` to leave room for it
- Data stored as multiscale OME-Zarr 0.5 (Zarr v3) - see [Preparing data](docs/preparing-data.md)

## Installation

Pick one of the options below. They all install the same package. The uv options require [uv](https://docs.astral.sh/uv/), which you can install by following its [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### conda

Create and activate a conda environment named `insegt3d`, then install the package:

```bash
conda create --name insegt3d python=3.13
conda activate insegt3d
pip install git+https://github.com/qim-center/insegt3d
```

### uv virtual environment

Create and activate a virtual environment in the current folder, then install the package:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install git+https://github.com/qim-center/insegt3d
```

### uv tool

Install InSegt3D into its own isolated environment and put the `insegt3d` command on your PATH, so there is nothing to activate:

```bash
uv tool install --python 3.13 git+https://github.com/qim-center/insegt3d
```

Update to the latest version with `uv tool upgrade insegt3d`, or remove it with `uv tool uninstall insegt3d`. If the `insegt3d` command is not found afterwards, run `uv tool update-shell` and restart your terminal.

## Quick start

Activate the environment you installed into (`conda activate insegt3d`, or `source .venv/bin/activate` from the folder where you created it). If you installed with `uv tool` there is nothing to activate. Then run:

```bash
insegt3d --project_folder "path/to/project_folder"
```

This creates the project folder if it does not exist, starts a server on a random free port, and prints a link to open in any web browser. To run InSegt3D on a remote machine or HPC cluster and use it from your own browser, see [Remote access](docs/remote-access.md).

In the interface:

1. Put the path to your data in **Path to data** and press **Load**. This accepts a single `.zarr` store, a folder containing several `.zarr` stores, an `http(s)` URL to a remote store, or a comma-separated list of such URLs.
2. Pick a volume under **Scan**.
3. Navigate and begin annotating (see [Controls](#controls)). Once at least one annotation has been made for each class, a model will begin training in the background.
4. Watch the **Live prediction overlay** improve as the model trains. Press <kbd>D</kbd> to toggle it on and off, or <kbd>Shift</kbd> + <kbd>Left Click</kbd> to accept part of the prediction as ground-truth annotation.
5. When you are happy with the model, press **Predict** to run it over every loaded volume, then view the result with the **Prediction overlay**. Tick **Also export tiff stack** first to write a tiff copy of each prediction alongside the zarr.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--project_folder` | `./default_project` | Where masks, annotations, checkpoints and predictions are stored |
| `--port` | random | Port to serve the interface on |
| `--host` | `localhost` | Address to serve on. Use `0.0.0.0` to allow access from other machines |
| `--server_base_path` | none | URL prefix to serve under, e.g. behind a reverse proxy |
| `--cache_gb` | `16` | Memory for cached volume data, in GiB. The app uses about 5 GiB on top of this, so lower it if the system starts swapping |

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
| Right Click + Drag | Erase (also the eraser end of a tablet pen) |
| Shift + Left Click | Push the displayed prediction overlay into the annotation map |
| Mouse Wheel | Adjust brush size |
| B / E / G / F / K | Draw / Erase / Fill / Flood / Keep tool |
| C | Next class colour |
| 1–9, 0 | Select class 1–10 |
| D | Toggle the live prediction overlay |
| Ctrl + Z | Undo last stroke |
| Ctrl + Y | Redo last stroke |

### Navigating

| Input | Action |
| --- | --- |
| Ctrl + Left Click + Drag | Pan |
| Ctrl + Right Click + Drag | Scroll through slices |
| Q / A | Step one slice up / down along the view normal |
| Ctrl + Middle Click + Drag | Rotate the slicing plane |
| Ctrl + Mouse Wheel | Zoom in and out |
| Space | Randomize the orientation of the slicing plane |
| Z / Y / X | View along the z, y or x axis, keeping the position and zoom |
| , / . | Go to the previous / next annotated slice |

Touch input is supported as well: one finger pans, two fingers rotate and pinch-zoom.

Shortcuts are paused while typing in a text or number field. Press Enter or Esc, or click the
viewport, to return to them. The keyboard button under the viewport lists every shortcut.

The **Viewport** panel shows the slicing plane inside the volume's bounding box. Drag its arrows
to move the plane, or the spheres on its rings to rotate it. Below it, **Z**, **Y** and **X** align
the view to an axis, **Center** returns to the starting view, and **Go to** moves to a typed
location (in full-resolution voxels) and slice normal. The arrows beside **Annotated slices** step
through the slices you have annotated in the current volume, framing their strokes and turning on
the annotation overlay.

### Classes

Projects start with two classes. Press **+** next to the class colours to add one (up to ten), or
**−** to remove the selected class. Removing a class erases its annotations and renumbers the
classes after it, which keep their colours. Classes can be changed at any time: the model keeps
what it has learned, and training resumes once every class has an annotation.

### Annotation modes

The tool buttons at the top of the **Annotation** panel switch between five ways of painting
(hover over them for their shortcuts):

- **Draw** - freehand brush strokes in the selected class.
- **Erase** - brush strokes clear annotations of any class. In Draw, Erase and Keep mode,
  right-click dragging or the eraser end of a tablet pen erases too.
- **Fill** - click inside a region fully enclosed by existing annotations to fill it.
- **Flood** - click a seed point and drag to grow an intensity-based flood fill. The drag
  distance sets the tolerance.
- **Keep** - brush strokes keep the live prediction inside them as annotation
  (the same thing Shift does temporarily while held).

### Display

The **Display** panel toggles three overlays and sets their opacity: the **Annotation overlay**,
the **Prediction overlay** (the result of **Predict**, once the current volume has one) and the
**Live prediction overlay**. Drag the range under the histogram to set the intensity window. It
starts at the 0.5th to 99.5th percentile of the volume.

## Training

Live training runs in the background while you annotate and is on by default. Under
**Advanced settings** you can turn it off, pick a different architecture or encoder (any
combination supported by
[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)),
set the **2.5D depth** (how many neighbouring slices the model sees) and the **Training
resolution** (which pyramid level it trains and predicts on), choose the loss function and
toggle SCNP, change the learning rate, batch size and steps per training burst, or reset the
model and the annotations.

Note that the architecture, encoder, 2.5D depth and training resolution are locked once
training has started. Use **Reset model** to change them.

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
Results are written to `<output>/predictions/<volume_name>__<id>`, where `<id>` is a short hash of the volume's location.
Each result is a zarr of per-class scores (`uint8`, 0-255) with the predicted class labels in a nested `labels` zarr.
Prediction runs at the resolution the model was trained on. Volumes without that level are skipped and listed in
`<output>/predictions/skipped_volumes.txt`.

| Option | Default | Description |
| --- | --- | --- |
| `--checkpoint` | *required* | Trained checkpoint (`model.ckpt`) |
| `--data` | *required* | Volume, folder of volumes, or `http(s)` URL |
| `--output` | *required* | Output directory |
| `--input-size` | `512` | Cubic block size used for inference |
| `--batch-size` | auto | Inference batch size. Defaults to the largest that fits in memory |
| `--overlap` | `0.25` | Fractional overlap between adjacent blocks |
| `--axes` | `0,1,2` | Axes to predict along and average over |
| `--export-tiff` | off | Also write each prediction as a tiff stack |
| `--temp-dir` | `<output>/temp` | Scratch directory for accumulation buffers |

Run `insegt3d predict --help` for the same list from the terminal.

### Exporting to tiff

Both **Also export tiff stack** in the interface and `--export-tiff` on the command line
write a second copy of the prediction next to the zarr, as
`<output>/predictions/<volume_name>__<id>_tiff/`. The folder holds one tiff per z slice, each
a channel-last `(y, x, c)` image whose channels are the per-class scores - the same data
the zarr holds.

## Project folder layout

```
project_folder/
├── annotations.json     # record of every annotated region and its camera pose
├── classes.json         # class colours, in class order
├── model.ckpt           # latest trained model checkpoint
├── masks/               # per-volume annotation masks (zarr)
├── predictions/         # full-volume predictions (zarr, plus optional tiff stacks)
└── temp/                # scratch space used during prediction
```

A project folder can be reopened at any time. The annotations, masks and the trained model are all picked up again on start-up.

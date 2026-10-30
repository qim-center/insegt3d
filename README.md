# InSegt3D

InSegt3D is a browser-based tool for segmenting large 3D volumes. You paint labels on 2D slices through the volume while a neural network trains on them in the background. Its prediction appears as an overlay, so you can see where the model is wrong and paint more where it needs help. When the result looks right, you apply the model to whole volumes.

Volumes are read in chunks from multiscale OME-Zarr stores, so they never need to fit in memory. InSegt3D has been tested on 200 GB volumes using a laptop with 32 GB of RAM and a 6 GB RTX A3000 GPU.

## Requirements

- Python 3.13
- A CUDA GPU or an Apple Silicon Mac. Without one, InSegt3D runs on the CPU, but training and prediction are very slow. Intel Macs are not supported.
- Volumes stored as multiscale OME-Zarr 0.5 (Zarr v3). [Preparing data](docs/preparing-data.md) shows how to convert tiff files and numpy arrays.

## Installation

The three options below install the same package. The uv options need [uv](https://docs.astral.sh/uv/getting-started/installation/).

### conda

```bash
conda create --name insegt3d python=3.13
conda activate insegt3d
pip install git+https://github.com/qim-center/insegt3d
```

### uv virtual environment

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install git+https://github.com/qim-center/insegt3d
```

### uv tool

This puts the `insegt3d` command on your PATH with its own environment, so there is nothing to activate.

```bash
uv tool install --python 3.13 git+https://github.com/qim-center/insegt3d
```

Update with `uv tool upgrade insegt3d` and remove with `uv tool uninstall insegt3d`. If the shell cannot find `insegt3d`, run `uv tool update-shell` and open a new terminal.

## Getting started

Activate the environment you installed into, then start InSegt3D.

```bash
insegt3d
```

InSegt3D prints a link to open in your browser. The start page asks for a project folder, which holds your annotations, model and predictions. Enter an existing folder to reopen it, or a new path to create one. To run InSegt3D on a remote machine or cluster, see [Remote access](docs/remote-access.md).

A typical session goes like this.

1. Enter the location of your data in **Volume path or URL** and press **Open data**. This can be a `.zarr` store, a folder of `.zarr` stores, an https URL, or several https URLs separated by commas. Plain http URLs are refused for security reasons.
2. Choose a volume in **Current volume**.
3. Paint some voxels of each class. Training starts in the background once every class has an annotation, and the bar under the view shows its progress.
4. Press <kbd>D</kbd> to show the live prediction. Paint more where it is wrong. Hold <kbd>Shift</kbd> and paint to accept the parts it gets right.
5. When the live prediction looks good, press **Predict all volumes**. Turn on **Prediction overlay** to view the result.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--project_folder` | start page | Open this project folder directly |
| `--port` | random free port | Port to serve the interface on |
| `--host` | `localhost` | Address to serve on. Use `0.0.0.0` to accept connections from other machines |
| `--server_base_path` | none | URL prefix to serve under, for use behind a reverse proxy |
| `--token` | none | Require this token to open the interface. See [Access token](docs/remote-access.md#access-token) |
| `--cache_gb` | `16` | Memory for cached volume data in GiB. InSegt3D uses about 5 GiB on top of this |

Lower `--cache_gb` if the system starts swapping. On a Mac the GPU shares system memory, so leave extra room there.

## Controls

| Input | Action |
| --- | --- |
| Left drag | Use the selected annotation tool |
| Right drag, or the eraser end of a pen | Erase |
| Shift + left drag | Keep the live prediction under the stroke |
| Mouse wheel | Change the brush size |
| B, E, G, F, K | Draw, Erase, Fill, Flood or Keep tool |
| 1 to 9, 0 | Select class 1 to 10 |
| C | Select the next class |
| D | Toggle the live prediction overlay |
| Ctrl + Z, Ctrl + Y | Undo or redo a stroke |
| Ctrl + left drag | Pan |
| Ctrl + right drag | Move through slices |
| Ctrl + middle drag | Rotate the slicing plane |
| Ctrl + mouse wheel | Zoom |
| Ctrl + double-click | Center the view on a point |
| Q, A | Step one slice forward or back |
| Z, Y, X | View along the z, y or x axis |
| Space | Pick a random slice orientation |
| Comma, period | Go to the previous or next annotated slice |
| H | Toggle the crosshair |

On a touch screen, one finger pans and two fingers rotate and zoom. Shortcuts pause while you type in a text field. Press Enter or Esc, or click the slice view, to resume them. **Shortcuts** in the top bar lists them all.

## The workspace

The left panel holds **Data source**, **Annotate** and **Predict**. The right panel holds **Navigate** and **Display**. On narrow screens the panels fold away, and **Tools** and **View** in the top bar bring them back.

The project name in the top bar lets you switch projects or shut down InSegt3D. Work in progress is saved first, and a running prediction is canceled. **Theme** switches between light and dark mode.

### Annotation tools

| Tool | What it does |
| --- | --- |
| Draw | Paints the selected class with a round brush |
| Erase | Removes annotations of any class |
| Fill | Fills an area enclosed by existing annotations with the selected class |
| Flood | Grows a fill from where you click based on intensity. Drag further to raise the tolerance |
| Keep | Copies the live prediction under the brush into the annotations |

Right drag erases in Draw, Erase and Keep. Holding Shift switches to Keep until you let go.

### Classes

A project starts with two classes and can have up to ten. Use the plus and minus buttons beside **Class** to add a class or remove the selected one. Removing a class erases its annotations and renumbers the classes after it, which keep their colors. Existing predictions keep the old classes. The model keeps what it has learned through these changes and resumes training once every class has an annotation.

### Working resolution

**Working resolution** in **Data source** picks the pyramid level used for annotating, training and predicting. A coarser level is faster on large volumes. Zooming in past the working level shows that level enlarged.

You can change the level at any time and the model keeps what it has learned. Moving to a finer level redraws your Draw and Erase strokes at the new resolution so they stay sharp. Fill, Flood and Keep strokes keep the resolution they were made at. Training and annotating pause until the update finishes, and the bar under the view shows its progress. Moving to a coarser level keeps your finer annotations, so moving back loses nothing.

### Navigate panel

The 3D view shows the slicing plane inside the volume. Drag its arrows to move the plane, or the spheres on its rings to rotate it. Below it are buttons to view along each axis, **Center view** to return to the starting view, and **Go to…** to jump to a location in full-resolution voxels. The arrows beside **Annotated slices** step through the slices you have annotated in the current volume.

### Display

**Display** has three overlays, each with an opacity slider.

- **Annotation overlay** shows your annotations.
- **Live prediction overlay** shows what the model currently predicts for the slice.
- **Prediction overlay** shows the saved result of the last prediction run on this volume.

**Crosshair** marks the center of the view. Rotating, zooming and the axis views all keep this point fixed, so you can Ctrl + double-click an object to center it and then look at it along each axis.

Drag the range under the histogram to set the intensity window. It starts at the 0.5th to 99.5th percentile of the volume's intensities.

### Predicting

**Prediction area** in **Predict** chooses what to predict. **All volumes** runs the model on every opened volume. **Selected regions** runs it only inside boxes you place in the current volume. The bar under the view shows the progress, and **Cancel** stops the run. Tick **Also export TIFF stacks** to also write each prediction as tiff files.

To set up a region, press **Add** to place a box around the current view. Turn on **Edit on slice** and use a Z, Y or X view to drag the box or its edges. You can also select a region and type its bounds in full-resolution voxels. Only checked regions are predicted.

A region run updates the existing prediction inside its boxes and leaves the rest in place. If that prediction was made at another working resolution or with a different number of classes, a new prediction covering only the regions replaces it. Canceling a region run can leave some finished blocks behind in the prediction.

### Model settings

**Model settings** in the top bar controls training.

| Setting | Default | Description |
| --- | --- | --- |
| Model architecture | Unet | Any architecture in [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) |
| Model encoder | efficientnet-b3 | Any encoder in segmentation-models-pytorch |
| 2.5D depth | 5 | Number of consecutive slices the model sees at once |
| Loss function | Dice+CE | Training loss |
| SCNP | on, size 3 | Adds [Same Class Neighbor Penalization](https://jmlipman.github.io/SCNP-SameClassNeighborPenalization/) to the loss |
| Learning rate | 0.001 | Optimizer step size |
| Batch size | 4 | Slices per training step |
| Steps per training burst | 20 | Training steps between checkpoint saves and live prediction updates |
| Live training | on | Train in the background while you annotate |

The architecture, encoder and 2.5D depth lock once training starts. **Reset model** discards the model so you can change them. **Reset annotations** deletes every annotation in the project.

## Batch prediction

`insegt3d predict` applies a trained model without the interface. This is useful for processing many volumes on a cluster.

```bash
insegt3d predict \
    --checkpoint path/to/project_folder/model.ckpt \
    --data path/to/volumes \
    --output path/to/output
```

| Option | Default | Description |
| --- | --- | --- |
| `--checkpoint` | required | Trained model, usually `model.ckpt` in a project folder |
| `--data` | required | A `.zarr` store, a folder of `.zarr` stores, or https URLs separated by commas |
| `--output` | required | Output folder |
| `--input-size` | `512` | Edge length of the cubic blocks the volume is split into |
| `--batch-size` | largest that fits | Slices per inference batch |
| `--overlap` | `0.25` | Overlap between neighboring blocks, as a fraction of the block size |
| `--axes` | `0,1,2` | Axes to slice along. The predictions along each axis are averaged |
| `--export-tiff` | off | Also write each prediction as tiff files |
| `--temp-dir` | `<output>/temp` | Folder for temporary files |

Prediction runs at the working resolution the model was trained at. Volumes without that level are skipped and listed in `skipped_volumes.txt` in the predictions folder.

### Output

Each volume's prediction goes to `predictions/<name>__<id>.zarr` in the output folder, where `<id>` is a short hash of the volume's location. The hash keeps volumes with the same name apart. The store holds a score from 0 to 255 for each class and voxel. A `labels` store inside it holds the predicted class of each voxel, numbered from 1.

The tiff export writes to `predictions/<name>__<id>_tiff/`, with one file per z slice. Each file holds the class scores as channels in `(y, x, c)` order. A region run writes one subfolder per region inside it instead, named after the region's bounds.

The interface uses the same layout inside the project folder.

## Project folder

```
project_folder/
├── annotations.json   # annotation strokes and the view each was drawn in
├── classes.json       # class colors in class order
├── regions.json       # prediction regions for each volume
├── settings.json      # working resolution
├── model.ckpt         # latest model checkpoint
├── masks/             # annotation masks for each volume
├── predictions/       # predictions for each volume
└── temp/              # temporary files used during prediction
```

Reopen a project folder to continue where you left off.

## Further reading

- [Preparing data](docs/preparing-data.md) covers converting tiff files and numpy arrays, axis order and voxel size.
- [Remote access](docs/remote-access.md) covers running InSegt3D on a workstation or cluster and using it from your own browser.

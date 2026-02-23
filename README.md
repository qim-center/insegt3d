
## Interactive U-Net

A segmentation tool that utilizes the U-Net deep learning architecture to quickly and efficiently segment 3D volumetric images. It utilizes the Zarr storage format to enable the segmentation of extremely large images.

### Installation:

`pip install git+https://github.com/laprade117/interactive-unet`

### Usage

After installation you can run the tool using:

`interactive-unet --project_folder "path/to/project_folder" --num_classes 2` 

This will create a project folder at the specified location then setup a user interface for a two-class segmentation task and provide a link that can be opened in any web browser. If you wish to segment more than two classes, increase the `--num_classes` argument to a larger value (maximum 10). A random port is used when creating the interface, however, it can be specified with `--port 9090`, if needed.

The tool expects data to be stored in multi-scale Zarr files with a 'uint8' datatype. Chunk sizes of 32x32x32 or 64x64x64 have been found to be optimal for efficient navigation with the tool. See the `Converting_to_zarr.ipynb` notebook for some scripts for converting to the correct formats.

### Keyboard Shortcuts

- **Left click**: Paint displayed color
- **Shift + Left Click**: Push displayed overlay onto annotation map

- **Mouse Wheel**: Adjust brush size
  
- **C**: Cycles through colors
- **D**: Toggle prediction overlay
  
- **Ctrl + Z**: Undo last paint stroke
- **Ctrl + Y**: Redo last paint stroke
  
- **Ctrl + Left Click + Drag**: Translation
- **Ctrl + Mouse Wheel**: Zoom in and out
- **Ctrl + Right Click + Drag**: Scroll through slices
- **Ctrl + Middle Mouse Button + Drag**: Rotate plane

### Local Setup (with Conda)

1. Create a conda environment:
`conda create --name unet python=3.11`

2. Activate the environment:
`conda activate unet`

3. Install the tool:
`pip install git+https://github.com/laprade117/interactive-unet`

4. Launch the tool:
`interactive-unet --project_folder "path/to/project_folder" --num_classes 2`

### DTU Thinlinc Setup (with Conda)

1. Activate an interactive GPU session:
`sxm2sh -X`

2. Create a conda environment:
`conda create --name unet python=3.11`

3. Activate the environment:
`conda activate unet`

4. Install the tool:
`pip install git+https://github.com/laprade117/interactive-unet`

5. Launch the tool:
`interactive-unet --project_folder "path/to/project_folder" --num_classes 2`

While running on Thinlinc, you can only access the tool from the browser within the the ThinLinc Client. Access outside of the ThinLinc client is not working.
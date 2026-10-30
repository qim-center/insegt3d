# Preparing data

InSegt3D reads multiscale [OME-Zarr 0.5](https://ngff.openmicroscopy.org/0.5/) stores. A multiscale store holds the volume at several resolutions, which lets InSegt3D show and train on large volumes without loading them into memory.

The helpers in [insegt3d/volume/io.py](../insegt3d/volume/io.py) convert numpy arrays and tiff files. They cover the common cases only. For other formats, use a converter such as [ngff-zarr](https://pypi.org/project/ngff-zarr/) or [bioformats2raw](https://github.com/glencoesoftware/bioformats2raw).

## Converting

| Source | Helper | Result |
| --- | --- | --- |
| A 3D, 4D or 5D numpy array | `write_zarr` | Same dimensions |
| A 3D, 4D or 5D tiff file | `convert_tiff_file` | Same dimensions |
| A folder of 2D tiff slices | `convert_tiff_stack` | `(z, y, x)` |
| A folder of 3D tiff volumes | `convert_tiff_stack` | `(c, z, y, x)` |
| A folder of 4D tiff volumes | `convert_tiff_stack` | `(t, c, z, y, x)` |

```python
from insegt3d.volume.io import write_zarr, convert_tiff_file, convert_tiff_stack

write_zarr(volume, 'volume.zarr')               # a numpy array
convert_tiff_file('volume.tif', 'volume.zarr')  # a single tiff file
convert_tiff_stack('slices/', 'volume.zarr')    # a folder of tiff files
```

`convert_tiff_stack` sorts the files by name, so pad slice numbers with zeros, as in `slice_0001.tif`.

## Axis order

A volume has three spatial axes, `z`, `y` and `x`. It can also have a channel axis `c` and a time axis `t` in front of them. OME-Zarr requires the order `t, c, z, y, x`. The viewer is 3D and shows only the first channel and time point, so to segment another channel, write it to a store of its own.

The helpers work out the axes from the shape. The last three dimensions are taken as `z, y, x`, a fourth as `c` and a fifth as `t`. Axis labels stored in tiff files are ignored. An ImageJ hyperstack, for example, is saved as `t, z, c, y, x` and would be read in the wrong order.

Pass `input_axes` to give the order of the dimensions in each source file.

```python
convert_tiff_file('hyperstack.tif', 'volume.zarr', input_axes='TZCYX')
convert_tiff_stack('slices/', 'volume.zarr', input_axes='CYX')  # each file is a multichannel slice
```

A folder of files is stacked along the one axis the files do not have themselves, so a folder of `CYX` slices becomes a `(c, z, y, x)` volume. `write_zarr` has no `input_axes` and expects the array in OME-Zarr order already. Use `numpy.transpose` to reorder it first.

## Voxel size

InSegt3D reads the voxel size of every level from the store. Volumes with non-cubic voxels display without distortion and the model trains on the true proportions. This also holds for pyramids that downsample some axes less than others.

The helpers write cubic voxels by default. Pass `voxel_size` in `z, y, x` order to change this. Any unit works as long as all three values use the same one.

```python
convert_tiff_stack('slices/', 'volume.zarr', voxel_size=(50, 12, 12))  # 50 nm sections, 12 nm pixels
```

## Memory

The helpers work in batches that stay under `mem_limit_gb`, which is 4 GB by default. This lets `convert_tiff_stack` convert stacks larger than RAM. `convert_tiff_file` can do the same for uncompressed tiff files, which it memory-maps. A compressed tiff file is read into memory in full.

```python
convert_tiff_stack('slices/', 'volume.zarr', mem_limit_gb=16)
```

## Exporting to tiff

`write_tiff_stack` writes a 3D or 4D array to a folder with one tiff file per z slice. The array can be a numpy or zarr array in OME-Zarr order. A `(c, z, y, x)` array gives `(y, x, c)` slices with the channels last. InSegt3D uses this function for the tiff export of predictions.

```python
from insegt3d.volume.io import write_tiff_stack

write_tiff_stack(volume, 'slices/')
```

Read such a folder back with `convert_tiff_stack('slices/', 'volume.zarr', input_axes='YXC')`.

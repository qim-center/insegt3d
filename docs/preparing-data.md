# Preparing data

The tool reads multiscale [OME-Zarr 0.5](https://ngff.openmicroscopy.org/0.5/) stores.

A volume may have **3, 4 or 5 dimensions**: the three spatial axes `z, y, x`, optionally preceded by a channel axis `c` and a time axis `t`. OME-Zarr fixes their order as `t, c, z, y, x`.

Note that the viewer itself is 3D: for a 4D or 5D store it shows the first channel and timepoint, so segmenting a different channel means writing that channel out on its own.

The helpers in [insegt3d/volume/io.py](../insegt3d/volume/io.py) cover the common cases. They are deliberately minimal - for anything beyond this, convert the data yourself with a tool such as [ngff-zarr](https://pypi.org/project/ngff-zarr/) or [bioformats2raw](https://github.com/glencoesoftware/bioformats2raw).

| Source | Helper | Result |
| --- | --- | --- |
| A 3D, 4D or 5D numpy array | `write_zarr` | same dimensions |
| A single 3D, 4D or 5D tiff | `convert_tiff_file` | same dimensions |
| A folder of 2D slices | `convert_tiff_stack` | `(z, y, x)` |
| A folder of 3D volumes | `convert_tiff_stack` | `(c, z, y, x)` |
| A folder of 4D volumes | `convert_tiff_stack` | `(t, c, z, y, x)` |

The one helper going the other way is `write_tiff_stack`, which writes a volume back out as a folder of tiff slices - see [Exporting back to tiff](#exporting-back-to-tiff).

An in-memory numpy array, already in `t, c, z, y, x` order:

```python
from insegt3d.volume.io import write_zarr

write_zarr(volume, 'path/to/output.zarr')
```

A single tiff holding the whole volume:

```python
from insegt3d.volume.io import convert_tiff_file

convert_tiff_file('path/to/volume.tif', 'path/to/output.zarr')
```

A folder of tiff files, stacked along one new dimension:

```python
from insegt3d.volume.io import convert_tiff_stack

convert_tiff_stack('path/to/tiff_stack/', 'path/to/output.zarr')
```

## Axis order

Dimensions are worked out from the **shape alone**: the last three are taken as `z, y, x`, a fourth as `c` and a fifth as `t`. Axis labels stored inside tiff files are ignored, so an ImageJ hyperstack - which is saved as `t, z, c, y, x` - is read in the wrong order unless you say otherwise.

`input_axes` overrides the assumption. It names the dimensions of a *single file*:

```python
convert_tiff_file('volume.tif', 'out.zarr', input_axes='TZCYX')  # an ImageJ hyperstack
convert_tiff_stack('slices/', 'out.zarr', input_axes='CYX')      # each slice is c, y, x
```

For a stack, the files are stacked along whichever axis they do not use themselves, so a folder of `CYX` slices becomes a `(c, z, y, x)` volume.

## Exporting back to tiff

`write_tiff_stack` is the inverse of `convert_tiff_stack`, and is what the **Also export
tiff stack** option and `insegt3d predict --export-tiff` use to write predictions as tiff:

```python
from insegt3d.volume.io import write_tiff_stack

write_tiff_stack(volume, 'path/to/tiff_stack/')
```

`volume` is a 3D or 4D numpy or zarr array in the OME-Zarr order. It is written as one
tiff per z slice, so a `(z, y, x)` volume gives 2D slices and a `(c, z, y, x)` volume
gives channel-last `(y, x, c)` slices - never more than 3 dimensions per file. Reading
the folder back with `convert_tiff_stack(..., input_axes='YXC')` restores the original
array.

## Memory

Both tiff converters read in batches that stay under `mem_limit_gb` (4 GB by default), so volumes much larger than RAM can be converted. `write_tiff_stack` batches its reads the same way. `convert_tiff_file` additionally attempts to memory-maps the source when the tiff is uncompressed, rather than reading it in full:

```python
convert_tiff_stack('path/to/tiff_stack/', 'path/to/output.zarr', mem_limit_gb=16)
```

---

Back to the [README](../README.md).
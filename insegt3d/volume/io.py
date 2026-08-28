import zarr
import tensorstore as ts

import numpy as np
import tifffile as tiff
from pathlib import Path
from scipy.ndimage import zoom
from urllib.parse import urlparse

# Axis order required by OME-Zarr 0.5: time, channel, then the spatial axes.
OME_AXES = ('t', 'c', 'z', 'y', 'x')
MAX_NDIM = len(OME_AXES)

# OME-Zarr axis "type" for each axis name
_AXIS_TYPES = {'t': 'time', 'c': 'channel'}

# Annotation masks are always uint8
MASK_DTYPE = 'uint8'

def default_axes(ndim):
    if not 3 <= ndim <= MAX_NDIM:
        raise ValueError(f"Expected 3 to {MAX_NDIM} dimensions, got {ndim}")

    return OME_AXES[MAX_NDIM - ndim:]

def transpose_order(input_axes, ndim, source=''):
    axes = default_axes(ndim)

    if input_axes is None:
        return tuple(range(ndim))

    input_axes = tuple(str(axis).lower() for axis in input_axes)

    if len(input_axes) != ndim or set(input_axes) != set(axes):
        raise ValueError(
            f"input_axes {input_axes} do not describe the {ndim} dimensions of "
            f"'{source}'; expected a permutation of {axes}")

    return tuple(input_axes.index(axis) for axis in axes)

def normalize_zarr_path(zarr_path):
    return str(zarr_path).strip().rstrip('/')

def get_zarr_group(zarr_path):
    zarr_path = normalize_zarr_path(zarr_path)

    if zarr_path.startswith(("http://", "https://")):
        store = zarr_path
    else:
        store = zarr.storage.LocalStore(zarr_path)
    return zarr.open_group(store=store, mode='r')

def get_scale_paths(zarr_path):
    root = get_zarr_group(zarr_path)
    multiscales = root.attrs['ome']['multiscales']
    return [dataset['path'] for dataset in multiscales[0]['datasets']]

def is_multiscale_zarr(zarr_path):
    zarr_path = Path(zarr_path)

    if not (zarr_path / 'zarr.json').is_file():
        return False

    try:
        get_scale_paths(zarr_path)
    except (KeyError, IndexError):
        return False

    return True

def create_empty_zarr(zarr_path, level_shapes, chunks=(64,64,64), shards=(256,256,256), dtype=MASK_DTYPE, scale=0.5):

    root = zarr.open(zarr_path, mode='w')

    axes = default_axes(len(level_shapes[0]))
    extra_dims = len(axes) - 3

    for i, shape in enumerate(level_shapes):
        root.create_array(
            name=str(i),
            shape=shape,
            chunks=(1,) * extra_dims + tuple(chunks),
            shards=(1,) * extra_dims + tuple(shards),
            dtype=dtype,
            dimension_names=axes,
            overwrite=True)

    write_multiscale_metadata(
        root, num_levels=len(level_shapes), scale=scale, name=Path(zarr_path).stem)

def read_zarr_as_tensorstore(zarr_path, ts_context=None, cache_size_mb=4096):

    zarr_path = normalize_zarr_path(zarr_path)

    if ts_context is None:
        ts_context = ts.Context({
            'cache_pool': {'total_bytes_limit': int(cache_size_mb * 1024**2)}
        })

    # Build kvstore (local or http)
    if zarr_path.startswith(("http://", "https://")):
        u = urlparse(zarr_path)
        kvstore = {
            "driver": "http",
            "base_url": f"{u.scheme}://{u.netloc}",
            "path": u.path.rstrip("/") + "/",
        }
    else:
        kvstore = {
            "driver": "file",
            "path": zarr_path,
        }

    data = ts.open({
        "driver": "zarr3",
        "kvstore": kvstore,
        "recheck_cached_data": False,
    }, context=ts_context).result()

    if data.ndim < 3:
        raise ValueError(f"Expected at least 3D data, got shape {data.shape}")

    return data[(0,) * (data.ndim - 3) + (slice(None),) * 3]

def read_multiscale_zarr(zarr_path, ts_context=None, cache_size_mb=4096):

    zarr_path = normalize_zarr_path(zarr_path)

    # Get per-level dataset paths in the multiscale zarr, finest to coarsest
    levels = get_scale_paths(zarr_path)

    images = []

    # Create multiscale image list
    for level in levels:
        path = f"{zarr_path}/{level}"
        image = read_zarr_as_tensorstore(path, ts_context=ts_context, cache_size_mb=cache_size_mb)
        images.append(image)

    return images

def read_multiscale_masks(mask_path, level_shapes, ts_context=None):

    mask_path = Path(mask_path)

    # Ensure masks folder exists
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    # Create mask Zarr if it doesn't exist.
    if not mask_path.exists():
        create_empty_zarr(
            zarr_path=str(mask_path),
            level_shapes=level_shapes,
            dtype=MASK_DTYPE)

    # Create multiscale mask list
    masks = []
    for level in range(len(level_shapes)):
        path = str(mask_path / str(level))
        mask = read_zarr_as_tensorstore(path, ts_context=ts_context)
        masks.append(mask)

    return masks

def write_level0_array(dst_file, shape, dtype, chunks, shards):

    axes = default_axes(len(shape))

    extra_dims = len(shape) - 3
    full_chunks = (1,) * extra_dims + tuple(chunks)
    full_shards = (1,) * extra_dims + tuple(shards)

    root = zarr.open(dst_file, mode='w')
    z0 = root.create_array(
        name='0',
        shape=shape,
        chunks=full_chunks,
        shards=full_shards,
        dtype=dtype,
        dimension_names=axes,
        overwrite=True)
    return root, z0

def write_zarr(volume, dst_file, dtype=None, multiscale=True, scale=0.5, chunks=(64,64,64), shards=(256,256,256), mem_limit_gb=4):
    volume_dtype = volume.dtype if dtype is None else dtype

    root, z0 = write_level0_array(dst_file, volume.shape, volume_dtype, chunks, shards)
    write_batches(z0, volume, mem_limit_gb, dtype=volume_dtype)

    # Clear some memory
    del z0, volume

    finish_zarr(root, dst_file, multiscale, scale)

def finish_zarr(root, dst_file, multiscale, scale):
    if multiscale:
        add_multiscales(dst_file, scale=scale)
    else:
        write_multiscale_metadata(root, num_levels=1, scale=scale, name=Path(dst_file).stem)

def items_per_batch(item_nbytes, mem_limit_gb):
    return max(1, int(mem_limit_gb * 1024**3) // max(int(item_nbytes), 1))

def write_batches(dst_vol, volume, mem_limit_gb, dtype=None):
    item_nbytes = int(np.prod(volume.shape[1:])) * volume.dtype.itemsize
    batch = items_per_batch(item_nbytes, mem_limit_gb)

    for start in range(0, volume.shape[0], batch):
        end = min(start + batch, volume.shape[0])

        # Materializes memory mapped or transposed data one batch at a time
        block = np.ascontiguousarray(volume[start:end])
        dst_vol[start:end] = block if dtype is None else block.astype(dtype, copy=False)

def add_multiscales(src_file, scale=0.5):

    if not (0 < scale < 1):
        raise ValueError(f"scale must be between 0 and 1 (exclusive), got {scale}")

    # Load root node
    root = zarr.open(src_file, mode='r+')

    axes = default_axes(root['0'].ndim)
    extra_dims = len(axes) - 3
    volume_shape = root['0'].shape
    spatial_shape = np.array(volume_shape[-3:])
    spatial_chunk = np.array(root['0'].chunks[-3:])
    chunk_shape = root['0'].chunks
    shard_shape = root['0'].shards

    # Number of downscale steps until the final size fits inside a chunk.
    num_steps = max(0, int(np.floor(np.log((spatial_shape / spatial_chunk).max()) / np.log(1 / scale))))

    # Create multiscale volume
    for i in range(num_steps):

        z0 = root[str(i)]

        z1_spatial_shape = tuple(int(x * scale) for x in z0.shape[-3:])
        z1_shape = tuple(z0.shape[:extra_dims]) + z1_spatial_shape
        z1 = root.create_array(name=str(i+1),
                               shape=z1_shape,
                               chunks=chunk_shape,
                               shards=shard_shape,
                               dtype=z0.dtype,
                               dimension_names=axes,
                               overwrite=True)
        downsample_volume(z0, z1, scale=scale, block_size=shard_shape[-3], order=0)

    write_multiscale_metadata(
        root, num_levels=num_steps + 1, scale=scale, name=Path(src_file).stem)

    # Clear some memory
    del root

def write_multiscale_metadata(root, num_levels, scale=0.5, name=None):
    axes = default_axes(root['0'].ndim)
    extra_dims = len(axes) - 3

    multiscale = {
        # Datasets are ordered finest to coarsest, as the standard requires
        'axes': [{'name': axis, 'type': _AXIS_TYPES.get(axis, 'space')} for axis in axes],
        'datasets': [
            {
                'path': str(i),
                'coordinateTransformations': [
                    # Leading channel/time axes are never downsampled
                    {'type': 'scale', 'scale': [1.0] * extra_dims + [(1 / scale) ** i] * 3}
                ],
            }
            for i in range(num_levels)
        ],
    }

    if name is not None:
        multiscale['name'] = str(name)

    root.attrs['ome'] = {
        'version': '0.5',
        'multiscales': [multiscale],
    }

def downsample_volume(src_vol, dst_vol, scale=0.5, block_size=512, order=0):
    src_shape = np.array(src_vol.shape[-3:]).astype(int)
    leading = (slice(None),) * (src_vol.ndim - 3)

    for i in range(0, src_shape[0], block_size):

        i0, i1 = i, min(i + block_size, src_shape[0])
        t_i0, t_i1 = int(i0 * scale), int(i1 * scale)
        if t_i1 == t_i0:
            continue

        for j in range(0, src_shape[1], block_size):

            j0, j1 = j, min(j + block_size, src_shape[1])
            t_j0, t_j1 = int(j0 * scale), int(j1 * scale)
            if t_j1 == t_j0:
                continue

            for k in range(0, src_shape[2], block_size):

                k0, k1 = k, min(k + block_size, src_shape[2])
                t_k0, t_k1 = int(k0 * scale), int(k1 * scale)
                if t_k1 == t_k0:
                    continue

                # 1.0 for any leading (channel/time) axes -- never downsampled.
                block_scale = (1.0,) * len(leading) + (
                    (t_i1 - t_i0) / (i1 - i0),
                    (t_j1 - t_j0) / (j1 - j0),
                    (t_k1 - t_k0) / (k1 - k0),
                )
                block = src_vol[leading + (slice(i0, i1), slice(j0, j1), slice(k0, k1))]

                # scipy's zoom has no float16 kernel
                if block.dtype == np.float16:
                    resized = zoom(block.astype(np.float32), block_scale, order=order).astype(np.float16)
                else:
                    resized = zoom(block, block_scale, order=order)

                dst_vol[leading + (slice(t_i0, t_i1), slice(t_j0, t_j1), slice(t_k0, t_k1))] = resized


def convert_tiff_file(
        src_file,
        dst_file,
        input_axes=None,
        multiscale=True,
        scale=0.5,
        chunks=(64,64,64),
        shards=(256,256,256),
        mem_limit_gb=4
    ):

    src_file = Path(src_file)

    try:
        volume = tiff.memmap(src_file, mode='r')
    except (ValueError, MemoryError):
        # Compressed or otherwise non-mappable data has to be read in full
        volume = tiff.imread(src_file)

    if volume.ndim < 3:
        raise ValueError(f"'{src_file}' is {volume.ndim}D; use convert_tiff_stack() for 2D slices")

    volume = volume.transpose(transpose_order(input_axes, volume.ndim, src_file))

    root, z0 = write_level0_array(dst_file, volume.shape, volume.dtype, chunks, shards)
    write_batches(z0, volume, mem_limit_gb)

    # Clear some memory
    del z0, volume

    finish_zarr(root, dst_file, multiscale, scale)

def convert_tiff_stack(
        src_folder,
        dst_file,
        input_axes=None,
        multiscale=True,
        scale=0.5,
        chunks=(64,64,64),
        shards=(256,256,256),
        mem_limit_gb=4
    ):

    tiff_files = sorted(Path(src_folder).glob("*.tif*"))

    if not tiff_files:
        raise ValueError(f"No tiff files found in '{src_folder}'")

    first_file = tiff.imread(tiff_files[0])
    input_shape = (len(tiff_files),) + first_file.shape

    # The files are stacked along the one axis they do not cover themselves
    if input_axes is not None:
        input_axes = tuple(str(axis).lower() for axis in input_axes)
        remaining = [axis for axis in default_axes(len(input_shape)) if axis not in input_axes]
        input_axes = tuple(remaining[:1]) + input_axes

    order = transpose_order(input_axes, len(input_shape), src_folder)
    volume_shape = tuple(input_shape[i] for i in order)

    # Where the stacking dimension ends up once reordered
    stack_dim = order.index(0)

    # Largest number of files that fit in memory at once
    files_per_write = items_per_batch(first_file.nbytes, mem_limit_gb)
    volume_dtype = first_file.dtype

    # Clear some memory
    del first_file

    # Copy original resolution to first level of the multiscale volume
    root, z0 = write_level0_array(dst_file, volume_shape, volume_dtype, chunks, shards)

    for start in range(0, input_shape[0], files_per_write):
        end = min(start + files_per_write, input_shape[0])
        block = np.stack([tiff.imread(f) for f in tiff_files[start:end]], axis=0)
        z0[(slice(None),) * stack_dim + (slice(start, end),)] = block.transpose(order)

    # Clear some memory
    del z0

    finish_zarr(root, dst_file, multiscale, scale)

def write_tiff_stack(volume, dst_folder, mem_limit_gb=4):

    if not 3 <= volume.ndim <= 4:
        raise ValueError(f"Expected a 3D or 4D volume, got {volume.ndim}D")

    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    num_slices = volume.shape[-3]
    digits = len(str(max(num_slices - 1, 1)))

    slice_nbytes = int(np.prod(volume.shape[-2:])) * (volume.shape[0] if volume.ndim == 4 else 1) * volume.dtype.itemsize
    slices_per_read = items_per_batch(slice_nbytes, mem_limit_gb)

    for start in range(0, num_slices, slices_per_read):
        end = min(start + slices_per_read, num_slices)

        block = volume[(slice(None),) * (volume.ndim - 3) + (slice(start, end),)]

        # Move the channel dimension last, leaving (z, y, x, c)
        if volume.ndim == 4:
            block = np.moveaxis(block, 0, -1)

        for i, image in enumerate(block, start=start):
            tiff.imwrite(dst_folder / f'{i:0{digits}d}.tif', np.ascontiguousarray(image))

import math
import zarr
import base64
import hashlib
import tensorstore as ts

import numpy as np
import tifffile as tiff
from pathlib import Path
from urllib.parse import urlparse

# Axis order required by OME-Zarr 0.5: time, channel, then the spatial axes
OME_AXES = ('t', 'c', 'z', 'y', 'x')
MAX_NDIM = len(OME_AXES)

_AXIS_TYPES = {'t': 'time', 'c': 'channel'}

MASK_DTYPE = 'uint8'

def default_axes(ndim):
    if not 3 <= ndim <= MAX_NDIM:
        raise ValueError(f"Expected 3 to {MAX_NDIM} dimensions, got {ndim}")

    return OME_AXES[MAX_NDIM - ndim:]

def transpose_order(input_axes, ndim, source=''):
    """Permutation that reorders `input_axes` into OME-Zarr order (identity if None)."""
    axes = default_axes(ndim)

    if input_axes is None:
        return tuple(range(ndim))

    input_axes = tuple(str(axis).lower() for axis in input_axes)

    if len(input_axes) != ndim or set(input_axes) != set(axes):
        raise ValueError(
            f"input_axes {input_axes} do not describe the {ndim} dimensions of "
            f"'{source}'. Expected a permutation of {axes}")

    return tuple(input_axes.index(axis) for axis in axes)

def normalize_zarr_path(zarr_path):
    return str(zarr_path).strip().rstrip('/')

def make_ts_context(cache_size_mb=4096):
    return ts.Context({
        'cache_pool': {'total_bytes_limit': int(cache_size_mb * 1024**2)},
        'http_request_concurrency': {'limit': 32}
    })

def is_http_url(zarr_path):
    return urlparse(normalize_zarr_path(zarr_path)).scheme in ("http", "https")

def url_id(url, length=10):
    digest = hashlib.sha256(url.encode("utf-8")).digest()
    token = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return token[:length]

def volume_folder_name(zarr_ref):
    """<stem>__<hash> for local paths and <last URL component>__<hash> for URLs, unique per location."""
    zarr_ref = normalize_zarr_path(zarr_ref)

    if is_http_url(zarr_ref):
        last = urlparse(zarr_ref).path.rstrip("/").split("/")[-1] or "remote"
        return f"{last}__{url_id(zarr_ref)}"

    path = Path(zarr_ref).resolve()
    return f"{path.stem}__{url_id(str(path))}"

def resolve_zarr_inputs(data_path):
    """Zarr volumes in data_path: a single store, every store in a folder, or comma-separated http(s) URLs."""
    data_path = str(data_path).strip()
    if not data_path:
        raise ValueError("Enter a path or URL to load.")

    parts = [p.strip() for p in data_path.split(",") if p.strip()]
    if parts and all(is_http_url(p) for p in parts):
        return parts
    if len(parts) > 1:
        raise ValueError("All comma-separated paths must be http(s) URLs.")

    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if not path.is_dir():
        raise ValueError(f"Data path must be a zarr directory, a folder of zarrs, or an http(s) URL: {data_path}")

    if is_multiscale_zarr(path):
        return [str(path)]

    zarr_files = [
        str(sub) for sub in sorted(path.iterdir())
        if sub.is_dir() and is_multiscale_zarr(sub)
    ]

    if not zarr_files:
        raise ValueError(f"No zarr volumes found under {data_path}")

    return zarr_files

def get_multiscale(zarr_path):
    return zarr.open_group(normalize_zarr_path(zarr_path), mode='r').attrs['ome']['multiscales'][0]

def get_scale_paths(zarr_path):
    return [dataset['path'] for dataset in get_multiscale(zarr_path)['datasets']]

def is_multiscale_zarr(zarr_path):
    zarr_path = Path(zarr_path)

    if not (zarr_path / 'zarr.json').is_file():
        return False

    try:
        get_scale_paths(zarr_path)
    except (KeyError, IndexError):
        return False

    return True

def _create_array(root, name, shape, dtype, chunks, shards=None):
    extra_dims = len(shape) - 3
    return root.create_array(
        name=name,
        shape=shape,
        chunks=(1,) * extra_dims + tuple(chunks),
        shards=(1,) * extra_dims + tuple(shards) if shards else None,
        dtype=dtype,
        dimension_names=default_axes(len(shape)),
        overwrite=True)

def create_empty_zarr(zarr_path, level_shapes, scales, translations, chunks=(64,64,64), shards=(256,256,256), dtype=MASK_DTYPE):

    root = zarr.open(zarr_path, mode='w')

    for i, shape in enumerate(level_shapes):
        _create_array(root, str(i), shape, dtype, chunks, shards)

    write_multiscale_metadata(root, scales, translations, name=Path(zarr_path).stem)

def read_zarr_as_tensorstore(zarr_path, axes=None, ts_context=None, cache_size_mb=4096, recheck=False):

    zarr_path = normalize_zarr_path(zarr_path)

    if ts_context is None:
        ts_context = make_ts_context(cache_size_mb)

    if is_http_url(zarr_path):
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
        "recheck_cached_data": recheck,
    }, context=ts_context).result()

    if data.ndim < 3:
        raise ValueError(f"Expected at least 3D data, got shape {data.shape}")

    if axes is not None:
        data = data.transpose(transpose_order(axes, data.ndim, source=zarr_path))

    # 4D/5D volumes show their first timepoint and channel
    return data[(0,) * (data.ndim - 3) + (slice(None),) * 3]

def read_multiscale_zarr(zarr_path, ts_context=None, cache_size_mb=4096, recheck=False):

    zarr_path = normalize_zarr_path(zarr_path)

    multiscale = get_multiscale(zarr_path)
    axes = tuple(str(axis['name']).lower() for axis in multiscale['axes'])
    spatial = list(transpose_order(axes, len(axes))[-3:])

    images, scales, translations = [], [], []
    for dataset in multiscale['datasets']:
        images.append(read_zarr_as_tensorstore(f"{zarr_path}/{dataset['path']}", axes=axes, ts_context=ts_context, cache_size_mb=cache_size_mb, recheck=recheck))
        scale, translation = _scale_translation(dataset['coordinateTransformations'] + multiscale.get('coordinateTransformations', []), len(axes))
        scales.append(scale[spatial])
        translations.append(translation[spatial])

    return images, np.array(scales), np.array(translations)

def _scale_translation(transformations, ndim):
    scale, translation = np.ones(ndim), np.zeros(ndim)
    for transformation in transformations:
        if transformation['type'] == 'scale':
            scale, translation = scale * transformation['scale'], translation * transformation['scale']
        elif transformation['type'] == 'translation':
            translation = translation + transformation['translation']
    return scale, translation

def read_multiscale_masks(mask_path, level_shapes, scales, translations, ts_context=None):

    mask_path = Path(mask_path)

    mask_path.parent.mkdir(parents=True, exist_ok=True)

    if not mask_path.exists():
        create_empty_zarr(
            zarr_path=str(mask_path),
            level_shapes=level_shapes,
            scales=scales,
            translations=translations,
            dtype=MASK_DTYPE)

    return [
        read_zarr_as_tensorstore(str(mask_path / str(level)), ts_context=ts_context, recheck="open")
        for level in range(len(level_shapes))
    ]

def remap_labels(root, lut):
    # Every level of every mask under root, rewriting only the shards that exist on disk
    for level in Path(root).glob('*/[0-9]*'):
        array = zarr.open_array(str(level), mode='r+')
        for shard in level.glob('c/*/*/*'):
            index = map(int, shard.relative_to(level / 'c').parts)
            box = tuple(slice(i * n, (i + 1) * n) for i, n in zip(index, array.shards))
            array[box] = lut[array[box]]

def write_level0_array(dst_file, shape, dtype, chunks, shards=None):
    root = zarr.open(dst_file, mode='w')
    return root, _create_array(root, '0', shape, dtype, chunks, shards)

def write_zarr(volume, dst_file, dtype=None, multiscale=True, voxel_size=(1.0, 1.0, 1.0), chunks=(64,64,64), shards=(256,256,256), mem_limit_gb=4):
    volume_dtype = volume.dtype if dtype is None else dtype

    root, z0 = write_level0_array(dst_file, volume.shape, volume_dtype, chunks, shards)
    write_batches(z0, volume, mem_limit_gb, dtype=volume_dtype)

    finish_zarr(root, dst_file, multiscale, voxel_size)

def finish_zarr(root, dst_file, multiscale, voxel_size):
    if multiscale:
        add_multiscales(dst_file, scale=voxel_size)
    else:
        write_multiscale_metadata(root, [voxel_size], name=Path(dst_file).stem)

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

def add_multiscales(src_file, scale=(1.0, 1.0, 1.0), translation=(0.0, 0.0, 0.0)):

    root = zarr.open(src_file, mode='r+')

    z0 = root['0']
    spatial_chunks = z0.chunks[-3:]
    spatial_shards = z0.shards[-3:] if z0.shards else None

    # Number of downscale steps until the final size fits inside a chunk
    num_steps = max(0, int(np.floor(np.log2((np.array(z0.shape[-3:]) / spatial_chunks).max()))))

    for i in range(num_steps):
        src = root[str(i)]
        dst_shape = src.shape[:-3] + tuple(s // 2 for s in src.shape[-3:])
        dst = _create_array(root, str(i + 1), dst_shape, src.dtype, spatial_chunks, spatial_shards)
        downsample_volume(src, dst, block_size=(spatial_shards or spatial_chunks)[0])

    write_multiscale_metadata(
        root, [np.multiply(scale, 2 ** i) for i in range(num_steps + 1)], [translation] * (num_steps + 1), name=Path(src_file).stem)

def write_multiscale_metadata(root, scales, translations=None, *, name):
    axes = default_axes(root['0'].ndim)
    extra_dims = len(axes) - 3
    translations = np.zeros_like(scales, dtype=float) if translations is None else translations

    multiscale = {
        'axes': [{'name': axis, 'type': _AXIS_TYPES.get(axis, 'space')} for axis in axes],
        'datasets': [
            {
                'path': str(i),
                'coordinateTransformations': [
                    # Leading channel/time axes are never downsampled
                    {'type': 'scale', 'scale': [1.0] * extra_dims + [float(s) for s in scale]},
                    {'type': 'translation', 'translation': [0.0] * extra_dims + [float(t) for t in translation]},
                ],
            }
            for i, (scale, translation) in enumerate(zip(scales, translations))
        ],
        'name': str(name),
    }

    root.attrs['ome'] = {
        'version': '0.5',
        'multiscales': [multiscale],
    }

def downsample_volume(src_vol, dst_vol, block_size=512):
    """Halves each spatial axis by keeping every second voxel, one block at a time."""
    leading = (slice(None),) * (src_vol.ndim - 3)
    shape = dst_vol.shape[-3:]
    step = max(1, block_size // 2)

    for index in np.ndindex(*(math.ceil(n / step) for n in shape)):
        dst = tuple(slice(i * step, min((i + 1) * step, n)) for i, n in zip(index, shape))
        src = tuple(slice(2 * s.start, 2 * s.stop, 2) for s in dst)
        dst_vol[leading + dst] = src_vol[leading + src]


def convert_tiff_file(
        src_file,
        dst_file,
        input_axes=None,
        multiscale=True,
        voxel_size=(1.0, 1.0, 1.0),
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
        raise ValueError(f"'{src_file}' is {volume.ndim}D. Use convert_tiff_stack() for 2D slices")

    volume = volume.transpose(transpose_order(input_axes, volume.ndim, src_file))
    write_zarr(volume, dst_file, multiscale=multiscale, voxel_size=voxel_size, chunks=chunks, shards=shards, mem_limit_gb=mem_limit_gb)

def convert_tiff_stack(
        src_folder,
        dst_file,
        input_axes=None,
        multiscale=True,
        voxel_size=(1.0, 1.0, 1.0),
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

    stack_dim = order.index(0)

    files_per_write = items_per_batch(first_file.nbytes, mem_limit_gb)

    root, z0 = write_level0_array(dst_file, volume_shape, first_file.dtype, chunks, shards)

    for start in range(0, input_shape[0], files_per_write):
        end = min(start + files_per_write, input_shape[0])
        block = np.stack([tiff.imread(f) for f in tiff_files[start:end]], axis=0)
        z0[(slice(None),) * stack_dim + (slice(start, end),)] = block.transpose(order)

    finish_zarr(root, dst_file, multiscale, voxel_size)

def write_tiff_stack(volume, dst_folder, mem_limit_gb=1):

    if not 3 <= volume.ndim <= 4:
        raise ValueError(f"Expected a 3D or 4D volume, got {volume.ndim}D")

    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    num_slices = volume.shape[-3]
    digits = len(str(max(num_slices - 1, 1)))

    slice_nbytes = int(np.prod(volume.shape[-2:])) * (volume.shape[0] if volume.ndim == 4 else 1) * volume.dtype.itemsize
    slices_per_read = items_per_batch(slice_nbytes, mem_limit_gb)

    chunks = getattr(volume, 'chunks', None)
    if chunks is not None:
        slices_per_read = max(1, min(slices_per_read, chunks[-3]))

    for start in range(0, num_slices, slices_per_read):
        end = min(start + slices_per_read, num_slices)

        block = volume[(slice(None),) * (volume.ndim - 3) + (slice(start, end),)]

        if volume.ndim == 4:
            block = np.moveaxis(block, 0, -1)

        for i, image in enumerate(block, start=start):
            tiff.imwrite(
                dst_folder / f'{i:0{digits}d}.tif', np.ascontiguousarray(image),
                photometric='minisblack', planarconfig='contig'
            )

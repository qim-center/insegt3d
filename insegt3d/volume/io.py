import zarr
import tensorstore as ts

import numpy as np
import tifffile as tiff
from pathlib import Path
from scipy.ndimage import zoom
from urllib.parse import urlparse

def get_levels(zarr_path):
    
    if str(zarr_path).startswith(("http://", "https://")):
        store = zarr_path
    else:
        store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r')

    levels = np.sort(np.array(list(root.array_keys())).astype(int))

    return levels

def create_empty_zarr(zarr_path, shapes, chunks=(64,64,64), shards=(256,256,256), dtype='uint8'):

    # Create zarr
    root = zarr.open(zarr_path, mode='w')

    # Create scales
    for i in range(len(shapes)):
        root.create_array(
            name=str(i),
            shape=shapes[i],
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            overwrite=True)
        
def read_zarr_as_tensorstore(zarr_path, ts_context=None, cache_size_mb=4096):
    """Open a local or remote Zarr store (v3 first, then v2)."""

    if ts_context is None:
        ts_context = {
            'cache_pool': {'total_bytes_limit': int(cache_size_mb * 1024**2)}
        }

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

    # Try Zarr v3 first
    try:
        return ts.open({
            "driver": "zarr3",
            "kvstore": kvstore,
            "context": ts_context,
            "recheck_cached_data": False,
        }).result()
    except Exception:
        pass

    # Fallback to Zarr v2
    data = ts.open({
        "driver": "zarr",
        "kvstore": kvstore,
        "context": ts_context,
        "recheck_cached_data": False,
    }).result()

    
    if data.ndim < 3:
        raise ValueError(f"Expected at least 3D data, got shape {data.shape}")
    data = data[(0,) * (data.ndim - 3) + (slice(None),) * 3]
    return data

def read_multiscale_zarr(zarr_path, mask_path=None, ts_context=None, cache_size_mb=4096):
    
    # Get levels in multiscale zarr
    levels = get_levels(zarr_path)

    # Get number of levels in multiscale zarr
    num_levels = len(levels)

    images = []

    # Create multiscale image list
    for level in levels:
        path = f"{zarr_path}/{str(level)}"
        image = read_zarr_as_tensorstore(path, ts_context=ts_context, cache_size_mb=cache_size_mb)
        images.append(image)

    if mask_path is not None:

        mask_path = Path(mask_path)

        # Ensure masks folder exists
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        masks = []
        
        # Create mask Zarr if it doesn't exist.
        if not mask_path.exists():
            shapes = [image.shape for image in images]
            create_empty_zarr(
                zarr_path=str(mask_path),
                shapes=shapes,
                dtype='uint8')
        
        # Create multiscale mask list
        for level in range(num_levels):
            path = str(Path(mask_path) / str(level))
            mask = read_zarr_as_tensorstore(path, ts_context=ts_context)
            masks.append(mask)
        
        return images, masks

    return images

def create_multiscale_zarr_from_tiff_stack(
        src_folder, 
        dst_file, 
        dtype=None, 
        v_min=None, 
        v_max=None, 
        scale=0.5, 
        chunks=(64,64,64), 
        shards=(256,256,256), 
        slices_per_write=64
    ):

    tiff_files = sorted(Path(src_folder).glob("*.tif*"))

    first_slice = tiff.imread(tiff_files[0])
    slice_shape = first_slice.shape
    volume_dtype = first_slice.dtype if dtype is None else dtype
    volume_shape = (len(tiff_files),) + slice_shape

    # Clear some memory
    del first_slice

    # If v_min/v_max are provided, we will clip/normalize before writing.
    # For integer output dtypes, normalization maps to the full dtype range.
    if (v_min is not None) and (v_max is not None) and (v_max == v_min):
        raise ValueError('v_max must be different from v_min')

    # Copy original resolution to first level of the multiscale volume
    root = zarr.open(dst_file, mode='w')
    z0 = root.create_array(name='0',
                        shape=volume_shape,
                        chunks=chunks,
                        shards=shards,
                        dtype=volume_dtype,
                        overwrite=True)

    for start in range(0, volume_shape[0], slices_per_write):
        end = min(start + slices_per_write, volume_shape[0])
        block = np.stack([tiff.imread(f) for f in tiff_files[start:end]], axis=0)

        if (v_min is not None) and (v_max is not None):
            block = block.astype(np.float32, copy=False)
            np.clip(block, v_min, v_max, out=block)
            block -= v_min
            block /= (v_max - v_min)

            if np.issubdtype(np.dtype(volume_dtype), np.integer):
                info = np.iinfo(np.dtype(volume_dtype))
                block = (block * info.max).round().astype(volume_dtype)
            else:
                block = block.astype(volume_dtype, copy=False)

            z0[start:end] = block
        else:
            z0[start:end] = block.astype(volume_dtype, copy=False)

    # Clear some memory
    del root, z0

    add_multiscales(dst_file, scale=scale)

def create_multiscale_zarr_from_numpy(volume, dst_file, scale=0.5, chunks=(64,64,64), shards=(256,256,256)):

    # Copy original resolution to first level of the multiscale volume
    root = zarr.open(dst_file, mode='w')
    z0 = root.create_array(name='0',
                        shape=volume.shape,
                        chunks=chunks,
                        shards=shards,
                        dtype=volume.dtype,
                        overwrite=True)
    z0[:] = volume

    # Clear some memory
    del root, z0, volume

    add_multiscales(dst_file, scale=scale)

def add_multiscales(src_file, scale=0.5):

    # Load root node 
    root = zarr.open(src_file, mode='r+')

    volume_shape = root['0'].shape
    chunk_shape = root['0'].chunks
    shard_shape = root['0'].shards
    
    # Number of downscale steps until the final size fits inside a chunk
    num_steps = int(np.floor(np.log((np.array(volume_shape) / np.array(chunk_shape)).max()) / np.log(1 / scale)))

    # Create multiscale volume
    for i in range(num_steps):
        
        z0 = root[str(i)]
        
        z1_shape = tuple(int(x * scale) for x in z0.shape)
        z1 = root.create_array(name=str(i+1),
                               shape=z1_shape,
                               chunks=chunk_shape,
                               shards=shard_shape,
                               dtype=z0.dtype,
                               overwrite=True)
        resize_volume(z0, z1, scale=scale, block_size=shard_shape[0], order=0)
        
    # Clear some memory
    del root, z0, z1

def resize_volume(src_vol, dst_vol, scale=0.5, block_size=512, order=0):
    
    src_shape = np.array(src_vol.shape).astype(int)
    
    for i in range(0, src_shape[0], block_size):
            
        i0, i1 = i, min(i + block_size, src_shape[0])
        t_i0, t_i1 = int(i0 * scale), int(i1 * scale)
        
        for j in range(0, src_shape[1], block_size):
            
            j0, j1 = j, min(j + block_size, src_shape[1])
            t_j0, t_j1 = int(j0 * scale), int(j1 * scale)
            
            for k in range(0, src_shape[2], block_size):
                
                k0, k1 = k, min(k + block_size, src_shape[2])
                t_k0, t_k1 = int(k0 * scale), int(k1 * scale)

                dst_vol[t_i0:t_i1, t_j0:t_j1, t_k0:t_k1] = zoom(src_vol[i0:i1, j0:j1, k0:k1], scale, order=order)
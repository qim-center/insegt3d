import zarr
import time
import shutil
import tempfile
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from insegt3d.volume.io import get_scale_paths, read_multiscale_zarr, write_level0_array, add_multiscales, write_tiff_stack, normalize_zarr_path, volume_folder_name
from insegt3d.volume.intensity import robust_percentile_range

TEMP_CHUNK = 128
OUTPUT_CHUNKS = (64, 64, 64)
OUTPUT_SHARDS = (256, 256, 256)


class PredictionCancelled(Exception):

    def __init__(self, volume_name=None):
        super().__init__(f"Prediction cancelled during volume: {volume_name}" if volume_name else "Prediction cancelled")
        self.volume_name = volume_name


def predict_block(model, block, num_classes=2, batch_size=8, axes=(0,1,2)):

    input_size = block.shape[0]

    device = next(model.parameters()).device
    slab_size = model.num_channels

    offsets = torch.arange(slab_size) - slab_size // 2
    lo, hi = robust_percentile_range(block)

    block = torch.from_numpy(block.astype(np.float32, copy=False))
    block_prediction = np.zeros((num_classes, input_size, input_size, input_size), dtype=np.float32)

    for axis in axes:

        with torch.inference_mode(), torch.autocast(device.type):

            block_t = torch.moveaxis(block, axis, 0)

            for i in range(0, input_size, batch_size):

                start, stop = max(0, i - slab_size // 2), min(input_size, i + batch_size + slab_size // 2)
                slices = ((block_t[start:stop].to(device=device, dtype=torch.float32) - lo) / (hi - lo)).clamp_(0.0, 1.0)

                # Each slice gets its slab_size neighbours as channels, clamped at the block edges
                idx = torch.arange(i, min(i + batch_size, input_size))[:, None] + offsets
                batch = slices[(idx.clamp(0, input_size - 1) - start).to(device)]

                batch_prediction = model(batch).float().cpu().numpy()

                # Move the batch dimension back to the sliced axis, giving (C, Z, Y, X)
                if axis == 0:
                    block_prediction[:, i:i+batch_size] += batch_prediction.transpose(1, 0, 2, 3)
                elif axis == 1:
                    block_prediction[:, :, i:i+batch_size] += batch_prediction.transpose(1, 2, 0, 3)
                elif axis == 2:
                    block_prediction[:, :, :, i:i+batch_size] += batch_prediction.transpose(1, 2, 3, 0)

    block_prediction /= len(axes)

    return block_prediction

def predict_volume(zarr_file, prediction_file, temp_folder, model, batch_size, input_size=512, num_classes=2, level=0, overlap=0.25, axes=(0,1,2), tiff_folder=None, progress_callback=None, cancel_event=None):

    start_time = time.time()

    zarr_name = prediction_file.name

    images, scales, translations = read_multiscale_zarr(zarr_file, cache_size_mb=512)
    volume = images[level]
    input_volume_shape = np.array(volume.shape)
    spatial_shape = tuple(input_volume_shape.astype(int).tolist())
    # Blocks are cubic in world space and resampled to input_size voxels per side
    block_shape = tuple(int(n) for n in np.maximum(1, np.round(input_size * scales[level].min() / scales[level])))
    window = gaussian_3d(block_shape)

    output_volume_shape = (num_classes,) + spatial_shape

    label_file = prediction_file / 'labels'

    temp_folder.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(dir=temp_folder))

    try:
        # The extra last channel accumulates the blending weights
        pred_root, pred = write_level0_array(
            str(scratch / 'pred.zarr'), (num_classes + 1,) + spatial_shape, 'float16', (TEMP_CHUNK,) * 3)

        block_coords, padded_block_coords, local_block_coords = get_block_coordinates(input_volume_shape, input_size=block_shape, overlap=overlap, align=TEMP_CHUNK)
        num_blocks = len(padded_block_coords)

        print(f'\nSegmenting {zarr_name}...')
        # The next block is read while the current one is predicted
        with ThreadPoolExecutor(max_workers=1) as reader:
            next_block = reader.submit(get_padded_block, volume, *padded_block_coords[0])

            for i in tqdm(range(num_blocks)):

                if cancel_event is not None and cancel_event.is_set():
                    raise PredictionCancelled(zarr_name)

                padded_block = next_block.result()
                if i + 1 < num_blocks:
                    next_block = reader.submit(get_padded_block, volume, *padded_block_coords[i + 1])

                block = resample(padded_block, (input_size,) * 3, 'trilinear')
                while True:
                    try:
                        prediction = predict_block(model, block, num_classes=num_classes, batch_size=batch_size, axes=axes)
                        break
                    except RuntimeError as e:
                        if "out of memory" not in str(e).lower() or batch_size == 1:
                            raise
                        batch_size //= 2
                        tqdm.write(f'Out of memory, retrying with batch size {batch_size}.')
                predicted_block = resample(prediction, block_shape, 'area')

                i0, j0, k0, i1, j1, k1 = block_coords[i]
                l_i0, l_j0, l_k0, l_i1, l_j1, l_k1 = local_block_coords[i]

                # Blend overlapping blocks with a Gaussian window that down-weights block edges
                weights = window[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1]
                windowed = predicted_block[:, l_i0:l_i1, l_j0:l_j1, l_k0:l_k1]
                windowed *= weights

                pred[:num_classes, i0:i1, j0:j1, k0:k1] += windowed
                pred[num_classes, i0:i1, j0:j1, k0:k1] += weights

                if progress_callback is not None:
                    completed = i + 1
                    elapsed = time.time() - start_time
                    eta_seconds = (elapsed / completed) * (num_blocks - completed)
                    progress_callback(completed, num_blocks, eta_seconds)

        del volume, images

        print('Postprocessing predictions...')

        prediction_file.mkdir(parents=True, exist_ok=True)
        root, final_predictions = write_level0_array(
            str(prediction_file), output_volume_shape, 'uint8',
            OUTPUT_CHUNKS, OUTPUT_SHARDS)
        label_root, final_labels = write_level0_array(
            str(label_file), spatial_shape, 'uint8',
            OUTPUT_CHUNKS, OUTPUT_SHARDS)

        # Normalize by the accumulated weights and take the argmax, one shard at a time
        eps = 1e-3
        for i0, j0, k0, i1, j1, k1 in get_shard_coordinates(input_volume_shape, shard_size=OUTPUT_SHARDS[0]):
            if cancel_event is not None and cancel_event.is_set():
                raise PredictionCancelled(zarr_name)

            p = pred[:, i0:i1, j0:j1, k0:k1].astype('float32')
            scores = 255 * p[:num_classes] / np.maximum(p[num_classes], eps)
            final_predictions[:, i0:i1, j0:j1, k0:k1] = np.clip(np.rint(scores), 0, 255).astype('uint8')
            final_labels[i0:i1, j0:j1, k0:k1] = (p[:num_classes].argmax(axis=0) + 1).astype('uint8')

        del pred, final_predictions, final_labels, root, pred_root, label_root

        add_multiscales(str(prediction_file), scale=scales[level], translation=translations[level])
        add_multiscales(str(label_file), scale=scales[level], translation=translations[level])

        if tiff_folder is not None:
            print('Writing tiff stack...')
            pred_level0 = get_scale_paths(prediction_file)[0]
            write_tiff_stack(zarr.open(str(prediction_file), mode='r')[pred_level0], tiff_folder)

    except BaseException:
        # Remove the partial output, including on cancellation
        if prediction_file.exists():
            shutil.rmtree(prediction_file)
        if tiff_folder is not None and tiff_folder.exists():
            shutil.rmtree(tiff_folder)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    time_elapsed = time.time() - start_time
    print(f'Completed volume {zarr_name} {tuple(input_volume_shape.astype(int).tolist())} in {time_elapsed}.')

    return batch_size


def predict_all_volumes(zarr_files, model, level, predictions_dir, temp_dir, input_size=512, batch_size=None, overlap=0.25, axes=(0,1,2), export_tiff=False, progress_callback=None, cancel_event=None):

    torch.set_float32_matmul_precision('medium')
    model.eval()

    # predict_volume halves the batch size until it fits in memory, and returns the size that fit
    batch_size = batch_size or input_size

    num_classes = model.num_classes
    predictions_dir, temp_dir = Path(predictions_dir), Path(temp_dir)

    try:
        num_volumes = len(zarr_files)
        skipped = []

        for vol_idx, zarr_file in enumerate(zarr_files):
            zarr_file = normalize_zarr_path(zarr_file)
            prediction_file = predictions_dir / volume_folder_name(zarr_file)
            zarr_name = prediction_file.name

            if len(get_scale_paths(zarr_file)) <= level:
                print(f'Skipping {zarr_name}: no level {level}.')
                skipped.append(zarr_name)
                continue

            if cancel_event is not None and cancel_event.is_set():
                raise PredictionCancelled()

            def volume_progress(block_idx, num_blocks, eta_seconds, vol_idx=vol_idx, name=zarr_name):
                if progress_callback is not None:
                    progress_callback(name, vol_idx, num_volumes, block_idx, num_blocks, eta_seconds)

            batch_size = predict_volume(
                zarr_file=zarr_file,
                prediction_file=prediction_file,
                temp_folder=temp_dir,
                model=model,
                batch_size=batch_size,
                input_size=input_size,
                num_classes=num_classes,
                level=level,
                overlap=overlap,
                axes=axes,
                tiff_folder=predictions_dir / f"{zarr_name.removesuffix('.zarr')}_tiff" if export_tiff else None,
                progress_callback=volume_progress,
                cancel_event=cancel_event
            )

        if skipped:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            (predictions_dir / 'skipped_volumes.txt').write_text(
                ''.join(f'{name}: no level {level}\n' for name in skipped))

        print('\nAll volumes segmented.\n')
    finally:
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()


def get_padded_block(volume, i0, j0, k0, i1, j1, k1):

    D, H, W = volume.shape
    block = np.asarray(volume[max(i0, 0):min(i1, D), max(j0, 0):min(j1, H), max(k0, 0):min(k1, W)])
    padding = ((max(0, -i0), max(0, i1 - D)), (max(0, -j0), max(0, j1 - H)), (max(0, -k0), max(0, k1 - W)))
    return np.pad(block, padding, mode='reflect')

def get_shard_coordinates(volume_shape, shard_size=128):
    starts = [np.arange(0, s, shard_size) for s in volume_shape]
    chunk_coordinates = np.stack(np.meshgrid(*starts, indexing='ij'), -1).reshape(-1, 3)
    chunk_coordinates = np.concatenate([chunk_coordinates,np.minimum(chunk_coordinates + shard_size, volume_shape)], axis=1)
    return chunk_coordinates

def resample(array, shape, mode):
    if array.shape[-3:] == tuple(shape):
        return array
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    resampled = F.interpolate(tensor.reshape((1, -1) + tensor.shape[-3:]), size=tuple(shape), mode=mode)
    return resampled.reshape(tensor.shape[:-3] + tuple(shape)).numpy()

def gaussian_3d(shape, sigma=0.125, eps=1e-3):
    """Separable Gaussian blending window, clipped at eps so no voxel gets zero weight."""

    g = []
    for size in shape:
        coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        profile = np.exp(-(coords**2) / (2 * (sigma * size)**2)).astype(np.float32)
        g.append(profile / profile.max())

    gaussian = g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :]
    return np.maximum(gaussian, eps, out=gaussian)

def get_block_coordinates(volume_shape, input_size=256, overlap=0.25, align=1):
    """
    Overlapping blocks covering the volume, centred so the overshoot is split between both ends.
    Returns their bounds clipped to the volume, unclipped, and the clipped bounds relative to each block's start.
    """

    input_size = np.broadcast_to(input_size, 3)
    step = input_size * (1 - overlap)
    blocks_per_axis = np.maximum(1, np.ceil((volume_shape - overlap * input_size) / step)).astype(int)
    padded_volume_shape = np.round(blocks_per_axis * input_size - (blocks_per_axis - 1) * input_size * overlap).astype(int)

    padding_shift = (padded_volume_shape - volume_shape) // 2 // align * align

    starts = np.floor(np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(blocks_per_axis, step)], indexing='ij'), -1).reshape(-1, 3) - padding_shift).astype(int)
    padded_block_coords = np.concatenate([starts, starts + input_size], axis=1)

    block_coords = np.concatenate([np.maximum(padded_block_coords[:, :3], 0), np.minimum(padded_block_coords[:, 3:], volume_shape)], axis=1)

    local_block_coords = block_coords - np.tile(padded_block_coords[:, :3], 2)

    return block_coords, padded_block_coords, local_block_coords

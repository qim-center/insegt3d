import zarr
import time
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch

from insegt3d.volume.io import get_scale_paths, write_level0_array, add_multiscales, write_multiscale_metadata, write_tiff_stack
from insegt3d.volume.intensity import robust_normalize


class PredictionCancelled(Exception):
    """Raised to unwind out of a prediction run when cancellation is requested."""

    def __init__(self, volume_name=None):
        super().__init__(f"Prediction cancelled during volume: {volume_name}" if volume_name else "Prediction cancelled")
        self.volume_name = volume_name


def find_max_batch_size(model, input_size=256, start=4, max_limit=512):

    batch_size = start
    best = start

    device = next(model.parameters()).device

    while batch_size <= max_limit:
        try:
            with torch.inference_mode():
                # Make a fake batch to test memory use
                test_batch = torch.zeros(
                    (batch_size, 1, input_size, input_size),
                    dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    device=device
                )
                _ = model(test_batch)

            best = batch_size
            batch_size *= 2  # Try next larger

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            break  # Too big, stop searching

    del test_batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return best

def predict_block(model, block, num_classes=2, batch_size=8, axes=(0,1,2)):

    input_size = block.shape[0]

    device = next(model.parameters()).device

    block_prediction = np.zeros((input_size, input_size, input_size, num_classes), dtype=np.float32)

    for axis in axes:

        with torch.inference_mode():

            block_t = torch.moveaxis(block, axis, 0)

            for i in range(0, input_size, batch_size):

                batch = block_t[i:i+batch_size].unsqueeze(1)

                dtype = torch.float16 if device.type == "cuda" else torch.float32
                batch = batch.to(device=device, dtype=dtype)

                batch_prediction = model(batch)
                batch_prediction = batch_prediction.permute(0, 2, 3, 1).float().cpu().numpy()

                # Accumulate predictions into correct orientation depending on axis
                if axis == 0:   # Z axis
                    block_prediction[i:i+batch_size, :, :, :] += batch_prediction
                elif axis == 1: # Y axis
                    block_prediction[:, i:i+batch_size, :, :] += batch_prediction.transpose(1, 0, 2, 3)
                elif axis == 2: # X axis
                    block_prediction[:, :, i:i+batch_size, :] += batch_prediction.transpose(1, 2, 0, 3)

    block_prediction /= len(axes)

    return block_prediction

def setup_model(model_path, input_size=512, batch_size=None):

    torch.set_float32_matmul_precision('medium')

    # Get CUDA device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    model = checkpoint['model'].to(device)
    model.eval()

    if device.type == "cuda":
        model = model.half()

    if batch_size is None:
        batch_size = find_max_batch_size(model, input_size=input_size, start=4, max_limit=input_size)
        print(f'Found optimal inference batch size of {batch_size}.')
    else:
        print(f'Using batch size of {batch_size}.')

    return model, batch_size, checkpoint['num_classes']


def predict_volume(zarr_file, prediction_file, temp_folder, model, window, input_size=512, num_classes=2, batch_size=None, overlap=0.25, axes=(0,1,2), tiff_folder=None, progress_callback=None, cancel_event=None):

    if batch_size is None:
        raise ValueError("batch_size must be provided")
    if model is None:
        raise ValueError("model must be provided")
    if window is None:
        window = gaussian_3d(input_size, sigma=0.125).astype('float32')

    start_time = time.time()

    level0 = get_scale_paths(zarr_file)[0]
    volume = zarr.open(str(zarr_file), mode='r')[level0] # Load highest resolution
    chunk_size = volume.chunks[-3]
    shard_size = volume.shards[-3]
    input_volume_shape = np.array(volume.shape[-3:])
    spatial_shape = tuple(input_volume_shape.astype(int).tolist())
    spatial_chunks = (chunk_size, chunk_size, chunk_size)
    spatial_shards = (shard_size, shard_size, shard_size)

    output_volume_shape = (num_classes,) + spatial_shape
    
    label_file = prediction_file / 'labels'

    # Ensure temp directory exists and is clean
    temp_folder.mkdir(parents=True, exist_ok=True)
    pred_folder = temp_folder / 'pred.zarr'
    if pred_folder.is_dir():
        shutil.rmtree(pred_folder)

    try:
        # Initialize temporary prediction volume.
        pred_root, pred = write_level0_array(
            str(pred_folder), output_volume_shape, 'float16',
            spatial_chunks, spatial_shards)
        write_multiscale_metadata(pred_root, num_levels=1, name='pred')

        # Get block coordinates
        block_coords, padded_block_coords, local_block_coords = get_block_coordinates(input_volume_shape, input_size=input_size, overlap=overlap)
        num_blocks = len(padded_block_coords)

        print(f'\nSegmenting {zarr_file.name}...')
        for i in tqdm(range(num_blocks)):

            if cancel_event is not None and cancel_event.is_set():
                raise PredictionCancelled(zarr_file.name)

            padded_block = get_padded_block(volume, *padded_block_coords[i])
            padded_block = torch.tensor(robust_normalize(padded_block))

            # predict_block() returns (D,H,W,C)
            predicted_block = predict_block(model, padded_block, num_classes=num_classes, batch_size=batch_size, axes=axes)

            i0, j0, k0, i1, j1, k1 = block_coords[i]
            l_i0, l_j0, l_k0, l_i1, l_j1, l_k1 = local_block_coords[i]

            windowed = predicted_block[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1, :] * window[l_i0:l_i1, l_j0:l_j1, l_k0:l_k1, None]

            # pred/final_predictions store (C,Z,Y,X)
            pred[:, i0:i1, j0:j1, k0:k1] += np.moveaxis(windowed, -1, 0)

            if progress_callback is not None:
                completed = i + 1
                elapsed = time.time() - start_time
                eta_seconds = (elapsed / completed) * (num_blocks - completed)
                progress_callback(completed, num_blocks, eta_seconds)

        del volume

        print('Postprocessing predictions...')

        prediction_file.mkdir(parents=True, exist_ok=True)
        root, final_predictions = write_level0_array(
            str(prediction_file), output_volume_shape, 'uint8',
            spatial_chunks, spatial_shards)
        label_root, final_labels = write_level0_array(
            str(label_file), spatial_shape, 'uint8',
            spatial_chunks, spatial_shards)

        # Normalize by shard
        eps = 1e-3
        for i0, j0, k0, i1, j1, k1 in get_shard_coordinates(input_volume_shape, shard_size=shard_size):
            if cancel_event is not None and cancel_event.is_set():
                raise PredictionCancelled(zarr_file.name)

            p = pred[:, i0:i1, j0:j1, k0:k1].astype('float32')
            w = np.maximum(compute_weight_map(window, block_coords, local_block_coords,
                                              (i0, j0, k0, i1, j1, k1)), eps)
            final_predictions[:, i0:i1, j0:j1, k0:k1] = (255 * p / w[None, ...]).astype('uint8')
            final_labels[i0:i1, j0:j1, k0:k1] = (p.argmax(axis=0) + 1).astype('uint8')

        del pred, final_predictions, final_labels, root, pred_root, label_root

        add_multiscales(str(prediction_file))
        add_multiscales(str(label_file))

        if tiff_folder is not None:
            print('Writing tiff stack...')
            pred_level0 = get_scale_paths(prediction_file)[0]
            write_tiff_stack(zarr.open(str(prediction_file), mode='r')[pred_level0], tiff_folder)

    except BaseException:
        # Remove the partial output for this volume
        if prediction_file.exists():
            shutil.rmtree(prediction_file)
        if tiff_folder is not None and tiff_folder.exists():
            shutil.rmtree(tiff_folder)
        raise
    finally:
        if temp_folder.exists():
            shutil.rmtree(temp_folder)

    time_elapsed = time.time() - start_time
    print(f'Completed volume {zarr_file.name} {tuple(input_volume_shape.astype(int).tolist())} in {time_elapsed}.')


def predict_all_volumes(zarr_files, project_path, model_path=None, predictions_dir=None, temp_dir=None, input_size=512, num_classes=None, batch_size=None, overlap=0.25, axes=(0,1,2), export_tiff=False, progress_callback=None, cancel_event=None):

    project_path = Path(project_path)

    # Load model
    model_path = Path(model_path) if model_path is not None else project_path / 'model.ckpt'
    model, batch_size, checkpoint_num_classes = setup_model(model_path, input_size=input_size, batch_size=batch_size)

    if num_classes is None:
        num_classes = checkpoint_num_classes

    try:
        # Precompute blending window for block size
        window = gaussian_3d(input_size, sigma=0.125).astype('float32')

        predictions_dir = Path(predictions_dir) if predictions_dir is not None else project_path / 'predictions'
        temp_dir = Path(temp_dir) if temp_dir is not None else project_path / 'temp'

        num_volumes = len(zarr_files)

        # Predict volumes
        for vol_idx, zarr_file in enumerate(zarr_files):
            zarr_file = Path(zarr_file)

            if cancel_event is not None and cancel_event.is_set():
                raise PredictionCancelled()

            def volume_progress(block_idx, num_blocks, eta_seconds, vol_idx=vol_idx, name=zarr_file.name):
                if progress_callback is not None:
                    progress_callback(name, vol_idx, num_volumes, block_idx, num_blocks, eta_seconds)

            predict_volume(
                zarr_file=zarr_file,
                prediction_file=predictions_dir / zarr_file.name,
                temp_folder=temp_dir,
                model=model,
                window=window,
                input_size=input_size,
                num_classes=num_classes,
                batch_size=batch_size,
                overlap=overlap,
                axes=axes,
                tiff_folder=(predictions_dir / f'{zarr_file.stem}_tiff') if export_tiff else None,
                progress_callback=volume_progress,
                cancel_event=cancel_event
            )

        print('\nAll volumes segmented.\n')
    finally:
        # Release the prediction model from GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------- Helper functions -----------------------------

def get_padded_block(volume, i0, j0, k0, i1, j1, k1):
    '''
    Extracts a block from a volume with reflection padding at the boundaries.
    '''

    volume_shape = volume.shape[-3:]
    extra_dims = volume.ndim - 3

    pad_before = [max(0, -i0), max(0, -j0), max(0, -k0)]
    pad_after  = [max(0, i1 - volume_shape[0]), max(0, j1 - volume_shape[1]), max(0, k1 - volume_shape[2])]

    # Clip indices into valid range
    c_i0, c_i1 = max(i0, 0), min(i1, volume_shape[0])
    c_j0, c_j1 = max(j0, 0), min(j1, volume_shape[1])
    c_k0, c_k1 = max(k0, 0), min(k1, volume_shape[2])

    # Load only the needed block from the zarr volume. Extra leading dims
    # (channel, time, ...) beyond the last 3 spatial ones are index 0.
    block = volume[(0,) * extra_dims + (slice(c_i0, c_i1), slice(c_j0, c_j1), slice(c_k0, c_k1))]

    # Pad to desired shape with reflection
    padding = ((pad_before[0], pad_after[0]),
               (pad_before[1], pad_after[1]),
               (pad_before[2], pad_after[2]))

    padded = np.pad(block, pad_width=padding, mode='reflect')

    return padded

def get_shard_coordinates(volume_shape, shard_size=128):
    '''
    Returns coordinates of all shards in the volume.
    '''
    starts = [np.arange(0, s, shard_size) for s in volume_shape]
    chunk_coordinates = np.stack(np.meshgrid(*starts, indexing='ij'), -1).reshape(-1, 3)
    chunk_coordinates = np.concatenate([chunk_coordinates,np.minimum(chunk_coordinates + shard_size, volume_shape)], axis=1)
    return chunk_coordinates

def compute_weight_map(window, block_coords, local_block_coords, region_coords):

    r_i0, r_j0, r_k0, r_i1, r_j1, r_k1 = region_coords

    weight = np.zeros((r_i1 - r_i0, r_j1 - r_j0, r_k1 - r_k0), dtype='float32')

    for block, local in zip(block_coords, local_block_coords):

        i0, j0, k0, i1, j1, k1 = block
        l_i0, l_j0, l_k0 = local[:3]

        # Overlap between the block and the region
        c_i0, c_i1 = max(i0, r_i0), min(i1, r_i1)
        c_j0, c_j1 = max(j0, r_j0), min(j1, r_j1)
        c_k0, c_k1 = max(k0, r_k0), min(k1, r_k1)

        if c_i0 >= c_i1 or c_j0 >= c_j1 or c_k0 >= c_k1:
            continue

        weight[c_i0-r_i0:c_i1-r_i0, c_j0-r_j0:c_j1-r_j0, c_k0-r_k0:c_k1-r_k0] += \
            window[c_i0-i0+l_i0:c_i1-i0+l_i0, c_j0-j0+l_j0:c_j1-j0+l_j0, c_k0-k0+l_k0:c_k1-k0+l_k0]

    return weight

def gaussian_3d(input_size, sigma=0.125, eps=1e-3):
    """
    Create a 3D Gaussian window for edge weighting.
    """

    # Adjust sigma based on input size
    sigma *= input_size

    # 1D Gaussian
    coords = np.arange(input_size, dtype=np.float32) - (input_size - 1) / 2.0
    g = np.exp(-(coords**2) / (2 * sigma**2)).astype(np.float32)
    g /= g.max()

    # 3D gaussian
    gaussian = g[:, None, None] * g[None, :, None] * g[None, None, :]

    # Normalize and clip
    gaussian /= gaussian.max()
    gaussian = np.clip(gaussian, max(gaussian.min(), eps), 1.0)

    return gaussian

def get_block_coordinates(volume_shape, input_size=256, overlap=0.25):

    blocks_per_axis = np.ceil((volume_shape - overlap * input_size) / (input_size - overlap * input_size)).astype(int)
    padded_volume_shape = np.round(blocks_per_axis * input_size - (blocks_per_axis - 1) * input_size * overlap).astype(int)

    padding_shift = (padded_volume_shape - volume_shape) // 2
    padding_shift = np.array(list(padding_shift) + list(padding_shift))

    block_coords = []
    padded_block_coords = []
    local_block_coords = []

    for i in range(blocks_per_axis[0]):

        p_i0 = i * input_size * (1 - overlap)
        p_i1 = p_i0 + input_size

        for j in range(blocks_per_axis[1]):

            p_j0 = j * input_size * (1 - overlap)
            p_j1 = p_j0 + input_size

            for k in range(blocks_per_axis[2]):

                p_k0 = k * input_size * (1 - overlap)
                p_k1 = p_k0 + input_size

                # padded block coords (outside of volume range)
                coords = np.array([p_i0, p_j0, p_k0, p_i1, p_j1, p_k1]) - padding_shift
                coords = coords.astype(int)
                padded_block_coords.append(coords)

                # block coords (clipped to volume)
                i0, j0, k0, i1, j1, k1 = coords
                i0_c, i1_c = max(0, i0), min(volume_shape[0], i1)
                j0_c, j1_c = max(0, j0), min(volume_shape[1], j1)
                k0_c, k1_c = max(0, k0), min(volume_shape[2], k1)
                block_coords.append([i0_c, j0_c, k0_c, i1_c, j1_c, k1_c])

                # local indices within block
                l_i0, l_i1 = i0_c - i0, i1_c - i0
                l_j0, l_j1 = j0_c - j0, j1_c - j0
                l_k0, l_k1 = k0_c - k0, k1_c - k0
                local_block_coords.append([l_i0, l_j0, l_k0, l_i1, l_j1, l_k1])

    padded_block_coords = np.array(padded_block_coords)
    block_coords = np.array(block_coords)
    local_block_coords = np.array(local_block_coords)

    return block_coords, padded_block_coords, local_block_coords

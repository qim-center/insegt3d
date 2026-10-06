import argparse
from pathlib import Path

import torch

from insegt3d.ml import predict2d
from insegt3d.ml.unet2d import UNet2D, load_checkpoint
from insegt3d.volume.io import resolve_zarr_inputs


def build_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='insegt3d predict',
        description='Run batch prediction on one or more zarr volumes using a trained checkpoint.'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to a trained model checkpoint (model.ckpt).'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='A single zarr volume, a folder containing multiple zarr volumes, or an http(s) zarr URL.'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory. Predictions are written to <output>/predictions/<volume_name>.'
    )
    parser.add_argument(
        '--input-size', type=int, default=512,
        help='Cubic block size used for inference (default: 512).'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Inference batch size. Defaults to the largest size that fits in memory.'
    )
    parser.add_argument(
        '--overlap', type=float, default=0.25,
        help='Fractional overlap between adjacent blocks (default: 0.25).'
    )
    parser.add_argument(
        '--axes', type=str, default='0,1,2',
        help='Comma-separated axes to predict along and average over, e.g. "0,1,2" for all three (default: 0,1,2).'
    )
    parser.add_argument(
        '--export-tiff', action='store_true',
        help='Also write each prediction as a folder of tiff slices at <output>/predictions/<volume_name>_tiff.'
    )
    parser.add_argument(
        '--temp-dir', type=str, default=None,
        help='Scratch directory for intermediate accumulation buffers. Defaults to <output>/temp.'
    )
    return parser


def run_predict(argv) -> None:
    parser = build_predict_parser()
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    try:
        zarr_files = resolve_zarr_inputs(args.data)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    try:
        axes = tuple(int(a) for a in args.axes.split(','))
    except ValueError:
        parser.error(f"--axes must be a comma-separated list of integers, got: {args.axes}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint)
    model = UNet2D.from_checkpoint(checkpoint).to(torch.accelerator.current_accelerator(check_available=True) or 'cpu')

    predict2d.predict_all_volumes(
        zarr_files,
        model,
        checkpoint['level'],
        predictions_dir=output / 'predictions',
        temp_dir=Path(args.temp_dir) if args.temp_dir else output / 'temp',
        input_size=args.input_size,
        batch_size=args.batch_size,
        overlap=args.overlap,
        axes=axes,
        export_tiff=args.export_tiff,
    )

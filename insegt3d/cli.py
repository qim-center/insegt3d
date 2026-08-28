import argparse
from pathlib import Path

from insegt3d.ml import predict2d
from insegt3d.volume.io import is_multiscale_zarr


def _is_http_url(path: str) -> bool:
    return path.startswith('http://') or path.startswith('https://')


def resolve_zarr_inputs(data_path: str) -> list:
    """
    Resolves --data into a list of zarr volume paths/URLs: a single zarr
    store, every zarr subfolder of a containing directory, or a single
    http(s) zarr URL.
    """
    data_path = str(data_path)

    if _is_http_url(data_path):
        return [data_path]

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if not path.is_dir():
        raise ValueError(f"Data path must be a zarr directory, a folder of zarrs, or an http(s) URL: {data_path}")

    # Is this directory itself a (multiscale) zarr store?
    if is_multiscale_zarr(path):
        return [str(path)]

    # Otherwise, treat it as a folder containing multiple zarr volumes.
    zarr_files = [
        str(sub) for sub in sorted(path.iterdir())
        if sub.is_dir() and is_multiscale_zarr(sub)
    ]

    if not zarr_files:
        raise ValueError(f"No zarr volumes found under {data_path}")

    return zarr_files


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
        '--num-classes', type=int, default=None,
        help='Number of classes. Defaults to the value stored in the checkpoint.'
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

    predict2d.predict_all_volumes(
        zarr_files,
        project_path=output,
        model_path=checkpoint,
        predictions_dir=output / 'predictions',
        temp_dir=Path(args.temp_dir) if args.temp_dir else output / 'temp',
        input_size=args.input_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        overlap=args.overlap,
        axes=axes,
        export_tiff=args.export_tiff,
    )

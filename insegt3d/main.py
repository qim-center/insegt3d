import argparse
import numpy as np
from nicegui import ui
from insegt3d.app import InteractiveSegmentationApp

def main():
    parser = argparse.ArgumentParser(
        description='Interactive U-Net Segmentation Tool'
    )
    parser.add_argument(
        '--project_folder',
        type=str,
        default=None,
        help='Location to store masks, predictions, model checkpoints, etc.'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port to run the application on'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        choices=range(2, 11),
        help='Number of classes (must be between 2 and 10)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='U-Net',
        choices=[
            "U-Net",
            "U-Net++",
            "FPN",
            "PSPNet",
            "DeepLabV3",
            "DeepLabV3+",
            "LinkNet",
            "MA-Net",
            "PAN",
            "UPerNet",
            "Segformer",
        ],
        help=(
            "Segmentation model architecture. One of: "
            "U-Net, U-Net++, FPN, PSPNet, DeepLabV3, "
            "DeepLabV3+, LinkNet, MA-Net, PAN, UPerNet, Segformer. "
        )
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='resnet34',
        help=(
            "Encoder backbone architecture. "
            "See https://smp.readthedocs.io/en/latest/encoders.html "
            "and https://smp.readthedocs.io/en/latest/encoders_timm.html "
            "for available options."
        )
    )
    args = parser.parse_args()

    # Use provided port or fall back to a random port between 20000-40000
    port = args.port if args.port else np.random.randint(20000, 40000)

    # Create the app instance (this builds the UI) - must happen before ui.run()

    @ui.page('/')
    def index():
        InteractiveSegmentationApp(args)

    # Start the server
    ui.run(port=port, show=False, reload=False)

if __name__ in {"__main__", "__mp_main__"}:
    main()
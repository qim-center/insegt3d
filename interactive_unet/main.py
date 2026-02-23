import argparse
import numpy as np
from nicegui import ui
from interactive_unet.app import InteractiveSegmentationApp

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
        choices=range(2, 11),
        required=True,
        help='Number of classes (must be between 2 and 10)'
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
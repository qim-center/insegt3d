import argparse
import numpy as np
from nicegui import ui, app
from insegt3d.app import InteractiveSegmentationApp

class StripRootPath:
    """
    ASGI middleware to strip the base path from incoming Nginx requests.
    """
    def __init__(self, asgi_app, root_path: str):
        self.asgi_app = asgi_app
        self.root_path = root_path.rstrip("/")

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path.startswith(self.root_path):
                scope["path"] = path[len(self.root_path):] or "/"
                scope["root_path"] = self.root_path
        await self.asgi_app(scope, receive, send)

def _normalize_base_path(base_path: str | None) -> str:
    if not base_path:
        return ''
    normalized = base_path.strip()
    if not normalized:
        return ''
    if not normalized.startswith('/'):
        normalized = f'/{normalized}'
    if len(normalized) > 1 and normalized.endswith('/'):
        normalized = normalized.rstrip('/')
    return normalized

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
        '--host',
        type=str,
        default='localhost',
        help='Host to run the application on (default: localhost)'
    )
    parser.add_argument(
        '--server_base_path',
        type=str,
        default='',
        help=(
            'Base path under which the app will be served. Default is root ("/")'
        )
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
    root_path = _normalize_base_path(args.server_base_path)

    if root_path:
        app.add_middleware(StripRootPath, root_path=root_path)

    @ui.page('/')
    def index():
        InteractiveSegmentationApp(args)

    # Start the server
    ui.run(host='0.0.0.0', port=port, show=False, reload=False, root_path=root_path)

if __name__ in {"__main__", "__mp_main__"}:
    main()
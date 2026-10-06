import sys
import ctypes
import socket
import argparse
from nicegui import ui, app
from insegt3d.app.app import InteractiveSegmentationApp

class StripRootPath:
    """ASGI middleware to strip the base path from requests forwarded by a reverse proxy such as Nginx."""
    def __init__(self, asgi_app, root_path: str):
        self.asgi_app = asgi_app
        self.root_path = root_path.rstrip("/")

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path == self.root_path or path.startswith(self.root_path + "/"):
                scope["path"] = path[len(self.root_path):] or "/"
                scope["root_path"] = self.root_path
        await self.asgi_app(scope, receive, send)

def _normalize_base_path(base_path: str | None) -> str:
    base_path = (base_path or '').strip().strip('/')
    return f'/{base_path}' if base_path else ''

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def _return_freed_memory_to_os():
    # glibc otherwise keeps freed chunk buffers in per-thread pools, which can grow past the cache size
    try:
        ctypes.CDLL('libc.so.6').mallopt(-3, 256 * 1024)  # M_MMAP_THRESHOLD
    except OSError:
        pass

def main():
    _return_freed_memory_to_os()
    argv = sys.argv[1:]

    if argv and argv[0] == 'predict':
        from insegt3d.cli import run_predict
        run_predict(argv[1:])
        return

    parser = argparse.ArgumentParser(
        description='Interactive U-Net Segmentation Tool',
        epilog='Run "insegt3d predict --help" for batch prediction from the command line.'
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
        '--cache_gb',
        type=float,
        default=16,
        help='Memory for cached volume data, in GiB (default: 16)'
    )
    args = parser.parse_args()

    port = args.port or _free_port()
    root_path = _normalize_base_path(args.server_base_path)

    if root_path:
        app.add_middleware(StripRootPath, root_path=root_path)

    segmentation_app = None

    def start():
        nonlocal segmentation_app
        segmentation_app = InteractiveSegmentationApp(args)

    app.on_startup(start)
    app.on_shutdown(lambda: segmentation_app.close())

    @ui.page('/')
    def index():
        segmentation_app.open_view()

    # JPEG frames barely compress, and deflating them would block the event loop
    ui.run(host=args.host, port=port, show=False, reload=False, root_path=root_path, ws_per_message_deflate=False)

if __name__ in {"__main__", "__mp_main__"}:
    main()
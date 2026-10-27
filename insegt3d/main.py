import os
import sys
import ctypes
import socket
import argparse
import asyncio
from pathlib import Path
from nicegui import ui, app
from insegt3d.app.app import InteractiveSegmentationApp
from insegt3d.app.ui.ui import setup_theme, toggle_theme
from insegt3d.auth import TokenAuth

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
    parser.add_argument(
        '--token',
        type=str,
        default=os.environ.get('INSEGT3D_TOKEN'),
        help=(
            'Require this token to access the application, given as a "token" URL parameter, an '
            '"Authorization: token ..." header or on a login page. Can also be set with the '
            'INSEGT3D_TOKEN environment variable, which keeps it out of the process list. '
            'Default is no token'
        )
    )
    args = parser.parse_args()

    port = args.port or _free_port()
    root_path = _normalize_base_path(args.server_base_path)

    # Added before StripRootPath, which then runs first and gives TokenAuth the stripped path
    if args.token:
        app.add_middleware(TokenAuth, token=args.token, cookie_name=f'insegt3d-token-{port}')

    if root_path:
        app.add_middleware(StripRootPath, root_path=root_path)

    segmentation_app = None
    transition_lock = asyncio.Lock()

    def start():
        nonlocal segmentation_app
        if args.project_folder:
            segmentation_app = InteractiveSegmentationApp(args, Path(args.project_folder).expanduser().resolve(), switch_project, shutdown_app)

    async def close_project():
        nonlocal segmentation_app
        async with transition_lock:
            if segmentation_app is not None:
                await segmentation_app.close()
                segmentation_app = None

    async def switch_project():
        await close_project()
        ui.navigate.reload()

    async def shutdown_app():
        await close_project()
        app.shutdown()

    async def stop():
        await close_project()

    app.on_startup(start)
    app.on_shutdown(stop)

    @ui.page('/')
    def index():
        if segmentation_app is not None:
            segmentation_app.open_view()
            return

        dark_mode = setup_theme()
        ui.page_title('InSegt3D · Choose a project')

        async def open_project():
            nonlocal segmentation_app
            value = (folder.value or '').strip()
            if not value:
                ui.notify('Enter a project folder path.', type='warning')
                return
            path = Path(value).expanduser().resolve()
            if path.exists() and not path.is_dir():
                ui.notify('That path is a file. Choose a folder.', type='warning')
                return
            open_button.disable()
            try:
                async with transition_lock:
                    if segmentation_app is None:
                        path.mkdir(parents=True, exist_ok=True)
                        segmentation_app = InteractiveSegmentationApp(args, path, switch_project, shutdown_app)
            except Exception as e:
                ui.notify(f'Could not open project: {e}', type='negative')
                open_button.enable()
                return
            ui.navigate.reload()

        with ui.element('main').classes('project-start'):
            ui.button(icon='contrast', on_click=lambda: toggle_theme(dark_mode)).props('flat round').classes('project-theme-button').tooltip('Switch between light and dark mode')
            with ui.column().classes('project-start-card'):
                ui.label('InSegt3D').classes('project-start-brand')
                ui.label('Start with a project').classes('project-start-title')
                ui.label('A project keeps your annotations, regions, model and predictions together. Open an existing folder or create a new one.').classes('project-start-copy')
                folder = ui.input('Project folder', placeholder=str(Path.cwd() / 'my_project')).props('outlined').classes('w-full')
                folder.on('keydown.enter', open_project)
                ui.label('Enter a folder path on the computer running InSegt3D. New folders are created automatically.').classes('project-start-hint')
                open_button = ui.button('Open project', icon='arrow_forward', on_click=open_project).props('unelevated no-caps').classes('project-start-button w-full')

    # JPEG frames barely compress, and deflating them would block the event loop
    ui.run(host=args.host, port=port, show=False, reload=False, root_path=root_path, ws_per_message_deflate=False)

if __name__ in {"__main__", "__mp_main__"}:
    main()

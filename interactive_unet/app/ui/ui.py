from nicegui import ui

from interactive_unet.app.ui.navigator import NavigatorWidget

class UIBuilder():

    def __init__(self, state, callbacks, input_handler):
        self.state = state
        self.callbacks = callbacks
        self.input_handler = input_handler

        # State references
        self.ui_state = self.state.ui
        self.annot = self.state.annot

        # Data
        self.input_path = None
        self.button_load = None
        self.select_scan = None

        # Annotation
        self.toggle_annotation_mode = None
        self.slider_brush_size = None
        self.button_palette = None
        self.button_undo = None
        self.button_redo = None

        # Navigation
        self.navigator = None

        # Viewport
        self.viewport = None

        # Keyboard handler
        self.keyboard = None

    def build(self):

        self._apply_browser_overrides()
        ui.page_title('Interactive 3D Segmentation')

        ui.add_head_html("""
        <style>
        html, body, #q-app {
        height: 100%;
        overflow: hidden;
        }
        </style>
        """, shared=True)

        with ui.column().classes('w-full h-screen'):

            with ui.row().classes('w-full h-[97%] gap-4 items-stretch'):
            
                # Left column toolbar
                with ui.column().classes('w-120 h-full shrink-0 overflow-auto p-1'):
                    self._create_data_card()
                    self._create_annotation_card()
                    self._create_navigation_card()

                # Center column viewer (padding added here)
                with ui.column().classes('flex-1 h-full p-1'):
                    self._create_viewport_card()

                # Right column toolbar
                with ui.column().classes('w-120 h-full shrink-0 overflow-auto p-1'):
                    self._create_info_card()
                    self._create_view_card()

        return self.viewport


    def _create_data_card(self):        
        
        with ui.card().classes('w-full p-3 gap-2'):

            with ui.expansion(value=True).props('dense filled').classes('w-full') as expansion:
                with expansion.add_slot('header'):
                    ui.label('Data').classes('w-full text-lg font-medium')
                ui.separator()
                
                self.input_path = ui.input(
                    label='Path to data',
                    value='path/to/zarr/files',
                ).classes('w-full')
                self.button_load = ui.button('Load', on_click=self.callbacks.load_zarr_files).classes('w-full')
                self.select_scan = ui.select({0: 'None'}, label='Scan', with_input=True, value=0, on_change=self.callbacks.select_scan).classes('w-full')
                self.button_predict = ui.button('Predict', on_click=self.callbacks.predict_volumes).classes('w-full')
                # self.select_num_classes = ui.select(
                #     {i: str(i) for i in range(2, 11)},
                #     label='Number of classes',
                #     with_input=False,
                #     value=2,
                #     on_change=self.callbacks.update_num_classes
                # ).classes('w-full')


    def _create_annotation_card(self):

        with ui.card().classes('w-full p-3 gap-2'):

            with ui.expansion(value=True).props('dense filled').classes('w-full') as expansion:
                with expansion.add_slot('header'):
                    ui.label('Annotation').classes('w-full text-lg font-medium')
                ui.separator()

                # Mode
                with ui.row().classes('w-full items-center gap-2'):
                    ui.label('Mode').classes('text-s text-gray-600 w-16 shrink-0')
                    self.toggle_annotation_mode = ui.toggle(
                        {0: 'Draw', 1: 'Overlay', 2: 'Flood'},
                        value=0,
                        on_change=self.callbacks.toggle_annotation_mode
                    ).props('dense spread').classes('flex-1')

                # Size
                with ui.row().classes('w-full items-center gap-2 mt-2'):
                    ui.label('Size').classes('text-s text-gray-600 w-16 shrink-0')
                    self.slider_brush_size = ui.slider(
                        min=1,
                        max=50,
                        value=3,
                        on_change=self.callbacks.update_brush_size
                    ).props('label dense').classes('flex-1')

                # Class
                with ui.row().classes('w-full items-center gap-2 mt-2'):
                    ui.label('Class').classes('text-s text-gray-600 w-16 shrink-0')
                    self._create_button_palette()

                # History
                with ui.row().classes('w-full gap-2 mt-2'):
                    self.button_undo = ui.button(
                        icon='undo',
                        on_click=self.callbacks.undo
                    ).props('dense outline').classes('flex-1').tooltip('Undo')
                    self.button_redo = ui.button(
                        icon='redo',
                        on_click=self.callbacks.redo
                    ).props('dense outline').classes('flex-1').tooltip('Redo')
    
    def _create_button_palette(self):
        
        size = 34

        self.button_palette = []

        # 1 row × 10 columns grid
        with ui.element('div').classes('grid grid-cols-10 grid-rows-1 gap-0'):
            for i, c in enumerate(self.annot.colors):
                btn = (
                    ui.button('', on_click=lambda i=i: self.callbacks.on_pick_color(i), color=None)
                    .props('unelevated dense')
                    .style(
                        f'background:{c} !important;'
                        f'width:{size}px !important; height:{size}px !important;'
                        f'min-width:{size}px !important; min-height:{size}px !important;'
                        f'padding:0 !important; margin:0 !important;'
                        f'line-height:{size}px !important;'
                        f'border:2px solid transparent; border-radius:0;'
                    )
                )
                self.button_palette.append(btn)

        self.callbacks.refresh_button_palette()

    def _create_navigation_card(self):

        with ui.card().classes('w-full p-3 gap-2'):

            with ui.expansion(value=True).props('dense filled').classes('w-full') as expansion:
                with expansion.add_slot('header'):
                    ui.label('Navigation').classes('w-full text-lg font-medium')
                ui.separator()
                self.navigator = NavigatorWidget(self.state)

    def _create_viewport_card(self):

        with ui.card().classes('w-full h-full p-3'):

            self.viewport = ui.interactive_image().classes('w-full h-full')
            self.viewport.on('viewport_resize', self.callbacks.on_viewport_resize)

        ui.add_body_html(f"""
        <script>
        (function () {{
        function attachResizeObserver(el) {{
            if (!el) return;

            const ro = new ResizeObserver(entries => {{
            for (const entry of entries) {{
                const r = entry.contentRect;
                el.dispatchEvent(new CustomEvent('viewport_resize', {{
                detail: {{ w: Math.round(r.width), h: Math.round(r.height) }}
                }}));
            }}
            }});

            ro.observe(el);
        }}

        window.addEventListener('load', () => {{
            const el = document.getElementById('{self.viewport.html_id}');
            attachResizeObserver(el);
        }});
        }})();
        </script>
        """, shared=True)

        # Event setup
        self.viewport.on('pointer_event', self.input_handler.on_pointer)
        ui.timer(
            0.1,
            lambda: self._attach_pointer_event(self.viewport, 'pointer_event'),
            once=True,
        )
        self.keyboard = ui.keyboard(on_key=self.input_handler.on_key)

    def _create_info_card(self):

        with ui.card().classes('w-full p-3 gap-2'):

            with ui.expansion(value=True).props('dense filled').classes('w-full') as expansion:
                with expansion.add_slot('header'):
                    ui.label('Info').classes('w-full text-lg font-medium')

                ui.separator()
                def info_row(title, initial_text, allow_line_breaks=False):
                    with ui.row().classes('w-full items-center gap-2 mt-2'):
                        ui.label(title).classes('text-s text-gray-600 w-24 shrink-0')
                        if allow_line_breaks:
                            return ui.label(initial_text).classes('whitespace-pre-line font-mono')
                        else:
                            return ui.label(initial_text).classes('font-mono')

                self.label_origin = info_row('Location', 'z: 256.0, y: 256.0, x: 256.0')
                self.label_rotation_u = info_row('Rotation - u', '1, 0, 0')
                self.label_rotation_v = info_row('Rotation - v', '0, 1, 0')
                self.label_rotation_w = info_row('Rotation - w', '0, 0, 1')
                self.label_zoom = info_row('Zoom', '1.0')
                self.label_shape = info_row('Shape', 'Z: 512, Y: 512, X: 512')

    def _create_view_card(self):

        with ui.card().classes('w-full p-3 gap-2'):

            with ui.expansion(value=True).props('dense filled').classes('w-full') as expansion:
                with expansion.add_slot('header'):
                    ui.label('View').classes('w-full text-lg font-medium')

                ui.separator()
                self.checkbox_prediction_overlay = ui.checkbox(
                    'Show prediction overlay',
                    value=False,
                    on_change=self.callbacks.toggle_prediction_overlay
                ).classes('text-base font-normal')

    def _attach_pointer_event(self, element, event_name: str):
        """
        Universal pointer + wheel bridge with multi-touch info.
        NO preventDefault/stopPropagation anywhere.

        Emits one NiceGUI event `event_name` with:
        type: 'down' | 'move' | 'up' | 'cancel' | 'wheel'
        x, y (element-relative), clientX, clientY,
        pointerType, pointerId, pressure, buttons, button, modifiers
        touches: active touch points (pointerType=='touch')
        touchCount
        wheel: deltaX/deltaY/deltaZ/deltaMode
        """
                
        ui.run_javascript(f"""
        (() => {{
        const root = document.getElementById('{element.html_id}');
        if (!root) return;

        const target = root.querySelector('img') || root;
        if (!target || target.__nicegui_universal_input_installed) return;
        target.__nicegui_universal_input_installed = true;

        target.style.touchAction = 'none';

        const EVENT_NAME = {event_name!r};

        // Track active pointers (for multi-touch)
        const active = new Map();

        // --- Minimal 2-finger gesture state ---
        const g = {{
            active: false,
            idA: null,
            idB: null,
            prevMidX: 0,
            prevMidY: 0,
            prevDist: 0,
            prevAng: 0,
        }};

        const rectXY = (clientX, clientY) => {{
            const r = target.getBoundingClientRect();
            return {{
            x: (clientX ?? 0) - r.left,
            y: (clientY ?? 0) - r.top,
            }};
        }};

        const touchesSnapshot = () => {{
            const out = [];
            for (const [id, p] of active) {{
            if (p.pointerType !== 'touch') continue;
            const xy = rectXY(p.clientX, p.clientY);
            out.push({{
                pointerId: id,
                clientX: p.clientX,
                clientY: p.clientY,
                x: xy.x,
                y: xy.y,
                pressure: p.pressure ?? 0.0,
            }});
            }}
            return out;
        }};

        const dispatch = (detail) => {{
            target.dispatchEvent(new CustomEvent(EVENT_NAME, {{
            bubbles: true,
            composed: true,
            detail,
            }}));
        }};

        const resetGesture = () => {{
            g.active = false;
            g.idA = g.idB = null;
        }};

        const wrapPi = (a) => {{
            if (a > Math.PI) a -= 2 * Math.PI;
            else if (a < -Math.PI) a += 2 * Math.PI;
            return a;
        }};

        // Compute 2-finger deltas (pan/zoom/rotate) from current touches
        const computeTwoFinger = (touches) => {{
            // defaults
            const out = {{
            pan_dx: 0,
            pan_dy: 0,
            zoom_factor: 1.0,
            rotation_rad: 0.0,
            hasTwoFinger: false,
            }};
            if (touches.length !== 2) {{
            resetGesture();
            return out;
            }}

            // stable order by pointerId
            let a = touches[0], b = touches[1];
            if (a.pointerId > b.pointerId) {{ const t = a; a = b; b = t; }}

            const midX = (a.x + b.x) * 0.5;
            const midY = (a.y + b.y) * 0.5;
            const dx = (b.x - a.x);
            const dy = (b.y - a.y);
            const dist = Math.hypot(dx, dy);
            const ang = Math.atan2(dy, dx);

            // initialize or rebind if IDs changed
            if (!g.active || g.idA !== a.pointerId || g.idB !== b.pointerId) {{
            g.active = true;
            g.idA = a.pointerId; g.idB = b.pointerId;
            g.prevMidX = midX; g.prevMidY = midY;
            g.prevDist = dist; g.prevAng = ang;
            out.hasTwoFinger = true;
            return out;
            }}

            // pan
            out.pan_dx = midX - g.prevMidX;
            out.pan_dy = midY - g.prevMidY;

            // zoom (ratio)
            if (g.prevDist > 1e-3) out.zoom_factor = dist / g.prevDist;

            // rotation
            out.rotation_rad = wrapPi(ang - g.prevAng);

            // store
            g.prevMidX = midX; g.prevMidY = midY;
            g.prevDist = dist; g.prevAng = ang;

            out.hasTwoFinger = true;
            return out;
        }};

        const emit = (type, ev, extra = {{}}) => {{
            const xy = rectXY(ev.clientX, ev.clientY);
            const touches = touchesSnapshot();
            const tf = computeTwoFinger(touches);

            dispatch({{
            type,
            x: xy.x,
            y: xy.y,
            clientX: ev.clientX ?? 0,
            clientY: ev.clientY ?? 0,
            pointerId: ev.pointerId ?? 0,
            pointerType: ev.pointerType || 'mouse',
            pressure: ev.pressure ?? 0.0,
            buttons: ev.buttons ?? 0,
            button: ev.button ?? 0,
            shiftKey: !!ev.shiftKey,
            ctrlKey: !!ev.ctrlKey,
            altKey: !!ev.altKey,
            metaKey: !!ev.metaKey,
            touches,
            touchCount: touches.length,
            pan_dx: tf.pan_dx,
            pan_dy: tf.pan_dy,
            zoom_factor: tf.zoom_factor,
            rotation_rad: tf.rotation_rad,
            hasTwoFinger: tf.hasTwoFinger,
            ...extra,
            }});
        }};

        const updateActive = (ev) => {{
            active.set(ev.pointerId, {{
            pointerType: ev.pointerType || 'mouse',
            clientX: ev.clientX ?? 0,
            clientY: ev.clientY ?? 0,
            pressure: ev.pressure ?? 0.0,
            }});
        }};

        target.addEventListener('pointerdown', (ev) => {{
            target.setPointerCapture(ev.pointerId);
            updateActive(ev);
            emit('down', ev);
        }});

        target.addEventListener('pointermove', (ev) => {{
            if (active.has(ev.pointerId)) updateActive(ev);
            emit('move', ev);
        }});

        const endPointer = (ev, type) => {{
            active.delete(ev.pointerId);
            resetGesture(); // baseline changes on up/cancel
            emit(type, ev);
        }};

        target.addEventListener('pointerup', (ev) => endPointer(ev, 'up'));
        target.addEventListener('pointercancel', (ev) => endPointer(ev, 'cancel'));

        target.addEventListener('wheel', (ev) => {{
            emit('wheel', ev, {{
            deltaX: ev.deltaX ?? 0,
            deltaY: ev.deltaY ?? 0,
            deltaZ: ev.deltaZ ?? 0,
            deltaMode: ev.deltaMode ?? 0,
            }});
        }}, {{ passive: true }});

        }})();
        """)

    # def _apply_browser_overrides(self):

    #     ui.add_body_html("""
    #     <script>
    #     document.addEventListener('DOMContentLoaded', () => {

    #     // Hard-disable page scrolling without touching pointer events
    #     document.documentElement.style.overflow = 'hidden';
    #     document.body.style.overflow = 'hidden';

    #     // Disable text selection / callouts globally (safe)
    #     document.documentElement.style.userSelect = 'none';
    #     document.documentElement.style.webkitUserSelect = 'none';
    #     document.documentElement.style.webkitTouchCallout = 'none';

    #     // Block context menu globally
    #     document.addEventListener('contextmenu', e => e.preventDefault(), { capture: true });

    #     // Block browser zoom (Ctrl/Cmd + wheel)
    #     document.addEventListener('wheel', e => {
    #         if (e.ctrlKey || e.metaKey) e.preventDefault();
    #     }, { passive: false, capture: true });

    #     // Block browser zoom (Alt/Cmd + wheel)
    #     document.addEventListener('wheel', e => {
    #         if (e.altKey) e.preventDefault();
    #     }, { passive: false, capture: true });

    #     // Block drag/drop
    #     for (const name of ['dragstart','dragover','drop']) {
    #         document.addEventListener(name, e => e.preventDefault(), { capture: true });
    #     }

    #     // Block Safari/iOS gesture zoom
    #     for (const name of ['gesturestart','gesturechange','gestureend']) {
    #         document.addEventListener(name, e => e.preventDefault(), { passive: false, capture: true });
    #     }

    #     // Block "selectstart" (text selection) — safe and helps long-press selection
    #     document.addEventListener('selectstart', e => e.preventDefault(), { capture: true });

    #     });
    #     </script>
    #     """)

    def _apply_browser_overrides(self):

        # Put the CSS + viewport in the HEAD (more reliable than body for touch behavior)
        ui.add_head_html("""
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">

        <style>
        /* Let the app own touch gestures everywhere */
        html, body, #q-app {
            touch-action: none !important;     /* key for preventing pinch/pan defaults */
            overscroll-behavior: none;
            -webkit-user-select: none;
            user-select: none;
            -webkit-touch-callout: none;
        }

        /* If you have a canvas area, you can also set: */
        /* canvas { touch-action: none !important; } */
        </style>
        """, shared=True)

        # Event-level blockers (covers Chrome trackpad pinch and older gesture paths)
        ui.add_body_html("""
        <script>
        (function () {
        const opts = { passive: false, capture: true };

        // Block browser zoom, but let the app receive the event
        window.addEventListener('wheel', (e) => {
            if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            // DO NOT stopPropagation / stopImmediatePropagation
            }
        }, opts);

        // Touchscreen pinch (2+ fingers): same idea
        const blockMultiTouch = (e) => {
            if (e.touches && e.touches.length > 1) {
            e.preventDefault();
            // DO NOT stopImmediatePropagation
            }
        };
        window.addEventListener('touchstart', blockMultiTouch, opts);
        window.addEventListener('touchmove',  blockMultiTouch, opts);

        // Keyboard zoom: prevent default, but don't nuke the event for the app
        window.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && (e.key === '+' || e.key === '-' || e.key === '=' || e.key === '0')) {
            e.preventDefault();
            // DO NOT stopImmediatePropagation
            }
        }, { capture: true });

        for (const name of ['gesturestart','gesturechange','gestureend']) {
            window.addEventListener(name, (e) => e.preventDefault(), opts);
        }
        })();
        </script>
        """, shared=True)
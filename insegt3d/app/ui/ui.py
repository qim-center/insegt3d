from contextlib import contextmanager

import segmentation_models_pytorch as smp
from nicegui import ui

from insegt3d.app.ui.navigator import NavigatorWidget
from insegt3d.app.ui.browser_bridge import apply_browser_overrides, attach_pointer_event

ARCHITECTURE_OPTIONS = {
    'Unet': 'U-Net',
    'U-Net++': 'U-Net++',
    'FPN': 'FPN',
    'PSPNet': 'PSPNet',
    'DeepLabV3': 'DeepLabV3',
    'DeepLabV3+': 'DeepLabV3+',
    'Linknet': 'Linknet',
    'MAnet': 'MAnet',
    'PAN': 'PAN',
    'UPerNet': 'UPerNet',
    'Segformer': 'Segformer',
    'DPT': 'DPT',
}

class UIBuilder:

    def __init__(self, state, callbacks, input_handler):
        self.state = state
        self.callbacks = callbacks
        self.input_handler = input_handler

        # State references
        self.ui_state = self.state.ui
        self.annot = self.state.annot

    @contextmanager
    def _card_section(self, title, expanded=True):
        with ui.card().classes('w-full p-3 gap-2'):
            with ui.expansion(value=expanded).props('dense filled').classes('w-full') as expansion:
                with expansion.add_slot('header'):
                    ui.label(title).classes('w-full text-lg font-medium')
                ui.separator()
                yield

    def build(self):

        apply_browser_overrides()
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

                with ui.column().classes('w-120 h-full shrink-0 overflow-auto p-1'):
                    self._create_data_card()
                    self._create_annotation_card()
                    self._create_prediction_card()
                    self._create_advanced_settings()

                with ui.column().classes('flex-1 h-full p-1 gap-2'):
                    self._create_viewport_card()
                    self._create_live_train_bar()

                with ui.column().classes('w-120 h-full shrink-0 overflow-auto p-1'):
                    self._create_viewport_controls_card()
                    self._create_display_card()

        return self.viewport

    def _create_data_card(self):

        with self._card_section('Data'):
            self.input_path = ui.input(
                label='Path to data',
                placeholder='path/to/zarr/files',
            ).classes('w-full')
            self.select_scan = ui.select({0: 'None'}, label='Scan', with_input=True, value=0, on_change=self.callbacks.select_scan).classes('w-full')
            self.button_load = ui.button('Load', on_click=self.callbacks.load_zarr_files).classes('w-full')

    def _create_annotation_card(self):

        with self._card_section('Annotation'):

            # Mode
            with ui.row().classes('w-full items-center gap-2'):
                ui.label('Mode').classes('text-s text-gray-600 w-16 shrink-0')
                self.toggle_annotation_mode = ui.toggle(
                    {0: 'Draw', 1: 'Overlay', 2: 'Flood', 3: 'Fill'},
                    value=0,
                    on_change=self.callbacks.toggle_annotation_mode
                ).props('dense spread').classes('flex-1')

            # Size
            with ui.row().classes('w-full items-center gap-2 mt-2'):
                ui.label('Size').classes('text-s text-gray-600 w-16 shrink-0')
                self.slider_brush_size = ui.slider(
                    min=1,
                    max=self.ui_state.max_brush_size(),
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
            for i, color in enumerate(self.annot.colors):
                button = (
                    ui.button('', on_click=lambda i=i: self.callbacks.on_pick_color(i), color=None)
                    .props('unelevated dense')
                    .style(
                        f'background:{color} !important;'
                        f'width:{size}px !important; height:{size}px !important;'
                        f'min-width:{size}px !important; min-height:{size}px !important;'
                        f'padding:0 !important; margin:0 !important;'
                        f'line-height:{size}px !important;'
                        f'border:2px solid transparent; border-radius:0;'
                    )
                )
                self.button_palette.append(button)

        self.callbacks.refresh_button_palette()

    def _create_prediction_card(self):

        with self._card_section('Prediction'):
            self.button_predict = ui.button('Predict', on_click=self.callbacks.predict_volumes).classes('w-full')

            self.checkbox_export_tiff = ui.checkbox(
                'Also export tiff stack',
                value=self.state.train.export_tiff,
                on_change=self.callbacks.toggle_export_tiff
            ).classes('text-base font-normal')

            self.label_predict_status = ui.label('').classes('text-sm font-normal')
            self.label_predict_status.set_visibility(False)

            self.progress_predict = ui.linear_progress(value=0.0, show_value=False).classes('w-full')
            self.progress_predict.set_visibility(False)

            self.label_predict_chunks = ui.label('').classes('text-xs text-gray-500')
            self.label_predict_chunks.set_visibility(False)

            self.button_cancel_predict = (
                ui.button('Cancel', on_click=self.callbacks.cancel_predict_volumes, color='negative')
                .props('outline')
                .classes('w-full')
            )
            self.button_cancel_predict.set_visibility(False)

    def _create_viewport_card(self):

        with ui.card().classes('w-full flex-1 min-h-0 p-3'):

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
        """, shared=False)

        # Event setup
        self.viewport.on('pointer_event', self.input_handler.on_pointer)
        ui.timer(
            0.1,
            lambda: attach_pointer_event(self.viewport, 'pointer_event'),
            once=True,
        )
        self.keyboard = ui.keyboard(on_key=self.input_handler.on_key)

    def _create_live_train_bar(self):

        with ui.card().classes('w-full shrink-0 p-2'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                ui.label('Live train').classes('text-xs text-gray-500 shrink-0')
                self.progress_live_train = ui.linear_progress(value=0.0, show_value=False).classes('flex-1')
                self.label_live_train_status = ui.label('').classes('w-56 shrink-0 text-right text-xs text-gray-500')

    def _create_viewport_controls_card(self):

        def info_grid_row(title, initial_text):
            ui.label(title).classes('text-xs text-gray-500')
            return ui.label(initial_text).classes('font-mono text-xs')

        with self._card_section('Viewport'):

            self.navigator = NavigatorWidget(self.state)

            ui.separator().classes('my-2')

            with ui.element('div').classes(
                'w-full grid grid-cols-[5rem_1fr] gap-x-3 gap-y-1.5 items-baseline '
                'rounded-lg bg-gray-50 px-3 py-2'
            ):
                self.label_origin = info_grid_row('Location', 'z: 256.0, y: 256.0, x: 256.0')

                self.label_zoom = info_grid_row('Zoom', '1.0')

                ui.label('Rotation').classes('text-xs text-gray-500 self-start')
                with ui.column().classes('gap-0 font-mono text-xs leading-tight'):
                    self.label_rotation_u = ui.label('u  1, 0, 0')
                    self.label_rotation_v = ui.label('v  0, 1, 0')
                    self.label_rotation_w = ui.label('w  0, 0, 1')

                self.label_shape = info_grid_row('Shape', 'Z: 512, Y: 512, X: 512')

    def _create_overlay_row(self, label, visible, alpha, on_toggle, on_opacity):
        with ui.row().classes('w-full items-center gap-2'):
            checkbox = ui.checkbox(
                label, value=visible, on_change=on_toggle
            ).classes('w-56 shrink-0 text-base font-normal')
            slider = ui.slider(
                min=0, max=1, step=0.05, value=alpha, on_change=on_opacity
            ).props('dense').classes('flex-1')
            percent = ui.label(f'{alpha:.0%}').classes('w-10 shrink-0 text-right text-xs text-gray-500')
            percent.bind_text_from(slider, 'value', backward=lambda v: f'{v:.0%}')
        return checkbox, slider

    def _create_display_card(self):

        with self._card_section('Display'):
            self.checkbox_mask_overlay, self.slider_mask_opacity = self._create_overlay_row(
                'Annotation overlay', self.ui_state.mask.visible, self.ui_state.mask.alpha,
                self.callbacks.toggle_mask_overlay, self.callbacks.update_mask_opacity
            )

            self.checkbox_saved_prediction_overlay, self.slider_saved_prediction_opacity = self._create_overlay_row(
                'Prediction overlay', self.ui_state.saved_prediction.visible, self.ui_state.saved_prediction.alpha,
                self.callbacks.toggle_saved_prediction_overlay, self.callbacks.update_saved_prediction_opacity
            )

            self.checkbox_saved_prediction_overlay.set_enabled(False)
            self.slider_saved_prediction_opacity.set_enabled(False)

            self.checkbox_prediction_overlay, self.slider_prediction_opacity = self._create_overlay_row(
                'Live prediction overlay', self.ui_state.prediction.visible, self.ui_state.prediction.alpha,
                self.callbacks.toggle_prediction_overlay, self.callbacks.update_prediction_opacity
            )

            ui.separator().classes('my-2')

            ui.label('Histogram').classes('text-s text-gray-600')
            self.image_histogram = ui.image('').classes('w-full').style('height:64px;')

            with ui.element('div').classes('relative w-full mt-3 mb-3'):
                self.range_intensity = ui.range(
                    min=0,
                    max=255,
                    step=1,
                    value={'min': 0, 'max': 255},
                    on_change=self.callbacks.update_intensity_range
                ).classes('w-full')
                self.label_intensity_low = ui.label('').classes(
                    'absolute -top-3 -translate-x-1/2 text-xs text-gray-500'
                )
                self.label_intensity_high = ui.label('').classes(
                    'absolute -bottom-3 -translate-x-1/2 text-xs text-gray-500'
                )

    def _create_advanced_settings(self):

        with self._card_section('Advanced settings', expanded=False):
            self.checkbox_live_training = ui.checkbox(
                'Live training',
                value=self.state.train.live_training_enabled,
                on_change=self.callbacks.toggle_live_training
            ).classes('text-base font-normal')

            self.select_architecture = ui.select(
                ARCHITECTURE_OPTIONS,
                label='Model architecture',
                with_input=True,
                value=self.state.train.architecture,
                on_change=self.callbacks.select_architecture
            ).classes('w-full')

            self.select_encoder = ui.select(
                smp.encoders.get_encoder_names(),
                label='Model encoder',
                with_input=True,
                value=self.state.train.encoder_name,
                on_change=self.callbacks.select_encoder
            ).classes('w-full')
            self.number_learning_rate = ui.number(
                label='Learning rate',
                step=0.001,
                value=self.state.train.lr,
                on_change=self.callbacks.update_learning_rate
            ).classes('w-full')
            self.number_batch_size = ui.number(
                label='Batch size',
                step=1,
                value=self.state.train.batch_size,
                format='%d',
                precision=0,
                on_change=self.callbacks.update_batch_size
            ).classes('w-full')
            self.number_steps_per_epoch = ui.number(
                label='Steps per training burst',
                step=1,
                value=self.state.train.steps_per_epoch,
                format='%d',
                precision=0,
                on_change=self.callbacks.update_steps_per_epoch
            ).classes('w-full')
            self.button_reset_model = ui.button(
                'Reset model',
                on_click=self.callbacks.reset_model
            ).classes('w-full')
            self.button_reset_annotations = ui.button(
                'Reset annotations',
                on_click=self.callbacks.reset_annotations
            ).classes('w-full')

        self.callbacks.set_model_lock()

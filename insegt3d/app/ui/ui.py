from contextlib import contextmanager

import segmentation_models_pytorch as smp
from nicegui import ui
from nicegui.elements.mixins.value_element import ValueElement

from insegt3d.app.ui.navigator import NavigatorWidget
from insegt3d.app.ui.browser_bridge import apply_browser_overrides, attach_pointer_event
from insegt3d.ml.metrics import LOSS_OPTIONS
from insegt3d.ml.unet2d import ARCHITECTURES

MODES = {
    'draw': ('sym_o_brush', 'Draw', 'Draw (B)'),
    'erase': ('sym_o_ink_eraser', 'Erase', 'Erase (E, or right-drag)'),
    'mask_fill': ('sym_o_format_color_fill', 'Fill', 'Fill a region enclosed by annotations (G)'),
    'flood': ('sym_o_auto_fix_normal', 'Flood', 'Flood fill, dragging from the seed sets the tolerance (F)'),
    'keep': ('sym_o_approval', 'Keep', 'Keep the live prediction under the stroke (K, or hold Shift)'),
}

SHORTCUTS = [
    ('Left drag', 'Use the selected tool'),
    ('Right drag, pen eraser', 'Erase'),
    ('Shift + left drag', 'Keep the live prediction under the stroke'),
    ('Mouse wheel', 'Brush size'),
    ('B / E / G / F / K', 'Draw / Erase / Fill / Flood / Keep'),
    ('1–9, 0', 'Select class 1–10'),
    ('C', 'Next class'),
    ('D', 'Toggle the live prediction overlay'),
    ('Ctrl + Z / Ctrl + Y', 'Undo / redo'),
    ('Ctrl + left drag', 'Pan'),
    ('Ctrl + right drag', 'Scroll through slices'),
    ('Ctrl + middle drag', 'Rotate the slicing plane'),
    ('Ctrl + mouse wheel', 'Zoom'),
    ('Q / A', 'Step one slice up / down'),
    ('Z / Y / X', 'View along the z / y / x axis'),
    (', / .', 'Previous / next annotated slice'),
    ('Space', 'Randomize the orientation'),
    ('Enter / Esc', 'Leave a text field'),
]

class UIBuilder:

    def __init__(self, state, callbacks, input_handler, scheduler):
        self.state = state
        self.callbacks = callbacks
        self.input_handler = input_handler
        self.scheduler = scheduler

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

        self._create_shortcuts_dialog()

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

        return self.viewport, self.overlay

    def _create_data_card(self):

        with self._card_section('Data'):
            self.input_path = ui.input(
                label='Path to data',
                placeholder='path/to/zarr/files',
                value=self.state.data.input_path,
            ).bind_value_to(self.state.data, 'input_path').classes('w-full')
            self.select_scan = ui.select(self.callbacks.scan_options(), label='Scan', with_input=True, value=self.state.data.zarr_idx, on_change=self.callbacks.select_scan).classes('w-full')
            self.button_load = ui.button('Load', on_click=self.callbacks.load_zarr_files).classes('w-full')
            self.button_load.bind_enabled_from(self.state.train, 'predicting', backward=lambda predicting: not predicting)

    def _create_annotation_card(self):

        with self._card_section('Annotation'):

            with ui.row().classes('w-full items-center gap-2'):
                self.toggle_annotation_mode = ValueElement(tag='q-btn-toggle', value=self.annot.mode).bind_value(self.annot, 'mode').props('stack no-caps')
                self.toggle_annotation_mode.props['options'] = [{'value': mode, 'icon': icon, 'label': label, 'slot': mode} for mode, (icon, label, _) in MODES.items()]
                for mode, (_, _, tip) in MODES.items():
                    with self.toggle_annotation_mode.add_slot(mode):
                        ui.tooltip(tip)
                ui.space()
                self.button_undo = ui.button(
                    icon='undo',
                    on_click=lambda: self.scheduler.request('undo')
                ).props('dense flat').tooltip('Undo (Ctrl + Z)')
                self.button_redo = ui.button(
                    icon='redo',
                    on_click=lambda: self.scheduler.request('redo')
                ).props('dense flat').tooltip('Redo (Ctrl + Y)')

            with ui.row().classes('w-full items-center gap-2 mt-2'):
                ui.label('Size').classes('text-s text-gray-600 w-16 shrink-0')
                self.slider_brush_size = ui.slider(
                    min=1,
                    max=self.ui_state.max_brush_size(),
                    value=self.annot.brush_size,
                    on_change=self.callbacks.update_brush_size
                ).props('label dense').classes('flex-1')

            with ui.row().classes('w-full items-center gap-2 mt-2'):
                ui.label('Class').classes('text-s text-gray-600 w-16 shrink-0')
                self._create_button_palette()

    def _create_button_palette(self):

        size = 30
        tile = (
            f'width:{size}px !important; height:{size}px !important;'
            f'min-width:{size}px !important; min-height:{size}px !important;'
            f'padding:0 !important; margin:0 !important;'
            f'line-height:{size}px !important;'
        )

        self.button_palette = []

        with ui.row().classes('gap-0 no-wrap'):
            for i in range(len(self.annot.colors)):
                button = (
                    ui.button(str(i + 1), on_click=lambda i=i: self.callbacks.on_pick_color(i), color=None)
                    .props('unelevated dense')
                    .style(tile + 'border:2px solid transparent; border-radius:0; color:black; text-shadow:0 0 3px white;')
                    .tooltip(f'Class {i + 1} ({(i + 1) % 10}, or C for the next class)')
                )
                self.button_palette.append(button)

            self.button_add_class = ui.button(icon='add', on_click=self.callbacks.add_class).props('flat dense').style(tile).tooltip('Add class')
            self.button_remove_class = ui.button(icon='remove', on_click=self.callbacks.remove_class).props('flat dense').style(tile).tooltip('Remove selected class')
            for button in (self.button_add_class, self.button_remove_class):
                button.bind_enabled_from(self.state.train, 'predicting', backward=lambda predicting: not predicting)

        with ui.dialog() as self.dialog_remove_class, ui.card():
            self.label_remove_class = ui.label('')
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=lambda: self.dialog_remove_class.submit(False)).props('flat')
                ui.button('Remove', color='negative', on_click=lambda: self.dialog_remove_class.submit(True))

        self.callbacks.refresh_button_palette()

    def _create_prediction_card(self):

        train = self.state.train

        with self._card_section('Prediction'):
            self.button_predict = ui.button('Predict', on_click=self.callbacks.predict_volumes).classes('w-full')
            self.button_predict.bind_text_from(train, 'predicting', backward=lambda predicting: 'Predicting...' if predicting else 'Predict')
            self.button_predict.bind_enabled_from(train, 'predicting', backward=lambda predicting: not predicting and train.level_available)

            self.checkbox_export_tiff = ui.checkbox(
                'Also export tiff stack',
                value=train.export_tiff,
            ).bind_value_to(train, 'export_tiff').classes('text-base font-normal')
            self.checkbox_export_tiff.bind_enabled_from(train, 'predicting', backward=lambda predicting: not predicting)

            self.label_predict_status = ui.label('').classes('text-sm font-normal')
            self.progress_predict = ui.linear_progress(value=0.0, show_value=False).classes('w-full')
            self.label_predict_chunks = ui.label('').classes('text-xs text-gray-500')
            self.button_cancel_predict = (
                ui.button('Cancel', on_click=self.callbacks.cancel_predict_volumes, color='negative')
                .props('outline')
                .classes('w-full')
            )

            for element in (self.label_predict_status, self.progress_predict, self.label_predict_chunks, self.button_cancel_predict):
                element.bind_visibility_from(train, 'predicting')

    def _create_viewport_card(self):

        with ui.card().classes('w-full flex-1 min-h-0 p-3'):

            self.viewport = ui.interactive_image(sanitize=False).classes('w-full h-full')
            self.overlay = self.viewport.add_layer()
            self.viewport.on('viewport_resize', self.callbacks.on_viewport_resize)

        ui.add_body_html(f"""
        <script>
        window.addEventListener('load', () => {{
            const el = document.getElementById('{self.viewport.html_id}');
            if (!el) return;

            new ResizeObserver(([entry]) => {{
                const r = entry.contentRect;
                el.dispatchEvent(new CustomEvent('viewport_resize', {{
                    detail: {{ w: Math.round(r.width), h: Math.round(r.height) }}
                }}));
            }}).observe(el);
        }});
        </script>
        """, shared=False)

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
                ui.button(icon='keyboard', on_click=self.dialog_shortcuts.open).props('dense flat').tooltip('Keyboard shortcuts')

    def _create_shortcuts_dialog(self):

        with ui.dialog() as self.dialog_shortcuts, ui.card():
            with ui.grid(columns='auto 1fr').classes('gap-x-6 gap-y-1'):
                for keys, action in SHORTCUTS:
                    ui.label(keys).classes('font-mono text-xs')
                    ui.label(action).classes('text-sm')

    def _create_viewport_controls_card(self):

        def info_grid_row(title, initial_text):
            ui.label(title).classes('text-xs text-gray-500')
            return ui.label(initial_text).classes('font-mono text-xs')

        with self._card_section('Viewport'):

            self.navigator = NavigatorWidget(self.state, self.scheduler)

            with ui.row().classes('w-full gap-2 no-wrap'):
                for axis in 'zyx':
                    ui.button(
                        axis.upper(),
                        on_click=lambda axis=axis: self.callbacks.align_view(axis)
                    ).props('dense outline').classes('flex-1').tooltip(f'View along the {axis} axis ({axis.upper()})')
                self.button_center = ui.button('Center', on_click=self.callbacks.center_view).props('dense outline').classes('flex-1')
                self.button_go_to = ui.button('Go to', on_click=self.callbacks.go_to).props('dense outline').classes('flex-1')

            with ui.row().classes('w-full items-center gap-2 no-wrap'):
                self.button_previous_plane = ui.button(
                    icon='chevron_left',
                    on_click=lambda: self.callbacks.step_plane(-1)
                ).props('dense outline').classes('flex-1').tooltip('Previous annotated slice (,)')
                self.label_plane = ui.label('Annotated slices').classes('w-44 shrink-0 text-center text-s text-gray-600')
                self.button_next_plane = ui.button(
                    icon='chevron_right',
                    on_click=lambda: self.callbacks.step_plane(1)
                ).props('dense outline').classes('flex-1').tooltip('Next annotated slice (.)')

            with ui.dialog() as self.dialog_go_to, ui.card():
                with ui.grid(columns='auto 1fr 1fr 1fr').classes('w-96 items-center'):
                    self.numbers_go_to = []
                    for title in ('Location', 'Normal'):
                        ui.label(title).classes('text-s text-gray-600')
                        self.numbers_go_to += [ui.number(axis) for axis in 'zyx']
                with ui.row().classes('w-full justify-end'):
                    ui.button('Cancel', on_click=lambda: self.dialog_go_to.submit(False)).props('flat')
                    ui.button('Go', on_click=lambda: self.dialog_go_to.submit(True))

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

    def _create_overlay_row(self, label, overlay):
        def render():
            self.callbacks.renderer.update()

        with ui.row().classes('w-full items-center gap-2'):
            checkbox = ui.checkbox(
                label, value=overlay.visible, on_change=render
            ).bind_value(overlay, 'visible').classes('w-56 shrink-0 text-base font-normal')
            slider = ui.slider(
                min=0, max=1, step=0.05, value=overlay.alpha, on_change=render
            ).bind_value_to(overlay, 'alpha').props('dense').classes('flex-1')
            percent = ui.label(f'{overlay.alpha:.0%}').classes('w-10 shrink-0 text-right text-xs text-gray-500')
            percent.bind_text_from(slider, 'value', backward=lambda v: f'{v:.0%}')
        return checkbox, slider

    def _create_display_card(self):

        with self._card_section('Display'):
            self.checkbox_mask_overlay, self.slider_mask_opacity = self._create_overlay_row(
                'Annotation overlay', self.ui_state.mask
            )

            self.checkbox_saved_prediction_overlay, self.slider_saved_prediction_opacity = self._create_overlay_row(
                'Prediction overlay', self.ui_state.saved_prediction
            )

            self.checkbox_saved_prediction_overlay.set_enabled(False)
            self.slider_saved_prediction_opacity.set_enabled(False)

            self.checkbox_prediction_overlay, self.slider_prediction_opacity = self._create_overlay_row(
                'Live prediction overlay', self.ui_state.prediction
            )
            self.checkbox_prediction_overlay.tooltip('Toggle with D')
            self.checkbox_prediction_overlay.on_value_change(lambda: self.scheduler.request('live_predict'))

            for element in (self.checkbox_prediction_overlay, self.slider_prediction_opacity):
                element.bind_enabled_from(self.state.train, 'predicting', backward=lambda predicting: not predicting)

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

        train = self.state.train

        with self._card_section('Advanced settings', expanded=False):
            ui.label('Model').classes('text-s text-gray-600')
            self.select_architecture = ui.select(
                list(ARCHITECTURES),
                label='Model architecture',
                with_input=True,
                value=train.architecture,
                on_change=self.callbacks.select_architecture
            ).classes('w-full')

            self.select_encoder = ui.select(
                smp.encoders.get_encoder_names(),
                label='Model encoder',
                with_input=True,
                value=train.encoder_name,
                on_change=self.callbacks.select_encoder
            ).classes('w-full')
            self.number_slab_size = ui.number(
                label='2.5D depth',
                min=1, max=15, step=2,
                value=train.slab_size,
                format='%d',
                precision=0,
                on_change=self.callbacks.update_slab_size
            ).classes('w-full')

            self.select_level = ui.select(
                self.callbacks.level_options(),
                label='Training resolution',
                value=train.level,
                on_change=self.callbacks.update_level
            ).classes('w-full')

            ui.separator().classes('my-2')

            ui.label('Loss').classes('text-s text-gray-600')
            self.select_loss_fn = ui.select(
                LOSS_OPTIONS,
                label='Loss function',
                value=train.loss_fn,
            ).bind_value_to(train, 'loss_fn').classes('w-full')
            with ui.row().classes('w-full items-center gap-2'):
                self.checkbox_scnp = ui.checkbox(
                    'SCNP', value=train.scnp_enabled,
                ).bind_value_to(train, 'scnp_enabled').classes('w-32 shrink-0 text-base font-normal')
                self.number_scnp_size = ui.number(
                    min=3, max=15, step=2,
                    value=train.scnp_size,
                    format='%d',
                    precision=0,
                    on_change=self.callbacks.update_scnp_size
                ).classes('flex-1')

            ui.separator().classes('my-2')

            ui.label('Optimization').classes('text-s text-gray-600')
            self.number_learning_rate = ui.number(
                label='Learning rate',
                min=0,
                step=0.001,
                value=train.lr,
            ).bind_value_to(train, 'lr', forward=lambda v: float(v or 0)).classes('w-full')
            self.number_batch_size = ui.number(
                label='Batch size',
                min=1,
                step=1,
                value=train.batch_size,
                format='%d',
                precision=0,
            ).bind_value_to(train, 'batch_size', forward=lambda v: max(1, int(v or 1))).classes('w-full')
            self.number_steps_per_epoch = ui.number(
                label='Steps per training burst',
                min=1,
                step=1,
                value=train.steps_per_epoch,
                format='%d',
                precision=0,
            ).bind_value_to(train, 'steps_per_epoch', forward=lambda v: max(1, int(v or 1))).classes('w-full')

            ui.separator().classes('my-2')

            self.checkbox_live_training = ui.checkbox(
                'Live training',
                value=train.live_training_enabled,
            ).bind_value_to(train, 'live_training_enabled').classes('text-base font-normal')
            with ui.row().classes('w-full gap-2'):
                self.button_reset_model = ui.button(
                    'Reset model',
                    on_click=lambda: self.scheduler.request('reset_model')
                ).classes('flex-1')
                self.button_reset_model.bind_enabled_from(train, 'predicting', backward=lambda predicting: not predicting)
                self.button_reset_annotations = ui.button(
                    'Reset annotations',
                    on_click=self.callbacks.reset_annotations
                ).classes('flex-1')

            with ui.dialog() as self.dialog_reset_annotations, ui.card():
                ui.label('Delete all annotations and masks in this project?')
                with ui.row().classes('w-full justify-end'):
                    ui.button('Cancel', on_click=lambda: self.dialog_reset_annotations.submit(False)).props('flat')
                    ui.button('Delete', color='negative', on_click=lambda: self.dialog_reset_annotations.submit(True))

        self.callbacks.set_model_lock()

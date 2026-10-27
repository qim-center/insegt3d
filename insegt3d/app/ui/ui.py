from contextlib import contextmanager
from pathlib import Path

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

MODE_HELP = {
    'draw': 'Paint the selected class on the slice.',
    'erase': 'Remove annotations with the brush.',
    'mask_fill': 'Fill a region enclosed by annotations.',
    'flood': 'Drag from a point to set the fill tolerance.',
    'keep': 'Paint the live prediction into the annotations.',
    'regions': 'Drag a region edge on an axis-aligned slice, or edit its coordinates below.',
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
    ('H', 'Toggle the crosshair'),
    ('Ctrl + Z / Ctrl + Y', 'Undo / redo'),
    ('Ctrl + left drag', 'Pan'),
    ('Ctrl + double-click', 'Center the view on the clicked point'),
    ('Ctrl + right drag', 'Scroll through slices'),
    ('Ctrl + middle drag', 'Rotate the slicing plane'),
    ('Ctrl + mouse wheel', 'Zoom'),
    ('Q / A', 'Step one slice up / down'),
    ('Z / Y / X', 'View along the z / y / x axis'),
    (', / .', 'Previous / next annotated slice'),
    ('Space', 'Randomize the orientation'),
    ('Enter / Esc', 'Leave a text field'),
]

# White with a dark edge, so it shows on both light and dark data
CROSSHAIR = (
    '<svg width="50" height="50" style="display: block">'
    '<path d="M25 0V50M0 25H50" stroke="black" stroke-width="3" stroke-opacity="0.4" />'
    '<path d="M25 0V50M0 25H50" stroke="white" stroke-width="1" stroke-opacity="0.8" />'
    '</svg>'
)

def setup_theme(on_change=None):
    stylesheet = Path(__file__).with_name('style.css').read_text()
    ui.add_head_html(
        f'<style>{stylesheet}</style>'
        "<script>document.documentElement.classList.toggle('dark-mode', "
        "localStorage.getItem('insegt3d_theme') === 'dark')</script>"
    )
    dark_mode = ui.dark_mode()
    ui.colors(primary='#fc7323', secondary='#747474', negative='#e63f47')

    async def restore():
        try:
            dark = await ui.run_javascript("localStorage.getItem('insegt3d_theme') === 'dark'", timeout=3)
        except TimeoutError:
            return
        if dark:
            dark_mode.enable()
            if on_change:
                on_change(True)

    ui.timer(0.1, restore, once=True)
    return dark_mode

def toggle_theme(dark_mode, on_change=None):
    dark_mode.toggle()
    dark = dark_mode.value
    theme = 'dark' if dark else 'light'
    ui.run_javascript(
        f"document.documentElement.classList.toggle('dark-mode', {str(dark).lower()}); "
        f"localStorage.setItem('insegt3d_theme', '{theme}')"
    )
    if on_change:
        on_change(dark)

class UIBuilder:

    def __init__(self, state, callbacks, input_handler, scheduler, on_switch=None, on_shutdown=None):
        self.state = state
        self.callbacks = callbacks
        self.input_handler = input_handler
        self.scheduler = scheduler
        self.on_switch = on_switch
        self.on_shutdown = on_shutdown

        self.ui_state = self.state.ui
        self.annot = self.state.annot

    @contextmanager
    def _section(self, title, description=None):
        with ui.element('section').classes('work-section'):
            ui.label(title).classes('section-title')
            if description:
                ui.label(description).classes('section-description')
            yield

    def build(self):

        apply_browser_overrides()
        self.dark_mode = setup_theme(self._apply_theme)
        ui.page_title('InSegt3D · Segmentation workspace')

        self._create_shortcuts_dialog()
        self._create_advanced_settings()
        self._create_project_dialogs()

        with ui.element('div').classes('app-shell'):
            with ui.element('header').classes('app-header'):
                with ui.row().classes('brand items-center no-wrap'):
                    ui.label('InSegt3D').classes('brand-name')
                    ui.label('Segmentation workspace').classes('brand-description')
                with ui.row().classes('header-actions items-center no-wrap'):
                    ui.button(self.state.data.project_path.name, icon='folder', on_click=self.dialog_project.open).props('flat no-caps').classes('project-button').tooltip(str(self.state.data.project_path))
                    ui.button('Tools', icon='brush', on_click=lambda: ui.run_javascript("document.body.classList.toggle('show-tools')")).props('flat no-caps').classes('panel-toggle tools-toggle')
                    ui.button('View', icon='tune', on_click=lambda: ui.run_javascript("document.body.classList.toggle('show-inspector')")).props('flat no-caps').classes('panel-toggle inspector-toggle')
                    ui.button('Shortcuts', icon='keyboard', on_click=self.dialog_shortcuts.open).props('flat no-caps').classes('header-button')
                    ui.button('Theme', icon='contrast', on_click=self._toggle_theme).props('flat no-caps').classes('header-button').tooltip('Switch between dark and light mode')
                    ui.button('Model settings', icon='settings', on_click=self.dialog_settings.open).props('flat no-caps').classes('header-button')

            with ui.element('div').classes('app-workspace'):
                with ui.element('aside').classes('side-panel tools-panel'):
                    self._create_data_card()
                    self._create_annotation_card()
                    self._create_prediction_card()

                with ui.element('main').classes('viewer-column'):
                    self._create_viewport_card()
                    self._create_live_train_bar()

                with ui.element('aside').classes('side-panel inspector-panel'):
                    self._create_viewport_controls_card()
                    self._create_display_card()

        return self.viewport, self.overlay

    def _toggle_theme(self):
        toggle_theme(self.dark_mode, self._apply_theme)

    def _apply_theme(self, dark):
        self.navigator.set_theme(dark)
        self.callbacks.refresh_button_palette()
        if self.callbacks._histogram_counts is not None:
            self.callbacks._refresh_histogram_image()

    def _focus_data_input(self):
        ui.run_javascript(
            f"document.body.classList.add('show-tools'); "
            f"document.getElementById('{self.input_path.html_id}').querySelector('input')?.focus()"
        )

    def _create_data_card(self):

        with self._section('Data source', 'Open an OME-Zarr volume or a folder of volumes.'):
            self.input_path = ui.input(
                label='Volume path or URL',
                placeholder='/path/to/data.zarr',
                value=self.state.data.input_path,
            ).bind_value_to(self.state.data, 'input_path').classes('w-full')
            self.button_load = ui.button('Open data', icon='folder_open', on_click=self.callbacks.load_zarr_files).props('unelevated no-caps').classes('w-full load-button')
            self.button_load.bind_enabled_from(self.state.train, 'busy', backward=lambda busy: not busy)
            self.select_scan = ui.select(self.callbacks.scan_options(), label='Current volume', with_input=True, value=self.state.data.zarr_idx, on_change=self.callbacks.select_scan).props('transition-duration=100').classes('w-full')
            self.select_scan.bind_enabled_from(self.state.train, 'changing_level', backward=lambda changing_level: not changing_level)
            self.select_level = ui.select(
                self.callbacks.level_options(), label='Working resolution', value=self.state.train.level,
                on_change=self.callbacks.update_level,
            ).props('transition-duration=100').classes('w-full').tooltip('Resolution used for annotations, training, and prediction')
            self.select_level.bind_enabled_from(self.state.train, 'busy', backward=lambda busy: not busy)
            ui.label('Use a coarser level for faster work on large volumes.').classes('section-note')

    def _create_annotation_card(self):

        with self._section('Annotate', 'Choose a tool and paint directly on the slice.'):

            with ui.row().classes('w-full items-center gap-2 no-wrap'):
                self.toggle_annotation_mode = ValueElement(tag='q-btn-toggle', value=self.annot.mode).bind_value(self.annot, 'mode').props('spread no-caps unelevated').classes('tool-toggle')
                self.toggle_annotation_mode.props['options'] = [{'value': mode, 'icon': icon, 'label': label, 'slot': mode} for mode, (icon, label, _) in MODES.items()]
                for mode, (_, _, tip) in MODES.items():
                    with self.toggle_annotation_mode.add_slot(mode):
                        ui.tooltip(tip)
            ui.label().bind_text_from(self.annot, 'mode', backward=lambda mode: MODE_HELP[mode]).classes('section-note')

            with ui.row().classes('w-full items-center gap-2 history-actions'):
                ui.label('History').classes('control-label')
                ui.space()
                self.button_undo = ui.button(
                    icon='undo',
                    on_click=lambda: self.scheduler.request('undo')
                ).props('dense flat').tooltip('Undo (Ctrl + Z)')
                self.button_redo = ui.button(
                    icon='redo',
                    on_click=lambda: self.scheduler.request('redo')
                ).props('dense flat').tooltip('Redo (Ctrl + Y)')

            with ui.row().classes('w-full items-center gap-2'):
                ui.label('Brush size').classes('control-label shrink-0')
                self.slider_brush_size = ui.slider(
                    min=1,
                    max=self.ui_state.max_brush_size(),
                    value=self.annot.brush_size,
                    on_change=self.callbacks.update_brush_size
                ).props('label dense').classes('flex-1')

            with ui.row().classes('w-full items-center gap-2 class-actions'):
                ui.label('Class').classes('control-label')
                ui.space()
                self.button_remove_class = ui.button(icon='remove', on_click=self.callbacks.remove_class).props('dense flat').tooltip('Remove selected class')
                self.button_add_class = ui.button(icon='add', on_click=self.callbacks.add_class).props('dense flat').tooltip('Add class')
                for button in (self.button_remove_class, self.button_add_class):
                    button.bind_enabled_from(self.state.train, 'busy', backward=lambda busy: not busy)
            self._create_button_palette()
            ui.label('Each color is a class the model learns to recognize.').classes('section-note')
            ui.label('Right drag erases · Shift paints from live prediction').classes('section-note')

    def _create_button_palette(self):

        size = 34
        tile = (
            f'width:{size}px !important; height:{size}px !important;'
            f'min-width:{size}px !important; min-height:{size}px !important;'
            f'padding:0 !important; margin:0 !important;'
            f'line-height:{size}px !important;'
        )

        self.button_palette = []

        with ui.row().classes('palette-grid'):
            for i in range(len(self.annot.colors)):
                button = (
                    ui.button(str(i + 1), on_click=lambda i=i: self.callbacks.on_pick_color(i), color=None)
                    .props('unelevated dense')
                    .style(tile + 'border:2px solid transparent; color:black; text-shadow:0 0 3px white;')
                    .tooltip(f'Class {i + 1} ({(i + 1) % 10}, or C for the next class)')
                )
                self.button_palette.append(button)

        with ui.dialog().props('transition-duration=160') as self.dialog_remove_class, ui.card():
            self.label_remove_class = ui.label('')
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=lambda: self.dialog_remove_class.submit(False)).props('flat')
                ui.button('Remove', color='negative', on_click=lambda: self.dialog_remove_class.submit(True))

        self.callbacks.refresh_button_palette()

    def _create_prediction_card(self):

        train = self.state.train

        with self._section('Predict', 'Apply the trained model to the volume or selected areas.'):
            ui.label('Prediction area').classes('control-label')
            self.toggle_predict_target = ui.toggle({'whole': 'All volumes', 'regions': 'Selected regions'}, value='regions' if self.state.regions.predict else 'whole', on_change=lambda e: self.callbacks.set_predict_target(e.value == 'regions')).props('spread no-caps').classes('prediction-target w-full')
            self.toggle_predict_target.bind_enabled_from(train, 'predicting', backward=lambda predicting: not predicting)
            ui.label('Regions use the current volume only. Existing predictions outside them stay in place.').classes('section-note').bind_visibility_from(self.state.regions, 'predict')
            self._create_regions_card()
            self.button_predict = ui.button('Predict regions' if self.state.regions.predict else 'Predict all volumes', icon='auto_awesome', on_click=self.callbacks.predict_volumes).props('unelevated no-caps').classes('w-full predict-button')
            self.button_predict.bind_text_from(train, 'predicting', backward=lambda predicting: 'Predicting…' if predicting else ('Predict regions' if self.state.regions.predict else 'Predict all volumes'))
            self.button_predict.bind_enabled_from(train, 'busy', backward=lambda busy: not busy and train.level_available)

            self.checkbox_export_tiff = ui.checkbox(
                'Also export TIFF stacks',
                value=train.export_tiff,
            ).bind_value_to(train, 'export_tiff').classes('text-base font-normal')
            self.checkbox_export_tiff.bind_enabled_from(train, 'predicting', backward=lambda predicting: not predicting)

            self.button_cancel_predict = (
                ui.button('Cancel', on_click=self.callbacks.cancel_predict_volumes, color='negative')
                .props('outline')
                .classes('w-full')
            )
            self.button_cancel_predict.bind_visibility_from(train, 'predicting')

    def _create_regions_card(self):
        with ui.element('div').classes('region-subsection').bind_visibility_from(self.state.regions, 'predict'):
            ui.label('Regions').classes('control-label')
            ui.label('Check the areas to predict in the current volume.').classes('section-note')
            with ui.row().classes('w-full items-center gap-2'):
                add_region = ui.button('Add', icon='add_box', on_click=self.callbacks.add_region).props('outline no-caps').classes('flex-1').tooltip('Add a region around the current view')
                edit_regions = ui.switch('Edit on slice', on_change=lambda e: self.callbacks.edit_regions(e.value)).bind_value_from(self.annot, 'mode', backward=lambda mode: mode == 'regions').classes('region-edit-switch')
                for control in (add_region, edit_regions):
                    control.bind_enabled_from(self.state.train, 'predicting', backward=lambda predicting: not predicting)
            ui.label('Drag inside a region to move it. Drag an edge to resize it. Use an axis view (Z, Y or X).').classes('section-note')
            self.list_regions = ui.column().classes('region-list w-full')
            with ui.column().classes('region-editor w-full') as self.editor_region:
                ui.label('Selected region · voxel bounds').classes('control-label')
                with ui.grid(columns=3).classes('region-coordinate-grid w-full'):
                    self.numbers_region = []
                    for axis in 'ZYX':
                        with ui.column().classes('gap-1'):
                            ui.label(axis).classes('region-axis')
                            for edge in ('Start', 'End'):
                                self.numbers_region.append(ui.number(edge, min=0, step=1).props('dense'))
                ui.button('Apply bounds', on_click=self.callbacks.update_region_box).props('flat no-caps').classes('w-full').bind_enabled_from(self.state.train, 'predicting', backward=lambda predicting: not predicting)
            self.editor_region.set_visibility(False)

    def _create_project_dialogs(self):
        with ui.dialog().props('transition-duration=160') as self.dialog_project, ui.card().classes('project-dialog'):
            ui.icon('folder_open').classes('project-dialog-icon')
            ui.label(self.state.data.project_path.name).classes('dialog-title')
            ui.label(str(self.state.data.project_path)).classes('project-path')
            ui.label('Annotations, regions, classes and model checkpoints are saved in this folder.').classes('section-note')
            with ui.row().classes('w-full gap-2 project-dialog-actions'):
                ui.button('Switch project', icon='swap_horiz', on_click=self._switch_project).props('outline no-caps').classes('flex-1')
                ui.button('Shut down app', icon='power_settings_new', on_click=self._shutdown_app).props('flat no-caps').classes('shutdown-button')
        with ui.dialog().props('transition-duration=160') as self.dialog_leave, ui.card().classes('project-dialog'):
            self.label_leave = ui.label().classes('dialog-title')
            ui.label('Current work will finish saving before this action completes. A running prediction will be canceled, and region blocks already written may remain.').classes('section-note')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Stay', on_click=lambda: self.dialog_leave.submit(False)).props('flat no-caps')
                ui.button('Continue', on_click=lambda: self.dialog_leave.submit(True)).props('unelevated no-caps')

    async def _switch_project(self):
        self.label_leave.text = 'Switch project?'
        if await self.dialog_leave:
            await self._leave(self.on_switch)

    async def _shutdown_app(self):
        self.label_leave.text = 'Shut down InSegt3D?'
        if await self.dialog_leave:
            await self._leave(self.on_shutdown)

    async def _leave(self, action):
        if self.annot.annotating:
            ui.notify('Finish the current brush stroke first.', type='warning')
            return
        self.dialog_project.close()
        with ui.dialog().props('persistent transition-duration=160') as waiting, ui.card().classes('project-dialog'):
            ui.spinner(size='lg')
            ui.label('Finishing current work…')
        waiting.open()
        try:
            await action()
        except Exception as e:
            waiting.close()
            ui.notify(f'Could not finish: {e}', type='negative')

    def _create_viewport_card(self):

        with ui.element('section').classes('viewer-frame'):
            with ui.element('div').classes('viewport-stage'):
                self.viewport = ui.interactive_image(sanitize=False).classes('slice-viewport')
                with self.viewport:
                    self.crosshair = (
                        ui.html(CROSSHAIR, sanitize=False)
                        .classes('absolute pointer-events-none')
                        .style('left: 50%; top: 50%; transform: translate(-50%, -50%)')
                    )
                self.overlay = self.viewport.add_layer()
                self.viewport.on('viewport_resize', self.callbacks.on_viewport_resize)

                with ui.element('div').classes('empty-view') as self.empty_view:
                    ui.label('Your volume appears here').classes('empty-title')
                    ui.label('Start with an OME-Zarr volume or a folder of volumes.').classes('empty-description')
                    ui.button('Enter data path', icon='folder_open', on_click=self._focus_data_input).props('unelevated no-caps').classes('empty-action')

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

        with ui.element('div').classes('training-bar'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                ui.icon('bolt').classes('training-icon')
                ui.label().bind_text_from(self.state.train, 'busy', backward=lambda busy: ('Predicting' if self.state.train.predicting else 'Updating annotations') if busy else 'Live training').classes('training-title shrink-0')
                self.progress_live_train = ui.linear_progress(value=0.0, show_value=False).classes('flex-1')
                self.label_live_train_status = ui.label('Add annotations for each class to begin training').classes('training-status')

    def _create_shortcuts_dialog(self):

        with ui.dialog().props('transition-duration=160') as self.dialog_shortcuts, ui.card().classes('shortcuts-card'):
            ui.label('Keyboard and pointer shortcuts').classes('dialog-title')
            with ui.grid(columns='auto 1fr').classes('gap-x-6 gap-y-1'):
                for keys, action in SHORTCUTS:
                    ui.label(keys).classes('shortcut-keys')
                    ui.label(action).classes('shortcut-action')

    def _create_viewport_controls_card(self):

        def info_grid_row(title, initial_text):
            ui.label(title).classes('readout-label')
            return ui.label(initial_text).classes('font-mono text-xs')

        with self._section('Navigate', 'Move through the volume and orient the slice.'):

            with ui.element('div').classes('navigator-wrap'):
                self.navigator = NavigatorWidget(self.state, self.scheduler)
                self.navigator.scene.classes('navigator-scene')
                with ui.element('div').classes('navigator-empty') as self.navigator_empty:
                    ui.icon('3d_rotation').classes('navigator-empty-icon')
                    ui.label('3D orientation appears after loading a volume.').classes('section-note')

            ui.label('View along axis').classes('control-label')
            with ui.row().classes('w-full gap-2 no-wrap axis-actions'):
                for axis in 'zyx':
                    ui.button(
                        axis.upper(),
                        on_click=lambda axis=axis: self.callbacks.align_view(axis)
                    ).props('dense outline').classes('flex-1').tooltip(f'View along the {axis} axis ({axis.upper()})')
            with ui.row().classes('w-full gap-2 no-wrap'):
                self.button_center = ui.button('Center view', icon='filter_center_focus', on_click=self.callbacks.center_view).props('dense flat no-caps').classes('flex-1')
                self.button_go_to = ui.button('Go to…', icon='my_location', on_click=self.callbacks.go_to).props('dense flat no-caps').classes('flex-1')

            with ui.row().classes('w-full items-center gap-2 no-wrap plane-actions'):
                self.button_previous_plane = ui.button(
                    icon='chevron_left',
                    on_click=lambda: self.callbacks.step_plane(-1)
                ).props('dense flat').tooltip('Previous annotated slice (,)')
                self.label_plane = ui.label('Annotated slices').classes('flex-1 text-center')
                self.button_next_plane = ui.button(
                    icon='chevron_right',
                    on_click=lambda: self.callbacks.step_plane(1)
                ).props('dense flat').tooltip('Next annotated slice (.)')

            with ui.dialog().props('transition-duration=160') as self.dialog_go_to, ui.card().classes('go-to-card'):
                with ui.grid(columns='auto 1fr 1fr 1fr').classes('go-to-grid items-center'):
                    self.numbers_go_to = []
                    for title in ('Location', 'Normal'):
                        ui.label(title).classes('control-label')
                        self.numbers_go_to += [ui.number(axis) for axis in 'zyx']
                with ui.row().classes('w-full justify-end'):
                    ui.button('Cancel', on_click=lambda: self.dialog_go_to.submit(False)).props('flat')
                    ui.button('Go', on_click=lambda: self.dialog_go_to.submit(True))

            with ui.element('div').classes('position-readout'):
                self.label_origin = info_grid_row('Location', 'z: 256.0, y: 256.0, x: 256.0')

                self.label_zoom = info_grid_row('Zoom', '1.0')

                ui.label('Rotation').classes('text-xs self-start')
                with ui.column().classes('gap-0 font-mono text-xs leading-tight'):
                    self.label_rotation_u = ui.label('u  1, 0, 0')
                    self.label_rotation_v = ui.label('v  0, 1, 0')
                    self.label_rotation_w = ui.label('w  0, 0, 1')

                self.label_shape = info_grid_row('Shape', 'Z: 512, Y: 512, X: 512')

    def _create_overlay_row(self, label, overlay):
        def render():
            self.callbacks.renderer.update()

        with ui.row().classes('w-full items-center gap-2 overlay-row'):
            checkbox = ui.checkbox(
                label, value=overlay.visible, on_change=render
            ).bind_value(overlay, 'visible').classes('overlay-label shrink-0')
            slider = ui.slider(
                min=0, max=1, step=0.05, value=overlay.alpha, on_change=render
            ).bind_value_to(overlay, 'alpha').props('dense').classes('flex-1')
            percent = ui.label(f'{overlay.alpha:.0%}').classes('opacity-value shrink-0 text-right')
            percent.bind_text_from(slider, 'value', backward=lambda v: f'{v:.0%}')
        return checkbox, slider

    def _create_display_card(self):

        with self._section('Display', 'Control what is visible over the image.'):
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

            self.checkbox_crosshair = ui.checkbox('Crosshair', value=False).classes('text-base font-normal').tooltip('Toggle with H')
            self.crosshair.bind_visibility_from(self.checkbox_crosshair, 'value')

            ui.label('Intensity window').classes('control-label')
            ui.label('Drag the range to adjust image contrast.').classes('section-note')
            self.image_histogram = ui.image('').classes('w-full').style('height:64px;')

            with ui.element('div').classes('relative w-full mt-3 mb-3'):
                self.range_intensity = ui.range(
                    min=0,
                    max=255,
                    step=1,
                    value={'min': 0, 'max': 255},
                    on_change=self.callbacks.update_intensity_range
                ).classes('w-full')
                self.label_intensity_low = ui.label('').classes('absolute -top-3 -translate-x-1/2 text-xs intensity-value')
                self.label_intensity_high = ui.label('').classes('absolute -bottom-3 -translate-x-1/2 text-xs intensity-value')

    def _create_advanced_settings(self):

        train = self.state.train

        with ui.dialog().props('transition-duration=160') as self.dialog_settings, ui.card().classes('settings-card'):
            with ui.row().classes('w-full items-center no-wrap'):
                with ui.column().classes('gap-0'):
                    ui.label('Model settings').classes('dialog-title')
                    ui.label('Training configuration for experienced users.').classes('section-description')
                ui.space()
                ui.button(icon='close', on_click=self.dialog_settings.close).props('flat round dense').tooltip('Close settings')
            ui.label('Model').classes('settings-heading')
            self.select_architecture = ui.select(
                list(ARCHITECTURES),
                label='Model architecture',
                with_input=True,
                value=train.architecture,
                on_change=self.callbacks.select_architecture
            ).props('transition-duration=100').classes('w-full')

            self.select_encoder = ui.select(
                smp.encoders.get_encoder_names(),
                label='Model encoder',
                with_input=True,
                value=train.encoder_name,
                on_change=self.callbacks.select_encoder
            ).props('transition-duration=100').classes('w-full')
            self.number_slab_size = ui.number(
                label='2.5D depth',
                min=1, max=15, step=2,
                value=train.slab_size,
                format='%d',
                precision=0,
                on_change=self.callbacks.update_slab_size
            ).classes('w-full')

            ui.separator().classes('my-2')

            ui.label('Loss').classes('settings-heading')
            self.select_loss_fn = ui.select(
                LOSS_OPTIONS,
                label='Loss function',
                value=train.loss_fn,
            ).bind_value_to(train, 'loss_fn').props('transition-duration=100').classes('w-full')
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

            ui.label('Optimization').classes('settings-heading')
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
                self.button_reset_model.bind_enabled_from(train, 'busy', backward=lambda busy: not busy)
                self.button_reset_annotations = ui.button(
                    'Reset annotations',
                    on_click=self.callbacks.reset_annotations
                ).classes('flex-1')

            with ui.dialog().props('transition-duration=160') as self.dialog_reset_annotations, ui.card():
                ui.label('Delete all annotations and masks in this project?')
                with ui.row().classes('w-full justify-end'):
                    ui.button('Cancel', on_click=lambda: self.dialog_reset_annotations.submit(False)).props('flat')
                    ui.button('Delete', color='negative', on_click=lambda: self.dialog_reset_annotations.submit(True))

        self.callbacks.set_model_lock()

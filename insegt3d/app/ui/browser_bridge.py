from nicegui import ui


def apply_browser_overrides():

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
    </style>
    """, shared=True)

    # Event-level blockers (covers Chrome trackpad pinch and older gesture paths)
    ui.add_body_html("""
    <script>
    (function () {
    const opts = { passive: false, capture: true };

    // Block the native right-click context menu, but let the app receive the pointer events
    window.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        // DO NOT stopPropagation / stopImmediatePropagation
    }, opts);

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


def attach_pointer_event(element, event_name: str):
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

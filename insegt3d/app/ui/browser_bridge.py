from nicegui import ui


def apply_browser_overrides():
    """Stops the browser's own zoom, scrolling, context menu, gestures and undo outside text fields, and keeps keyboard shortcuts out of clicked controls."""

    ui.add_head_html("""
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">

    <style>
    html, body, #q-app {
        height: 100%;
        overflow: hidden;
        touch-action: none !important;
        overscroll-behavior: none;
        -webkit-user-select: none;
        user-select: none;
        -webkit-touch-callout: none;
    }
    </style>
    """)

    ui.add_body_html("""
    <script>
    (function () {
    const opts = { passive: false, capture: true };

    window.addEventListener('contextmenu', (e) => e.preventDefault(), opts);

    window.addEventListener('wheel', (e) => {
        if (e.ctrlKey || e.metaKey) e.preventDefault();
    }, opts);

    const blockMultiTouch = (e) => {
        if (e.touches && e.touches.length > 1) e.preventDefault();
    };
    window.addEventListener('touchstart', blockMultiTouch, opts);
    window.addEventListener('touchmove',  blockMultiTouch, opts);

    window.addEventListener('keydown', (e) => {
        const el = document.activeElement;
        const ctrl = e.ctrlKey || e.metaKey;
        if (ctrl && ['+', '-', '=', '0'].includes(e.key)) e.preventDefault();
        if (ctrl && ['z', 'y'].includes(e.key.toLowerCase()) && !el?.matches('input, textarea')) e.preventDefault();
        if (e.key === 'Escape' || (e.key === 'Enter' && el?.matches('input') && !el.closest('.q-select'))) el?.blur();
    }, { capture: true });

    window.addEventListener('click', (e) => {
        if (!e.detail || e.target.closest('input, textarea, .q-field')) return;
        setTimeout(() => document.activeElement?.blur());
    }, { capture: true });

    for (const name of ['gesturestart','gesturechange','gestureend']) {
        window.addEventListener(name, (e) => e.preventDefault(), opts);
    }
    })();
    </script>
    """)


def attach_pointer_event(element, event_name: str):
    """Forwards pointer, touch and wheel input on `element` to Python as `event_name` events (see PointerEvent)."""

    ui.run_javascript(f"""
    (() => {{
    const root = document.getElementById('{element.html_id}');
    if (!root) return;

    const target = root.querySelector('img') || root;
    if (!target || target.__nicegui_universal_input_installed) return;
    target.__nicegui_universal_input_installed = true;

    target.style.touchAction = 'none';

    const EVENT_NAME = {event_name!r};

    const active = new Map();

    const g = {{
        active: false,
        idA: null,
        idB: null,
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
        out.push({{ pointerId: id, x: xy.x, y: xy.y }});
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

    // Zoom ratio and rotation since the previous two-finger event
    const computeTwoFinger = (touches) => {{
        const out = {{
        zoom_factor: 1.0,
        rotation_rad: 0.0,
        }};
        if (touches.length !== 2) {{
        resetGesture();
        return out;
        }}

        let a = touches[0], b = touches[1];
        if (a.pointerId > b.pointerId) {{ const t = a; a = b; b = t; }}

        const dx = (b.x - a.x);
        const dy = (b.y - a.y);
        const dist = Math.hypot(dx, dy);
        const ang = Math.atan2(dy, dx);

        if (!g.active || g.idA !== a.pointerId || g.idB !== b.pointerId) {{
        g.active = true;
        g.idA = a.pointerId; g.idB = b.pointerId;
        g.prevDist = dist; g.prevAng = ang;
        return out;
        }}

        if (g.prevDist > 1e-3) out.zoom_factor = dist / g.prevDist;

        out.rotation_rad = wrapPi(ang - g.prevAng);

        g.prevDist = dist; g.prevAng = ang;

        return out;
    }};

    let pendingMove = null;

    const flushMove = () => {{
        const ev = pendingMove;
        pendingMove = null;
        if (ev) emit('move', ev);
    }};

    const emit = (type, ev, extra = {{}}) => {{
        if (type !== 'move' && pendingMove) flushMove();

        const xy = rectXY(ev.clientX, ev.clientY);
        const touches = touchesSnapshot();
        const tf = computeTwoFinger(touches);

        dispatch({{
        type,
        x: xy.x,
        y: xy.y,
        pointerType: ev.pointerType || 'mouse',
        button: ev.button ?? 0,
        shiftKey: !!ev.shiftKey,
        ctrlKey: !!ev.ctrlKey,
        altKey: !!ev.altKey,
        touchCount: touches.length,
        zoom_factor: tf.zoom_factor,
        rotation_rad: tf.rotation_rad,
        ...extra,
        }});
    }};

    const updateActive = (ev) => {{
        active.set(ev.pointerId, {{
        pointerType: ev.pointerType || 'mouse',
        clientX: ev.clientX ?? 0,
        clientY: ev.clientY ?? 0,
        }});
    }};

    // Double clicks are detected from presses, since macOS treats ctrl + click as a right click and sends no dblclick
    let lastDown = null;

    target.addEventListener('pointerdown', (ev) => {{
        target.setPointerCapture(ev.pointerId);
        updateActive(ev);
        const double = !!lastDown && ev.button === lastDown.button && ev.timeStamp - lastDown.timeStamp < 500
            && Math.hypot(ev.clientX - lastDown.clientX, ev.clientY - lastDown.clientY) < 5;
        lastDown = double ? null : ev;
        emit('down', ev, {{ double }});
    }});

    // Moves are coalesced to one per animation frame
    target.addEventListener('pointermove', (ev) => {{
        if (active.has(ev.pointerId)) updateActive(ev);
        if (!pendingMove) requestAnimationFrame(flushMove);
        pendingMove = ev;
    }});

    const endPointer = (ev, type) => {{
        active.delete(ev.pointerId);
        resetGesture();
        emit(type, ev);
    }};

    target.addEventListener('pointerup', (ev) => endPointer(ev, 'up'));
    target.addEventListener('pointercancel', (ev) => endPointer(ev, 'cancel'));

    target.addEventListener('wheel', (ev) => {{
        emit('wheel', ev, {{ deltaY: ev.deltaY ?? 0 }});
    }}, {{ passive: true }});

    }})();
    """)

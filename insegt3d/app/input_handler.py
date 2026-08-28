from dataclasses import dataclass
from typing import Dict, List

class InputHandler:

    def __init__(self, state, tools):
        self.state = state
        self.tools = tools

    async def on_pointer(self, e):
        p = self.state.pointer

        e = PointerEvent.from_dict(e.args['detail'])

        for t in self.tools:
            if not t.ignore_pointer:
                await t.on_pointer(e)

        p.x = e.x
        p.y = e.y

    async def on_key(self, e):
        for t in self.tools:
            if not t.ignore_key:
                await t.on_key(e)

@dataclass(slots=True)
class PointerEvent:
    event_type: str  # "down" | "move" | "up" | "cancel" | "wheel"

    # Element-relative coords
    x: float
    y: float

    # Viewport coords
    client_x: float
    client_y: float

    # Pointer info
    pointer_id: int
    pointer_type: str  # "mouse" | "touch" | "pen"
    pressure: float
    buttons: int
    button: int

    # Modifiers
    shift: bool
    ctrl: bool
    alt: bool
    meta: bool

    # Multitouch snapshot (only touch pointers)
    touches: List[Dict[str, float]]
    touch_count: int

    # 2-finger gesture deltas (touch_count==2)
    pan_dx: float = 0.0
    pan_dy: float = 0.0
    zoom_factor: float = 1.0
    rotation_rad: float = 0.0
    has_two_finger: bool = False

    # Wheel
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0
    delta_mode: int = 0

    # Event types -------------------------
    @property
    def down(self):
        return self.event_type == "down"

    @property
    def move(self):
        return self.event_type == "move"

    @property
    def up(self):
        return self.event_type == "up"

    @property
    def wheel(self):
        return self.event_type == "wheel"

    @property
    def cancel(self):
        return self.event_type == "cancel"

    # Pointer types -------------------------
    @property
    def mouse(self):
        return self.pointer_type == "mouse"

    @property
    def touch(self):
        return self.pointer_type == "touch"

    @property
    def pen(self):
        return self.pointer_type == "pen"

    # Convenience -------------------------
    @property
    def one_finger(self):
        return self.touch_count == 1

    @property
    def two_finger(self):
        return self.has_two_finger or self.touch_count == 2

    # Create from dict -------------------------
    @classmethod
    def from_dict(cls, d):
        return cls(
            event_type=d.get("type", "move"),
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            client_x=float(d.get("clientX", 0.0)),
            client_y=float(d.get("clientY", 0.0)),

            pointer_id=int(d.get("pointerId", 0)),
            pointer_type=d.get("pointerType", "mouse"),
            pressure=float(d.get("pressure", 0.0)),
            buttons=int(d.get("buttons", 0)),
            button=int(d.get("button", 0)),

            shift=bool(d.get("shiftKey", False)),
            ctrl=bool(d.get("ctrlKey", False)),
            alt=bool(d.get("altKey", False)),
            meta=bool(d.get("metaKey", False)),

            touches=list(d.get("touches", [])),
            touch_count=int(d.get("touchCount", 0)),

            pan_dx=float(d.get("pan_dx", 0.0)),
            pan_dy=float(d.get("pan_dy", 0.0)),
            zoom_factor=float(d.get("zoom_factor", 1.0)),
            rotation_rad=float(d.get("rotation_rad", 0.0)),
            has_two_finger=bool(d.get("hasTwoFinger", False)),

            delta_x=float(d.get("deltaX", 0.0)),
            delta_y=float(d.get("deltaY", 0.0)),
            delta_z=float(d.get("deltaZ", 0.0)),
            delta_mode=int(d.get("deltaMode", 0)),
        )

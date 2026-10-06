from dataclasses import dataclass

class InputHandler:

    def __init__(self, tools):
        self.tools = tools

    async def on_pointer(self, e):
        e = PointerEvent.from_dict(e.args['detail'])

        for t in self.tools:
            await t.on_pointer(e)

    async def on_key(self, e):
        for t in self.tools:
            await t.on_key(e)

@dataclass(slots=True)
class PointerEvent:
    event_type: str  # "down" | "move" | "up" | "cancel" | "wheel"

    x: float
    y: float

    pointer_type: str  # "mouse" | "touch" | "pen"
    button: int

    shift: bool
    ctrl: bool
    alt: bool

    touch_count: int

    # Two-finger gesture change since the previous event
    zoom_factor: float = 1.0
    rotation_rad: float = 0.0

    delta_y: float = 0.0

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
    def mouse(self):
        return self.pointer_type == "mouse"

    @property
    def touch(self):
        return self.pointer_type == "touch"

    @property
    def pen(self):
        return self.pointer_type == "pen"

    @property
    def primary(self):
        return (self.mouse and self.button == 0) or (self.pen and not self.eraser)

    @property
    def eraser(self):
        # Button 5 is a pen's eraser end
        return (self.mouse and self.button == 2) or (self.pen and self.button == 5)

    @property
    def one_finger(self):
        return self.touch_count == 1

    @property
    def two_finger(self):
        return self.touch_count == 2

    @classmethod
    def from_dict(cls, d):
        return cls(
            event_type=d.get("type", "move"),
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),

            pointer_type=d.get("pointerType", "mouse"),
            button=int(d.get("button", 0)),

            shift=bool(d.get("shiftKey", False)),
            ctrl=bool(d.get("ctrlKey", False)),
            alt=bool(d.get("altKey", False)),

            touch_count=int(d.get("touchCount", 0)),

            zoom_factor=float(d.get("zoom_factor", 1.0)),
            rotation_rad=float(d.get("rotation_rad", 0.0)),

            delta_y=float(d.get("deltaY", 0.0)),
        )

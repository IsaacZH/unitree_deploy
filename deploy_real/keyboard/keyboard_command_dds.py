from dataclasses import dataclass
import time

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class KeyboardCommand_(idl.IdlStruct, typename="KeyboardCommand_"):
    stamp_sec: types.uint32
    stamp_nanosec: types.uint32
    buttons: types.sequence[types.uint8]
    lx: types.float32
    ly: types.float32
    rx: types.float32
    ry: types.float32
    has_target_update: types.uint8
    target_world: types.sequence[types.float32]


def create_keyboard_command_message(buttons, lx, ly, rx, ry, target_world=None):
    t = time.time()
    has_target_update = target_world is not None
    target = [0.0, 0.0, 0.0]
    if has_target_update:
        target = [float(v) for v in target_world[:3]]

    return KeyboardCommand_(
        stamp_sec=int(t),
        stamp_nanosec=int((t % 1) * 1e9),
        buttons=[int(v) for v in buttons[:16]],
        lx=float(lx),
        ly=float(ly),
        rx=float(rx),
        ry=float(ry),
        has_target_update=1 if has_target_update else 0,
        target_world=target,
    )

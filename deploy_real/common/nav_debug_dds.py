from dataclasses import dataclass
import time

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class NavDebug_(idl.IdlStruct, typename="NavDebug_"):
    stamp_sec: types.uint32
    stamp_nanosec: types.uint32
    target_dir_b: types.sequence[types.float32]
    target_speed_b: types.sequence[types.float32]


def create_nav_debug_message(target_dir_b, target_speed_b):
    t = time.time()
    return NavDebug_(
        stamp_sec=int(t),
        stamp_nanosec=int((t % 1) * 1e9),
        target_dir_b=[float(v) for v in target_dir_b[:3]],
        target_speed_b=[float(v) for v in target_speed_b[:3]],
    )

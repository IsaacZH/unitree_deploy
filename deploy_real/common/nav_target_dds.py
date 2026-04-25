from dataclasses import dataclass
import time

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class NavTarget_(idl.IdlStruct, typename="NavTarget_"):
    stamp_sec: types.uint32
    stamp_nanosec: types.uint32
    target_world: types.sequence[types.float32]


def create_nav_target_message(target_world):
    t = time.time()
    return NavTarget_(
        stamp_sec=int(t),
        stamp_nanosec=int((t % 1) * 1e9),
        target_world=[float(v) for v in target_world[:3]],
    )


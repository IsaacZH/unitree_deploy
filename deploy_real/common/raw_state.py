from dataclasses import dataclass
from typing import Any


@dataclass
class RawState:
    remote: Any
    imu: Any
    motor_state: Any
    high_state: Any
    navigation_manager: Any
    last_action: Any
    nav_last_action: Any
    depth_feature: Any

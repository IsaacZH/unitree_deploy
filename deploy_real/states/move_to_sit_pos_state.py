import time
import numpy as np
from typing import Optional
from common.command_helper import create_zero_cmd
from .base_state import BaseState


class MoveToSitPosState(BaseState):
    """State that moves the robot to sit position using interpolation."""

    def __init__(self, controller):
        super().__init__(controller)
        self.start_time = None
        self.kps = None
        self.kds = None
        self.ts = None
        self.qs = None
        self._completion_logged = False

    def enter(self):
        print("Moving to sit pos.")

        create_zero_cmd(self.controller.low_cmd)
        self.controller.send_cmd(self.controller.low_cmd)
        time.sleep(self.config.control_dt)

        self.kps = self.config.sitdown_kp
        self.kds = self.config.sitdown_kd
        self.ts = self.config.sitdown_ts
        self.qs = self.config.sitdown_qs

        dof_size = len(self.config.joint_ids_map)
        q0 = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            q0[i] = self.controller.low_state.motor_state[i].q
        self.qs[0] = q0.tolist()

        self.start_time = time.time()
        self._completion_logged = False

    def execute(self) -> Optional[str]:
        elapsed_time = time.time() - self.start_time
        q = self._linear_interpolate(elapsed_time, self.ts, self.qs)

        dof_size = len(self.config.joint_ids_map)
        for j in range(dof_size):
            self.controller.low_cmd.motor_cmd[j].q = q[j]
            self.controller.low_cmd.motor_cmd[j].qd = 0
            self.controller.low_cmd.motor_cmd[j].kp = self.kps[j]
            self.controller.low_cmd.motor_cmd[j].kd = self.kds[j]
            self.controller.low_cmd.motor_cmd[j].tau = 0

        self.controller.send_cmd(self.controller.low_cmd)

        if elapsed_time >= self.ts[-1]:
            if not self._completion_logged:
                print("Completed moving to sit position.")
                self._completion_logged = True
            return "zero_torque"

        time.sleep(self.config.control_dt)
        return None

    def exit(self):
        print("Exit move_to_sit_pos state.")

    def _linear_interpolate(self, t, ts, qs):
        if t <= ts[0]:
            return np.array(qs[0], dtype=np.float32)
        if t >= ts[-1]:
            return np.array(qs[-1], dtype=np.float32)

        for i in range(len(ts) - 1):
            if t >= ts[i] and t <= ts[i + 1]:
                alpha = (t - ts[i]) / (ts[i + 1] - ts[i])
                result = []
                for j in range(len(qs[i])):
                    result.append(qs[i][j] * (1 - alpha) + qs[i + 1][j] * alpha)
                return np.array(result, dtype=np.float32)

        return np.array(qs[-1], dtype=np.float32)

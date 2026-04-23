import time
from typing import Optional
from common.command_helper import create_zero_cmd
from common.remote_controller import KeyMap
from .base_state import BaseState


class ZeroTorqueState(BaseState):
    """State that applies zero torque and waits for start signal."""

    def enter(self):
        """Enter zero torque state and reset."""
        print("Enter zero torque state.")
        print("Waiting for the start signal...")

    def execute(self) -> Optional[str]:
        """
        Execute zero torque control.

        Returns:
            "move_to_default_pos" when start button is pressed.
        """
        # Send zero torque command
        create_zero_cmd(self.controller.low_cmd)
        self.controller.send_cmd(self.controller.low_cmd)

        # Check for start signal
        if self.controller.remote_controller.button[KeyMap.start] == 1:
            return "move_to_default_pos"

        time.sleep(self.config.control_dt)
        return None

    def exit(self):
        """Exit zero torque state."""
        print("Exit zero torque state.")

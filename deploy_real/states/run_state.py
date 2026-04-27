import time
import torch
from typing import Optional
from common.remote_controller import KeyMap
from common.command_helper import create_damping_cmd
from common.raw_state import RawState
from .base_state import BaseState


class RunState(BaseState):
    """Main control loop state that runs the policy."""

    def enter(self):
        """Enter run state."""
        print("Enter run state. Starting policy control.")

    def execute(self) -> Optional[str]:
        """
        Execute the control loop with policy inference.

        Returns:
            "exit" when select button is pressed, otherwise None.
        """
        # Build raw state from robot data
        raw_state = self._build_raw_state()

        # Get observations from encoder and current state
        current_obs, encoder_output = self.controller.obs_manager.forward(raw_state)
        if raw_state.nav_last_action is not None:
            self.controller.nav_last_action = raw_state.nav_last_action.copy()
        self.controller.publish_nav_debug(raw_state)

        # Assemble policy input from observations
        policy_obs = self.controller.policy_input_manager.forward(current_obs, encoder_output)
        obs_tensor = torch.from_numpy(policy_obs.copy()).unsqueeze(0)

        # Run policy inference
        action = self.controller.actor(obs_tensor).detach().numpy().squeeze()

        # Process action (clip, scale, offset)
        target_dof_pos = self.controller.action_manager.forward(action)

        # Write commands to all joints
        for i, motor_idx in enumerate(self.config.joint_ids_map):
            self.controller.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.controller.low_cmd.motor_cmd[motor_idx].qd = 0
            self.controller.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i]
            self.controller.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
            self.controller.low_cmd.motor_cmd[motor_idx].tau = 0

        # Update observation manager with last action
        self.controller.obs_manager.set_last_action(action)

        # Send command to robot
        self.controller.send_cmd(self.controller.low_cmd)

        # B button triggers sit-down sequence.
        if self.controller.remote_controller.button[KeyMap.B] == 1:
            return "move_to_sit_pos"

        # Check for exit signal
        if self.controller.remote_controller.button[KeyMap.select] == 1:
            return "exit"

        time.sleep(self.config.control_dt)
        return None

    def exit(self):
        """Exit run state and apply damping."""
        print("Exit run state.")
        create_damping_cmd(self.controller.low_cmd)
        self.controller.send_cmd(self.controller.low_cmd)

    def _build_raw_state(self):
        """Build raw state object from robot data."""
        return RawState(
            remote=self.controller.remote_controller,
            imu=self.controller.low_state.imu_state,
            motor_state=self.controller.low_state.motor_state,
            high_state=self.controller.high_state,
            navigation_manager=self.controller.navigation_manager,
            last_action=self.controller.obs_manager.last_action,
            nav_last_action=getattr(self.controller, "nav_last_action", None),
            depth_feature=(
            self.controller.depth_observer.get_latest()
            if self.controller.depth_observer is not None
            else None
            ),
        )

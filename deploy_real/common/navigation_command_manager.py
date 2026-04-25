import numpy as np
import torch
import time
from scipy.spatial.transform import Rotation as R

from common.remote_controller import KeyMap
from common.term_pipeline import collect_scaled_terms


class NavigationCommandManager:
    def __init__(self, navigation_cfg, terms_registry):
        self._cfg = navigation_cfg
        self._terms = terms_registry

        self.enabled = bool(self._cfg["enabled"])
        self.control_source = "remote"
        self._switch_button_prev = 0
        self._switch_button_id = self._resolve_switch_button(self._cfg["switch_button"])

        self._policy = None
        self._policy_expected_obs_dim = None
        self._policy_device = self._cfg["device"]
        self._command_rate_hz = float(self._cfg["command_rate_hz"])
        if self._command_rate_hz <= 0.0:
            raise ValueError(f"navigation.command_rate_hz must be > 0, got {self._command_rate_hz}")
        self._command_period_s = 1.0 / self._command_rate_hz
        self._last_command_time = 0.0
        self._has_cached_command = False
        self._cached_navigation_command = np.zeros(3, dtype=np.float32)
        self._fixed_target = np.asarray(self._cfg["fixed_target_position"], dtype=np.float32)
        self._use_raw_actions = bool(self._cfg["use_raw_actions"])
        self._action_scale = np.asarray(self._cfg["action_scale"], dtype=np.float32).ravel()
        self._action_offset = np.asarray(self._cfg["action_offset"], dtype=np.float32).ravel()
        self._policy_scaling = np.asarray(self._cfg["policy_scaling"], dtype=np.float32).ravel()
        self._policy_bias = np.asarray(self._cfg["policy_bias"], dtype=np.float32).ravel()
        self._policy_distr_type = str(self._cfg["policy_distr_type"])
        self._low_pass_filter_cfg = self._cfg["low_pass_filter"]
        self._low_pass_enabled = bool(self._low_pass_filter_cfg["enabled"])
        self._low_pass_alpha = np.asarray(self._low_pass_filter_cfg["alpha"], dtype=np.float32).ravel()
        self._prev_filtered_velocity_command = np.zeros(3, dtype=np.float32)
        if self._action_scale.shape[0] != 3:
            raise ValueError(
                f"navigation.action_scale must be 3D [sx, sy, sw], got shape {self._action_scale.shape}"
            )
        if self._action_offset.shape[0] != 3:
            raise ValueError(
                f"navigation.action_offset must be 3D [ox, oy, ow], got shape {self._action_offset.shape}"
            )
        if self._policy_scaling.shape[0] != 3:
            raise ValueError(
                f"navigation.policy_scaling must be 3D [vx, vy, wz], got shape {self._policy_scaling.shape}"
            )
        if self._policy_bias.shape[0] != 3:
            raise ValueError(
                f"navigation.policy_bias must be 3D [bx, by, bw], got shape {self._policy_bias.shape}"
            )
        if self._low_pass_alpha.shape[0] != 3:
            raise ValueError(
                f"navigation.low_pass_filter.alpha must be 3D [ax, ay, aw], got shape {self._low_pass_alpha.shape}"
            )

        self._policy_input = self._cfg["policy_input"]
        self._input_names = list(self._policy_input.keys())
        self._input_clips = [self._policy_input[name]["clip"] for name in self._input_names]
        self._input_scales = [self._policy_input[name]["scale"] for name in self._input_names]
        self._load_policy()

    def _resolve_switch_button(self, button_name):
        if isinstance(button_name, int):
            return int(button_name)
        if isinstance(button_name, str) and hasattr(KeyMap, button_name):
            return int(getattr(KeyMap, button_name))
        raise ValueError(f"Invalid navigation.switch_button: {button_name}")

    def _load_policy(self):
        if not self.enabled:
            return
        policy_path = self._cfg["policy_path"]
        if not policy_path:
            raise ValueError("navigation.policy_path is required when navigation.enabled is true.")
        nav_module = torch.jit.load(policy_path, map_location=self._policy_device)
        self._policy = nav_module
        self._policy.eval()
        for p in self._policy.parameters():
            p.requires_grad_(False)
        if hasattr(nav_module, "num_image_features") and hasattr(nav_module, "actor_proprioceptive_input_dim"):
            self._policy_expected_obs_dim = int(nav_module.num_image_features) + int(
                nav_module.actor_proprioceptive_input_dim
            )
        print(f"[Navigation] Loaded navigation policy: {policy_path}")

    def update_control_source(self, buttons):
        if not self.enabled:
            return
        switch_pressed = int(buttons[self._switch_button_id])
        if switch_pressed == 1 and self._switch_button_prev == 0:
            self.control_source = "navigation" if self.control_source == "remote" else "remote"
            print(f"[Navigation] Control source -> {self.control_source}")
        self._switch_button_prev = switch_pressed

    def use_navigation_command(self):
        return self.enabled and self.control_source == "navigation"

    def get_target_position(self):
        return self._fixed_target.copy()

    def set_target_position(self, target):
        target = np.asarray(target, dtype=np.float32).ravel()
        if target.shape[0] != 3:
            raise ValueError(f"target_position must be 3D [x, y, z], got shape {target.shape}")
        self._fixed_target = target.copy()

    def reset_policy(self):
        if self._policy is None:
            raise RuntimeError("Navigation policy is not loaded.")
        print("[Navigation] Resetting navigation policy.")
        self._policy.reset()
        self._prev_filtered_velocity_command[:] = 0.0
        self._has_cached_command = False
        self._last_command_time = 0.0

    @staticmethod
    def compute_goal_command_body(goal_position_world, robot_position_world, robot_quat_wxyz):
        """Match IsaacLab goal command update:
        goal_in_body -> direction(3) + log_distance(1).
        """
        goal_w = np.asarray(goal_position_world, dtype=np.float32).ravel()
        robot_w = np.asarray(robot_position_world, dtype=np.float32).ravel()
        quat_wxyz = np.asarray(robot_quat_wxyz, dtype=np.float32).ravel()
        if goal_w.shape[0] != 3:
            raise ValueError(f"goal_position_world must be 3D [x, y, z], got shape {goal_w.shape}")
        if robot_w.shape[0] != 3:
            raise ValueError(f"robot_position_world must be 3D [x, y, z], got shape {robot_w.shape}")
        if quat_wxyz.shape[0] != 4:
            raise ValueError(f"robot_quat_wxyz must be 4D [w, x, y, z], got shape {quat_wxyz.shape}")

        # SciPy uses xyzw order.
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        goal_in_body = R.from_quat(quat_xyzw).inv().apply(goal_w - robot_w).astype(np.float32)
        distance = np.linalg.norm(goal_in_body)
        direction = goal_in_body / max(distance, 1e-6)
        log_distance = np.array([np.log(distance + 1.0)], dtype=np.float32)
        return np.concatenate([direction.astype(np.float32), log_distance], axis=0), goal_in_body

    def _term_value_provider(self, name, raw_state, config):
        return self._terms[name](raw_state, config)

    def _apply_low_pass_filter(self, velocity_command):
        if not self._low_pass_enabled:
            return velocity_command
        alpha = self._low_pass_alpha
        filtered = alpha * self._prev_filtered_velocity_command + (1.0 - alpha) * velocity_command
        self._prev_filtered_velocity_command = filtered.astype(np.float32).copy()
        return filtered

    def build_navigation_obs(self, raw_state, config):
        nav_obs = collect_scaled_terms(
            raw_state=raw_state,
            config=config,
            names=self._input_names,
            clips=self._input_clips,
            scales=self._input_scales,
            value_provider=self._term_value_provider,
        )
        return np.concatenate(nav_obs, axis=0).astype(np.float32)

    def get_navigation_velocity_command(self, raw_state, config):
        target = np.asarray(self.get_target_position(), dtype=np.float32).ravel()
        if target.shape[0] != 3:
            raise ValueError(
                f"navigation.fixed_target_position must be 3D [x, y, z], got shape {target.shape}"
            )
        now = time.monotonic()
        if self._has_cached_command and (now - self._last_command_time) < self._command_period_s:
            return self._cached_navigation_command.copy()

        nav_obs = self.build_navigation_obs(raw_state, config)
        if self._policy_expected_obs_dim is not None and nav_obs.shape[0] != self._policy_expected_obs_dim:
            raise ValueError(
                "Navigation obs dim mismatch: "
                f"built={nav_obs.shape[0]} expected={self._policy_expected_obs_dim}. "
                "Please align navigation.policy_input terms with training play export."
            )
        nav_obs_tensor = torch.from_numpy(nav_obs).unsqueeze(0).to(self._policy_device)
        with torch.inference_mode():
            nav_cmd = self._policy(nav_obs_tensor).cpu().numpy().squeeze()

        nav_cmd = np.asarray(nav_cmd, dtype=np.float32).ravel()
        if nav_cmd.shape[0] != 3:
            raise ValueError(f"Navigation command must be 3D (vx, vy, wz), got shape {nav_cmd.shape}")

        if not self._use_raw_actions:
            nav_cmd = nav_cmd * self._action_scale + self._action_offset

        if self._policy_distr_type == "gaussian":
            nav_cmd = np.tanh(nav_cmd)
        elif self._policy_distr_type == "beta":
            nav_cmd = (nav_cmd - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown navigation.policy_distr_type: {self._policy_distr_type}")

        if hasattr(raw_state, "high_state") and raw_state.high_state is not None and hasattr(raw_state.high_state, "velocity"):
            base_lin_vel = np.asarray(raw_state.high_state.velocity, dtype=np.float32).ravel()[:3]
            vel_xyz = float(np.linalg.norm(base_lin_vel))
        else:
            vel_xyz = 0.0

        nav_cmd = (nav_cmd + vel_xyz * self._policy_bias) * self._policy_scaling
        nav_cmd = self._apply_low_pass_filter(nav_cmd)
        self._cached_navigation_command = np.asarray(nav_cmd, dtype=np.float32).copy()
        self._has_cached_command = True
        self._last_command_time = now
        return self._cached_navigation_command.copy()

import numpy as np
import torch

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
        self._kp = np.asarray(self._cfg["kp"], dtype=np.float32)
        self._fixed_target = np.asarray(self._cfg["fixed_target_position"], dtype=np.float32)
        self._max_speed = np.asarray(self._cfg["max_speed"], dtype=np.float32)

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
            print("[Navigation] No policy_path configured. Using fixed-target proportional command.")
            return
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
        print("[Navigation] Resetting navigation policy.")
        self._policy.reset()

    def _term_value_provider(self, name, raw_state, config):
        return self._terms[name](raw_state, config)

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

        if self._policy is not None:
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
        else:
            nav_cmd = self._kp * target[:3]

        nav_cmd = np.asarray(nav_cmd, dtype=np.float32).ravel()
        if nav_cmd.shape[0] != 3:
            raise ValueError(f"Navigation command must be 3D (vx, vy, wz), got shape {nav_cmd.shape}")
        return np.clip(nav_cmd, -self._max_speed, self._max_speed)

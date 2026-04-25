import numpy as np
import torch
import torch.nn.functional as F


class ObservationManager:
    def __init__(self, config, term_registry, encoder):
        self.config = config
        self.term_registry = term_registry
        self.encoder = encoder

        self.use_encoder = config.use_encoder
        self.history_length = config.encoder_input.get("history_length")

        self._build_obs_definitions()

        self.history_buffer = None
        self.last_action = np.zeros(len(config.default_joint_pos), dtype=np.float32)
        self.reset()

    def _build_obs_definitions(self):
        self.obs_names = []
        self.obs_clips = []
        self.obs_scales = []
        self.obs_sizes = []

        for name, obs_config in self.config.encoder_input.items():
            if name == "history_length":
                continue
            self.obs_names.append(name)
            self.obs_clips.append(obs_config["clip"])
            self.obs_scales.append(obs_config["scale"])

        self.obs_offsets = {}
        offset = 0
        for name in self.obs_names:
            size = self._get_term_size(name)
            self.obs_offsets[name] = offset
            self.obs_sizes.append(size)
            offset += size
        self.num_obs_current = offset

    def _get_term_size(self, name):
        dof = len(self.config.default_joint_pos)
        if name in ("velocity_commands", "base_ang_vel", "projected_gravity"):
            return 3
        if name in ("joint_pos_rel", "joint_vel_rel", "last_action"):
            return dof
        if name == "depth_image":
            if self.config.depth_camera is None:
                raise ValueError("depth_image term requires depth_camera config")
            if "feature_dim" in self.config.depth_camera and "resolution" in self.config.depth_camera:
                width, height = self.config.depth_camera["resolution"]
                # Depth encoder outputs spatial features (C, H', W') that are flattened.
                # For the current encoder stack and 64x40 input this is 64x5x8.
                fmap_w = max(1, int(width) // 8)
                fmap_h = max(1, int(height) // 8)
                return int(self.config.depth_camera["feature_dim"]) * fmap_w * fmap_h
            if "feature_dim" in self.config.depth_camera:
                return int(self.config.depth_camera["feature_dim"])
            if "resolution" in self.config.depth_camera:
                width, height = self.config.depth_camera["resolution"]
                return int(width) * int(height)
            raise ValueError("depth_image term size cannot be inferred from depth_camera config")
        raise ValueError(f"Unsupported observation term for static sizing: {name}")

    def reset(self):
        self.history_buffer = np.zeros(self.history_length * self.num_obs_current, dtype=np.float32)

    def set_last_action(self, action):
        self.last_action = action.copy()

    def _apply_clip_scale(self, values, clip, scale):
        values = np.clip(values, clip[0], clip[1])
        values = values * np.array(scale, dtype=np.float32)
        return values

    def _get_term_value(self, name):
        if name == "last_action":
            return self.last_action
        term_fn = self.term_registry[name]
        return term_fn(self.raw_state, self.config)

    def get_obs_slice(self, name):
        offset = self.obs_offsets[name]
        idx = self.obs_names.index(name)
        size = self.obs_sizes[idx]
        return slice(offset, offset + size)

    def forward(self, raw_state):
        self.raw_state = raw_state

        obs_parts = []
        for i, name in enumerate(self.obs_names):
            values = self._get_term_value(name)
            if not isinstance(values, np.ndarray):
                values = np.array(values, dtype=np.float32)
            values = self._apply_clip_scale(values, self.obs_clips[i], self.obs_scales[i])
            obs_parts.append(values)

        current_obs = np.concatenate(obs_parts).astype(np.float32)

        if self.use_encoder:
            self.history_buffer = np.concatenate((current_obs, self.history_buffer[:-self.num_obs_current]))
            obs_encoder_tensor = torch.from_numpy(self.history_buffer.copy()).unsqueeze(0).float()
            encoder_output = self.encoder(obs_encoder_tensor)
            return current_obs, encoder_output
        else:
            return current_obs, None


class PolicyInputManager:
    def __init__(self, config, obs_offsets, obs_sizes, obs_names):
        self.config = config
        self.obs_offsets = obs_offsets
        self.obs_sizes = obs_sizes
        self.obs_names = obs_names
        self._build_definitions()

    def _build_definitions(self):
        self.input_names = []
        self.input_clips = []
        self.input_scales = []

        for name, pi_config in self.config.policy_input.items():
            self.input_names.append(name)
            self.input_clips.append(pi_config["clip"])
            self.input_scales.append(pi_config["scale"])

    def _apply_clip_scale(self, values, clip, scale):
        values = np.clip(values, clip[0], clip[1])
        values = values * np.array(scale, dtype=np.float32)
        return values

    def forward(self, current_obs, encoder_output):
        parts = []

        for i, name in enumerate(self.input_names):
            if name == "encoder_output":
                enc = encoder_output.detach().squeeze(0)
                vel = enc[:3]
                latent = F.normalize(enc[3:], dim=-1, p=2)
                values = torch.cat((vel, latent), dim=-1).numpy()
                values = self._apply_clip_scale(values, self.input_clips[i], self.input_scales[i])
            else:
                offset = self.obs_offsets[name]
                idx = self.obs_names.index(name)
                size = self.obs_sizes[idx]
                # Terms in current_obs are already clip/scale processed in ObservationManager.
                values = current_obs[offset:offset + size]
                # Keep single scaling pass to match pre-refactor behavior.
            parts.append(values)

        return np.concatenate(parts)

import numpy as np


class ActionManager:
    def __init__(self, config):
        self.config = config
        self._build_action_definitions()

    def _build_action_definitions(self):
        for action_type, action_config in self.config.actions.items():
            self.action_type = action_type
            self.action_clip = action_config["clip"]
            self.action_scale = action_config["scale"]
            self.action_offset = np.array(action_config["offset"], dtype=np.float32)
            self.joint_ids = action_config.get("joint_ids")
            self.joint_names = action_config.get("joint_names", [".*"])
            break

    def _apply_clip_scale(self, values, clip, scale):
        values = np.clip(values, clip[0], clip[1])
        values = values * np.array(scale, dtype=np.float32)
        return values

    def forward(self, policy_action):
        action = policy_action.copy()

        clip_min = np.array([c[0] for c in self.action_clip], dtype=np.float32)
        clip_max = np.array([c[1] for c in self.action_clip], dtype=np.float32)
        action = np.clip(action, clip_min, clip_max)
        action = action * np.array(self.action_scale, dtype=np.float32)

        target_dof_pos = self.config.default_joint_pos + action

        return target_dof_pos
